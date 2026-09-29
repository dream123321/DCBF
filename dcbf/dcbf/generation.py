from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ase.io import iread, write
import yaml
from tqdm import tqdm

from .candidate_pool import (
    CandidatePool,
    count_cfg_structures,
    generation_selection_added_marker_path,
    generation_selection_cache_path,
    resolve_candidate_trigger,
)
from .das.calc_ensemble_ambiguity import ambiguity_extract, check_lmp_error, get_force_ambiguity
from .das.das_update_ambiguity import (
    af_limit_record,
    af_limit_update,
    das_update_ambiguity,
    record_yaml,
)
from .das.logger import setup_logger
from .das.main_calc import check_and_modify_calc_dir, main_calc
from .das.mkdir import mkdir_vasp
from .das.other import end_yaml, mkdir, remove, touch
from .das.sample_xyz import sample_main
from .das.scf_lmp_data import collect_dft_data
from .das.train_mlp import (
    pre_train_mlp,
    start_train,
    training_task_dirs,
    update_mlp_from_current_batch,
)
from .das.work_dir import (
    bsub_dir,
    check_filter_xyz_0,
    check_finish,
    check_scf,
    delete_dump,
    scf_dir,
    submit_lammps_task,
    work_deepest_dir,
)
from .encode.mlp_encode_sample_flow import main_sample_flow
from .bootstrap import WorkspaceBootstrapper
from .high_precision_training import HighPrecisionTrainer
from .npt_volume_filter import (
    build_volume_intake_stats,
    case_dir_context,
    format_volume_summary_message,
    format_volume_truncation_message,
    load_npt_volume_filter_report,
    new_frame_volume_detail,
    npt_frame_volume_guard,
    truncated_frame_count,
    write_npt_volume_filter_report,
)
from .path_names import DFT_WORK_DIR, MD_WORK_DIR, SUS2_MODEL_DIR
from .runtime_config import build_scheduler_spec, load_runtime_config
from .training_dataset import TrainingDatasetStore
from .raw_dft_archive import RawDFTArchiveManager, load_collection_cache, write_collection_cache
from .das.scf_filter_sources import load_scf_filter_sources


GENERATION_DESCRIPTOR_STORE_PATHS = (
    ("train_mlp", "database_descriptor_store"),
    ("train_mlp", "gen_0_database_descriptor_store"),
    (MD_WORK_DIR, "md_descriptor_store"),
    (MD_WORK_DIR, "gen_0_md_descriptor_store"),
)


def cleanup_generation_descriptor_stores(workspace, logger):
    workspace = Path(workspace)
    removed = []
    failed = []
    for relative_parts in GENERATION_DESCRIPTOR_STORE_PATHS:
        path = workspace.joinpath(*relative_parts)
        if not path.exists() and not path.is_symlink():
            continue
        try:
            if path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                raise OSError("expected a descriptor-store directory")
            removed.append(str(path))
        except OSError as exc:
            failed.append({"path": str(path), "error": str(exc)})
            logger.warning(
                "Generation descriptor-store cleanup failed: path=%s error=%s",
                path,
                exc,
            )
    if removed:
        logger.info(
            "Generation descriptor stores cleaned: removed=%s failed=%s",
            len(removed),
            len(failed),
        )
    return {"removed": removed, "failed": failed}


class GenerationRunner:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.run_root = self.workspace.parent.parent
        self.logger = setup_logger()
        self.parameter = self._load_parameter_yaml()
        self.runtime_config = load_runtime_config(self.workspace)
        self.scheduler = build_scheduler_spec(self.runtime_config["scheduler"])
        self.scf2xyz = self._resolve_scf_handler()
        self.candidate_pool = CandidatePool(self.run_root)

    def _load_parameter_yaml(self):
        with open(self.workspace / "parameter.yaml", "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return WorkspaceBootstrapper.apply_parameter_defaults(data)

    def _resolve_scf_handler(self):
        scf_cal_engine = self.scheduler.scf_cal_engine
        if scf_cal_engine == "abacus":
            from .das.abacus_main_xyz import abacus_main_xyz as scf2xyz
        elif scf_cal_engine == "cp2k":
            from .das.cp2k_main_xyz import cp2k_main_xyz as scf2xyz
        elif scf_cal_engine == "qe":
            from .das.qe_main_xyz import qe_main_xyz as scf2xyz
        elif scf_cal_engine == "vasp":
            from .das.vasp_main_xyz import vasp_main_xyz as scf2xyz
        else:
            raise ValueError(f"{scf_cal_engine} does not exist")
        return scf2xyz

    @property
    def generation_index(self):
        return int(self.workspace.name.replace("gen_", ""))

    def _load_end_state(self):
        try:
            return end_yaml(str(self.workspace / "end.yaml"))
        except Exception:
            return None, None, None, None, None

    def _validate_training_outputs(self):
        gen_num = self.generation_index
        train_dirs = []
        for train_dir in training_task_dirs(self.workspace):
            logout_path = Path(train_dir) / "logout"
            if logout_path.exists():
                train_dirs.append(train_dir)

        if not train_dirs:
            return

        if gen_num == 0 and len(train_dirs) == 1:
            return

        for train_dir in train_dirs:
            log_file = os.path.join(train_dir, "logout")
            with open(log_file, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
            for line in lines:
                if "Killed" in line:
                    self.logger.info(f"Scaling_mlp training error, please check! : {log_file}")
                    touch(str(self.workspace), "__error__")
                    raise RuntimeError(f"Scaling_mlp training error: {log_file}")
            for line in lines[-25:]:
                if "nan" in line:
                    self.logger.info(f"Scaling_mlp training error, please check! : {log_file}")
                    touch(str(self.workspace), "__error__")
                    raise RuntimeError(f"Scaling_mlp training error: {log_file}")

    def _collect_training_metrics_from_workspace(self, workspace: Path):
        workspace = Path(workspace).resolve()
        train_root = workspace / "train_mlp"
        if not train_root.exists():
            return []

        metrics_list = []
        for train_dir in training_task_dirs(workspace):
            logout_path = Path(train_dir) / "logout"
            if not logout_path.exists():
                continue
            metrics = HighPrecisionTrainer._extract_training_metrics(logout_path)
            if any(value is not None for value in metrics.values()):
                metrics_list.append(metrics)
        return metrics_list

    @staticmethod
    def _average_training_metrics(metrics_list):
        averaged = {
            "energy_mae_mev_per_atom": None,
            "force_mae_mev_per_a": None,
            "stress_mae_ev": None,
        }
        for key in averaged:
            values = [metrics[key] for metrics in metrics_list if metrics.get(key) is not None]
            if values:
                averaged[key] = sum(values) / len(values)
        return averaged

    def _emit_training_summary(self, metrics_list, log_tag="training.summary"):
        if not metrics_list:
            return False
        metrics = self._average_training_metrics(metrics_list)
        parts = []
        if metrics["energy_mae_mev_per_atom"] is not None:
            parts.append(f"energy_mae={metrics['energy_mae_mev_per_atom']:.3f} meV/atom")
        if metrics["force_mae_mev_per_a"] is not None:
            parts.append(f"force_mae={metrics['force_mae_mev_per_a']:.3f} meV/A")
        if metrics["stress_mae_ev"] is not None:
            parts.append(f"stress_mae={metrics['stress_mae_ev']:.3f} eV")

        if parts:
            self.logger.info(f"[{log_tag}] {' '.join(parts)}")
            return True
        return False

    def _emit_gen0_training_summary(self):
        if self.generation_index != 0:
            return

        metrics_list = self._collect_training_metrics_from_workspace(self.workspace)
        if not metrics_list:
            main_num = int(self.workspace.parent.name.replace("main_", ""))
            if main_num > 0:
                previous_main_dir = self.workspace.parent.parent / f"main_{main_num - 1}"
                if previous_main_dir.exists():
                    previous_gen_dirs = sorted(
                        [
                            path
                            for path in previous_main_dir.iterdir()
                            if path.is_dir() and path.name.startswith("gen_")
                        ],
                        key=lambda path: int(path.name.replace("gen_", "")),
                    )
                    if previous_gen_dirs:
                        metrics_list = self._collect_training_metrics_from_workspace(previous_gen_dirs[-1])

        self._emit_training_summary(metrics_list, log_tag="training.summary")

    def _emit_candidate_update_training_summary(self):
        metrics_list = self._collect_training_metrics_from_workspace(self.workspace)
        self._emit_training_summary(metrics_list, log_tag="training.summary.candidate_update")

    def _train_models(self):
        ele = self.parameter["ele"]
        ele_model = 1 if self.parameter["sort_ele"] else 2
        mlp_nums = 1 if self.parameter["mlp_encode_model"] else self.parameter["mlp_nums"]
        if not self.parameter["mlp_encode_model"] and mlp_nums < 3:
            raise ValueError("During the use of DAS, mlp_nums should be greater than or equal to 3")

        label = pre_train_mlp(
            str(self.workspace),
            mlp_nums,
            ele,
            ele_model,
            self.logger,
            self.scheduler,
        )
        if label:
            start_train(str(self.workspace), self.scheduler.task_submission_method, mlp_nums, self.logger)
        self._validate_training_outputs()
        self._emit_gen0_training_summary()

    def _prepare_md(self):
        mkdir_vasp(
            str(self.workspace),
            self.parameter["mlp_MD"],
            self.parameter["ele"],
            tuple(eval(self.parameter["size"])),
            1 if self.parameter["mlp_encode_model"] else self.parameter["mlp_nums"],
            self.parameter["sort_ele"],
            self.parameter["nvt_lattice_scaling_factor"],
            self.parameter["mlp_encode_model"],
            self.scheduler,
        )
        dirs_1 = work_deepest_dir(str(self.workspace))
        dirs_2 = bsub_dir(str(self.workspace))
        submit_lammps_task(str(self.workspace), self.logger, self.scheduler.task_submission_method)
        check_finish(dirs_1, self.logger, "All MD calculations have been completed")
        for directory in dirs_1:
            error, message = check_lmp_error(directory)
            if error:
                self.logger.warning(f"LAMMPS runtime error detected: {message}")
        return dirs_1, dirs_2

    def _find_latest_existing_scf_filter_xyz(self):
        current_generation = self.generation_index
        for previous_index in range(current_generation - 1, -1, -1):
            candidate = self.workspace.parent / f"gen_{previous_index}" / DFT_WORK_DIR / "scf_filter.xyz"
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
        return None

    def _select_by_ambiguity(self, dirs_1, dirs_2, end_state):
        end_threshold_low, end_threshold_high, end_n, end_cluster_threshold_init, end_k = end_state
        threshold_low = self.parameter["threshold_low"]
        threshold_high = self.parameter["threshold_high"]
        sample = self.parameter["sample"]
        n = sample["n"]
        cluster_threshold_init = sample["cluster_threshold_init"]
        k = sample["k"]
        volume_factor = self.parameter["npt_max_cell_volume_filter_factor"]
        volume_details = {}
        thresholds_changed = (
            end_threshold_low != threshold_low
            or end_threshold_high != threshold_high
        )
        # Volume stability is an independent intake guard for NPT only.  It
        # must still run when ambiguity thresholds are unchanged, but an
        # NVT-only generation should retain the old fast path.
        has_npt_directory = any(
            os.path.basename(os.path.dirname(os.path.normpath(directory))).lower()
            == "npt"
            for directory in dirs_1
        )
        volume_filter_may_apply = volume_factor is not None and has_npt_directory
        need_ambiguity_pass = thresholds_changed or volume_filter_may_apply

        last_gen = "gen_" + str(self.generation_index - 1)
        last_gen_path = self.workspace.parent / last_gen
        mtp_path = self.workspace / SUS2_MODEL_DIR

        if need_ambiguity_pass:
            for directory in dirs_2:
                os.chdir(directory)
                for file_name in glob.glob("*filter*"):
                    remove(file_name)

            af_adaptive = None
            if self.parameter["das_ambiguity"] and thresholds_changed:
                if self.generation_index == 0:
                    xyz = None
                    model_fns = None
                else:
                    latest_scf_filter_xyz = self._find_latest_existing_scf_filter_xyz()
                    if latest_scf_filter_xyz is None:
                        xyz = None
                        self.logger.info(
                            "No previous scf_filter.xyz was found. Reusing the configured DAS defaults for the adaptive threshold."
                        )
                    else:
                        xyz = str(latest_scf_filter_xyz)
                        source_generation = latest_scf_filter_xyz.parent.parent.name
                        self.logger.info(
                            f"Reuse latest available scf_filter.xyz from {source_generation}: {latest_scf_filter_xyz}"
                        )
                    model_fns = glob.glob(os.path.join(mtp_path, "current*"))
                yaml_file = os.path.join(self.workspace, "parameter.yaml")
                new_af_limit = af_limit_update(str(self.workspace), yaml_file)
                af_adaptive, label = das_update_ambiguity(
                    ele=self.parameter["ele"],
                    sort_ele=self.parameter["sort_ele"],
                    af_default=self.parameter["af_default"],
                    af_limit=new_af_limit,
                    af_failed=self.parameter["af_failed"],
                    over_fitting_factor=self.parameter["over_fitting_factor"],
                    logger=self.logger,
                ).run(xyz, model_fns, self.generation_index)
                af_limit_record(str(self.workspace), label)
                threshold_low = float(af_adaptive)
                threshold_high = self.parameter["af_failed"]

            select_stru_num = 0
            for directory_name in tqdm(dirs_1):
                directory = os.path.join(self.workspace, directory_name)
                total_stru = get_force_ambiguity(directory)
                '''体积截断：膨胀的帧在写filter.xyz之前就丢掉'''
                volume_guard = npt_frame_volume_guard(directory, volume_factor)
                volume_detail = (
                    new_frame_volume_detail() if volume_guard is not None else None
                )
                num, structures, interval, hist = ambiguity_extract(
                    directory,
                    "force.0.dump",
                    "af.out",
                    threshold_low,
                    threshold_high,
                    self.parameter["ele"],
                    self.parameter["sort_ele"],
                    self.parameter["end"],
                    self.parameter["num_elements"],
                    volume_guard=volume_guard,
                    volume_detail=volume_detail,
                )
                path_parts = os.path.normpath(directory).split(os.sep)
                final_path = os.sep.join(path_parts[-3:])
                filter_path = os.path.join(os.sep.join(path_parts[:-2]), "filter.xyz")
                write(filter_path, structures, format="extxyz", append=True)
                select_stru_num += num
                error, message = check_lmp_error(directory)
                if error:
                    self.logger.warning(message)
                if volume_detail is not None:
                    volume_details[directory] = volume_detail
                    if volume_detail["dropped"]:
                        structure_name, case_name = case_dir_context(directory)
                        self.logger.warning(
                            format_volume_truncation_message(
                                structure_name,
                                case_name,
                                volume_detail,
                                volume_factor,
                                message if error else "",
                            )
                        )
                self.logger.info(
                    f"{final_path}: According to the ambiguity in the {round(threshold_low, 3)}-{threshold_high} range , "
                    f"{num} structures are selected from {total_stru} structures. Interval:{interval} Statistical number:{hist}"
                )

            volume_stats = build_volume_intake_stats(volume_details)
            write_npt_volume_filter_report(self.workspace, volume_factor, volume_stats)
            if volume_stats["removed_count"]:
                self.logger.warning(
                    format_volume_summary_message(volume_factor, volume_stats)
                )

            yaml_file = os.path.join(self.workspace, "parameter.yaml")
            adaptive_value = float(af_adaptive) if af_adaptive is not None else None
            if thresholds_changed:
                record_yaml(yaml_file, adaptive_value, int(select_stru_num))
        else:
            self.logger.info("end_threshold equals threshold: skip to select the structure by ambiguity")

        if (
            end_threshold_low == threshold_low
            and end_threshold_high == threshold_high
            and end_n == n
            and end_cluster_threshold_init == cluster_threshold_init
            and end_k == k
        ):
            self.logger.info("(threshold, n, cluster_threshold_init, k) parameters are equal: skip to select the structure by MBTR+Brich")
            # A volume-only pass can reach this early return when the
            # ambiguity settings are unchanged.  Persist the truncation
            # outcome before returning so an affected seed is not lost from
            # the next-generation MD list.
            self._keep_volume_truncated_structures_for_next_md(
                dirs_2, volume_details
            )
            return

        for directory in dirs_2:
            os.chdir(directory)
            if os.path.isfile("filter.xyz") and os.path.getsize("filter.xyz") != 0:
                num = len(list(iread("filter.xyz")))
                if n * k <= num:
                    select, total = sample_main(
                        os.getcwd(),
                        n=n,
                        threshold_init=cluster_threshold_init,
                        k=k,
                        clustering_by_ambiguity=sample["clustering_by_ambiguity"],
                    )
                    name = os.path.basename(directory)
                    self.logger.info(f"{name}: selected {select} structures from {total} structures in data by MBTR+Brich.")
                else:
                    shutil.copy("filter.xyz", f"{num}_sample_filter.xyz")
        self._keep_volume_truncated_structures_for_next_md(dirs_2, volume_details)

    def _keep_volume_truncated_structures_for_next_md(self, structure_dirs, volume_details):
        '''膨胀过的结构视为采样未收敛：没有筛出样本也要进入下一轮MD'''
        dropped_by_structure = {}
        for case_dir, detail in volume_details.items():
            if not detail["dropped"]:
                continue
            structure_dir = os.path.dirname(os.path.dirname(os.path.abspath(case_dir)))
            dropped_by_structure[structure_dir] = (
                dropped_by_structure.get(structure_dir, 0) + int(detail["dropped"])
            )
        if not dropped_by_structure:
            return

        for structure_dir in structure_dirs:
            key = os.path.abspath(str(structure_dir))
            dropped = dropped_by_structure.get(key)
            if dropped is None:
                continue
            if sorted(Path(structure_dir).glob("*_sample_filter.xyz")):
                continue
            # collect_dft_data 只看结构目录里有没有 *sample*filter* 文件，
            # 写一个空样本文件把这个结构留在下一轮的 stru.pkl 里。
            (Path(structure_dir) / "0_sample_filter.xyz").write_text("", encoding="utf-8")
            self.logger.warning(
                "%s: all %s volume-truncated frames were sampled away. Keep this "
                "structure for the next MD round",
                os.path.basename(key),
                dropped,
            )

    def _select_by_encoding(self, dirs_1):
        sample_xyz_list = glob.glob(os.path.join(self.workspace, MD_WORK_DIR, "*_sample_filter.xyz"))
        if len(sample_xyz_list) == 0:
            main_sample_flow(
                str(self.workspace),
                dirs_1,
                self.parameter["dq_width"],
                self.parameter["dq_width_method"],
                self.parameter.get("dq_width_factor", 1.0),
                self.parameter["body_list"],
                self.parameter["ele"],
                self.parameter["sort_ele"],
                self.parameter["mtp_type"],
                self.parameter["selection_budget_schedule"],
                self.parameter["coverage_threshold_schedule"],
                self.parameter["coverage_rate_method"],
                self.logger,
                self.parameter.get("report_per_configuration_details", False),
                self.parameter.get("plateau_generations"),
                self.parameter.get("min_coverage_delta"),
                self.parameter.get("state_population", 0),
                self.parameter.get("report_state_population_zero_baseline", False),
                self.parameter.get("mean_descriptor_enabled", False),
                self.parameter.get("mean_descriptor_state_population", 0),
                self.parameter.get("npt_max_cell_volume_filter_factor"),
            )
        elif len(sample_xyz_list) == 1:
            self.logger.info(f"*_sample_filter.xyz already exists.({sample_xyz_list[0]})")
        else:
            raise ValueError("Multiple *_sample_filter.xyz, Please delete!")

    def _persist_end_state(self):
        shutil.copy(self.workspace / "parameter.yaml", self.workspace / "end.yaml")

    def _load_mode_selected_atoms(self):
        if self.parameter["mlp_encode_model"]:
            selected_atoms = []
            sample_xyz_list = sorted(glob.glob(os.path.join(self.workspace, MD_WORK_DIR, "*_sample_filter.xyz")))
            for sample_xyz in sample_xyz_list:
                if os.path.getsize(sample_xyz) == 0:
                    continue
                selected_atoms.extend(list(iread(sample_xyz)))
            return selected_atoms
        return collect_dft_data("no", str(self.workspace))

    def _materialize_generation_selection(self):
        cache_path = generation_selection_cache_path(self.workspace)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return list(iread(str(cache_path), index=":"))

        selected_atoms = self._load_mode_selected_atoms()
        if selected_atoms:
            write(str(cache_path), selected_atoms, format="extxyz")
        return selected_atoms

    def _append_generation_selection_to_candidate_pool(self):
        selected_atoms = self._materialize_generation_selection()
        selected_count = len(selected_atoms)
        added_marker = generation_selection_added_marker_path(self.workspace)
        if selected_count > 0 and not added_marker.exists():
            self.candidate_pool.append_atoms(selected_atoms)
            added_marker.write_text(str(selected_count), encoding="utf-8")
        return selected_count, self.candidate_pool.count()

    def _candidate_trigger_threshold(self):
        train_cfg = TrainingDatasetStore.from_generation(self.workspace).global_cfg
        base_structure_count = count_cfg_structures(train_cfg)
        threshold, description = resolve_candidate_trigger(
            self.parameter.get("candidate_trigger"),
            base_structure_count,
        )
        return threshold, description, base_structure_count

    def _run_candidate_batch_stage(self):
        selected_count, candidate_count = self._append_generation_selection_to_candidate_pool()
        threshold, trigger_description, base_structure_count = self._candidate_trigger_threshold()

        self.logger.info(
            "candidate pool status: current_gen_selected=%s candidate_total=%s trigger=%s base_train_structures=%s",
            selected_count,
            candidate_count,
            trigger_description,
            base_structure_count,
        )

        if selected_count == 0:
            npt_filter_report = load_npt_volume_filter_report(self.workspace)
            truncated_frames = truncated_frame_count(npt_filter_report)
            if truncated_frames > 0:
                '''膨胀过的结构视为未收敛：本代没选出结构也要继续下一轮MD'''
                self.logger.info(
                    "No structures were selected in this generation, but %s MD frames were "
                    "truncated by the NPT cell-volume filter (factor=%s). Skip DFT, reuse the "
                    "current MLIP, and continue to the next generation so that the expanded "
                    "structures are sampled again",
                    truncated_frames,
                    npt_filter_report.get("factor"),
                )
                return
            self.logger.info("No structures were selected in this generation. The active learning loop ends")
            touch(str(self.workspace), "__end__")
            return

        if candidate_count < threshold:
            self.logger.info(
                "candidate pool below trigger: %s < %s. Skip DFT and reuse the previous MLIP in the next generation",
                candidate_count,
                threshold,
            )
            return

        candidate_atoms = self.candidate_pool.load_atoms()
        if not candidate_atoms:
            self.logger.info("candidate pool is empty after trigger evaluation. Skip DFT")
            return

        if not self._resume_existing_scf_if_possible():
            dft_work_path = self.workspace / DFT_WORK_DIR
            scf_path = dft_work_path / "scf"
            mkdir(str(dft_work_path))
            mkdir(str(scf_path))
            os.chdir(scf_path)
            for item in [name for name in os.listdir() if name != "total_sample_filter.xyz"]:
                remove(item)
            self._submit_scf_jobs(candidate_atoms)

        self.logger.info("Waiting for DFT/SCF tasks to complete...")
        check_finish(
            scf_dir(str(self.workspace)),
            self.logger,
            "All scf calculations have been completed",
            pending_warning_hours=self.parameter["dft"].get("pending_warning_hours"),
        )

        ok_count, len_count, no_success_path, force_count, force_of_force_count_0 = self._collect_scf_results(
            self.parameter["dft"]["force_threshold"]
        )
        with open(self.workspace / "no_success_path.json", "w", encoding="utf-8") as handle:
            json.dump(no_success_path, handle)

        gen = self.workspace.name
        if force_count == 0:
            self.logger.info(
                f"candidate batch DFT completed: {gen} | {self.scheduler.scf_cal_engine}_completed_number:{ok_count} | "
                f"Successful_collection_structure/scf_convergent_number:{len_count} | "
                f"force_threshold_number({self.parameter['dft']['force_threshold']}):{force_count} | "
                f"minimum_max_force:{force_of_force_count_0}"
            )
        else:
            self.logger.info(
                f"candidate batch DFT completed: {gen} | {self.scheduler.scf_cal_engine}_completed_number:{ok_count} | "
                f"Successful_collection_structure/scf_convergent_number:{len_count} | "
                f"force_threshold_number({self.parameter['dft']['force_threshold']}):{force_count}"
            )

        ele = self.parameter["ele"]
        ele_model = 1 if self.parameter["sort_ele"] else 2
        mlp_nums = 1 if self.parameter["mlp_encode_model"] else self.parameter["mlp_nums"]
        update_mlp_from_current_batch(str(self.workspace), mlp_nums, ele, ele_model, self.logger, self.scheduler)
        self._validate_training_outputs()
        self._emit_candidate_update_training_summary()
        self.candidate_pool.clear()
        self.logger.info("candidate pool cleared after successful MLIP update")

    def _submit_scf_jobs(self, total_atom_list):
        dft_work_path = self.workspace / DFT_WORK_DIR
        scf_path = dft_work_path / "scf"
        mkdir(str(dft_work_path))
        mkdir(str(scf_path))
        os.chdir(scf_path)
        calc_dir_num = check_and_modify_calc_dir(str(self.workspace), total_atom_list, self.parameter["dft"]["calc_dir_num"])
        num = main_calc(total_atom_list, calc_dir_num, str(self.workspace), self.scheduler)
        if not check_scf(str(self.workspace)):
            subprocess.run([sys.executable, "start_calc.py"], check=True)
        self.logger.info(f"The {num} structures are divided into {calc_dir_num} dft calculation tasks to be submitted")

    def _existing_scf_task_state(self):
        scf_filter_root = self.workspace / DFT_WORK_DIR / "scf" / "filter"
        if not scf_filter_root.exists():
            return None

        task_dirs = []
        for first_level in sorted(scf_filter_root.iterdir()):
            if not first_level.is_dir():
                continue
            for second_level in sorted(first_level.iterdir()):
                if second_level.is_dir():
                    task_dirs.append(second_level)

        if not task_dirs:
            return None

        started = sum((task_dir / "__start__").exists() for task_dir in task_dirs)
        completed = sum((task_dir / "__ok__").exists() for task_dir in task_dirs)
        return {
            "task_dirs": task_dirs,
            "task_count": len(task_dirs),
            "started": started,
            "completed": completed,
            "scf_path": self.workspace / DFT_WORK_DIR / "scf",
        }

    def _resume_existing_scf_if_possible(self):
        state = self._existing_scf_task_state()
        if state is None:
            return False

        self.logger.info(
            "Existing SCF task directories detected. Reusing them without regeneration: tasks=%s started=%s completed=%s",
            state["task_count"],
            state["started"],
            state["completed"],
        )

        if state["completed"] == state["task_count"]:
            self.logger.info("Existing SCF tasks are already complete. Skipping resubmission.")
            return True

        if state["started"] > 0:
            return True

        start_calc = state["scf_path"] / "start_calc.py"
        if start_calc.exists():
            self.logger.info("Existing SCF task directories were found but not started. Submitting existing tasks once.")
            subprocess.run([sys.executable, "start_calc.py"], check=True, cwd=state["scf_path"])
            return True

        self.logger.warning("Existing SCF task directories were found but start_calc.py is missing. Regenerating SCF tasks.")
        return False

    def _collect_scf_results(self, force_threshold):
        current = self.workspace / DFT_WORK_DIR / "scf" / "filter"
        out_name = self.workspace / DFT_WORK_DIR / "scf_filter.xyz"
        ori_out_name = self.workspace / DFT_WORK_DIR / "ori_scf_filter.xyz"
        archive_manager = RawDFTArchiveManager(
            self.workspace,
            self.scheduler.scf_cal_engine,
            self.logger,
        )
        cached = load_collection_cache(self.workspace, out_name, current)
        if cached is not None:
            source_manifest = load_scf_filter_sources(out_name, required=False)
            if source_manifest is None:
                self.logger.warning(
                    "Cached SCF collection predates strict raw-DFT source mapping; "
                    "existing tasks will not be archived or cleaned automatically."
                )
                return cached
            archived_tasks = archive_manager.archive_completed_tasks(current, source_manifest)
            excluded_tasks = archive_manager.archive_excluded_tasks(current, source_manifest)
            if len(archived_tasks) != int(source_manifest["frame_count"]):
                raise RuntimeError(
                    "Raw DFT archive count does not match scf_filter.xyz: "
                    f"archives={len(archived_tasks)} frames={source_manifest['frame_count']}"
                )
            archive_manager.cleanup_archived_tasks(archived_tasks)
            archive_manager.cleanup_excluded_tasks(excluded_tasks)
            return cached

        remove(str(out_name))
        result = self.scf2xyz(str(current), str(out_name), str(ori_out_name), force_threshold)
        ok_count, len_count, no_success_path, force_count, _ = result
        source_manifest = load_scf_filter_sources(out_name, required=True)
        archived_tasks = archive_manager.archive_completed_tasks(current, source_manifest)
        excluded_tasks = archive_manager.archive_excluded_tasks(current, source_manifest)
        if len(archived_tasks) != int(source_manifest["frame_count"]):
            raise RuntimeError(
                "Raw DFT archive count does not match scf_filter.xyz: "
                f"archives={len(archived_tasks)} frames={source_manifest['frame_count']}"
            )
        if len_count == 0:
            with open(self.workspace / "no_success_path.json", "w", encoding="utf-8") as handle:
                json.dump(no_success_path, handle)
            archive_manager.cleanup_archived_tasks(archived_tasks)
            archive_manager.cleanup_excluded_tasks(excluded_tasks)
            raise RuntimeError(
                "No successful SCF structures were collected. "
                f"completed_tasks={ok_count}, force_threshold_count={force_count}. "
                f"See {self.workspace / 'no_success_path.json'}"
            )
        write_collection_cache(self.workspace, result, current)
        archive_manager.cleanup_archived_tasks(archived_tasks)
        archive_manager.cleanup_excluded_tasks(excluded_tasks)
        return result

    def _run_scf_stage_without_encoding(self, end_state):
        end_threshold_low, end_threshold_high, end_n, end_cluster_threshold_init, end_k = end_state
        sample = self.parameter["sample"]
        if not check_filter_xyz_0(str(self.workspace)):
            self.logger.info("The active learning loop ends")
            touch(str(self.workspace), "__end__")
            return

        if (
            end_threshold_low == self.parameter["threshold_low"]
            and end_threshold_high == self.parameter["threshold_high"]
            and end_n == sample["n"]
            and end_cluster_threshold_init == sample["cluster_threshold_init"]
            and end_k == sample["k"]
            and self.parameter["dft"]["calc_dir_num"]
        ):
            self.logger.info("(calc_dir_num, threshold, n, cluster_threshold_init, k) parameters are equal: skip scf calculations")
        else:
            if not self._resume_existing_scf_if_possible():
                total_atom_list = collect_dft_data("no", str(self.workspace))
                scf_path = self.workspace / DFT_WORK_DIR / "scf"
                os.chdir(scf_path)
                for item in [name for name in os.listdir() if name != "total_sample_filter.xyz"]:
                    remove(item)
                self._submit_scf_jobs(total_atom_list)

        self.logger.info("Waiting for DFT/SCF tasks to complete...")
        check_finish(
            scf_dir(str(self.workspace)),
            self.logger,
            "All scf calculations have been completed",
            pending_warning_hours=self.parameter["dft"].get("pending_warning_hours"),
        )

        ok_count, len_count, no_success_path, force_count, _ = self._collect_scf_results(self.parameter["dft"]["force_threshold"])
        with open(self.workspace / "no_success_path.json", "w", encoding="utf-8") as handle:
            json.dump(no_success_path, handle)
        gen = self.workspace.name
        self.logger.info(
            f"Active learning continues: {gen} | {self.scheduler.scf_cal_engine}_completed_number:{ok_count} | "
            f"Successful_collection_structure/scf_convergent_number:{len_count} | "
            f"force_threshold_number({self.parameter['dft']['force_threshold']}):{force_count}"
        )

    def _run_scf_stage_with_encoding(self):
        sample_xyz_list = glob.glob(os.path.join(self.workspace, MD_WORK_DIR, "*_sample_filter.xyz"))
        if len(sample_xyz_list) == 0:
            self.logger.info("The active learning loop ends")
            touch(str(self.workspace), "__end__")
            return

        sample_xyz = sample_xyz_list[0]
        if os.path.getsize(sample_xyz) == 0:
            self.logger.info("No structures were selected for SCF. The active learning loop ends")
            touch(str(self.workspace), "__end__")
            return

        if not self._resume_existing_scf_if_possible():
            total_atom_list = list(iread(sample_xyz))
            dft_work_path = self.workspace / DFT_WORK_DIR
            scf_path = self.workspace / DFT_WORK_DIR / "scf"
            mkdir(str(dft_work_path))
            mkdir(str(scf_path))
            os.chdir(scf_path)
            for item in [name for name in os.listdir() if name != "total_sample_filter.xyz"]:
                remove(item)
            self._submit_scf_jobs(total_atom_list)

        self.logger.info("Waiting for DFT/SCF tasks to complete...")
        check_finish(
            scf_dir(str(self.workspace)),
            self.logger,
            "All scf calculations have been completed",
            pending_warning_hours=self.parameter["dft"].get("pending_warning_hours"),
        )

        ok_count, len_count, no_success_path, force_count, force_of_force_count_0 = self._collect_scf_results(
            self.parameter["dft"]["force_threshold"]
        )
        with open(self.workspace / "no_success_path.json", "w", encoding="utf-8") as handle:
            json.dump(no_success_path, handle)

        gen = self.workspace.name
        if force_count == 0:
            self.logger.info(
                f"Active learning continues: {gen} | {self.scheduler.scf_cal_engine}_completed_number:{ok_count} | "
                f"Successful_collection_structure/scf_convergent_number:{len_count} | "
                f"force_threshold_number({self.parameter['dft']['force_threshold']}):{force_count} | "
                f"minimum_max_force:{force_of_force_count_0}"
            )
        else:
            self.logger.info(
                f"Active learning continues: {gen} | {self.scheduler.scf_cal_engine}_completed_number:{ok_count} | "
                f"Successful_collection_structure/scf_convergent_number:{len_count} | "
                f"force_threshold_number({self.parameter['dft']['force_threshold']}):{force_count}"
            )

    def run(self):
        end_state = self._load_end_state()
        self._train_models()
        dirs_1, dirs_2 = self._prepare_md()
        if self.parameter["mlp_encode_model"]:
            self._select_by_encoding(dirs_1)
        else:
            self._select_by_ambiguity(dirs_1, dirs_2, end_state)
        self._persist_end_state()
        self._run_candidate_batch_stage()
        delete_dump(dirs_1)
        cleanup_generation_descriptor_stores(self.workspace, self.logger)
        touch(str(self.workspace), "__ok__")
