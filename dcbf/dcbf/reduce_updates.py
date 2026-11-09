"""Recoverable multi-round Reduce -> optional SCF -> SUS2 continuation.

The public configuration uses readable names. Each committed round keeps its
selection, labeling, training and pool files so that a run can be audited or
resumed without repeating completed scheduler jobs.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from .encode.coverage_policy import (
    determine_structure_budget_with_stage,
    stable_unique,
    strict_budget_selection,
    validate_selection_schedules,
)
from .fast_extxyz import complement_indices, index_xyz_frames, write_indexed_frames
from .labelled_data import inspect_labels
from .memory_guard import descriptor_guard
from .reduce_execution import ReduceExecutor
from .reduce_state import fault, read_json, save_json, sha, work_lock
from .sus2_training import DEFAULT_FINAL_ITERATIONS, DEFAULT_SUBSEQUENT_ITERATIONS


def validate_max_update_rounds(max_rounds, completed_rounds):
    """Validate the round safety limit against a checkpoint's committed rounds.

    ``max_update_rounds`` is deliberately excluded from the durable settings
    signature so it can be raised in the middle of a run.  A value below the
    number of already committed rounds cannot be honoured, because those
    rounds happened; reject it instead of silently reporting a contradictory
    checkpoint.
    """
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds <= 0:
        raise ValueError("reduce.max_update_rounds must be a positive integer")
    completed_rounds = int(completed_rounds)
    if max_rounds < completed_rounds:
        raise ValueError(
            f"reduce.max_update_rounds ({max_rounds}) is below the "
            f"{completed_rounds} round(s) already committed in this checkpoint; "
            f"set it to at least {completed_rounds} or use a new work_dir"
        )
    return max_rounds


def plan_update_round(
    coverage,
    budget_schedule,
    threshold_schedule,
    completed_rounds=0,
    max_rounds=None,
):
    """Derive one update round's budget and stop reason from coverage.

    The coverage targets own the stop decision: once every element reaches the
    final stage threshold the run stops with ``coverage_target_reached`` even
    when the safety limit is also exhausted.  Otherwise the budget is the first
    stage whose target is still unmet, and ``max_update_rounds`` only stops a
    run that has not converged.
    """
    budget, converged, thresholds, stage_index, stage_target = (
        determine_structure_budget_with_stage(
            coverage,
            budget_schedule,
            threshold_schedule,
        )
    )
    if converged:
        return {
            "coverage": [float(value) for value in coverage],
            "converged": True,
            "budget": 0,
            "stage": None,
            "stage_index": None,
            "stage_target": [float(value) for value in stage_target],
            "thresholds": thresholds,
            "stop_reason": "coverage_target_reached",
        }
    completed_rounds = int(completed_rounds)
    stop_reason = None
    if max_rounds is not None and completed_rounds >= int(max_rounds):
        stop_reason = "max_update_rounds_reached"
    return {
        "coverage": [float(value) for value in coverage],
        "converged": False,
        "budget": int(budget),
        "stage": int(stage_index) + 1,
        "stage_index": int(stage_index),
        "stage_target": [float(value) for value in stage_target],
        "thresholds": thresholds,
        "stop_reason": stop_reason,
    }



def _update_signatures_compatible(saved, current):
    """Ignore absolute config and snapshot locations when resuming after a move."""
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    saved = deepcopy(saved)
    current = deepcopy(current)
    for signature in (saved, current):
        signature.pop("config_dir", None)
        for item in (signature.get("inputs") or {}).values():
            if isinstance(item, dict):
                item.pop("path", None)
        block = (signature.get("config") or {}).get("reduce")
        if isinstance(block, dict):
            # Older checkpoints saved this deprecated cumulative cap in the
            # signature.  The key is ignored now, so it must not make an
            # otherwise identical checkpoint look changed.
            block.pop("max_total_new_structures", None)
    return saved == current


def updates_requested(config):
    block = config.get("reduce") or {}
    for key in ("update_model_after_selection", "run_scf_labeling"):
        if key in block and not isinstance(block[key], bool):
            raise ValueError(f"reduce.{key} must be true or false")
    final = (config.get("training") or {}).get("final") or {}
    return bool(
        block.get("update_model_after_selection")
        or block.get("run_scf_labeling")
        or final.get("enabled")
    )


def positive_iterations(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def concatenate(output, inputs):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".partial")
    with temporary.open("wb") as target:
        needs_separator = False
        for path in inputs:
            path = Path(path)
            if not path.exists() or not path.stat().st_size:
                continue
            with path.open("rb") as source:
                source.seek(-1, os.SEEK_END)
                last_byte = source.read(1)
                source.seek(0)
                if needs_separator:
                    target.write(b"\n")
                shutil.copyfileobj(source, target, length=1024 * 1024)
                needs_separator = last_byte != b"\n"
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, output)


class UpdatingReducer:
    def __init__(self, reducer):
        self.parent = reducer
        self.config = reducer.config
        self.options = dict(self.config.get("reduce") or {})
        self.update_model = bool(self.options.get("update_model_after_selection", False))
        self.run_scf = bool(self.options.get("run_scf_labeling", False))
        self.max_rounds = validate_max_update_rounds(
            int(self.options.get("max_update_rounds", 1)),
            0,
        )
        self.budget_schedule = reducer.selection_budget_by_coverage
        self.threshold_schedule = reducer.coverage_target_by_stage
        if self.budget_schedule is None or self.threshold_schedule is None:
            raise ValueError(
                "reduce.selection_budget_by_coverage and "
                "reduce.coverage_target_by_stage are required when "
                "update_model_after_selection or run_scf_labeling is enabled; "
                "one-shot Reduce may omit both to keep the full selection"
            )
        self.final = dict((self.config.get("training") or {}).get("final") or {})
        self.final_enabled = bool(self.final.get("enabled", False))
        if self.final_enabled:
            positive_iterations(
                self.final.get("iterations", DEFAULT_FINAL_ITERATIONS),
                "training.final.iterations",
            )
        self.iterations = None
        if self.update_model:
            self.iterations = positive_iterations(
                (self.config.get("training") or {}).get("sampling", {}).get(
                    "subsequent_iterations", DEFAULT_SUBSEQUENT_ITERATIONS
                ),
                "training.sampling.subsequent_iterations",
            )
        self.root = reducer.work_dir / "updates"
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.root / "reduce_update_checkpoint.json"
        self.inputs = {
            "pool": reducer.input_xyz,
            "seed": reducer.current_xyz,
            "initial_model": reducer.model_path,
        }
        for path in (reducer.output_xyz, reducer.remain_xyz, reducer.report_json):
            if path is not None and path in [p for p in self.inputs.values() if p is not None]:
                raise ValueError("Reduce outputs must not overwrite input dataset/model files")
        self.elements = list(reducer.elements)
        # Normalise the paired schedules once; each round then derives its
        # budget and stage from measured coverage, never from a round counter.
        self.budget_schedule, self.threshold_schedule = validate_selection_schedules(
            self.budget_schedule,
            self.threshold_schedule,
            len(self.elements),
        )
        # The executor is needed to prepare SCF templates, while the template
        # hashes are part of the durable checkpoint signature.  Initialize the
        # field before the first signature pass, then refresh the signature
        # after preparation has produced the real hashes.
        self.template_hashes = None
        self.signature = self._signature()
        token = hashlib.sha256(
            json.dumps(self.signature, sort_keys=True).encode()
        ).hexdigest()
        self.executor = ReduceExecutor(
            self.config, reducer.config_dir, self.root, self.elements, token
        )
        if self.update_model:
            self.executor.preflight_training(reducer.model_path)
            labels_mode = (self.config.get("training") or {}).get("labels", "auto")
            if reducer.current_xyz is not None:
                inspect_labels(reducer.current_xyz, labels_mode)
            if not self.run_scf:
                inspect_labels(reducer.input_xyz, labels_mode)
        self.template_hashes = self.executor.prepare_templates() if self.run_scf else None
        self.signature = self._signature()
        self.index = index_xyz_frames(reducer.input_xyz)
        if not len(self.index):
            raise ValueError("Empty candidate pool")

    def _signature(self):
        config = deepcopy(self.config)
        block = config.setdefault("reduce", {})
        for key in (
            "selected_structures_xyz",
            "remaining_candidates_xyz",
            "report_json",
            "work_dir",
            "keep_intermediate",
            # Only a safety limit: it may be raised between rounds, so it must
            # not invalidate the durable settings signature.  ``_load``
            # validates it against the committed round count instead.
            "max_update_rounds",
        ):
            block.pop(key, None)
        return {
            "version": 2,
            "config": config,
            "config_dir": str(self.parent.config_dir),
            "inputs": {
                key: {"path": str(path), "sha256": sha(path)} if path else None
                for key, path in self.inputs.items()
            },
            "elements": self.elements,
            "model_sha256": sha(self.parent.model_path),
            "template_hashes": self.template_hashes,
        }

    def _save(self, state):
        state["updated_at"] = time.time()
        state["max_update_rounds_effective"] = self.max_rounds
        state.setdefault("stop_reason", state.get("status", "running"))
        save_json(self.checkpoint, state)

    def _load(self):
        if self.checkpoint.exists():
            state = read_json(self.checkpoint)
            if state.get("version") != 2 or not _update_signatures_compatible(
                state.get("signature"), self.signature
            ):
                raise RuntimeError(
                    "Reduce update inputs/settings changed; checkpoint preserved. "
                    "Use a new work_dir."
                )
            # max_update_rounds is excluded from the signature so it can be
            # raised mid-run; it still cannot drop below committed rounds.
            validate_max_update_rounds(self.max_rounds, state.get("next_round", 0))
            for key in ("current_pool", "current_dataset", "current_model"):
                state[key] = str(self._resolve_state_path(state[key], state[f"{key}_sha256"]))
                if sha(state[key]) != state[f"{key}_sha256"]:
                    raise RuntimeError(f"Current Reduce file changed after the last commit: {key}")
            return state
        snapshot = self.root / "snapshots"
        snapshot.mkdir(parents=True, exist_ok=True)
        pool = snapshot / "pool_initial.xyz"
        dataset = snapshot / "reference_initial.xyz"
        model = snapshot / "model_initial.sus2"
        concatenate(pool, [self.parent.input_xyz])
        if self.parent.current_xyz is not None:
            concatenate(dataset, [self.parent.current_xyz])
        else:
            dataset.write_text("", encoding="utf-8")
        shutil.copy2(self.parent.model_path, model)
        state = {
            "version": 2, "signature": self.signature, "status": "running",
            "stop_reason": "running",
            "max_update_rounds_effective": self.max_rounds,
            "selection_budget_by_coverage": list(self.budget_schedule),
            "coverage_target_by_stage": [list(row) for row in self.threshold_schedule],
            "next_round": 0, "total_new_structures": 0,
            "current_pool": str(pool), "current_pool_sha256": sha(pool),
            "pool_source_indices": list(range(len(self.index))),
            "current_dataset": str(dataset), "current_dataset_sha256": sha(dataset),
            "current_model": str(model), "current_model_sha256": sha(model),
            "rounds": [], "active_round": None,
        }
        self._save(state)
        return state

    def _resolve_state_path(self, raw_path, expected_sha256=None):
        path = Path(raw_path)
        if path.exists():
            return path
        if not path.is_absolute():
            return self.root / path

        parts = path.parts
        anchor = self.root.name
        if anchor in parts:
            candidate = self.root.joinpath(*parts[parts.index(anchor) + 1:])
            if candidate.exists():
                return candidate

        if expected_sha256:
            for candidate in self.root.rglob(path.name):
                if candidate.is_file() and sha(candidate) == expected_sha256:
                    return candidate
        return path

    def _plan_coverage(self, coverage, completed_rounds):
        """Budget/stage/stop plan for one measured coverage vector."""
        return plan_update_round(
            coverage,
            self.budget_schedule,
            self.threshold_schedule,
            completed_rounds=completed_rounds,
            max_rounds=self.max_rounds,
        )

    def _selection_config(self, state, directory):
        config = deepcopy(self.config)
        block = config.setdefault("reduce", {})
        block["selection_mode"] = (
            self.parent.mode if Path(state["current_dataset"]).stat().st_size else "candidate_only"
        )
        block["candidate_pool_xyz"] = str(state["current_pool"])
        block["reference_training_xyz"] = (
            str(state["current_dataset"])
            if Path(state["current_dataset"]).stat().st_size else None
        )
        block["starting_model_path"] = str(state["current_model"])
        block["selected_structures_xyz"] = str(directory / "selected_raw.xyz")
        block["remaining_candidates_xyz"] = str(directory / "remaining_raw.xyz")
        block["work_dir"] = str(directory / "descriptor_work")
        block["report_json"] = str(directory / "selection_engine_report.json")
        block["update_model_after_selection"] = False
        block["run_scf_labeling"] = False
        config.pop("training", None)
        config.pop("dft", None)
        config.setdefault("parameter", {})["ele"] = list(self.elements)
        config["parameter"]["sort_ele"] = False
        return config

    def _measure_coverage(self, directory, pool, dataset, model, tag):
        """Measure coverage of ``pool`` by ``dataset`` with a real descriptor pass.

        A dedicated single-pass reducer reuses the same descriptor store cache
        as selection (the round work directory is shared), and its
        ``measure_coverage`` follows the sampling coverage chain.
        """
        directory = Path(directory)
        config = self._selection_config(
            {
                "current_pool": pool,
                "current_dataset": dataset,
                "current_model": model,
            },
            directory,
        )
        config_path = directory / f"coverage_{tag}.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        from .reduce import DCBFReducer
        core = DCBFReducer(config_path)
        with descriptor_guard(directory / "descriptor_work", task_type="reduce_update_coverage"):
            measured = core.measure_coverage(Path(dataset), Path(pool))
        return {
            "coverage": [float(value) for value in measured["coverage"]],
            "bodies": measured.get("bodies") or {},
        }

    def _begin_round(self, state):
        """Prepare one round, or report why the run must stop.

        Coverage of ``dataset_before`` against ``pool_before`` is measured
        first.  Reaching the final stage threshold returns ``(None, plan)``
        with ``coverage_target_reached`` before any selection runs.  Otherwise
        the plan's budget is the first unmet stage's budget, so stages advance
        from measured coverage instead of from a round counter.
        """
        number = int(state["next_round"]) + 1
        directory = self.root / f"round_{number:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        pool_before = directory / "pool_before.xyz"
        model_before = directory / "model_before.sus2"
        dataset_before = directory / "reference_before.xyz"
        concatenate(pool_before, [state["current_pool"]])
        shutil.copy2(state["current_model"], model_before)
        concatenate(dataset_before, [state["current_dataset"]])
        pool_index = index_xyz_frames(pool_before)
        source_indices = list(state["pool_source_indices"])
        if len(pool_index) != len(source_indices):
            raise RuntimeError("Reduce pool source map does not match pool frame count")
        config = self._selection_config(
            {**state, "current_pool": pool_before, "current_dataset": dataset_before,
             "current_model": model_before}, directory
        )
        config_path = directory / "selection_reduce.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        from .reduce import DCBFReducer
        core = DCBFReducer(config_path)
        with descriptor_guard(
            directory / "descriptor_work", task_type="reduce_update_coverage"
        ):
            measured = core.measure_coverage(dataset_before, pool_before)
        coverage = [float(value) for value in measured["coverage"]]
        coverage_by_body = measured.get("bodies") or {}
        plan = self._plan_coverage(coverage, int(state["next_round"]))
        if plan["stop_reason"] is not None:
            save_json(directory / "selection_report.json", {
                "round": number,
                "selection_mode": self.parent.mode,
                "candidate_pool_before": str(pool_before),
                "reference_training_before": str(dataset_before),
                "model_before": str(model_before),
                "coverage": coverage,
                "coverage_by_body": coverage_by_body,
                "stage": plan["stage"],
                "stage_index": plan["stage_index"],
                "stage_target": plan["stage_target"],
                "stop_reason": plan["stop_reason"],
                "max_update_rounds_effective": self.max_rounds,
                "selected_local_indices": [],
                "selected_source_indices": [],
                "selected_count": 0,
                "round_budget": None,
                "selection_budget_by_coverage": self.budget_schedule,
                "coverage_target_by_stage": self.threshold_schedule,
                "selection_seconds": 0.0,
                "effective_config": core._build_effective_config(),
            })
            return None, plan

        started = time.monotonic()
        with descriptor_guard(directory / "descriptor_work", task_type="reduce_update_selection"):
            if core.direct_self_dedup or core.current_xyz is None:
                local_selected = core._select_direct_indices(pool_before)
            else:
                local_selected = core._select_indices(
                    dataset_before, pool_before, stage_label=f"round_{number:04d}"
                )
        # Keep the min-cover contribution order from ``_select_*`` and only
        # then truncate with the stage budget.
        local_selected = stable_unique(
            int(item) for item in local_selected if 0 <= int(item) < len(pool_index)
        )
        budget = plan["budget"]
        local_selected = strict_budget_selection(local_selected, budget)
        # XYZ frame indices must be strictly increasing.  The source-id list is
        # consumed positionally together with this file by the SCF/label path,
        # so it must use the same ascending order; the contribution order stays
        # in the report and checkpoint for budget/audit purposes.
        selected_for_io = sorted(local_selected)
        selected_source = [source_indices[item] for item in selected_for_io]
        write_indexed_frames(directory / "selected_before_labeling.xyz", [(pool_index, selected_for_io)])
        write_indexed_frames(
            directory / "pool_after_selection.xyz",
            [(pool_index, complement_indices(len(pool_index), set(local_selected)))]
        )
        save_json(directory / "selection_report.json", {
            "round": number, "selection_mode": self.parent.mode,
            "candidate_pool_before": str(pool_before),
            "reference_training_before": str(dataset_before),
            "model_before": str(model_before),
            "coverage": coverage,
            "coverage_by_body": coverage_by_body,
            "stage": plan["stage"],
            "stage_index": plan["stage_index"],
            "stage_target": plan["stage_target"],
            "stop_reason": None,
            "max_update_rounds_effective": self.max_rounds,
            "selected_local_indices": local_selected,
            "selected_local_indices_io": selected_for_io,
            "selected_source_indices": selected_source,
            "selected_count": len(local_selected),
            "round_budget": budget,
            "selection_budget_by_coverage": self.budget_schedule,
            "coverage_target_by_stage": self.threshold_schedule,
            "selection_seconds": time.monotonic() - started,
            "effective_config": core._build_effective_config(),
        })
        round_state = {
            "round": number, "directory": str(directory), "stage": "selected",
            "coverage": coverage,
            "coverage_by_body": coverage_by_body,
            "coverage_stage": plan["stage"],
            "coverage_stage_index": plan["stage_index"],
            "coverage_stage_target": plan["stage_target"],
            "round_budget": budget,
            "max_update_rounds_effective": self.max_rounds,
            "pool_before": str(pool_before), "pool_before_sha256": sha(pool_before),
            "pool_source_indices": source_indices,
            "selected_local_indices": local_selected,
            "selected_local_indices_io": selected_for_io,
            "selected_source_indices": selected_source,
            "dataset_before": str(dataset_before),
            "dataset_before_sha256": sha(dataset_before),
            "model_before": str(model_before),
            "model_before_sha256": sha(model_before),
            "accepted_source_indices": [],
        }
        save_json(directory / "round_state.json", round_state)
        state["active_round"] = str(directory)
        self._save(state)
        fault("after_selection")
        return round_state, plan

    def _process_round(self, state, round_state):
        directory = Path(round_state["directory"])
        selected = directory / "selected_before_labeling.xyz"
        source_ids = list(round_state["selected_source_indices"])
        stage = round_state["stage"]
        labels_mode = (self.config.get("training") or {}).get("labels", "auto")
        if stage == "selected":
            if source_ids and self.run_scf:
                labels_path, accepted = self.executor.scf(directory, selected, source_ids)
            elif source_ids:
                inspect_labels(selected, labels_mode)
                labels_path = directory / "labeled_after_scf.xyz"
                concatenate(labels_path, [selected])
                accepted = list(source_ids)
                save_json(directory / "scf_report.json", {
                    "status": "reused_input_labels", "accepted_source_indices": accepted,
                })
            else:
                labels_path = directory / "labeled_after_scf.xyz"
                labels_path.write_text("", encoding="utf-8")
                accepted = []
                save_json(directory / "scf_report.json", {
                    "status": "no_structures_selected", "accepted_source_indices": [],
                })
            if Path(labels_path) != directory / "labeled_after_scf.xyz":
                concatenate(directory / "labeled_after_scf.xyz", [labels_path])
            round_state.update(
                stage="labeled",
                labels_path=str(directory / "labeled_after_scf.xyz"),
                labels_sha256=sha(directory / "labeled_after_scf.xyz"),
                accepted_source_indices=list(accepted),
            )
            save_json(directory / "round_state.json", round_state)
            self._save(state)
            fault("after_labels")
            stage = "labeled"
        if stage == "labeled":
            merged = directory / "dataset_after.xyz"
            concatenate(merged, [round_state["dataset_before"], round_state["labels_path"]])
            round_state.update(stage="merged", dataset_after=str(merged),
                               dataset_after_sha256=sha(merged))
            save_json(directory / "round_state.json", round_state)
            self._save(state)
            fault("after_merge")
            stage = "merged"
        if stage == "merged":
            if self.update_model and round_state["accepted_source_indices"]:
                model_after = self.executor.train(
                    directory / "training", round_state["dataset_after"],
                    round_state["model_before"], self.iterations,
                )
                run_yaml = directory / "training" / "run.yaml"
                if run_yaml.exists():
                    shutil.copy2(run_yaml, directory / "continuation.yaml")
            else:
                model_after = Path(round_state["model_before"])
            if Path(model_after) != directory / "model_after.sus2":
                shutil.copy2(model_after, directory / "model_after.sus2")
            round_state.update(stage="trained",
                               model_after=str(directory / "model_after.sus2"),
                               model_after_sha256=sha(directory / "model_after.sus2"))
            save_json(directory / "round_state.json", round_state)
            self._save(state)
            fault("before_commit")
            stage = "trained"
        if stage == "trained":
            pool_before_index = index_xyz_frames(round_state["pool_before"])
            source_ids = round_state["pool_source_indices"]
            accepted_set = set(round_state["accepted_source_indices"])
            accepted_local = {local for local, source in enumerate(source_ids) if source in accepted_set}
            pool_after = directory / "pool_after.xyz"
            write_indexed_frames(pool_after, [
                (pool_before_index, complement_indices(len(pool_before_index), accepted_local))
            ])
            round_state.update(stage="complete", pool_after=str(pool_after),
                               pool_after_sha256=sha(pool_after))
            # End-of-round coverage: the dataset after merging this round's
            # labels against the pool that remains.  It is the value the next
            # round would start from and decides whether the run can stop now.
            coverage_after = self._measure_coverage(
                directory,
                pool_after,
                Path(round_state["dataset_after"]),
                Path(round_state["model_after"]),
                f"round_{int(round_state['round']):04d}_after",
            )
            after_plan = self._plan_coverage(
                coverage_after["coverage"],
                int(round_state["round"]),
            )
            round_state.update(
                coverage_after=coverage_after["coverage"],
                coverage_by_body_after=coverage_after["bodies"],
                coverage_stage_after=after_plan["stage"],
                coverage_stage_target_after=after_plan["stage_target"],
                coverage_target_reached_after=after_plan["converged"],
            )
            save_json(directory / "round_state.json", round_state)
            selection_report_path = directory / "selection_report.json"
            if selection_report_path.exists():
                selection_report = read_json(selection_report_path)
                selection_report.update({
                    "coverage_after": coverage_after["coverage"],
                    "coverage_by_body_after": coverage_after["bodies"],
                    "stage_after": after_plan["stage"],
                    "stage_target_after": after_plan["stage_target"],
                    "coverage_target_reached_after": after_plan["converged"],
                })
                save_json(selection_report_path, selection_report)
            state["rounds"].append({
                "round": round_state["round"], "directory": str(directory),
                "selected_count": len(round_state["selected_source_indices"]),
                "accepted_count": len(round_state["accepted_source_indices"]),
                "coverage": round_state.get("coverage"),
                "coverage_after": coverage_after["coverage"],
                "stage": round_state.get("coverage_stage"),
                "stage_target": round_state.get("coverage_stage_target"),
                "stage_after": after_plan["stage"],
                "stage_target_after": after_plan["stage_target"],
                "coverage_target_reached_after": after_plan["converged"],
                "round_budget": round_state.get("round_budget"),
                "max_update_rounds_effective": self.max_rounds,
                "stop_reason": None,
                "model_before": round_state["model_before"],
                "model_after": round_state["model_after"],
                "pool_after": str(pool_after),
                "dataset_after": round_state["dataset_after"],
            })
            state["total_new_structures"] += len(round_state["accepted_source_indices"])
            state["current_pool"] = str(pool_after)
            state["current_pool_sha256"] = sha(pool_after)
            state["pool_source_indices"] = [source for source in source_ids if source not in accepted_set]
            state["current_dataset"] = round_state["dataset_after"]
            state["current_dataset_sha256"] = sha(round_state["dataset_after"])
            state["current_model"] = round_state["model_after"]
            state["current_model_sha256"] = sha(round_state["model_after"])
            state["next_round"] = int(round_state["round"])
            state["active_round"] = None
            self._save(state)
            fault("after_commit")
        return round_state

    def _ensure_after_coverage(self, round_state):
        """Fill end-of-round coverage for a checkpoint committed by older code.

        A resumed round whose ``stage`` is already ``complete`` has all files
        on disk; if the checkpoint predates the coverage fields, measure them
        once and persist so reports stay complete without redoing any job.
        """
        if round_state.get("coverage_after") is not None:
            return round_state
        if round_state.get("stage") != "complete":
            return round_state
        directory = Path(round_state["directory"])
        measured = self._measure_coverage(
            directory,
            Path(round_state["pool_after"]),
            Path(round_state["dataset_after"]),
            Path(round_state["model_after"]),
            f"round_{int(round_state['round']):04d}_after",
        )
        after_plan = self._plan_coverage(
            measured["coverage"],
            int(round_state["round"]),
        )
        round_state.update(
            coverage_after=measured["coverage"],
            coverage_by_body_after=measured["bodies"],
            coverage_stage_after=after_plan["stage"],
            coverage_stage_target_after=after_plan["stage_target"],
            coverage_target_reached_after=after_plan["converged"],
        )
        save_json(directory / "round_state.json", round_state)
        return round_state

    def _finish_training(self, state):
        if not self.final_enabled or not self.update_model:
            return
        output_name = str(self.final.get("model_name", "final.sus2"))
        if Path(output_name).name != output_name or not output_name.endswith(".sus2"):
            raise ValueError("training.final.model_name must be a .sus2 filename")
        output = self.executor.train(
            self.root / "final_training", state["current_dataset"],
            state["current_model"], int(self.final.get("iterations", DEFAULT_FINAL_ITERATIONS)),
            output_name,
        )
        state["final_training"] = {"output_model": str(output),
                                   "output_sha256": sha(output),
                                   "dataset_sha256": state["current_dataset_sha256"]}
        state["current_model"] = str(output)
        state["current_model_sha256"] = sha(output)
        self._save(state)

    def _publish(self, state, started):
        labeled_paths = [
            Path(item["directory"]) / "labeled_after_scf.xyz"
            for item in state["rounds"]
            if (Path(item["directory"]) / "labeled_after_scf.xyz").exists()
        ]
        concatenate(self.parent.output_xyz, labeled_paths)
        if self.parent.remain_xyz is not None:
            concatenate(self.parent.remain_xyz, [state["current_pool"]])
        shutil.copy2(state["current_model"], self.root / "latest.sus2")
        report = {
            "selection_mode": self.parent.mode,
            "update_model_after_selection": self.update_model,
            "run_scf_labeling": self.run_scf,
            "total_new_structures": state["total_new_structures"],
            "max_update_rounds": self.max_rounds,
            "max_update_rounds_effective": self.max_rounds,
            "selection_budget_by_coverage": list(self.budget_schedule),
            "coverage_target_by_stage": [list(row) for row in self.threshold_schedule],
            "stop_reason": state.get("stop_reason", state["status"]),
            "input_count": len(self.index),
            "selected_count": sum(item["selected_count"] for item in state["rounds"]),
            "remaining_count": len(state["pool_source_indices"]),
            "current_model": state["current_model"],
            "current_model_sha256": state["current_model_sha256"],
            "rounds": state["rounds"],
            "checkpoint": str(self.checkpoint),
            "selected_structures_xyz": str(self.parent.output_xyz),
            "remaining_candidates_xyz": str(self.parent.remain_xyz) if self.parent.remain_xyz is not None else None,
            "report_json": str(self.parent.report_json),
            "elapsed_seconds": time.monotonic() - started,
            "input_config": self.config,
        }
        save_json(self.parent.report_json, report)
        return report

    def _stop(self, state, reason):
        """Record a terminal reason on the checkpoint and the last round report."""
        state["status"] = reason
        state["stop_reason"] = reason
        if not state["rounds"]:
            return
        last = state["rounds"][-1]
        last["stop_reason"] = reason
        report_path = Path(last["directory"]) / "selection_report.json"
        if not report_path.exists():
            return
        try:
            report = read_json(report_path)
            report["stop_reason"] = reason
            save_json(report_path, report)
        except (OSError, ValueError):
            pass

    def _rounds_completed_at_limit(self, state):
        """Best-effort convergence check when the safety limit is already reached.

        Every committed round records its end-of-round coverage, so a restart
        at the limit can still report ``coverage_target_reached`` when that
        round already met the final target.
        """
        rounds = state.get("rounds") or []
        if not rounds:
            return False
        coverage = rounds[-1].get("coverage_after")
        if not coverage:
            return False
        plan = self._plan_coverage(coverage, int(state.get("next_round", 0)))
        return bool(plan["converged"])

    def run(self):
        started = time.monotonic()
        with work_lock(self.root / "run.lock"):
            state = self._load()
            try:
                if state["active_round"] is not None:
                    round_state = read_json(Path(state["active_round"]) / "round_state.json")
                    round_state = self._process_round(state, round_state)
                    self._ensure_after_coverage(round_state)
                while True:
                    if not state["pool_source_indices"]:
                        self._stop(state, "candidate_pool_empty")
                        break
                    if int(state["next_round"]) >= self.max_rounds:
                        reason = (
                            "coverage_target_reached"
                            if self._rounds_completed_at_limit(state)
                            else "max_update_rounds_reached"
                        )
                        self._stop(state, reason)
                        break
                    round_state, plan = self._begin_round(state)
                    if round_state is None:
                        # Coverage already met the final stage target; no
                        # selection or labeling is launched for this round.
                        self._stop(state, plan["stop_reason"])
                        break
                    round_state = self._process_round(state, round_state)
                    accepted = int(state["rounds"][-1]["accepted_count"])
                    if accepted == 0:
                        self._stop(state, "no_new_structures")
                        break
                    after_plan = self._plan_coverage(
                        round_state.get("coverage_after") or [],
                        int(round_state["round"]),
                    )
                    if after_plan["converged"]:
                        self._stop(state, "coverage_target_reached")
                        break
                    if int(state["next_round"]) >= self.max_rounds:
                        self._stop(state, "max_update_rounds_reached")
                        break
                    state["status"] = "running"
                    state["stop_reason"] = "running"
                self._finish_training(state)
                self._save(state)
                result = self._publish(state, started)
                print(
                    f"[reduce.updates] {state['status']}; "
                    f"total_new_structures={state['total_new_structures']}; "
                    f"model={state['current_model']}",
                    flush=True,
                )
                return result
            except BaseException as exc:
                state["last_error"] = f"{type(exc).__name__}: {exc}"
                state["status"] = "interrupted"
                state["stop_reason"] = "interrupted"
                self._save(state)
                raise

