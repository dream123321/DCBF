from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re
import subprocess

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers, chemical_symbols
from ase.filters import ExpCellFilter
from ase.io import iread, read, write
from ase.optimize import BFGS, BFGSLineSearch, FIRE, GPMin, LBFGS, LBFGSLineSearch, MDMin, QuasiNewton
import numpy as np

from .cli_defaults import defaults_epilog, format_section_defaults, get_section_defaults, set_section_defaults
from .das.file_conversion import write_normalized_extxyz, xyz2cfg
from .efs_distribution import EFSDistributionAnalyzer
from .high_precision_tools import plot_errors_main, predict_xyz_main
from .high_precision_training import TRAINING_TEMPLATE_CHOICES
from .runtime_config import build_scheduler_spec, normalize_scheduler_config
from .tensor_utils import GPA_TO_EV_PER_A3, coerce_tensor_3x3

try:
    from ase.filters import FrechetCellFilter
except ImportError:
    FrechetCellFilter = None


try:
    from pymlip.core import MTPCalactor, PyConfiguration

    PYMLIP_AVAILABLE = True
except ImportError:
    MTPCalactor = None
    PyConfiguration = None
    PYMLIP_AVAILABLE = False


RELAX_OPTIMIZERS = {
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "FIRE": FIRE,
    "MDMin": MDMin,
    "GPMin": GPMin,
    "LBFGSLineSearch": LBFGSLineSearch,
    "BFGSLineSearch": BFGSLineSearch,
    "QuasiNewton": QuasiNewton,
}
RELAX_OUTPUT_FORMATS = ("cif", "vasp", "xyz", "extxyz", "pdb", "json", "xsf")


class SUS2Calculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, potential, ele_list, compute_stress=True, stress_weight=1.0, **kwargs):
        if not PYMLIP_AVAILABLE:
            raise ImportError("pymlip is required for `dcbf relax`")
        super().__init__(**kwargs)
        self.potential = str(potential)
        self.compute_stress = compute_stress
        self.stress_weight = float(stress_weight)
        self.unique_numbers = [atomic_numbers[element] for element in ele_list]
        self.mtpcalc = MTPCalactor(self.potential)

    def calculate(self, atoms=None, properties=None, system_changes=None):
        properties = properties or ["energy"]
        system_changes = system_changes or self.all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        cfg = PyConfiguration.from_ase_atoms(atoms, unique_numbers=self.unique_numbers)
        volume = atoms.cell.volume if atoms.cell.volume > 0 else 1.0
        self.mtpcalc.calc(cfg)
        self.results["energy"] = np.array(cfg.energy)
        self.results["forces"] = np.asarray(cfg.force, dtype=float)

        if self.compute_stress and getattr(cfg, "stresses", None) is not None:
            stresses = np.asarray(cfg.stresses, dtype=float)
            self.results["stress"] = -np.array(
                [
                    stresses[0, 0],
                    stresses[1, 1],
                    stresses[2, 2],
                    stresses[1, 2],
                    stresses[0, 2],
                    stresses[0, 1],
                ]
            ) * self.stress_weight / volume


def add_high_precision_subparsers(subparsers):
    _build_train_parser(subparsers)
    _build_relax_parser(subparsers)
    _build_efs_distri_parser(subparsers)
    _build_predict_xyz_parser(subparsers)
    _build_plot_errors_parser(subparsers)


def _required_epilog(required_text, defaults_section=None):
    sections = [f"Required input:\n  {required_text}"]
    if defaults_section:
        sections.append(defaults_epilog(defaults_section))
    return "\n\n".join(sections)


def _build_train_parser(subparsers):
    defaults = get_section_defaults("train")
    parser = subparsers.add_parser(
        "train",
        help="generate SUS2 training files from one xyz/extxyz dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog("exactly one xyz/extxyz dataset, or `set key=value ...` to update defaults.", "train"),
    )
    parser.add_argument("items", nargs="*", help='input xyz file, or `set key=value ...`')
    parser.add_argument("--template", choices=sorted(TRAINING_TEMPLATE_CHOICES), help=f"training template type (current: {defaults['template']})")
    parser.add_argument("--min-dist", type=float, dest="min_dist", help=f"override min_dist (current: {defaults['min_dist']})")
    parser.add_argument("--max-dist", type=float, dest="max_dist", help=f"override max_dist (current: {defaults['max_dist']})")
    parser.add_argument("--radial-basis-size", type=int, dest="radial_basis_size", help=f"override radial_basis_size (current: {defaults['radial_basis_size']})")
    parser.add_argument("--backend", choices=["bsub", "sbatch", "lsf", "slurm"], help=f"scheduler backend (current: {defaults['backend']})")
    parser.add_argument("--queue", help=f"queue/partition name (current: {defaults['queue']})")
    parser.add_argument("--cores", type=int, help=f"MPI core count (current: {defaults['cores']})")
    parser.add_argument("--ptile", type=int, help=f"tasks per node (current: {defaults['ptile']})")
    parser.add_argument("--max-iter", type=int, dest="max_iter", help=f"training max iterations (current: {defaults['max_iter']})")
    parser.add_argument("--sus2-exe", dest="sus2_exe", help=f"SUS2 executable path (current: {defaults['sus2_exe']})")
    parser.add_argument("--train-env", dest="train_env", help=f"shell snippet loaded before training (current: {defaults['train_env']})")
    parser.add_argument("--work-dir", dest="work_dir", help=f"output directory (current: {defaults['work_dir']})")
    parser.add_argument("--elements", nargs="+", help=f"element list override (current: {defaults['elements']})")
    parser.add_argument("--keep-order", dest="keep_order", action="store_true", help=f"keep element order (current: {defaults['keep_order']})")
    parser.add_argument("--no-keep-order", dest="keep_order", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--submit", dest="submit", action="store_true", help=f"submit job after writing files (current: {defaults['submit']})")
    parser.add_argument("--no-submit", dest="submit", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(keep_order=None, submit=None)


def _build_relax_parser(subparsers):
    defaults = get_section_defaults("relax")
    parser = subparsers.add_parser(
        "relax",
        help="relax structures with one SUS2/MTP potential",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog(
            "at least one input structure; model must be provided via --model or saved default.",
            "relax",
        ),
    )
    parser.add_argument("items", nargs="*", help='input structure path(s), or `set key=value ...`')
    parser.add_argument("--model", help=f"potential path (current: {defaults['model']})")
    parser.add_argument("--elements", nargs="+", help=f"element list override (current: {defaults['elements']})")
    parser.add_argument("--keep-order", dest="keep_order", action="store_true", help=f"keep explicit element order (current: {defaults['keep_order']})")
    parser.add_argument("--no-keep-order", dest="keep_order", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--optimizer", choices=sorted(RELAX_OPTIMIZERS), help=f"optimizer name (current: {defaults['optimizer']})")
    parser.add_argument("--fmax", type=float, help=f"force convergence threshold (current: {defaults['fmax']})")
    parser.add_argument("--steps", type=int, help=f"max optimization steps (current: {defaults['steps']})")
    parser.add_argument("--relax-cell", dest="relax_cell", action="store_true", help=f"relax cell as well (current: {defaults['relax_cell']})")
    parser.add_argument("--no-relax-cell", dest="relax_cell", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--cell-filter", choices=["exp", "frechet"], help=f"cell filter type (current: {defaults['cell_filter']})")
    parser.add_argument("--pressure", type=float, help=f"external pressure in GPa (current: {defaults['pressure']})")
    parser.add_argument("--stress-weight", type=float, dest="stress_weight", help=f"stress weight (current: {defaults['stress_weight']})")
    parser.add_argument("--output", help="single output file or output directory for multiple inputs")
    parser.add_argument("--output-format", choices=RELAX_OUTPUT_FORMATS, dest="output_format", help=f"output format (current: {defaults['output_format']})")
    parser.add_argument("--log-file", dest="log_file", help=f"optimizer logfile path (current: {defaults['log_file']})")
    parser.add_argument("--single", dest="single", action="store_true", help=f"only read the first structure from each input (current: {defaults['single']})")
    parser.add_argument("--no-single", dest="single", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--batch", dest="batch", action="store_true", help=f"expand wildcard inputs explicitly (current: {defaults['batch']})")
    parser.add_argument("--no-batch", dest="batch", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(keep_order=None, relax_cell=None, single=None, batch=None)


def _build_efs_distri_parser(subparsers):
    defaults = get_section_defaults("efs_distri")
    parser = subparsers.add_parser(
        "efs-distri",
        help="plot energy/force/stress distributions from xyz/extxyz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog("exactly one xyz/extxyz file, or `set key=value ...` to update defaults.", "efs_distri"),
    )
    parser.add_argument("items", nargs="*", help='input xyz file, or `set key=value ...`')
    parser.add_argument("--force-threshold", type=float, dest="force_threshold", help=f"frame filter by max force (current: {defaults['force_threshold']})")
    parser.add_argument("--bins", type=int, help=f"default histogram bin count (current: {defaults['bins']})")
    parser.add_argument("--energy-bins", type=int, dest="energy_bins", help=f"energy histogram bins (current: {defaults['energy_bins']})")
    parser.add_argument("--force-bins", type=int, dest="force_bins", help=f"force histogram bins (current: {defaults['force_bins']})")
    parser.add_argument("--stress-bins", type=int, dest="stress_bins", help=f"stress histogram bins (current: {defaults['stress_bins']})")
    parser.add_argument("--density", dest="density", action="store_true", help=f"plot probability density (current: {defaults['density']})")
    parser.add_argument("--no-density", dest="density", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--fit", dest="fit", action="store_true", help=f"show normal fit on density plots (current: {defaults['fit']})")
    parser.add_argument("--no-fit", dest="fit", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--figsize", type=float, nargs=2, help=f"figure size width height (current: {defaults['figsize']})")
    parser.add_argument("--dpi", type=int, help=f"output dpi (current: {defaults['dpi']})")
    parser.add_argument("--log-y", dest="log_y", action="store_true", help=f"use logarithmic y axis (current: {defaults['log_y']})")
    parser.add_argument("--no-log-y", dest="log_y", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--output", help=f"output image path (current: {defaults['output']})")
    parser.set_defaults(density=None, fit=None, log_y=None)


def _build_predict_xyz_parser(subparsers):
    defaults = get_section_defaults("predict_xyz")
    parser = subparsers.add_parser(
        "predict-xyz",
        help="run the bundled prediction tool that writes predicted xyz/extxyz outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog("exactly one input structure file, or `set key=value ...` to update defaults.", "predict_xyz"),
    )
    parser.add_argument("items", nargs="*", help='input structure file, or `set key=value ...`')
    parser.add_argument("--calc-type", "-c", choices=["nep", "mace", "chgnet", "dp", "m3gnet", "mattersim", "sus2"], dest="calc_type", help=f"calculator type (current: {defaults['calc_type']})")
    parser.add_argument("--model", "-m", help=f"model or potential path (current: {defaults['model']})")
    parser.add_argument("--elements", "-e", nargs="+", help=f"element list override (current: {defaults['elements']})")
    parser.add_argument("--device", "-d", choices=["cpu", "cuda"], help=f"compute device (current: {defaults['device']})")
    parser.add_argument("--output", "-o", help=f"output directory (current: {defaults['output']})")
    parser.add_argument("--format", "-f", help=f"output format (current: {defaults['format']})")
    parser.add_argument("--append", "-a", dest="append", action="store_true", help=f"append to existing output (current: {defaults['append']})")
    parser.add_argument("--no-append", dest="append", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--log-level", "-l", choices=["DEBUG", "INFO", "WARNING", "ERROR"], dest="log_level", help=f"log level (current: {defaults['log_level']})")
    parser.add_argument("--suffix", "-s", help=f"output filename suffix (current: {defaults['suffix']!r})")
    parser.add_argument("--num-workers", "-n", type=int, dest="num_workers", help=f"CPU worker count (current: {defaults['num_workers']})")
    parser.add_argument(
        "--split-workers",
        type=int,
        dest="split_workers",
        help=f"split xyz/extxyz into N files, predict them in parallel, then concatenate (current: {defaults['split_workers']})",
    )
    parser.set_defaults(append=None)


def _build_plot_errors_parser(subparsers):
    defaults = get_section_defaults("plot_errors")
    parser = subparsers.add_parser(
        "plot-errors",
        help="run the bundled SUS2/MLIP error-plot tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog(
            "exactly two files: DFT reference and MLIP prediction, or `set key=value ...` to update defaults.",
            "plot_errors",
        ),
    )
    parser.add_argument("items", nargs="*", help='DFT file + MLIP file, or `set key=value ...`')
    parser.add_argument("--mlip-name", dest="mlip_name", help=f"MLIP name shown in labels (current: {defaults['mlip_name']})")
    parser.add_argument("--elements", nargs="+", help=f"element list used for cfg conversion (current: {defaults['elements']})")
    parser.add_argument("--keep-temp", dest="keep_temp", action="store_true", help=f"keep temporary converted xyz files (current: {defaults['keep_temp']})")
    parser.add_argument("--no-keep-temp", dest="keep_temp", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--force-mode", choices=["magnitude", "components"], dest="force_mode", help=f"force display mode (current: {defaults['force_mode']})")
    parser.add_argument("--num-processes", type=int, dest="num_processes", help=f"worker process count (current: {defaults['num_processes']})")
    parser.add_argument("--output", help=f"output image file (current: {defaults['output']})")
    parser.add_argument("--stress-unit", choices=["eV", "ev", "GPa", "gpa"], dest="stress_unit", help=f"stress display unit (current: {defaults['stress_unit']})")
    parser.add_argument("--skip-structure-indices", type=int, nargs="*", dest="skip_structure_indices", help=f"structure indices skipped from plotting (current: {defaults['skip_structure_indices']})")
    parser.add_argument("--axis-padding", type=float, dest="axis_padding", help=f"axis padding fraction (current: {defaults['axis_padding']})")
    parser.add_argument("--signed-axis-symmetric", dest="signed_axis_symmetric", action="store_true", help=f"use symmetric signed axes (current: {defaults['signed_axis_symmetric']})")
    parser.add_argument("--no-signed-axis-symmetric", dest="signed_axis_symmetric", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--figsize", type=float, nargs=2, metavar=("WIDTH", "HEIGHT"), help=f"figure size (current: {defaults['figsize']})")
    parser.add_argument("--dpi", type=int, help=f"output image DPI (current: {defaults['dpi']})")
    parser.add_argument("--cmap", help=f"colormap name (current: {defaults['cmap']})")
    parser.add_argument("--scatter-size", type=int, dest="scatter_size", help=f"scatter point size (current: {defaults['scatter_size']})")
    parser.add_argument("--bins", type=int, help=f"density histogram bins (current: {defaults['bins']})")
    parser.add_argument("--fontsize", type=int, help=f"base font size (current: {defaults['fontsize']})")
    parser.add_argument("--tick-labelsize", type=int, dest="tick_labelsize", help=f"tick label size (current: {defaults['tick_labelsize']})")
    parser.add_argument("--legend-fontsize", type=int, dest="legend_fontsize", help=f"legend font size (current: {defaults['legend_fontsize']})")
    parser.add_argument("--title-fontsize", type=int, dest="title_fontsize", help=f"title font size (current: {defaults['title_fontsize']})")
    parser.add_argument("--annotation-fontsize", type=int, dest="annotation_fontsize", help=f"annotation font size (current: {defaults['annotation_fontsize']})")
    parser.add_argument("--cbar-fontsize", type=int, dest="cbar_fontsize", help=f"colorbar font size (current: {defaults['cbar_fontsize']})")
    parser.add_argument("--cbar-tick-size", type=int, dest="cbar_tick_size", help=f"colorbar tick label size (current: {defaults['cbar_tick_size']})")
    parser.add_argument("--linewidth", type=float, help=f"line width (current: {defaults['linewidth']})")
    parser.add_argument("--show-r2", dest="show_r2", action="store_true", help=f"show R² (current: {defaults['show_r2']})")
    parser.add_argument("--no-show-r2", dest="show_r2", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--save-data", dest="save_data", action="store_true", help=f"save raw comparison data (current: {defaults['save_data']})")
    parser.add_argument("--no-save-data", dest="save_data", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(keep_temp=None, show_r2=None, save_data=None, signed_axis_symmetric=None, skip_structure_indices=None)


def handle_train_command(args):
    items = list(args.items)
    if items and items[0] == "set":
        return _handle_set_command("train", items[1:])
    if len(items) != 1:
        raise SystemExit("dcbf train requires exactly one input xyz/extxyz file")

    options = get_section_defaults("train")
    for key in ("template", "min_dist", "max_dist", "radial_basis_size", "backend", "queue", "cores", "ptile", "max_iter", "sus2_exe", "train_env", "work_dir", "keep_order", "submit"):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    if args.elements is not None:
        options["elements"] = list(args.elements)

    report = prepare_training_workspace(Path(items[0]), options)
    print(f"input_xyz: {report['input_xyz']}")
    print(f"work_dir: {report['work_dir']}")
    print(f"train_cfg: {report['train_cfg']}")
    print(f"template: {report['template_path']}")
    print(f"rendered_template: {report['rendered_template']}")
    print(f"submission_script: {report['script_path']}")
    print(f"submit_command: {report['submit_command']}")
    print(f"elements: {', '.join(report['elements'])}")
    if report["submitted"]:
        print("job_submitted: true")
    return 0


def handle_relax_command(args):
    items = list(args.items)
    if items and items[0] == "set":
        return _handle_set_command("relax", items[1:])
    if not items:
        raise SystemExit("dcbf relax requires at least one input structure")

    options = get_section_defaults("relax")
    for key in ("model", "keep_order", "optimizer", "fmax", "steps", "relax_cell", "cell_filter", "pressure", "stress_weight", "output_format", "log_file", "single", "batch"):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    if args.elements is not None:
        options["elements"] = list(args.elements)

    model_path = options.get("model")
    if not model_path:
        raise SystemExit("dcbf relax requires a model path; set it with `dcbf relax set model=/path/to/p.mtp`")

    input_paths = _expand_input_items(items, options.get("batch", False))
    if not input_paths:
        raise SystemExit("No input files matched")

    output_override = getattr(args, "output", None)
    if len(input_paths) > 1 and output_override and Path(output_override).suffix:
        raise SystemExit("When relaxing multiple inputs, --output must be a directory")

    for input_path in input_paths:
        output_path = _resolve_relax_output_path(Path(input_path), output_override, options["output_format"], multiple=len(input_paths) > 1)
        relax_structures(
            input_path=Path(input_path),
            model_path=Path(model_path),
            output_path=output_path,
            elements=options.get("elements"),
            keep_order=bool(options.get("keep_order")),
            optimizer_name=options["optimizer"],
            fmax=float(options["fmax"]),
            steps=int(options["steps"]),
            relax_cell=bool(options["relax_cell"]),
            cell_filter=options["cell_filter"],
            pressure_gpa=float(options["pressure"]),
            stress_weight=float(options["stress_weight"]),
            logfile=options.get("log_file"),
            single=bool(options["single"]),
            output_format=options.get("output_format"),
        )
        print(f"relaxed: {input_path} -> {output_path}")
    return 0


def handle_efs_distri_command(args):
    items = list(args.items)
    if items and items[0] == "set":
        return _handle_set_command("efs_distri", items[1:])
    if len(items) != 1:
        raise SystemExit("dcbf efs-distri requires exactly one input xyz/extxyz file")

    options = get_section_defaults("efs_distri")
    for key in ("force_threshold", "bins", "energy_bins", "force_bins", "stress_bins", "density", "fit", "dpi", "log_y", "output"):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    if args.figsize is not None:
        options["figsize"] = [float(args.figsize[0]), float(args.figsize[1])]

    input_path = Path(items[0]).resolve()
    output_path = Path(options["output"]) if options.get("output") else input_path.with_name(f"{input_path.stem}_efs_distribution.jpg")
    analyzer = EFSDistributionAnalyzer(input_path, force_threshold=options["force_threshold"])
    analyzer.print_summary()
    analyzer.plot_distribution(
        bins=int(options["bins"]),
        bin_list=_build_bin_list(options),
        figsize=tuple(float(item) for item in options["figsize"]),
        density=bool(options["density"]),
        output_file=output_path,
        show_fit=bool(options["fit"]),
        log_y=bool(options["log_y"]),
        dpi=int(options["dpi"]),
    )
    print(f"plot_saved: {output_path}")
    return 0


def handle_predict_xyz_command(args):
    items = list(args.items)
    if items and items[0] == "set":
        return _handle_set_command("predict_xyz", items[1:])
    if len(items) != 1:
        raise SystemExit("dcbf predict-xyz requires exactly one input structure file")

    options = get_section_defaults("predict_xyz")
    for key in ("calc_type", "model", "device", "output", "format", "append", "log_level", "suffix", "num_workers", "split_workers"):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    if args.elements is not None:
        options["elements"] = list(args.elements)

    argv = [items[0], "--calc_type", str(options["calc_type"])]
    if options.get("model"):
        argv.extend(["--model", str(options["model"])])
    if options.get("elements"):
        argv.extend(["--ele_list", *options["elements"]])
    argv.extend(
        [
            "--device",
            str(options["device"]),
            "--output",
            str(options["output"]),
            "--format",
            str(options["format"]),
            "--log-level",
            str(options["log_level"]),
            "--suffix",
            str(options["suffix"]),
            "--num-workers",
            str(int(options["num_workers"])),
            "--split-workers",
            str(int(options["split_workers"])),
        ]
    )
    if options.get("append"):
        argv.append("--append")
    predict_xyz_main(argv)
    return 0


def handle_plot_errors_command(args):
    items = list(args.items)
    if items and items[0] == "set":
        return _handle_set_command("plot_errors", items[1:])
    if len(items) != 2:
        raise SystemExit("dcbf plot-errors requires exactly two input files: DFT and MLIP prediction")

    options = get_section_defaults("plot_errors")
    for key in (
        "mlip_name",
        "keep_temp",
        "force_mode",
        "num_processes",
        "output",
        "stress_unit",
        "axis_padding",
        "signed_axis_symmetric",
        "dpi",
        "cmap",
        "scatter_size",
        "bins",
        "fontsize",
        "tick_labelsize",
        "legend_fontsize",
        "title_fontsize",
        "annotation_fontsize",
        "cbar_fontsize",
        "cbar_tick_size",
        "linewidth",
        "show_r2",
        "save_data",
    ):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    if args.figsize is not None:
        options["figsize"] = [float(args.figsize[0]), float(args.figsize[1])]
    if args.elements is not None:
        options["elements"] = list(args.elements)
    if args.skip_structure_indices is not None:
        options["skip_structure_indices"] = list(args.skip_structure_indices)

    argv = [
        items[0],
        items[1],
        "--mlip-name",
        str(options["mlip_name"]),
        "--keep-temp",
        str(bool(options["keep_temp"])),
        "--force-mode",
        str(options["force_mode"]),
        "--num-processes",
        str(int(options["num_processes"])),
        "--output",
        str(options["output"]),
        "--stress-unit",
        str(options["stress_unit"]),
        "--axis-padding",
        str(float(options["axis_padding"])),
        "--signed-axis-symmetric",
        str(bool(options["signed_axis_symmetric"])),
        "--figsize",
        str(float(options["figsize"][0])),
        str(float(options["figsize"][1])),
        "--dpi",
        str(int(options["dpi"])),
        "--cmap",
        str(options["cmap"]),
        "--scatter-size",
        str(int(options["scatter_size"])),
        "--bins",
        str(int(options["bins"])),
        "--fontsize",
        str(int(options["fontsize"])),
        "--tick-labelsize",
        str(int(options["tick_labelsize"])),
        "--legend-fontsize",
        str(int(options["legend_fontsize"])),
        "--title-fontsize",
        str(int(options["title_fontsize"])),
        "--annotation-fontsize",
        str(int(options["annotation_fontsize"])),
        "--cbar-fontsize",
        str(int(options["cbar_fontsize"])),
        "--cbar-tick-size",
        str(int(options["cbar_tick_size"])),
        "--linewidth",
        str(float(options["linewidth"])),
        "--show-r2",
        str(bool(options["show_r2"])),
        "--save-data",
        str(bool(options["save_data"])),
    ]
    if options.get("elements"):
        argv.extend(["--elements", *options["elements"]])
    if options.get("skip_structure_indices"):
        argv.extend(["--skip-structure-indices", *[str(index) for index in options["skip_structure_indices"]]])
    plot_errors_main(argv)
    return 0


def _handle_set_command(section, assignments):
    if not assignments:
        print(format_section_defaults(section))
        return 0
    updated = set_section_defaults(section, assignments)
    for key in sorted(updated):
        value = updated[key]
        if isinstance(value, list):
            rendered = ",".join(str(item) for item in value)
        elif value is None:
            rendered = "<auto>"
        else:
            rendered = str(value)
        print(f"{key}={rendered}")
    return 0


def _expand_input_items(items, batch_enabled):
    expanded = []
    for item in items:
        if batch_enabled or any(token in item for token in "*?[]"):
            matches = sorted(glob.glob(item))
            if matches:
                expanded.extend(matches)
                continue
        expanded.append(item)
    return expanded


def _resolve_elements(xyz_path, explicit_elements=None, keep_order=False):
    if explicit_elements:
        elements = list(explicit_elements)
        if not keep_order:
            elements = sorted(elements, key=lambda element: atomic_numbers[element])
        return elements

    element_set = set()
    for atoms in iread(str(xyz_path), index=":"):
        element_set.update(atoms.get_chemical_symbols())
    return [chemical_symbols[number] for number in sorted(atomic_numbers[element] for element in element_set)]


def prepare_training_workspace(input_xyz, options):
    input_xyz = Path(input_xyz).resolve()
    if not input_xyz.exists():
        raise FileNotFoundError(f"Input xyz file does not exist: {input_xyz}")

    work_dir = _resolve_training_work_dir(input_xyz, options.get("work_dir"))
    work_dir.mkdir(parents=True, exist_ok=True)

    elements = _resolve_elements(input_xyz, explicit_elements=options.get("elements"), keep_order=bool(options.get("keep_order")))
    train_cfg = work_dir / "train.cfg"
    template_path = Path(__file__).resolve().parent / "training_assets" / TRAINING_TEMPLATE_CHOICES[options["template"]]
    rendered_template = work_dir / "hyx.mtp"
    xyz2cfg(elements, 2 if options.get("keep_order") else 1, str(input_xyz), str(train_cfg))
    _render_training_template(
        template_path=template_path,
        output_path=rendered_template,
        species_count=len(elements),
        min_dist=options.get("min_dist"),
        max_dist=options.get("max_dist"),
        radial_basis_size=options.get("radial_basis_size"),
    )

    backend = str(options["backend"]).strip().lower()
    backend = {"lsf": "bsub", "slurm": "sbatch"}.get(backend, backend)
    scheduler = build_scheduler_spec(
        normalize_scheduler_config(
            {
                "submission_backend": backend,
                "train_sus_queue": options["queue"],
                "train_sus_cores": options["cores"],
                "train_sus_ptile": options["ptile"],
                "sus2_mlp_exe": options["sus2_exe"],
                "original_command": "",
                "subsequent_command": "",
                "lmp_exe": "lmp",
                "scf_cal_engine": "abacus",
            }
        )
    )
    script_name = "bsub.lsf" if backend == "bsub" else "sbatch.slurm"
    submit_command = f"bsub < {script_name}" if backend == "bsub" else f"sbatch {script_name}"
    script_path = work_dir / script_name
    script_path.write_text(
        _build_training_script(
            scheduler=scheduler,
            sus2_exe=options["sus2_exe"],
            max_iter=int(options["max_iter"]),
            train_env=options.get("train_env"),
        ),
        encoding="utf-8",
    )

    mapping = {element: index for index, element in enumerate(elements)}
    report = {
        "input_xyz": str(input_xyz),
        "work_dir": str(work_dir),
        "train_cfg": str(train_cfg),
        "template_path": str(template_path),
        "rendered_template": str(rendered_template),
        "script_path": str(script_path),
        "submit_command": submit_command,
        "elements": elements,
        "element_mapping": mapping,
        "template": options["template"],
        "min_dist": options.get("min_dist"),
        "max_dist": options.get("max_dist"),
        "radial_basis_size": options.get("radial_basis_size"),
        "submitted": False,
    }
    report_path = work_dir / "train_meta.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if options.get("submit"):
        subprocess.run(submit_command, cwd=work_dir, shell=True, check=True)
        report["submitted"] = True
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _resolve_training_work_dir(input_xyz, configured_work_dir):
    if configured_work_dir:
        work_dir = Path(configured_work_dir)
        if not work_dir.is_absolute():
            work_dir = input_xyz.parent / work_dir
        return work_dir.resolve()
    return (input_xyz.parent / f"{input_xyz.stem}_train").resolve()


def _render_training_template(template_path, output_path, species_count, min_dist=None, max_dist=None, radial_basis_size=None):
    text = template_path.read_text(encoding="utf-8")
    replacements = {
        "species_count": species_count,
        "min_dist": min_dist,
        "max_dist": max_dist,
        "radial_basis_size": radial_basis_size,
    }
    for key, value in replacements.items():
        if value is None and key != "species_count":
            continue
        text, count = re.subn(
            rf"(^\s*{key}\s*=\s*)([-+0-9.eE]+)",
            rf"\g<1>{value}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if count == 0:
            raise ValueError(f"Could not find {key} in template: {template_path}")
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _build_training_script(scheduler, sus2_exe, max_iter, train_env=None):
    train_env_text = f"{train_env.strip()}\n" if train_env else ""
    command = (
        f"mpirun -n {scheduler.train_sus_cores} {sus2_exe} train hyx.mtp train.cfg "
        f"--stdd-weight=0.0 --std-weight=0.0000 --max-iter={max_iter} "
        "--do-lin --curr-pot-name=current.mtp --trained-pot-name=p.mtp"
    )
    return (
        "#!/bin/bash\n"
        f"{scheduler.bsub_script_train_sus_job_name}dcbf_train\n"
        f"{scheduler.bsub_script_train_sus}"
        f"{train_env_text}"
        "start_time=$(date +%s.%N)\n"
        f"COMMAND_std=\"{command}\"\n"
        "$COMMAND_std > logout 2>&1\n"
        "status=$?\n"
        "end_time=$(date +%s.%N)\n"
        "runtime=$(echo \"$end_time - $start_time\" | bc)\n"
        "echo \"total_runtime:$runtime s\" >> time.txt\n"
        "exit $status\n"
    )


def relax_structures(input_path, model_path, output_path, elements=None, keep_order=False, optimizer_name="BFGSLineSearch", fmax=0.05, steps=500, relax_cell=True, cell_filter="exp", pressure_gpa=0.0, stress_weight=1.0, logfile=None, single=False, output_format=None):
    if not PYMLIP_AVAILABLE:
        raise ImportError("pymlip is required for `dcbf relax`")

    input_path = Path(input_path).resolve()
    model_path = Path(model_path).resolve()
    output_path = Path(output_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    resolved_elements = _resolve_relax_elements(input_path, elements, keep_order)
    structures = read(str(input_path), index=0 if single else ":")
    if isinstance(structures, Atoms):
        structures = [structures]
    else:
        structures = list(structures)

    calculator = SUS2Calculator(model_path, resolved_elements, stress_weight=stress_weight)
    optimizer_cls = RELAX_OPTIMIZERS[optimizer_name]
    processed = []
    for index, atoms in enumerate(structures):
        atoms = atoms.copy()
        atoms.calc = calculator
        target = atoms
        if relax_cell:
            scalar_pressure = float(pressure_gpa) * GPA_TO_EV_PER_A3
            if cell_filter == "frechet":
                if FrechetCellFilter is None:
                    raise ImportError("FrechetCellFilter is not available in the current ASE version")
                target = FrechetCellFilter(atoms, scalar_pressure=scalar_pressure)
            else:
                target = ExpCellFilter(atoms, scalar_pressure=scalar_pressure)
        optimizer = optimizer_cls(target, logfile=_resolve_relax_logfile(logfile, input_path, index))
        optimizer.run(fmax=float(fmax), steps=int(steps))
        relaxed_atoms = target.atoms if hasattr(target, "atoms") else atoms
        processed.append(_annotate_structure(relaxed_atoms))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_structures(output_path, processed, output_format=output_format)


def _resolve_relax_elements(input_path, explicit_elements, keep_order):
    if explicit_elements:
        elements = list(explicit_elements)
        if not keep_order:
            elements = sorted(elements, key=lambda element: atomic_numbers[element])
        return elements

    detected = set()
    for atoms in iread(str(input_path), index=":"):
        detected.update(atoms.get_chemical_symbols())
    return [chemical_symbols[number] for number in sorted(atomic_numbers[element] for element in detected)]


def _resolve_relax_output_path(input_path, output_override, output_format, multiple=False):
    if output_override:
        output_override = Path(output_override)
        if multiple or output_override.is_dir() or not output_override.suffix:
            output_override.mkdir(parents=True, exist_ok=True)
            filename = _default_output_name(input_path, suffix="_relaxed", output_format=output_format)
            return (output_override / filename).resolve()
        return output_override.resolve()
    return (input_path.parent / _default_output_name(input_path, suffix="_relaxed", output_format=output_format)).resolve()


def _default_output_name(input_path, suffix, output_format=None):
    if output_format:
        extension = ".xyz" if output_format == "extxyz" else f".{output_format}"
        return f"{input_path.stem}{suffix}{extension}"

    extension = input_path.suffix or ".xyz"
    if extension.lower() == ".extxyz":
        extension = ".xyz"
    return f"{input_path.stem}{suffix}{extension}"


def _resolve_relax_logfile(logfile, input_path, index):
    if logfile:
        path = Path(logfile)
        if path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            return str(path / f"{input_path.stem}_{index}.relax.log")
        return str(path)
    return str(input_path.with_name(f"{input_path.stem}_{index}.relax.log"))


def _annotate_structure(atoms):
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    stress_matrix = coerce_tensor_3x3(atoms.get_stress(voigt=False))
    virial = -stress_matrix * atoms.get_volume()
    annotated = Atoms(atoms.get_chemical_symbols(), positions=atoms.get_positions(), cell=atoms.get_cell(), pbc=atoms.get_pbc())
    annotated.arrays["forces"] = forces
    annotated.info["energy"] = energy
    annotated.info["stress"] = stress_matrix.reshape(-1).tolist()
    annotated.info["virial"] = virial.reshape(3, 3).tolist()
    return annotated


def _write_structures(output_path, structures, output_format=None):
    fmt = (output_format or output_path.suffix.lstrip(".") or "xyz").lower()
    if fmt in {"xyz", "extxyz"}:
        write_normalized_extxyz(output_path, structures, append=False)
        return
    if len(structures) == 1:
        write(str(output_path), structures[0], format=fmt)
        return
    write(str(output_path), structures, format=fmt)


def _build_bin_list(options):
    energy_bins = options.get("energy_bins")
    force_bins = options.get("force_bins")
    stress_bins = options.get("stress_bins")
    if energy_bins is None and force_bins is None and stress_bins is None:
        return None
    base = int(options["bins"])
    return [
        int(energy_bins if energy_bins is not None else base),
        int(force_bins if force_bins is not None else base),
        int(stress_bins if stress_bins is not None else base),
    ]
