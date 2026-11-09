from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from ase.data import atomic_numbers, chemical_symbols
from ase.io import iread

from .cli_defaults import defaults_epilog, format_section_defaults, get_section_defaults, set_section_defaults
from .cli_examples import append_command_example
from .das.file_conversion import xyz2cfg
from .efs_distribution import EFSDistributionAnalyzer
from .high_precision_tools import plot_errors_main, predict_xyz_main
from .high_precision_training import TRAINING_TEMPLATE_CHOICES
from .sus2_training import load_model_yaml, write_run_yaml
from .runtime_config import build_scheduler_spec, normalize_package_activation, normalize_scheduler_config
from .runtime_paths import resolve_program, runtime_identity


def _example_dir_candidates(root):
    yield root / "example"
    yield root / "source" / "DCBF" / "example"
    yield root / "source" / "OCBF_stability" / "example"


def resolve_example_dir():
    """Locate the source example directory, honoring DCBF_ROOT/DCBF_HOME.

    Source installs place the package in site-packages while ``example/``
    stays in the checkout.  The environment variables let ``dcbf train``
    resolve its default SUS2 YAML without requiring an absolute path.
    """
    checked = set()

    def should_check(candidate):
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in checked:
            return False
        checked.add(resolved)
        return True

    for env_name in ("DCBF_ROOT", "DCBF_HOME"):
        raw_root = os.environ.get(env_name)
        if not raw_root:
            continue
        for candidate in _example_dir_candidates(Path(raw_root)):
            if should_check(candidate) and candidate.is_dir():
                return candidate

    for parent in Path(__file__).resolve().parents:
        for candidate in _example_dir_candidates(parent):
            if should_check(candidate) and candidate.is_dir():
                return candidate

    checked_text = "\n".join(str(path) for path in sorted(checked))
    raise FileNotFoundError(
        "Could not locate the source example directory; set DCBF_ROOT to the "
        "checkout root. Checked:\n"
        f"{checked_text}"
    )


def add_high_precision_subparsers(subparsers):
    _build_train_parser(subparsers)
    _build_efs_distri_parser(subparsers)
    _build_predict_xyz_parser(subparsers)
    _build_plot_errors_parser(subparsers)


def _required_epilog(required_text, defaults_section=None, command=None):
    sections = [f"Required input:\n  {required_text}"]
    if defaults_section:
        sections.append(defaults_epilog(defaults_section))
    text = "\n\n".join(sections)
    return append_command_example(text, command) if command else text


def _build_train_parser(subparsers):
    defaults = get_section_defaults("train")
    parser = subparsers.add_parser(
        "train",
        help="generate SUS2 training files from one xyz/extxyz dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog(
            "exactly one xyz/extxyz dataset, or `set key=value ...` to update defaults.",
            "train",
            "train",
        ),
    )
    parser.add_argument("items", nargs="*", help='input xyz file, or `set key=value ...`')
    parser.add_argument('--model', help='continue a complete .sus2 checkpoint, or builtin:sus2_87; omit to initialize from YAML')
    parser.add_argument("--template", choices=sorted(TRAINING_TEMPLATE_CHOICES), help=f"training template type (current: {defaults['template']})")
    parser.add_argument(
        "--model-yaml",
        dest="model_yaml",
        help=("SUS2 architecture YAML (static-radial or coordination-effect; "
              f"current: {defaults['model_yaml'] or '<bundled static default>'})"),
    )
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


def _build_efs_distri_parser(subparsers):
    defaults = get_section_defaults("efs_distri")
    parser = subparsers.add_parser(
        "efs-distri",
        help="plot energy/force/stress distributions from xyz/extxyz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog(
            "exactly one xyz/extxyz file, or `set key=value ...` to update defaults.",
            "efs_distri",
            "efs-distri",
        ),
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
    parser.add_argument(
        "--output",
        help=(
            "output image path (current: "
            f"{defaults['output'] or '<cwd>/<input-stem>_efs_distribution.jpg'}); "
            "relative paths are resolved from the current directory"
        ),
    )
    parser.set_defaults(density=None, fit=None, log_y=None)


def _build_predict_xyz_parser(subparsers):
    defaults = get_section_defaults("predict_xyz")
    parser = subparsers.add_parser(
        "predict-xyz",
        help="run the bundled prediction tool that writes predicted xyz/extxyz outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog(
            "exactly one input structure file, or `set key=value ...` to update defaults.",
            "predict_xyz",
            "predict-xyz",
        ),
    )
    parser.add_argument("items", nargs="*", help='input structure file, or `set key=value ...`')
    parser.add_argument("--calc-type", "-c", choices=["sus2"], dest="calc_type", help=f"calculator type (current: {defaults['calc_type']})")
    parser.add_argument("--model", "-m", help=f"model or potential path, or builtin:sus2_87 (current: {defaults['model']})")
    parser.add_argument("--elements", "-e", nargs="+", help="element order for a custom model; builtin models use their registry order")
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
            "plot-errors",
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
    for key in ("template", "model_yaml", "min_dist", "max_dist", "radial_basis_size", "backend", "queue", "cores", "ptile", "max_iter", "sus2_exe", "train_env", "work_dir", "keep_order", "submit"):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    if args.elements is not None:
        options["elements"] = list(args.elements)

    if args.model is not None:
        options['model'] = args.model
        if any(getattr(args, k, None) is not None for k in ('min_dist', 'max_dist', 'radial_basis_size')):
            raise ValueError('Continuation inherits its architecture; omit radial/min/max architecture overrides')

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
    if options.get("output"):
        output_path = Path(options["output"]).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path = output_path.resolve()
    else:
        output_path = (Path.cwd() / f"{input_path.stem}_efs_distribution.jpg").resolve()
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
    calc_type = str(options.get("calc_type", "")).strip().lower()
    if calc_type != "sus2":
        raise SystemExit("dcbf predict-xyz supports only --calc-type sus2")
    if str(options.get("device", "cpu")).strip().lower() != "cpu":
        raise SystemExit("SUS2 CPU-only mode does not support GPU prediction")
    model = options.get("model")
    if model:
        from .builtin_models import registered_elements, resolve_model_path
        from .encode.model_metadata import read_sus2_descriptor_metadata

        resolved_model = resolve_model_path(model)
        elements = registered_elements(resolved_model, options.get("elements"))
        if elements is None:
            elements = options.get("elements")
            if not elements:
                raise SystemExit(
                    "dcbf predict-xyz requires --elements for a custom .sus2 model; "
                    "builtin models use their verified registry element order"
                )
            schema = read_sus2_descriptor_metadata(resolved_model)
            if len(elements) != schema.species_count or len(set(elements)) != len(elements):
                raise SystemExit(
                    "dcbf predict-xyz --elements must list the complete model element "
                    f"order ({schema.species_count} species, no duplicates)"
                )
        options["model"] = str(resolved_model)
        options["elements"] = list(elements)

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
    options = dict(options)
    options["sus2_exe"] = resolve_program("mlp-sus2", options.get("sus2_exe"))
    input_xyz = Path(input_xyz).resolve()
    if not input_xyz.exists():
        raise FileNotFoundError(f"Input xyz file does not exist: {input_xyz}")

    work_dir = _resolve_training_work_dir(input_xyz, options.get("work_dir"))
    work_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = work_dir / "train.cfg"
    checkpoint = None
    checkpoint_kind = None
    if options.get('model'):
        from .builtin_models import resolve_model_path, registered_elements
        from .encode.model_metadata import read_sus2_descriptor_metadata
        from .reduce_execution import model_kind as detect_checkpoint_kind
        from .sus2_training import check_complete_checkpoint
        from .labelled_data import write_training_cfg

        checkpoint = check_complete_checkpoint(resolve_model_path(options['model']))
        checkpoint_kind = detect_checkpoint_kind(checkpoint)

    model_yaml = options.get("model_yaml")
    if model_yaml:
        model_yaml = Path(model_yaml).expanduser().resolve()
    else:
        example_dir = resolve_example_dir()
        yaml_name = (
            "sus2_87-continue.yaml"
            if checkpoint_kind == "coordination-effect"
            else "sus2-static.yaml"
        )
        model_yaml = example_dir / yaml_name
    model_document, model_kind = load_model_yaml(model_yaml)

    if checkpoint is not None:
        if checkpoint_kind != model_kind:
            raise ValueError('model_yaml static/effect kind does not match continuation checkpoint')
        if checkpoint in {work_dir/n for n in ('current.sus2', 'best.sus2', 'final.sus2')}:
            raise ValueError('Choose a new --work-dir; continuation outputs must not overwrite the seed')
        elements = registered_elements(checkpoint, options.get('elements'))
        if elements is None:
            elements = options.get('elements')
            sidecar = Path(str(checkpoint)+'.elements.json')
            if elements is None and sidecar.is_file():
                from .reduce_state import sha
                record = json.loads(sidecar.read_text())
                if record['model_sha256'] != sha(checkpoint):
                    raise ValueError('Checkpoint element sidecar SHA-256 mismatch')
                elements = record['elements']
        schema = read_sus2_descriptor_metadata(checkpoint)
        if elements is None or len(elements) != schema.species_count or len(set(elements)) != len(elements):
            raise ValueError('Continuation requires the complete checkpoint element order, not just dataset elements; use --elements')
        elements = list(elements)
        label_stats = write_training_cfg(input_xyz, train_cfg, elements)
        if label_stats['effective'] == 'ef':
            model_document.setdefault('loss', {})['virial_weight'] = 0
    else:
        elements = _resolve_elements(input_xyz, explicit_elements=options.get('elements'), keep_order=bool(options.get('keep_order')))
        xyz2cfg(elements, 2 if options.get('keep_order') else 1, str(input_xyz), str(train_cfg))
    run_yaml = work_dir / "run.yaml"
    write_run_yaml(
        run_yaml,
        model_document,
        train_cfg=train_cfg,
        output_dir=work_dir,
        iterations=int(options["max_iter"]),
        continuation_model=checkpoint,
        output_name="current.sus2",
        best_name="best.sus2",
        elements=elements,
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
            train_section=_render_train_scheduler_section(
                scheduler, options.get("cores"), options.get("ptile")
            ),
        ),
        encoding="utf-8",
    )

    mapping = {element: index for index, element in enumerate(elements)}
    report = {
        "input_xyz": str(input_xyz),
        "work_dir": str(work_dir),
        "train_cfg": str(train_cfg),
        "template_path": str(model_yaml),
        "rendered_template": str(run_yaml),
        "script_path": str(script_path),
        "submit_command": submit_command,
        "elements": elements,
        "element_mapping": mapping,
        "template": options["template"],
        "model_kind": model_kind,
        "continuation_model": str(checkpoint) if checkpoint else None,
        "min_dist": options.get("min_dist"),
        "max_dist": options.get("max_dist"),
        "radial_basis_size": options.get("radial_basis_size"),
        "submitted": False,
        "runtime": runtime_identity(),
    }
    report_path = work_dir / "train_meta.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (work_dir/'model_elements.json').write_text(json.dumps({'elements': elements}, indent=2)+'\n')

    if options.get("submit"):
        subprocess.run(submit_command, cwd=work_dir, shell=True, check=True)
        report["submitted"] = True
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def _resolve_training_work_dir(input_xyz, configured_work_dir):
    if configured_work_dir:
        work_dir = Path(configured_work_dir).expanduser()
        if not work_dir.is_absolute():
            work_dir = Path.cwd() / work_dir
        return work_dir.resolve()
    return (input_xyz.parent / f"{input_xyz.stem}_train").resolve()


def _render_train_scheduler_section(scheduler, cores, ptile):
    section = scheduler.bsub_script_train_sus
    if cores is not None and ptile is not None:
        return section
    kept_lines = []
    for line in section.splitlines(keepends=True):
        stripped = line.strip()
        if cores is None and stripped.startswith(("#BSUB -n ", "#SBATCH -n ")):
            continue
        if stripped.startswith("NP=${SLURM_NTASKS:-") and "None" in stripped:
            kept_lines.append("NP=${SLURM_NTASKS:-1}\n")
            continue
        if ptile is None and (
            stripped.startswith('#BSUB -R "span[ptile=')
            or stripped.startswith("#SBATCH --ntasks-per-node=")
        ):
            continue
        kept_lines.append(line)
    return "".join(kept_lines)


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


def _build_training_script(scheduler, sus2_exe, max_iter, train_env=None, train_section=None):
    if not train_env:
        dcbf_home = os.environ.get("DCBF_HOME")
        if dcbf_home:
            train_env = 'source "${DCBF_HOME}/activate.sh"'
    train_env_text = f"{normalize_package_activation(train_env)}\n" if train_env else ""
    command = (
        'mpirun -np "${NP:-1}" '
        f"{shlex.quote(str(scheduler.sus2_mlp_exe or sus2_exe))} train --config run.yaml"
    )
    sidecars = shlex.quote(sys.executable) + ' -c ' + shlex.quote(
        "from dcbf.sus2_training import write_element_sidecars; write_element_sidecars('.')")
    if train_section is None:
        train_section = scheduler.bsub_script_train_sus
    return (
        "#!/bin/bash\n"
        f"{scheduler.bsub_script_train_sus_job_name}dcbf_train\n"
        f"{train_section}"
        f"{train_env_text}"
        "start_time=$(date +%s.%N)\n"
        f"{command} > logout 2>&1\n"
        "status=$?\n"
        f"if [ \"$status\" -eq 0 ]; then {sidecars}; status=$?; fi\n"
        "end_time=$(date +%s.%N)\n"
        "runtime=$(echo \"$end_time - $start_time\" | bc)\n"
        "echo \"total_runtime:$runtime s\" >> time.txt\n"
        "exit $status\n"
    )

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
