from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ASSET_DIR = Path(__file__).resolve().parent / "training_assets"


def _run_asset_module(module_name: str, script_name: str, argv=None):
    script_path = ASSET_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Training asset script does not exist: {script_path}")

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {script_path}")

    module = importlib.util.module_from_spec(spec)
    previous_argv = sys.argv[:]
    try:
        sys.modules[module_name] = module
        if argv is not None:
            sys.argv = [script_name, *argv]
        spec.loader.exec_module(module)
        if not hasattr(module, "main") and not hasattr(module, "sus2_plot_errors_main"):
            raise AttributeError(f"No runnable entry function found in {script_path}")
        if hasattr(module, "main"):
            return module.main()
        return module.sus2_plot_errors_main()
    finally:
        sys.argv = previous_argv
        sys.modules.pop(module_name, None)


def predict_xyz_main(argv=None):
    from .training_assets import scf_sus2_mace_chgnet_nep_mattersim_dp_v3

    previous_argv = sys.argv[:]
    try:
        if argv is not None:
            sys.argv = ["scf_sus2_mace_chgnet_nep_mattersim_dp_v3.py", *argv]
        return scf_sus2_mace_chgnet_nep_mattersim_dp_v3.main()
    finally:
        sys.argv = previous_argv


def plot_errors_main(argv=None):
    from . import sus2_plot_errors_v3

    previous_argv = sys.argv[:]
    try:
        if argv is not None:
            sys.argv = ["sus2_plot_errors-v3.py", *argv]
        return sus2_plot_errors_v3.sus2_plot_errors_main()
    finally:
        sys.argv = previous_argv


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("Usage: python -m dcbf.high_precision_tools {predict|plot} [args...]")

    command = argv[0]
    args = argv[1:]
    if command == "predict":
        return predict_xyz_main(args)
    if command == "plot":
        return plot_errors_main(args)
    raise SystemExit(f"Unsupported command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
