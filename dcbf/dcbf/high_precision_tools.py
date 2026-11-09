from __future__ import annotations

from pathlib import Path
import sys


def predict_xyz_main(argv=None):
    import argparse
    from concurrent.futures import ThreadPoolExecutor
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import iread, write
    from .native_calculator import NativeSUS2Calculator

    parser = argparse.ArgumentParser(description='SUS2-SH native CPU prediction')
    parser.add_argument('input')
    parser.add_argument('--calc_type', default='sus2', choices=['sus2'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--ele_list', nargs='+', required=False, default=None)
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--output', default='out_files')
    parser.add_argument('--format', default='extxyz')
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--split-workers', type=int, default=1)
    parser.add_argument('--log-level', default='INFO')
    args = parser.parse_args(argv)
    workers = max(args.num_workers, args.split_workers)
    if min(args.num_workers, args.split_workers) <= 0:
        raise ValueError('Prediction worker counts must be positive')
    output = Path(args.output)
    if output.suffix.lower() not in {'.xyz', '.extxyz', '.traj'}:
        output.mkdir(parents=True, exist_ok=True)
        suffix = '_' + args.suffix if args.suffix else ''
        extension = 'xyz' if args.format in {'xyz', 'extxyz'} else args.format
        output = output / f'{Path(args.input).stem}{suffix}.{extension}'
    output.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(atoms):
        atoms = atoms.copy()
        calc = NativeSUS2Calculator(args.model, args.ele_list)
        atoms.calc = calc
        energy, forces, stress = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
        atoms.info.pop('virial', None)
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
        calc.close()
        return atoms

    frames = iread(args.input, index=':')
    if workers == 1:
        predictions = [evaluate(atoms) for atoms in frames]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            predictions = list(pool.map(evaluate, frames))
    write(output, predictions, format=args.format, append=args.append)
    print(f'predicted: {len(predictions)} frames -> {output}')
    return 0


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
