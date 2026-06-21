from __future__ import annotations

from pathlib import Path

from ase.io import iread
import matplotlib.pyplot as plt
import numpy as np

from .tensor_utils import extract_stress_matrix_gpa


class EFSDistributionAnalyzer:
    def __init__(self, xyz_file, force_threshold=None):
        self.xyz_file = str(xyz_file)
        self.force_threshold = force_threshold
        self.energies = []
        self.forces = []
        self.stresses = []
        self.stats = {
            "energy_count": 0,
            "force_count": 0,
            "stress_count": 0,
            "filtered_count": 0,
            "total_frames": 0,
        }
        self._extract_data()

    def _extract_data(self):
        for frame in iread(self.xyz_file):
            self.stats["total_frames"] += 1
            if not self._include_frame(frame):
                self.stats["filtered_count"] += 1
                continue

            try:
                energy = frame.get_potential_energy() / frame.get_global_number_of_atoms()
                self.energies.append(float(energy))
                self.stats["energy_count"] += 1
            except Exception:
                pass

            try:
                forces = frame.get_forces()
                self.forces.extend(np.asarray(forces, dtype=float).reshape(-1))
                self.stats["force_count"] += 1
            except Exception:
                pass

            try:
                stress = extract_stress_matrix_gpa(frame)
                if stress is not None:
                    self.stresses.extend(np.asarray(stress, dtype=float).reshape(-1))
                    self.stats["stress_count"] += 1
            except Exception:
                pass

    def _include_frame(self, frame):
        if self.force_threshold is None:
            return True
        try:
            forces = np.asarray(frame.get_forces(), dtype=float)
            return np.max(np.linalg.norm(forces, axis=1)) <= float(self.force_threshold)
        except Exception:
            return True

    def get_statistics(self):
        stats = {"processing": dict(self.stats)}
        if self.energies:
            stats["energy"] = self._metric_block(np.asarray(self.energies, dtype=float))
        if self.forces:
            stats["force"] = self._metric_block(np.asarray(self.forces, dtype=float))
        if self.stresses:
            stats["stress"] = self._metric_block(np.asarray(self.stresses, dtype=float))
        return stats

    @staticmethod
    def _metric_block(values):
        mean = float(np.mean(values))
        return {
            "mean": mean,
            "std": float(np.std(values)),
            "mad": float(np.mean(np.abs(values - mean))),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": int(values.size),
        }

    def print_summary(self):
        stats = self.get_statistics()
        proc = stats["processing"]
        print("=" * 70)
        print("EFS DISTRIBUTION ANALYSIS SUMMARY")
        print("=" * 70)
        print("\nProcessing Statistics:")
        print(f"  Total frames processed: {proc['total_frames']}")
        print(f"  Frames filtered by force threshold: {proc['filtered_count']}")

        if "energy" in stats:
            self._print_metric_section("Energy Statistics (eV/atom)", proc["energy_count"], stats["energy"])
        if "force" in stats:
            self._print_metric_section("Force Statistics (eV/A)", proc["force_count"], stats["force"])
        if "stress" in stats:
            self._print_metric_section("Stress Statistics (GPa)", proc["stress_count"], stats["stress"])
        print("\n" + "=" * 70)

    @staticmethod
    def _print_metric_section(title, structure_count, block):
        print(f"\n{title}:")
        print(f"  Number of structures: {structure_count}")
        print(f"  Range: [{block['min']:.4f}, {block['max']:.4f}]")
        print(f"  Mean: {block['mean']:.4f}")
        print(f"  Std Dev: {block['std']:.4f}")
        print(f"  Mean Absolute Deviation (MAD): {block['mad']:.4f}")

    def plot_distribution(
        self,
        bins=120,
        bin_list=None,
        figsize=(30, 10),
        density=False,
        output_file="efs_distribution.jpg",
        show_fit=False,
        log_y=False,
        dpi=300,
    ):
        self._setup_plot_style()
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        if bin_list is None:
            energy_bins = force_bins = stress_bins = bins
        else:
            energy_bins, force_bins, stress_bins = bin_list

        stats = self.get_statistics()
        if self.energies:
            self._plot_single_distribution(
                axes[0],
                self.energies,
                "Energy (eV/atom)",
                "skyblue",
                energy_bins,
                density,
                stats.get("energy", {}),
                show_fit,
                log_y,
            )
        if self.forces:
            self._plot_single_distribution(
                axes[1],
                self.forces,
                "Force (eV/A)",
                "lightgreen",
                force_bins,
                density,
                stats.get("force", {}),
                show_fit,
                log_y,
            )
        if self.stresses:
            self._plot_single_distribution(
                axes[2],
                self.stresses,
                "Stress (GPa)",
                "salmon",
                stress_bins,
                density,
                stats.get("stress", {}),
                show_fit,
                log_y,
            )

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return fig

    def _plot_single_distribution(self, ax, data, xlabel, color, bins, density, stats_dict, show_fit, log_y):
        ylabel = "Probability Density" if density else "Frequency"
        if log_y:
            ylabel += " (log scale)"
            ax.set_yscale("log")

        ax.hist(data, bins=bins, alpha=0.7, color=color, edgecolor="black", density=density)
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")

        if stats_dict:
            legend_text = (
                f"mean={stats_dict['mean']:.3f}\n"
                f"std={stats_dict['std']:.3f}\n"
                f"mad={stats_dict['mad']:.3f}"
            )
            ax.plot([], [], " ", label=legend_text)
            ax.legend(loc="best", frameon=True, fontsize=20, handlelength=0, handletextpad=0)

        if density and show_fit and len(data) > 1:
            try:
                from scipy.stats import norm

                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 200)
                fit = norm.pdf(x, stats_dict["mean"], stats_dict["std"])
                if log_y:
                    fit = np.maximum(fit, 1e-300)
                ax.plot(x, fit, "r--", linewidth=3)
            except Exception:
                pass

    @staticmethod
    def _setup_plot_style():
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams.update(
            {
                "font.size": 28,
                "axes.titlesize": 28,
                "axes.titleweight": "bold",
                "axes.labelsize": 28,
                "axes.labelweight": "bold",
                "xtick.labelsize": 26,
                "ytick.labelsize": 26,
                "legend.fontsize": 20,
                "legend.title_fontsize": 22,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "axes.linewidth": 3,
                "xtick.major.width": 3,
                "ytick.major.width": 3,
                "xtick.major.size": 12,
                "ytick.major.size": 12,
                "xtick.minor.width": 2,
                "ytick.minor.width": 2,
                "xtick.minor.size": 6,
                "ytick.minor.size": 6,
            }
        )
