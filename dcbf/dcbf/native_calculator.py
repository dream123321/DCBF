"""In-process ASE calculator for the bundled SUS2-SH CPU evaluator.

The public DCBF calculator name is retained for callers, but evaluation is
performed by ``pysuperlinear.SuperlinearCalculator``. The model is loaded
once per Python process and each ASE calculation stays inside the C++ backend;
no temporary CFG files or ``mlp-sus2`` subprocesses are created.
"""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .builtin_models import registered_elements, resolve_model_path
from .encode.model_metadata import read_sus2_descriptor_metadata
from .sus2_training import check_complete_checkpoint

try:
    from pysuperlinear import SuperlinearCalculator as _SuperlinearCalculator
except (ImportError, OSError) as exc:  # pragma: no cover - broken install
    _SUPERLINEAR_IMPORT_ERROR = exc
    _SuperlinearCalculator = None
else:
    _SUPERLINEAR_IMPORT_ERROR = None


class NativeSUS2Calculator(Calculator):
    """Persistent native SUS2-SH calculator exposed through ASE.

    ``ele_list`` must be the complete model species order. The bundled model
    registry supplies that order automatically for ``builtin:sus2_87``.
    Each worker process owns one backend instance, so this calculator is safe
    for the dataset builder's process based ``parallel_workers`` mode.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, potential, ele_list=None, compute_stress=True, stress_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.potential = str(check_complete_checkpoint(resolve_model_path(potential)))
        schema = read_sus2_descriptor_metadata(self.potential)
        self.elements = registered_elements(self.potential, ele_list) or list(ele_list or [])
        if len(self.elements) != schema.species_count or len(set(self.elements)) != len(self.elements):
            raise ValueError("ele_list must give the full SUS2 model species order")
        self.compute_stress = bool(compute_stress)
        self.stress_weight = float(stress_weight)
        if _SuperlinearCalculator is None:
            raise ImportError(
                "The bundled in-process SUS2 ASE backend is unavailable; "
                "install the package runtime with pysuperlinear."
            ) from _SUPERLINEAR_IMPORT_ERROR
        self.backend = _SuperlinearCalculator(self.potential, species=tuple(self.elements))

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        requested = tuple(properties or ("energy", "forces", "stress"))
        self.backend.calculate(self.atoms, properties=requested, system_changes=system_changes)
        self.results = {
            key: (value.copy() if isinstance(value, np.ndarray) else value)
            for key, value in self.backend.results.items()
        }
        if not self.compute_stress:
            self.results.pop("stress", None)
        elif "stress" in self.results and self.stress_weight != 1.0:
            self.results["stress"] = np.asarray(self.results["stress"]) * self.stress_weight
        if any(
            not np.isfinite(value).all()
            for value in self.results.values()
            if isinstance(value, np.ndarray)
        ):
            raise RuntimeError("Native SUS2 returned non-finite energy, forces or stress")

    def close(self):
        """Release the backend reference; kept for compatibility with old callers."""
        self.backend = None


# Explicit name for new code while preserving the old import path.
SuperlinearSUS2Calculator = NativeSUS2Calculator
