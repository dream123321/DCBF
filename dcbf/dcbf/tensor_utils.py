from __future__ import annotations

import numpy as np


EV_PER_A3_TO_GPA = 160.21766208
GPA_TO_EV_PER_A3 = 1.0 / EV_PER_A3_TO_GPA
KBAR_TO_EV_PER_A3 = 0.0006241509073


def _normalize_unit(unit):
    if unit is None:
        return "auto"
    normalized = str(unit).strip().lower()
    normalized = normalized.replace(" ", "").replace("_", "").replace("-", "")
    normalized = normalized.replace("angstrom", "ang")
    normalized = normalized.replace("å", "a")
    normalized = normalized.replace("^", "")
    mapping = {
        "auto": "auto",
        "eva3": "ev_a3",
        "ev/a3": "ev_a3",
        "ev/ang3": "ev_a3",
        "ev/a3)": "ev_a3",
        "gpa": "gpa",
        "kbar": "kbar",
    }
    return mapping.get(normalized, normalized)


def coerce_tensor_3x3(value):
    array = np.asarray(value, dtype=float)
    if array.shape == (3, 3):
        return array
    flat = array.reshape(-1)
    if flat.size == 6:
        xx, yy, zz, yz, xz, xy = flat
        return np.array(
            [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
            dtype=float,
        )
    if flat.size == 9:
        return flat.reshape(3, 3)
    raise ValueError(f"Expected a 6- or 9-component tensor, got shape {array.shape}")


def _stress_unit_from_info(info):
    for key in ("stress_unit", "stress_units", "stress-unit"):
        if key in info:
            return info[key]
    if "stress_GPa" in info:
        return "gpa"
    return "auto"


def stress_to_ev_per_a3(stress, unit="auto"):
    matrix = coerce_tensor_3x3(stress)
    normalized = _normalize_unit(unit)
    if normalized in {"auto", "ev_a3"}:
        return matrix
    if normalized == "gpa":
        return matrix * GPA_TO_EV_PER_A3
    if normalized == "kbar":
        return matrix * KBAR_TO_EV_PER_A3
    raise ValueError(f"Unsupported stress unit: {unit!r}")


def stress_to_gpa(stress, unit="auto"):
    return stress_to_ev_per_a3(stress, unit=unit) * EV_PER_A3_TO_GPA


def extract_virial_matrix(atoms):
    info = getattr(atoms, "info", {})
    if "virial" in info:
        return coerce_tensor_3x3(info["virial"])

    stress_value = None
    stress_unit = _stress_unit_from_info(info)
    if "stress_GPa" in info:
        stress_value = info["stress_GPa"]
        stress_unit = "gpa"
    elif "stress" in info:
        stress_value = info["stress"]
    else:
        try:
            stress_value = atoms.get_stress(voigt=False)
            stress_unit = "ev_a3"
        except Exception:
            try:
                stress_value = atoms.get_stress(voigt=True)
                stress_unit = "ev_a3"
            except Exception:
                return None

    volume = float(atoms.get_volume())
    if volume <= 0:
        return None
    stress_matrix = stress_to_ev_per_a3(stress_value, unit=stress_unit)
    return -stress_matrix * volume


def extract_stress_matrix_gpa(atoms):
    info = getattr(atoms, "info", {})
    if "stress_GPa" in info:
        return coerce_tensor_3x3(info["stress_GPa"])
    if "stress" in info:
        return stress_to_gpa(info["stress"], unit=_stress_unit_from_info(info))

    try:
        return stress_to_gpa(atoms.get_stress(voigt=False), unit="ev_a3")
    except Exception:
        try:
            return stress_to_gpa(atoms.get_stress(voigt=True), unit="ev_a3")
        except Exception:
            virial = extract_virial_matrix(atoms)
            volume = float(atoms.get_volume())
            if virial is None or volume <= 0:
                return None
            return (-virial / volume) * EV_PER_A3_TO_GPA
