"""Validated SUS2-SH outer descriptor schema, independent of model dimensions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


class ModelMetadataError(ValueError):
    """Raised when a model cannot provide a safe descriptor schema."""


BODY_ORDER_NAMES = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
}


def _header_value(text: str, key: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(key)}\s*=\s*(.*)$", text)
    return match.group(1).strip() if match else None


def _integer(text: str, key: str, required: bool = True) -> int | None:
    value = _header_value(text, key)
    if value is None:
        if required:
            raise ModelMetadataError(f"SUS2 model is missing required metadata: {key}")
        return None
    match = re.search(r"[-+]?\d+", value)
    if match is None:
        raise ModelMetadataError(f"SUS2 metadata {key} is not an integer: {value!r}")
    return int(match.group(0))


def _bool(text: str, key: str, default: bool = False) -> bool:
    value = _header_value(text, key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ModelMetadataError(f"SUS2 metadata {key} is not a boolean: {value!r}")


def _integer_list(text: str, key: str) -> tuple[int, ...]:
    value = _header_value(text, key)
    if value is None:
        raise ModelMetadataError(f"SUS2 model is missing required metadata: {key}")
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        raise ModelMetadataError(f"SUS2 metadata {key} has no brace-delimited list")
    return tuple(int(item) for item in re.findall(r"[-+]?\d+", value[start + 1:end]))


def _record_list(text: str, key: str) -> tuple[tuple[int, ...], ...]:
    value = _header_value(text, key)
    if value is None:
        raise ModelMetadataError(f"SUS2 model is missing required metadata: {key}")
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        raise ModelMetadataError(f"SUS2 metadata {key} has no record list")
    records = []
    for raw_record in re.findall(r"\{([^{}]+)\}", value[start + 1:end]):
        values = tuple(int(item) for item in re.findall(r"[-+]?\d+", raw_record))
        if not values:
            raise ModelMetadataError(f"SUS2 metadata {key} contains an empty record")
        records.append(values)
    return tuple(records)


@dataclass(frozen=True)
class Sus2DescriptorMetadata:
    path: str
    model_version: str
    potential_tag: str
    species_count: int
    scalar_count: int
    alpha_moment_mapping: tuple[int, ...]
    sh_scalar_info: tuple[tuple[int, ...], ...]
    joint_readout_enabled: bool
    joint_scalar_count: int
    header_schema: tuple[tuple[str, str], ...]

    @property
    def output_width(self) -> int:
        return self.scalar_count + self.joint_scalar_count

    @property
    def body_columns(self) -> dict[str, tuple[int, ...]]:
        columns = {
            BODY_ORDER_NAMES[order]: tuple(
                i for i, record in enumerate(self.sh_scalar_info) if record[0] == order
            )
            for order in sorted(BODY_ORDER_NAMES)
        }
        # ``all`` means every outer scalar in the order emitted by
        # calc-descriptors.  Joint/readout scalars are deliberately excluded.
        # This is the default for sampling so a model with body orders 5/6 is
        # not silently reduced to the historical two/three/four subset.
        columns["all"] = tuple(range(self.scalar_count))
        return columns

    @property
    def has_coordination_gate(self) -> bool:
        return any(
            key.startswith(("two_layer_gate_", "coordination_"))
            for key, _ in self.header_schema
        )

    @property
    def gate_metadata(self) -> dict[str, str]:
        return {
            key: value
            for key, value in self.header_schema
            if key.startswith(("two_layer_gate_", "coordination_"))
        }

    def signature(self) -> dict:
        payload = {
            "model_format": "sus2",
            "parser_schema": "sus2-sh-strict-v3",
            "sus2_version": self.model_version,
            "potential_tag": self.potential_tag,
            "species_count": self.species_count,
            "scalar_count": self.scalar_count,
            "alpha_moment_mapping": list(self.alpha_moment_mapping),
            "sh_scalar_info": [list(record) for record in self.sh_scalar_info],
            "body_columns": {key: list(value) for key, value in self.body_columns.items()},
            "joint_readout_enabled": self.joint_readout_enabled,
            "joint_scalar_count": self.joint_scalar_count,
            "output_width": self.output_width,
            "descriptor_blocks": {
                "outer": {"offset": 0, "width": self.scalar_count},
                "joint_readout": {"offset": self.scalar_count, "width": self.joint_scalar_count},
            },
            "gate_metadata": self.gate_metadata,
            # Keep this JSON-native: manifests deserialize nested tuples as
            # lists, which would make every otherwise-identical cache miss.
            "header_schema": [list(item) for item in self.header_schema],
        }
        payload["schema_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload


def read_sus2_descriptor_metadata(model_path) -> Sus2DescriptorMetadata:
    path = Path(model_path).resolve()
    if not path.is_file():
        raise ModelMetadataError(f"SUS2 model does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ModelMetadataError(f"SUS2 model is not a readable text model: {path}") from exc

    model_version = (_header_value(text, "version") or "").strip()
    if not model_version:
        raise ModelMetadataError(f"SUS2 model is missing required metadata: version")
    potential_tag = (_header_value(text, "potential_tag") or "").strip()
    if potential_tag != "SUS2-SH":
        raise ModelMetadataError(
            f"Expected potential_tag=SUS2-SH in {path}, got {potential_tag or '<missing>'!r}"
        )
    scalar_count = _integer(text, "alpha_scalar_moments")
    species_count = _integer(text, "species_count")
    moment_count = _integer(text, "alpha_moments_count")
    if scalar_count <= 0 or species_count <= 0 or moment_count <= 0:
        raise ModelMetadataError("SUS2 scalar/species/moment counts must be positive")
    sh_count = _integer(text, "sh_scalar_info_count")
    mapping = _integer_list(text, "alpha_moment_mapping")
    scalar_info = _record_list(text, "sh_scalar_info")
    if scalar_count != sh_count or sh_count != len(scalar_info):
        raise ModelMetadataError(
            "SUS2 descriptor metadata count mismatch: "
            f"alpha_scalar_moments={scalar_count}, sh_scalar_info_count={sh_count}, "
            f"records={len(scalar_info)}"
        )
    if len(mapping) != scalar_count:
        raise ModelMetadataError(
            f"alpha_moment_mapping has {len(mapping)} entries, expected {scalar_count}"
        )
    if len(set(mapping)) != len(mapping):
        raise ModelMetadataError("alpha_moment_mapping contains duplicate entries")
    if any(index < 0 or index >= moment_count for index in mapping):
        raise ModelMetadataError("alpha_moment_mapping contains an out-of-range moment index")
    if any(len(record) != 7 or not 2 <= record[0] <= 6 for record in scalar_info):
        raise ModelMetadataError("Invalid sh_scalar_info record: expected body_order, q[5], intermediate_l")

    joint_enabled = _bool(text, "joint_readout_enabled")
    joint_count = _integer(text, "joint_readout_scalar_count", required=joint_enabled) or 0
    if joint_count < 0:
        raise ModelMetadataError("joint_readout_scalar_count must be non-negative")

    schema_keys = (
        "version", "potential_name", "potential_tag", "sh_l_max", "sh_k_max",
        "sh_body_order", "sh_body_l_max", "sh_parity", "radial_basis_type",
        "radial_basis_size", "outer_k_rank", "sh_factor_pruning", "readout_mode",
        "joint_readout_enabled", "joint_readout_scalar_count",
    )
    header_pairs = [
        (key, _header_value(text, key) or "")
        for key in schema_keys
        if _header_value(text, key) is not None
    ]
    # Coordination-effect stores gate metadata in the header.  Keep all
    # structural gate keys in the cache signature, but omit large coefficient
    # arrays which are already covered by the model SHA-256.
    for line in text.splitlines():
        if "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if (
            key.startswith(("two_layer_gate_", "coordination_", "joint_", "readout_"))
            and not key.endswith(("_coeffs", "_weights", "_coefficients"))
            and key not in {item[0] for item in header_pairs}
        ):
            header_pairs.append((key, _header_value(text, key) or ""))
    header_schema = tuple(sorted(header_pairs))
    return Sus2DescriptorMetadata(
        path=str(path),
        model_version=model_version,
        potential_tag=potential_tag,
        species_count=species_count,
        scalar_count=scalar_count,
        alpha_moment_mapping=mapping,
        sh_scalar_info=scalar_info,
        joint_readout_enabled=joint_enabled,
        joint_scalar_count=joint_count,
        header_schema=header_schema,
    )


def validate_descriptor_output_width(descriptor_paths, expected_width: int, species_count=None) -> None:
    """Reject every descriptor row that disagrees with the model schema."""
    for raw_path in descriptor_paths:
        path = Path(raw_path)
        saw_frame = False
        with path.open("rb") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                if not line.lstrip().startswith(b"#start"):
                    continue
                saw_frame = True
                try:
                    atom_count = int(line.split()[1])
                except (IndexError, ValueError) as exc:
                    raise ModelMetadataError(
                        f"Invalid #start header in descriptor output: {path}"
                    ) from exc
                if atom_count <= 0:
                    raise ModelMetadataError(f"Descriptor frame atom count must be positive: {path}")
                for atom_index in range(atom_count):
                    row = handle.readline()
                    if not row.strip():
                        raise ModelMetadataError(
                            f"Descriptor output has a missing atom row in {path}: "
                            f"frame atom index {atom_index}"
                        )
                    fields = row.split()
                    width = len(fields) - 1
                    if width != int(expected_width):
                        raise ModelMetadataError(
                            f"Descriptor column count mismatch for {path}: "
                            f"output={width}, model_metadata={expected_width}"
                        )
                    try:
                        atom_type = int(fields[0])
                    except ValueError as exc:
                        raise ModelMetadataError(f"Invalid descriptor atom type in {path}") from exc
                    if atom_type < 0 or (species_count is not None and atom_type >= species_count):
                        raise ModelMetadataError(f"Descriptor atom type {atom_type} is outside the model mapping")
        if not saw_frame:
            raise ModelMetadataError(
                f"Descriptor output contains no #start frame: {path}"
            )
