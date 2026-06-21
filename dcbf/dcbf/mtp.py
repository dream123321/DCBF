from __future__ import annotations

from importlib import resources


SUPPORTED_MTP_TYPES = {"l2k2", "l2k3"}


def normalize_mtp_type(mtp_type):
    value = str(mtp_type).strip()
    if value.endswith(".mtp"):
        value = value[:-4]
    value = value.lower()
    if value not in SUPPORTED_MTP_TYPES:
        raise ValueError(f"mtp_type must be one of {sorted(SUPPORTED_MTP_TYPES)}, got {mtp_type!r}")
    return value


def mtp_template_path(mtp_type):
    normalized = normalize_mtp_type(mtp_type)
    return resources.files("dcbf.mtp_templates").joinpath(f"{normalized}.mtp")


def render_mtp_template(mtp_type, species_count):
    template = mtp_template_path(mtp_type).read_text(encoding="utf-8")
    lines = template.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if "species_count" in line:
            lines[index] = f"species_count = {species_count}\n"
            break
    return "".join(lines)


def write_mtp_template(destination, mtp_type, species_count):
    destination.write_text(render_mtp_template(mtp_type, species_count), encoding="utf-8")
