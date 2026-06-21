from __future__ import annotations

import csv
import itertools
from pathlib import Path

from ase.data import chemical_symbols

from .cli_defaults import format_section_defaults, get_section_defaults, set_section_defaults


SUMMARY_FIELDS = (
    "material_id",
    "formula_pretty",
    "chemsys",
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "output_path",
)
QUERY_FIELDS = list(SUMMARY_FIELDS[:-1]) + ["structure"]
VALID_ELEMENT_SYMBOLS = {symbol for symbol in chemical_symbols if symbol}


def handle_mp_search_command(args):
    items = list(args.items)
    if items and items[0] == "set":
        return _handle_set_command(items[1:])
    if not items:
        raise SystemExit("dcbf mp-search requires at least one element symbol")

    options = get_section_defaults("mp_search")
    for key in ("api_key", "output_dir", "csv_name"):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value

    api_key = str(options.get("api_key") or "").strip()
    if not api_key:
        raise SystemExit("dcbf mp-search requires an API key; set it with `dcbf mp-search set api_key=...`")

    output_root = _resolve_output_root(options["output_dir"])
    csv_name = str(options["csv_name"]).strip() or "summary.csv"
    elements = _normalize_elements(items)
    results = run_mp_search(elements, api_key=api_key, output_root=output_root, csv_name=csv_name)

    print(f"output_root: {results['output_root']}")
    print(f"total_combinations: {len(results['combinations'])}")
    print(f"total_structures: {results['total_structures']}")
    for combo_report in results["combinations"]:
        print(
            "combo: {chemsys} structures={count} dir={directory} csv={csv_path}".format(
                chemsys=combo_report["chemsys"],
                count=combo_report["count"],
                directory=combo_report["directory"],
                csv_path=combo_report["csv_path"],
            )
        )
    return 0


def run_mp_search(elements, api_key: str, output_root: Path, csv_name: str):
    try:
        from pymatgen.core import Structure
        from pymatgen.ext.matproj import MPRester
    except ImportError as exc:
        raise RuntimeError("pymatgen is required for `dcbf mp-search`") from exc

    output_root.mkdir(parents=True, exist_ok=True)
    combination_reports = []
    total_structures = 0

    with MPRester(api_key=api_key) as rester:
        for combo in _combination_sequence(elements):
            chemsys = "-".join(combo)
            directory = output_root / chemsys
            directory.mkdir(parents=True, exist_ok=True)

            try:
                docs = rester.summary_search(chemsys=chemsys, _fields=QUERY_FIELDS)
            except Exception as exc:
                raise RuntimeError(f"Materials Project query failed for {chemsys}: {exc}") from exc

            rows = []
            for doc in sorted(docs, key=lambda item: str(item.get("material_id", ""))):
                material_id = str(doc.get("material_id") or "").strip()
                structure = _coerce_structure(doc.get("structure"), Structure)
                if not material_id or structure is None:
                    continue

                output_path = directory / f"{material_id}.vasp"
                structure.to(filename=str(output_path), fmt="poscar")
                rows.append(
                    {
                        "material_id": material_id,
                        "formula_pretty": _stringify(doc.get("formula_pretty")),
                        "chemsys": _stringify(doc.get("chemsys")) or chemsys,
                        "band_gap": _stringify(doc.get("band_gap")),
                        "formation_energy_per_atom": _stringify(doc.get("formation_energy_per_atom")),
                        "energy_above_hull": _stringify(doc.get("energy_above_hull")),
                        "is_stable": _stringify(doc.get("is_stable")),
                        "output_path": str(output_path.resolve()),
                    }
                )

            csv_path = directory / csv_name
            _write_summary_csv(csv_path, rows)
            combination_reports.append(
                {
                    "chemsys": chemsys,
                    "directory": str(directory.resolve()),
                    "csv_path": str(csv_path.resolve()),
                    "count": len(rows),
                }
            )
            total_structures += len(rows)

    return {
        "output_root": str(output_root.resolve()),
        "combinations": combination_reports,
        "total_structures": total_structures,
    }


def _handle_set_command(assignments):
    if not assignments:
        print(format_section_defaults("mp_search"))
        return 0
    updated = set_section_defaults("mp_search", assignments)
    for key in sorted(updated):
        value = updated[key]
        if value is None:
            rendered = "<auto>"
        else:
            rendered = str(value)
        print(f"{key}={rendered}")
    return 0


def _normalize_elements(raw_elements):
    normalized = []
    seen = set()
    for raw_element in raw_elements:
        text = str(raw_element).strip()
        if not text:
            continue
        element = text[:1].upper() + text[1:].lower()
        if element not in VALID_ELEMENT_SYMBOLS:
            raise SystemExit(f"Invalid element symbol: {raw_element}")
        if element not in seen:
            seen.add(element)
            normalized.append(element)
    if not normalized:
        raise SystemExit("dcbf mp-search requires at least one valid element symbol")
    return normalized


def _combination_sequence(elements):
    return list(
        itertools.chain.from_iterable(
            itertools.combinations(elements, size) for size in range(1, len(elements) + 1)
        )
    )


def _resolve_output_root(raw_output_dir):
    output_root = Path(raw_output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    return output_root.resolve()


def _coerce_structure(value, structure_cls):
    if value is None:
        return None
    if isinstance(value, structure_cls):
        return value
    if isinstance(value, dict):
        return structure_cls.from_dict(value)
    return None


def _stringify(value):
    if value is None:
        return ""
    return str(value)


def _write_summary_csv(csv_path: Path, rows):
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
