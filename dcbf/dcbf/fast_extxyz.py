from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Iterator, Sequence

from ase.data import atomic_numbers, chemical_symbols


CFG_CELL_LINE = "{:16.8f} {:16.8f} {:16.8f}\n"
CFG_ATOM_LINE_POS_ONLY = " {:6d} {:6d} {:16.8f} {:16.8f} {:16.8f}\n"

_PROPERTY_RE = re.compile(rb'(?:^|\s)Properties=(?:"([^"]+)"|(\S+))')
_LATTICE_RE = re.compile(rb'(?:^|\s)Lattice=(?:"([^"]+)"|(\S+))')


class UnsupportedFastXYZ(ValueError):
    """The file is valid XYZ but needs ASE for semantic parsing."""


class MalformedXYZ(ValueError):
    """The file does not contain complete XYZ frame blocks."""


@dataclass
class XYZFrameIndex:
    path: Path
    starts: array
    ends: array
    atom_counts: array
    elements: set[str]
    descriptor_compatible: bool
    descriptor_reason: str | None = None

    def __len__(self) -> int:
        return len(self.starts)

    def all_indices(self) -> range:
        return range(len(self))


def supports_raw_xyz(path) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in {".xyz", ".extxyz"}


def _parse_properties(header: bytes) -> tuple[int, int, int]:
    match = _PROPERTY_RE.search(header)
    if match is None:
        raise UnsupportedFastXYZ("EXTXYZ header does not define Properties")
    raw = match.group(1) or match.group(2)
    try:
        tokens = raw.decode("ascii").split(":")
    except UnicodeDecodeError as exc:
        raise UnsupportedFastXYZ("Properties contains non-ASCII field names") from exc
    if len(tokens) % 3 != 0:
        raise UnsupportedFastXYZ("Properties is not a sequence of name:type:columns triples")

    offset = 0
    species_offset = None
    atomic_number_offset = None
    position_offset = None
    total_columns = 0
    for index in range(0, len(tokens), 3):
        name = tokens[index].strip().lower()
        try:
            columns = int(tokens[index + 2])
        except ValueError as exc:
            raise UnsupportedFastXYZ("Properties contains a non-integer column count") from exc
        if columns <= 0:
            raise UnsupportedFastXYZ("Properties contains a non-positive column count")
        if name in {"species", "symbol"} and columns == 1:
            species_offset = offset
        elif name in {"z", "atomic_number"} and columns == 1:
            atomic_number_offset = offset
        elif name in {"pos", "position", "positions"} and columns == 3:
            position_offset = offset
        offset += columns
        total_columns = offset

    element_offset = species_offset if species_offset is not None else atomic_number_offset
    if element_offset is None:
        raise UnsupportedFastXYZ("Properties does not contain species:S:1 or Z:I:1")
    if position_offset is None:
        raise UnsupportedFastXYZ("Properties does not contain pos:R:3")
    return element_offset, position_offset, total_columns


def _parse_lattice(header: bytes) -> tuple[float, ...] | None:
    match = _LATTICE_RE.search(header)
    if match is None:
        # Non-periodic EXTXYZ commonly omits Lattice.  The fast path keeps
        # these frames and emits a CFG without a Supercell block, matching the
        # ASE conversion semantics for pbc=False.
        return None
    raw = match.group(1) or match.group(2)
    try:
        values = tuple(float(value) for value in raw.split())
    except ValueError as exc:
        raise MalformedXYZ("Lattice contains a non-numeric value") from exc
    if len(values) != 9 or not all(math.isfinite(value) for value in values):
        raise MalformedXYZ("Lattice must contain nine finite values")
    return values


def _decode_element(token: bytes) -> str:
    try:
        text = token.decode("ascii")
    except UnicodeDecodeError as exc:
        raise MalformedXYZ("Element token is not ASCII") from exc
    if text in atomic_numbers:
        return text
    try:
        number = int(text)
    except ValueError as exc:
        raise MalformedXYZ(f"Unknown element token: {text!r}") from exc
    if number <= 0 or number >= len(chemical_symbols) or not chemical_symbols[number]:
        raise MalformedXYZ(f"Invalid atomic number: {number}")
    return chemical_symbols[number]


def index_xyz_frames(path, collect_elements: bool = True) -> XYZFrameIndex:
    path = Path(path).resolve()
    if not supports_raw_xyz(path):
        raise UnsupportedFastXYZ(f"unsupported filename suffix: {path.suffix or '<none>'}")

    starts = array("Q")
    ends = array("Q")
    atom_counts = array("Q")
    elements: set[str] = set()
    descriptor_compatible = True
    descriptor_reason = None

    with path.open("rb", buffering=1024 * 1024) as handle:
        frame_index = 0
        while True:
            start = handle.tell()
            first = handle.readline()
            while first and not first.strip():
                start = handle.tell()
                first = handle.readline()
            if not first:
                break
            try:
                atom_count = int(first.strip())
            except ValueError as exc:
                raise MalformedXYZ(
                    f"{path}: frame {frame_index} has an invalid atom-count line"
                ) from exc
            if atom_count <= 0:
                raise MalformedXYZ(f"{path}: frame {frame_index} has atom_count={atom_count}")

            header = handle.readline()
            if not header:
                raise MalformedXYZ(f"{path}: frame {frame_index} is missing its header")

            try:
                schema = _parse_properties(header)
                _parse_lattice(header)
            except UnsupportedFastXYZ as exc:
                schema = exc
            if isinstance(schema, Exception):
                descriptor_compatible = False
                if descriptor_reason is None:
                    descriptor_reason = str(schema)

            for atom_index in range(atom_count):
                line = handle.readline()
                if not line:
                    raise MalformedXYZ(
                        f"{path}: frame {frame_index} is truncated at atom {atom_index}"
                    )
                if collect_elements and not isinstance(schema, Exception):
                    columns = line.split()
                    element_offset, _, total_columns = schema
                    if len(columns) < total_columns:
                        raise MalformedXYZ(
                            f"{path}: frame {frame_index} atom {atom_index} has fewer columns than Properties"
                        )
                    elements.add(_decode_element(columns[element_offset]))

            starts.append(start)
            ends.append(handle.tell())
            atom_counts.append(atom_count)
            frame_index += 1

    return XYZFrameIndex(
        path=path,
        starts=starts,
        ends=ends,
        atom_counts=atom_counts,
        elements=elements,
        descriptor_compatible=descriptor_compatible,
        descriptor_reason=descriptor_reason,
    )


def _contiguous_runs(indices: Iterable[int], total: int) -> Iterator[tuple[int, int]]:
    iterator = iter(indices)
    try:
        first = int(next(iterator))
    except StopIteration:
        return
    if first < 0 or first >= total:
        raise IndexError(f"frame index out of range: {first}")
    run_start = first
    previous = first
    for raw_index in iterator:
        index = int(raw_index)
        if index < 0 or index >= total:
            raise IndexError(f"frame index out of range: {index}")
        if index <= previous:
            raise ValueError("frame indices must be strictly increasing")
        if index != previous + 1:
            yield run_start, previous + 1
            run_start = index
        previous = index
    yield run_start, previous + 1


def _copy_byte_span(source, destination, start: int, end: int) -> None:
    source.seek(start)
    remaining = end - start
    while remaining > 0:
        payload = source.read(min(4 * 1024 * 1024, remaining))
        if not payload:
            raise MalformedXYZ("unexpected EOF while copying an indexed XYZ frame")
        destination.write(payload)
        remaining -= len(payload)


def write_indexed_frames(
    destination,
    selections: Sequence[tuple[XYZFrameIndex, Iterable[int]]],
) -> int:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            buffering=1024 * 1024,
            dir=destination.parent,
            prefix=f".{destination.name}.fastxyz-",
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            for frame_index, indices in selections:
                with frame_index.path.open("rb", buffering=1024 * 1024) as source:
                    for start_index, stop_index in _contiguous_runs(indices, len(frame_index)):
                        _copy_byte_span(
                            source,
                            output,
                            frame_index.starts[start_index],
                            frame_index.ends[stop_index - 1],
                        )
                        written += stop_index - start_index
        os.replace(temp_path, destination)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return written


def complement_indices(total: int, selected: set[int]) -> Iterator[int]:
    for index in range(total):
        if index not in selected:
            yield index


def _ordered_elements(elements: Sequence[str], sort_elements: bool) -> list[str]:
    if sort_elements:
        return [chemical_symbols[number] for number in sorted(atomic_numbers[item] for item in elements)]
    return list(elements)


def _parse_frame_geometry(source, frame_index: XYZFrameIndex, index: int):
    source.seek(frame_index.starts[index])
    first = source.readline()
    atom_count = int(first.strip())
    header = source.readline()
    element_offset, position_offset, total_columns = _parse_properties(header)
    lattice = _parse_lattice(header)
    atoms = []
    for atom_index in range(atom_count):
        line = source.readline()
        if not line:
            raise MalformedXYZ(
                f"{frame_index.path}: frame {index} is truncated at atom {atom_index}"
            )
        columns = line.split()
        if len(columns) < total_columns:
            raise MalformedXYZ(
                f"{frame_index.path}: frame {index} atom {atom_index} has fewer columns than Properties"
            )
        symbol = _decode_element(columns[element_offset])
        try:
            position = tuple(float(columns[position_offset + offset]) for offset in range(3))
        except ValueError as exc:
            raise MalformedXYZ(
                f"{frame_index.path}: frame {index} atom {atom_index} has an invalid position"
            ) from exc
        if not all(math.isfinite(value) for value in position):
            raise MalformedXYZ(
                f"{frame_index.path}: frame {index} atom {atom_index} has a non-finite position"
            )
        atoms.append((symbol, position))
    return lattice, atoms


def write_position_cfg_parts(
    frame_index: XYZFrameIndex,
    part_paths: Sequence[Path],
    ranges: Sequence[tuple[int, int]],
    elements: Sequence[str],
    sort_elements: bool,
) -> None:
    if not frame_index.descriptor_compatible:
        raise UnsupportedFastXYZ(frame_index.descriptor_reason or "unsupported EXTXYZ descriptor fields")
    if len(part_paths) != len(ranges):
        raise ValueError("part_paths and ranges must have the same length")

    ordered_elements = _ordered_elements(elements, sort_elements)
    type_map = {element: index for index, element in enumerate(ordered_elements)}
    handles = []
    try:
        for path in part_paths:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            handles.append(path.open("w", encoding="utf-8"))
        with frame_index.path.open("rb", buffering=1024 * 1024) as source:
            for part_number, (start, stop) in enumerate(ranges):
                output = handles[part_number]
                for index in range(start, stop):
                    lattice, atoms = _parse_frame_geometry(source, frame_index, index)
                    output.write("BEGIN_CFG\n")
                    output.write(" Size\n")
                    output.write("  {:6} \n".format(len(atoms)))
                    if lattice is not None:
                        output.write(" Supercell \n")
                        output.write(CFG_CELL_LINE.format(*lattice[0:3]))
                        output.write(CFG_CELL_LINE.format(*lattice[3:6]))
                        output.write(CFG_CELL_LINE.format(*lattice[6:9]))
                    output.write("AtomData:  id type       cartes_x      cartes_y      cartes_z     \n")
                    for atom_index, (symbol, position) in enumerate(atoms, start=1):
                        if symbol not in type_map:
                            raise MalformedXYZ(
                                f"{frame_index.path}: frame {index} contains element {symbol!r} "
                                "outside the configured MTP element mapping"
                            )
                        output.write(
                            CFG_ATOM_LINE_POS_ONLY.format(
                                atom_index,
                                type_map[symbol],
                                position[0],
                                position[1],
                                position[2],
                            )
                        )
                    output.write("END_CFG \n")
    except Exception:
        for handle in handles:
            handle.close()
        for path in part_paths:
            Path(path).unlink(missing_ok=True)
        raise
    finally:
        for handle in handles:
            if not handle.closed:
                handle.close()


def write_position_cfg_part(
    frame_index: XYZFrameIndex,
    part_path: Path,
    frame_range: tuple[int, int],
    elements: Sequence[str],
    sort_elements: bool,
) -> None:
    """Process-pool entry point for one independent CFG shard."""
    write_position_cfg_parts(
        frame_index,
        [part_path],
        [frame_range],
        elements,
        sort_elements,
    )
