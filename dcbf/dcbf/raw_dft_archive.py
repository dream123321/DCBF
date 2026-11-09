from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path, PurePosixPath

from .runtime_config import find_runtime_config, load_json_config
from .training_dataset import resolve_summary_bundle_dir


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_manifest_lock(manifest_path):
    lock_path = Path(manifest_path).with_name(".archive.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as handle:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _strip_comment(line):
    return line.split("#", 1)[0].split("!", 1)[0].strip()


def _parse_bool(value):
    return str(value).strip().strip(".").lower() in {"true", "t", "yes", "1", "on"}


def _read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


ZSTD_LEVEL = 19
ZSTD_THREADS = 1
ARCHIVE_BUFFER_SIZE = 1024 * 1024


@lru_cache(maxsize=1)
def _bundled_zstd_executable():
    candidates = []
    configured_root = os.environ.get("DCBF_HOME")
    if configured_root:
        candidates.append(Path(configured_root).expanduser() / "runtime" / "dcbf_env" / "bin" / "zstd")

    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if parent.name == "dcbf_env":
            candidates.append(parent / "bin" / "zstd")
        candidates.append(parent / "runtime" / "dcbf_env" / "bin" / "zstd")

    checked = []
    for executable in candidates:
        executable = executable.resolve()
        if executable in checked:
            continue
        checked.append(executable)
        if executable.is_file() and os.access(executable, os.X_OK):
            return executable
    expected = checked[0] if checked else Path("runtime/dcbf_env/bin/zstd")
    raise RuntimeError(
        "Bundled zstd executable is unavailable; expected "
        f"{expected}. Run install.sh again before archiving raw DFT data."
    )


@lru_cache(maxsize=1)
def _zstd_version():
    executable = _bundled_zstd_executable()
    result = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Cannot query bundled zstd version: {result.stderr.strip() or result.stdout.strip()}"
        )
    match = re.search(r"\bv?(\d+\.\d+\.\d+)\b", result.stdout)
    return match.group(1) if match else result.stdout.strip()


def _archive_format(path):
    name = Path(path).name.lower()
    if name.endswith(".tar.zst") or name.endswith(".tar.zst.tmp"):
        return "tar.zst"
    if name.endswith(".tar.gz") or name.endswith(".tar.gz.tmp"):
        return "tar.gz"
    raise ValueError(f"Unsupported raw DFT archive format: {path}")


def _safe_member_path(member):
    name = member.name
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Unsafe path in raw DFT archive: {name!r}")
    if member.issym() or member.islnk() or member.ischr() or member.isblk() or member.isfifo():
        raise RuntimeError(f"Unsupported link or device in raw DFT archive: {name!r}")
    if not member.isfile() and not member.isdir():
        raise RuntimeError(f"Unsupported member type in raw DFT archive: {name!r}")
    return path


def _consume_tar_stream(archive, output_dir=None):
    names = []
    seen = set()
    for member in archive:
        path = _safe_member_path(member)
        normalized = path.as_posix()
        if normalized in seen:
            raise RuntimeError(f"Duplicate member in raw DFT archive: {member.name!r}")
        seen.add(normalized)
        if member.isdir():
            if output_dir is not None:
                output_dir.joinpath(*path.parts).mkdir(parents=True, exist_ok=True)
            continue

        names.append(member.name)
        source = archive.extractfile(member)
        if source is None:
            raise RuntimeError(f"Cannot read raw DFT archive member: {member.name!r}")
        if output_dir is None:
            while source.read(ARCHIVE_BUFFER_SIZE):
                pass
            continue

        target = output_dir.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "xb") as handle:
            shutil.copyfileobj(source, handle, length=ARCHIVE_BUFFER_SIZE)
        os.chmod(target, member.mode & 0o777)
    return sorted(names)


def _read_archive_members(path, output_dir=None):
    path = Path(path)
    archive_format = _archive_format(path)
    if archive_format == "tar.gz":
        with tarfile.open(path, "r|gz") as archive:
            return _consume_tar_stream(archive, output_dir=output_dir)

    executable = _bundled_zstd_executable()
    process = subprocess.Popen(
        [str(executable), "-q", "-d", "-c", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            names = _consume_tar_stream(archive, output_dir=output_dir)
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        returncode = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if returncode != 0:
        raise RuntimeError(f"Bundled zstd could not decompress {path}: {stderr.strip()}")
    return names


def _test_zstd_archive(path):
    executable = _bundled_zstd_executable()
    result = subprocess.run(
        [str(executable), "-q", "-t", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Raw DFT zstd integrity check failed for {path}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def _archive_root_relative(path, root_name):
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    indices = [index for index, part in enumerate(parts) if part == root_name]
    if not indices:
        return None
    return PurePosixPath(*parts[indices[-1] + 1 :]).as_posix()


def _raw_dft_relative(path):
    return _archive_root_relative(path, "raw_dft")


def _find_manifest_record(archive_path):
    archive_path = Path(archive_path).resolve()
    manifest_path = None
    for parent in [archive_path.parent, *archive_path.parents]:
        candidate = parent / "manifest.json"
        if candidate.is_file() and parent.name in {"raw_dft", "raw_dft_excluded"}:
            manifest_path = candidate
            break
    if manifest_path is None:
        return None, None, None

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_relative = archive_path.relative_to(manifest_path.parent.resolve()).as_posix()
    matches = []
    for key, record in payload.get("entries", {}).items():
        recorded_path = record.get("archive")
        if not recorded_path:
            continue
        if _archive_root_relative(recorded_path, manifest_path.parent.name) == current_relative:
            matches.append((key, record))
    if len(matches) > 1:
        raise RuntimeError(f"Multiple raw DFT manifest records match {archive_path}")
    if not matches:
        return manifest_path, None, None
    return manifest_path, matches[0][0], matches[0][1]


def verify_raw_dft_archive(archive_path):
    archive_path = Path(archive_path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Raw DFT archive does not exist: {archive_path}")
    archive_format = _archive_format(archive_path)
    if archive_format == "tar.zst":
        _test_zstd_archive(archive_path)
    files = _read_archive_members(archive_path)
    digest = _sha256(archive_path)
    manifest_path, manifest_key, record = _find_manifest_record(archive_path)
    if record is not None:
        if digest != record.get("sha256"):
            raise RuntimeError(f"Raw DFT manifest SHA-256 mismatch for {archive_path}")
        if files != sorted(record.get("files", [])):
            raise RuntimeError(f"Raw DFT manifest member-list mismatch for {archive_path}")
    return {
        "archive": str(archive_path),
        "archive_format": archive_format,
        "size_bytes": archive_path.stat().st_size,
        "sha256": digest,
        "files": files,
        "manifest": str(manifest_path) if manifest_path else None,
        "manifest_key": manifest_key,
        "manifest_matched": record is not None,
    }


def _default_extract_dir(archive_path):
    archive_path = Path(archive_path)
    name = archive_path.name
    for suffix in (".tar.zst", ".tar.gz"):
        if name.lower().endswith(suffix):
            return archive_path.with_name(name[: -len(suffix)])
    raise ValueError(f"Unsupported raw DFT archive format: {archive_path}")


def extract_raw_dft_archive(archive_path, output_dir=None):
    archive_path = Path(archive_path).resolve()
    verification = verify_raw_dft_archive(archive_path)
    destination = (
        _default_extract_dir(archive_path)
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    if destination.exists():
        raise FileExistsError(f"Raw DFT extraction target already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        extracted = _read_archive_members(archive_path, output_dir=destination)
        if extracted != verification["files"]:
            raise RuntimeError(f"Raw DFT extraction member-list mismatch for {archive_path}")
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return {**verification, "output_dir": str(destination)}


def _check_archive_disk_space(destination, files):
    required = sum(path.stat().st_size for path in files) + ARCHIVE_BUFFER_SIZE
    available = shutil.disk_usage(Path(destination).parent).free
    if available < required:
        raise RuntimeError(
            "Insufficient disk space for raw DFT archive: "
            f"required_at_most={required} available={available} destination={destination}"
        )


def _write_zstd_tar_members(temporary, members, output_fd=None):
    executable = _bundled_zstd_executable()
    output_context = (
        os.fdopen(output_fd, "wb")
        if output_fd is not None
        else open(temporary, "xb")
    )
    with output_context as output:
        process = subprocess.Popen(
            [
                str(executable),
                "-q",
                f"-{ZSTD_LEVEL}",
                f"-T{ZSTD_THREADS}",
                "-c",
            ],
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=subprocess.PIPE,
        )
        try:
            with tarfile.open(fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for path, archive_name in members:
                    archive.add(path, arcname=archive_name, recursive=False)
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            returncode = process.wait()
        except (OSError, tarfile.TarError) as exc:
            try:
                process.stdin.close()
            except OSError:
                pass
            process.kill()
            process.wait()
            raise RuntimeError(
                f"Failed to stream tar data through bundled zstd for {temporary}: {exc}"
            ) from exc
        except BaseException:
            try:
                process.stdin.close()
            except OSError:
                pass
            process.kill()
            process.wait()
            raise
    if returncode != 0:
        raise RuntimeError(
            f"Bundled zstd failed while creating {temporary}: {stderr.strip()}"
        )


def _write_zstd_tar(temporary, files, task, output_fd=None):
    members = [
        (path, path.relative_to(task).as_posix())
        for path in files
    ]
    _write_zstd_tar_members(temporary, members, output_fd=output_fd)


def _directory_archive_members(directory):
    members = []
    files = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.relative_to(directory).as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"Refusing to archive symbolic link: {path}")
        if path.is_dir():
            members.append((path, path.relative_to(directory).as_posix()))
        elif path.is_file():
            members.append((path, path.relative_to(directory).as_posix()))
            files.append(path)
        else:
            raise RuntimeError(f"Refusing to archive special filesystem entry: {path}")
    return members, files


def pack_raw_dft_directory(directory, output_path=None):
    source_input = Path(directory).expanduser()
    if source_input.is_symlink():
        raise RuntimeError(f"Refusing to archive symbolic-link directory: {source_input}")
    source = source_input.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Raw DFT source directory does not exist: {source}")

    destination = (
        source.with_name(source.name + ".tar.zst")
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    if not destination.name.lower().endswith(".tar.zst"):
        raise ValueError(f"Raw DFT pack output must end with .tar.zst: {destination}")
    if destination == source or destination.is_relative_to(source):
        raise ValueError(f"Raw DFT pack output must be outside the source directory: {destination}")
    if destination.exists():
        raise FileExistsError(f"Raw DFT archive already exists: {destination}")

    members, files = _directory_archive_members(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _check_archive_disk_space(destination, files)
    _bundled_zstd_executable()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tar.zst.tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        _write_zstd_tar_members(temporary, members, output_fd=descriptor)
        verification = verify_raw_dft_archive(temporary)
        expected_files = sorted(path.relative_to(source).as_posix() for path in files)
        if verification["files"] != expected_files:
            raise RuntimeError(f"Raw DFT packed member-list mismatch for {source}")
        os.link(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink(missing_ok=True)
    return {
        **verification,
        "archive": str(destination),
        "source_dir": str(source),
    }


class RawDFTArchiveManager:
    MANIFEST_VERSION = 1
    KEEP_AFTER_CLEANUP = {
        "__ok__",
        "__start__",
        "__failed__",
        "logout",
        "time.txt",
        "__raw_archive__.json",
        "__raw_archive_excluded__.json",
    }

    def __init__(self, workspace, engine, logger, stage=None, run_dir=None, config=None):
        self.workspace = Path(workspace).resolve()
        if run_dir is not None and config is not None:
            self.run_dir = Path(run_dir).resolve()
            self.config = dict(config)
        else:
            runtime_path = find_runtime_config(self.workspace)
            self.run_dir = runtime_path.parent
            self.config = load_json_config(runtime_path)
        self.engine = str(engine).strip().lower()
        self.logger = logger
        self.stage = stage or self._infer_stage()
        self.bundle_dir = resolve_summary_bundle_dir(self.run_dir, self.config)
        self.raw_root = self.bundle_dir / "raw_dft"
        self.manifest_path = self.raw_root / "manifest.json"
        self.excluded_root = self.bundle_dir / "raw_dft_excluded"
        self.excluded_manifest_path = self.excluded_root / "manifest.json"

    def _infer_stage(self):
        try:
            relative = self.workspace.relative_to(self.run_dir)
        except ValueError:
            return "unknown"
        parts = relative.parts
        if len(parts) >= 2 and parts[0].startswith("main_") and parts[1].startswith("gen_"):
            return f"{parts[0]}/{parts[1]}"
        if "init_dataset_build" in parts:
            return "init_dataset_build"
        return "/".join(parts) or "root"

    def _load_manifest_from(self, manifest_path):
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            return {"version": self.MANIFEST_VERSION, "entries": {}}
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("version") != self.MANIFEST_VERSION:
            raise RuntimeError(f"Unsupported raw-DFT manifest: {manifest_path}")
        return payload

    def _save_manifest_to(self, manifest_path, manifest):
        _atomic_json(manifest_path, manifest)

    def _load_manifest(self):
        return self._load_manifest_from(self.manifest_path)

    def _save_manifest(self, manifest):
        self._save_manifest_to(self.manifest_path, manifest)

    @staticmethod
    def task_dirs(scf_filter_root):
        root = Path(scf_filter_root)
        if not root.exists():
            return []
        result = []
        for group in sorted(path for path in root.iterdir() if path.is_dir()):
            for task in sorted(
                (path for path in group.iterdir() if path.is_dir()),
                key=lambda item: (0, int(item.name)) if item.name.isdigit() else (1, item.name),
            ):
                result.append(task)
        return result

    def _key(self, task):
        return f"{self.engine}/{self.stage}/{task.parent.name}/{task.name}"

    def _archive_path(self, task):
        return self.raw_root / self.engine / Path(self.stage) / task.parent.name / f"task_{task.name}.tar.zst"

    def _excluded_archive_path(self, task):
        return self.excluded_root / self.engine / Path(self.stage) / task.parent.name / f"task_{task.name}.tar.zst"

    def _record_archive_path(self, record):
        recorded = Path(record["archive"])
        if recorded.is_file():
            return recorded
        relative = _raw_dft_relative(recorded)
        if relative:
            relocated = self.raw_root / Path(relative)
            if relocated.is_file():
                return relocated
        return recorded

    def _record_excluded_archive_path(self, record):
        recorded = Path(record["archive"])
        if recorded.is_file():
            return recorded
        parts = recorded.parts
        if "raw_dft_excluded" in parts:
            index = max(i for i, part in enumerate(parts) if part == "raw_dft_excluded")
            relocated = self.excluded_root / Path(*parts[index + 1 :])
            if relocated.is_file():
                return relocated
        return recorded

    @staticmethod
    def _resolve_source_task(scf_filter_root, source_task):
        root = Path(scf_filter_root).resolve()
        task = (root / str(source_task)).resolve()
        try:
            task.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Unsafe SCF source task path: {source_task}") from exc
        if not task.is_dir():
            raise RuntimeError(f"SCF source task directory is missing: {task}")
        return task

    def _validate_source_manifest(self, scf_filter_root, source_manifest):
        frames = list(source_manifest.get("frames") or [])
        excluded = list(source_manifest.get("excluded") or [])
        selected_paths = [str(item.get("source_task")) for item in frames]
        excluded_paths = [str(item.get("source_task")) for item in excluded]
        if len(selected_paths) != len(set(selected_paths)):
            raise RuntimeError("SCF filter source manifest contains duplicate selected tasks")
        if set(selected_paths) & set(excluded_paths):
            raise RuntimeError("SCF filter source manifest selects and excludes the same task")

        completed = {
            task.relative_to(Path(scf_filter_root).resolve()).as_posix()
            for task in self.task_dirs(scf_filter_root)
            if (task / "__ok__").exists()
        }
        mapped = set(selected_paths) | set(excluded_paths)
        if completed != mapped:
            missing = sorted(completed - mapped)
            extra = sorted(mapped - completed)
            raise RuntimeError(
                "SCF filter source manifest does not cover completed tasks: "
                f"missing={missing} extra={extra}"
            )
        return frames, excluded

    def _vasp_files(self, task):
        incar = task / "INCAR"
        if not incar.exists():
            incar = self.run_dir / "init" / "INCAR"
        tags = {}
        for line in _read_text(incar).splitlines():
            clean = _strip_comment(line)
            if "=" in clean:
                key, value = clean.split("=", 1)
                tags[key.strip().upper()] = value.strip()
        lcharg = _parse_bool(tags.get("LCHARG", "false"))
        laechg = _parse_bool(tags.get("LAECHG", "false"))
        chgcar = task / "CHGCAR"
        charge = [chgcar] if chgcar.exists() else []
        return (
            [task / "POSCAR", task / "vasprun.xml"],
            charge,
            lcharg,
            f"LCHARG={lcharg}, LAECHG={laechg}, archive_policy=CHGCAR_only",
        )

    def _abacus_files(self, task):
        input_path = task / "INPUT"
        if not input_path.exists():
            input_path = self.run_dir / "init" / "INPUT"
        out_chg = None
        for line in _read_text(input_path).splitlines():
            words = _strip_comment(line).split()
            if len(words) >= 2 and words[0].lower() == "out_chg":
                try:
                    out_chg = int(float(words[1]))
                except ValueError:
                    pass
        requested = out_chg is not None and out_chg > 0
        logs = sorted(task.glob("OUT.*/running_scf.log"))
        if not logs:
            logs = [task / "OUT.ABACUS" / "running_scf.log"]
        charge = []
        for path in task.glob("OUT.*/*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if "_ini" in name:
                continue
            if re.fullmatch(r"spin\d+_chg(?:\.cube)?", name) or re.fullmatch(
                r"chg(?:s\d+)?\.cube", name
            ):
                charge.append(path)
        charge = sorted(set(charge))
        return [task / "STRU", *logs], charge, requested, f"out_chg={0 if out_chg is None else out_chg}"

    def _qe_files(self, task):
        qe_input = task / "qe.in"
        text = _read_text(qe_input)
        disk_match = re.search(r"\bdisk_io\s*=\s*['\"]?([^,'\"\s/]+)", text, re.IGNORECASE)
        disk_io = disk_match.group(1).lower() if disk_match else "low"
        prefix_match = re.search(r"\bprefix\s*=\s*['\"]([^'\"]+)", text, re.IGNORECASE)
        outdir_match = re.search(r"\boutdir\s*=\s*['\"]([^'\"]+)", text, re.IGNORECASE)
        prefix = prefix_match.group(1) if prefix_match else "pwscf"
        outdir = outdir_match.group(1) if outdir_match else "."
        save_dir = (task / outdir / f"{prefix}.save").resolve()
        charge = sorted(save_dir.glob("charge-density.*")) if save_dir.exists() else []
        schema = [save_dir / "data-file-schema.xml"] if (save_dir / "data-file-schema.xml").is_file() else []
        requested = disk_io != "none"
        return (
            [qe_input, task / "logout", *schema],
            charge,
            requested,
            f"disk_io={disk_io}, prefix={prefix}, schema_files={len(schema)}",
        )

    def _cp2k_files(self, task):
        input_path = task / "cp2k.inp"
        requested = bool(re.search(r"^\s*&E_DENSITY_CUBE\b", _read_text(input_path), re.IGNORECASE | re.MULTILINE))
        charge = sorted(
            {
                path
                for path in task.iterdir()
                if path.is_file()
                and path.suffix.lower() == ".cube"
                and "electron_density" in path.name.lower()
            }
        )
        return [task / "cp2k.xyz", task / "logout"], charge, requested, "E_DENSITY_CUBE=" + str(requested)

    def _select_files(self, task):
        selectors = {
            "vasp": self._vasp_files,
            "abacus": self._abacus_files,
            "qe": self._qe_files,
            "cp2k": self._cp2k_files,
        }
        if self.engine not in selectors:
            raise ValueError(f"Unsupported DFT archive engine: {self.engine}")
        fixed, charge, requested, detail = selectors[self.engine](task)
        fixed = [Path(path) for path in fixed]
        missing = [str(path) for path in fixed if not path.exists()]
        if missing:
            raise RuntimeError(f"Cannot archive completed {self.engine} task {task}; missing required files: {missing}")
        files = []
        for path in [*fixed, *charge]:
            if path.is_file() and path not in files:
                files.append(path)
        return files, charge, requested, detail

    def _diagnostic_files(self, task):
        if self.engine == "vasp":
            candidates = [
                task / "POSCAR",
                task / "INCAR",
                task / "KPOINTS",
                task / "logout",
                task / "OUTCAR",
                task / "OSZICAR",
            ]
        elif self.engine == "abacus":
            candidates = [
                task / "STRU",
                task / "INPUT",
                task / "KPT",
                task / "logout",
                *sorted(task.glob("OUT.*/running_scf.log")),
            ]
        elif self.engine == "qe":
            candidates = [task / "qe.in", task / "logout"]
        elif self.engine == "cp2k":
            candidates = [task / "cp2k.xyz", task / "cp2k.inp", task / "logout"]
        else:
            raise ValueError(f"Unsupported DFT archive engine: {self.engine}")
        candidates.extend(
            task / name for name in ("__start__", "__ok__", "__failed__", "time.txt")
        )
        files = []
        for path in candidates:
            path = Path(path)
            if path.is_file() and path not in files:
                files.append(path)
        if not files:
            raise RuntimeError(f"No diagnostic files are available for excluded DFT task {task}")
        return files

    @staticmethod
    def _record_matches_source(record, source_record, scf_filter_sha256):
        if source_record is None:
            return True
        return (
            int(record.get("source_frame_index", -1)) == int(source_record["frame_index"])
            and record.get("scf_filter_sha256") == scf_filter_sha256
            and float(record.get("max_force", float("nan"))) == float(source_record["max_force"])
            and record.get("selection_reason") == source_record["selection_reason"]
        )

    @staticmethod
    def _verified_record_archive(record, resolver, task):
        if not record:
            return None
        archive_path = resolver(record)
        if not archive_path.is_file():
            return None
        if _sha256(archive_path) != record.get("sha256"):
            raise RuntimeError(f"Raw DFT archive SHA-256 mismatch for task {task}: {archive_path}")
        verification = verify_raw_dft_archive(archive_path)
        if verification["files"] != sorted(record.get("files", [])):
            raise RuntimeError(f"Raw DFT archive member-list mismatch for task {task}: {archive_path}")
        return archive_path

    def _write_task_archive(self, task, destination, files):
        destination.parent.mkdir(parents=True, exist_ok=True)
        _check_archive_disk_space(destination, files)
        _bundled_zstd_executable()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tar.zst.tmp",
            dir=str(destination.parent),
        )
        temporary = Path(temporary_name)
        try:
            _write_zstd_tar(temporary, files, task, output_fd=descriptor)
            verification = verify_raw_dft_archive(temporary)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        expected_names = sorted(path.relative_to(task).as_posix() for path in files)
        if verification["files"] != expected_names:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Raw DFT archive verification failed for {task}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = verify_raw_dft_archive(destination)
            if existing["files"] != expected_names:
                raise RuntimeError(f"Existing raw DFT archive does not match task {task}: {destination}")
        finally:
            temporary.unlink(missing_ok=True)
        return expected_names

    def archive_completed_tasks(self, scf_filter_root, source_manifest=None):
        with _exclusive_manifest_lock(self.manifest_path):
            return self._archive_completed_tasks_unlocked(scf_filter_root, source_manifest)

    def _archive_completed_tasks_unlocked(self, scf_filter_root, source_manifest=None):
        manifest = self._load_manifest()
        if source_manifest is None:
            task_records = [
                (task, None)
                for task in self.task_dirs(scf_filter_root)
                if (task / "__ok__").exists()
            ]
            scf_filter_sha256 = None
        else:
            frames, _ = self._validate_source_manifest(scf_filter_root, source_manifest)
            task_records = [
                (self._resolve_source_task(scf_filter_root, item["source_task"]), item)
                for item in frames
            ]
            scf_filter_sha256 = source_manifest.get("scf_filter_sha256")

        archived = []
        charge_not_requested = []
        for task, source_record in task_records:
            if not (task / "__ok__").exists():
                raise RuntimeError(f"Selected SCF source task is not complete: {task}")
            key = self._key(task)
            record = manifest["entries"].get(key)
            archive_path = self._verified_record_archive(record, self._record_archive_path, task)
            if archive_path is not None:
                if not self._record_matches_source(record, source_record, scf_filter_sha256):
                    raise RuntimeError(
                        f"Existing raw DFT archive metadata does not match scf_filter.xyz: {task}"
                    )
                archived.append(task)
                continue

            destination = self._archive_path(task)
            files, charge_files, charge_requested, detail = self._select_files(task)
            warning = None
            if not charge_requested:
                warning = f"Charge-density output was not requested for {self.engine} task {task} ({detail})"
                charge_not_requested.append(task)
            elif not charge_files:
                warning = f"Charge-density output was requested but no charge file was found for {self.engine} task {task} ({detail})"
            elif self.engine == "qe" and not any(path.name == "data-file-schema.xml" for path in files):
                warning = (
                    f"QE charge density was found without data-file-schema.xml for task {task} "
                    f"({detail})"
                )
            # A missing charge-density request is the same setting for every
            # task in this batch.  Keep per-task detail in the manifest but log
            # it once at the end instead of once per task (logout/app.log spam).
            if warning and charge_requested:
                self.logger.warning("[raw_dft] %s", warning)

            expected_names = self._write_task_archive(task, destination, files)
            record = {
                "status": "archived",
                "task": str(task),
                "archive": str(destination),
                "archive_format": "tar.zst",
                "compression": {
                    "codec": "zstd",
                    "level": ZSTD_LEVEL,
                    "threads": ZSTD_THREADS,
                    "version": _zstd_version(),
                },
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
                "files": expected_names,
                "charge_requested": charge_requested,
                "charge_files": [path.relative_to(task).as_posix() for path in charge_files],
                "charge_setting": detail,
                "warning": warning,
                "cleaned": False,
            }
            if source_record is not None:
                record.update(
                    {
                        "source_frame_index": int(source_record["frame_index"]),
                        "scf_filter_sha256": scf_filter_sha256,
                        "max_force": float(source_record["max_force"]),
                        "selection_reason": source_record["selection_reason"],
                    }
                )
            manifest["entries"][key] = record
            self._save_manifest(manifest)
            archived.append(task)
        if charge_not_requested:
            self.logger.warning(
                "[raw_dft] charge-density output was not requested for %d %s task(s); "
                "per-task details are recorded in %s",
                len(charge_not_requested),
                self.engine,
                self.manifest_path,
            )
        return archived

    def archive_excluded_tasks(self, scf_filter_root, source_manifest):
        with _exclusive_manifest_lock(self.excluded_manifest_path):
            return self._archive_excluded_tasks_unlocked(scf_filter_root, source_manifest)

    def _archive_excluded_tasks_unlocked(self, scf_filter_root, source_manifest):
        _, excluded = self._validate_source_manifest(scf_filter_root, source_manifest)
        manifest = self._load_manifest_from(self.excluded_manifest_path)
        archived = []
        for source_record in excluded:
            task = self._resolve_source_task(scf_filter_root, source_record["source_task"])
            reason = source_record.get("reason")
            if reason not in {"collection_failed", "force_threshold_excluded", "missing_efs"}:
                raise RuntimeError(f"Unsupported excluded DFT task reason {reason!r}: {task}")
            key = self._key(task)
            record = manifest["entries"].get(key)
            archive_path = self._verified_record_archive(
                record, self._record_excluded_archive_path, task
            )
            if archive_path is not None:
                if (
                    record.get("reason") != reason
                    or record.get("scf_filter_sha256") != source_manifest.get("scf_filter_sha256")
                ):
                    raise RuntimeError(
                        f"Existing excluded DFT archive metadata does not match scf_filter.xyz: {task}"
                    )
                archived.append(task)
                continue

            destination = self._excluded_archive_path(task)
            files = self._diagnostic_files(task)
            expected_names = self._write_task_archive(task, destination, files)
            record = {
                "status": "excluded_archived",
                "task": str(task),
                "archive": str(destination),
                "archive_format": "tar.zst",
                "compression": {
                    "codec": "zstd",
                    "level": ZSTD_LEVEL,
                    "threads": ZSTD_THREADS,
                    "version": _zstd_version(),
                },
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
                "files": expected_names,
                "reason": reason,
                "detail": source_record.get("detail"),
                "max_force": source_record.get("max_force"),
                "scf_filter_sha256": source_manifest.get("scf_filter_sha256"),
                "charge_files": [],
                "cleaned": False,
            }
            manifest["entries"][key] = record
            self._save_manifest_to(self.excluded_manifest_path, manifest)
            archived.append(task)
        return archived

    def _cleanup_tasks(self, tasks, manifest_path, resolver, marker_name):
        with _exclusive_manifest_lock(manifest_path):
            return self._cleanup_tasks_unlocked(tasks, manifest_path, resolver, marker_name)

    def _cleanup_tasks_unlocked(self, tasks, manifest_path, resolver, marker_name):
        manifest = self._load_manifest_from(manifest_path)
        for task in tasks:
            key = self._key(task)
            record = manifest["entries"].get(key)
            archive_path = self._verified_record_archive(record, resolver, task)
            if archive_path is None:
                raise RuntimeError(f"Refusing to clean unverified DFT task: {task}")
            for child in task.iterdir():
                if child.name in self.KEEP_AFTER_CLEANUP:
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            marker = task / marker_name
            _atomic_json(
                marker,
                {
                    "archive": str(archive_path),
                    "archive_format": record.get("archive_format", _archive_format(archive_path)),
                    "sha256": record["sha256"],
                },
            )
            record["cleaned"] = True
            self._save_manifest_to(manifest_path, manifest)

    def cleanup_archived_tasks(self, tasks):
        self._cleanup_tasks(
            tasks,
            self.manifest_path,
            self._record_archive_path,
            "__raw_archive__.json",
        )

    def cleanup_excluded_tasks(self, tasks):
        self._cleanup_tasks(
            tasks,
            self.excluded_manifest_path,
            self._record_excluded_archive_path,
            "__raw_archive_excluded__.json",
        )


def collection_cache_path(workspace):
    return Path(workspace) / "dft" / "collection_result.json"


def _task_signature(scf_filter_root):
    if scf_filter_root is None:
        return None
    root = Path(scf_filter_root)
    signature = []
    for task in RawDFTArchiveManager.task_dirs(root):
        marker = task / "__ok__"
        signature.append(
            {
                "path": str(task.relative_to(root)),
                "ok_mtime_ns": marker.stat().st_mtime_ns if marker.exists() else None,
            }
        )
    return signature


def write_collection_cache(workspace, result, scf_filter_root=None):
    serializable = []
    for value in result:
        if isinstance(value, (int, float, str)) or value is None:
            serializable.append(value)
        elif isinstance(value, list):
            serializable.append([str(item) for item in value])
        else:
            try:
                serializable.append(float(value))
            except (TypeError, ValueError):
                serializable.append(str(value))
    path = collection_cache_path(workspace)
    _atomic_json(path, {"result": serializable, "task_signature": _task_signature(scf_filter_root)})
    return path


def load_collection_cache(workspace, required_output, scf_filter_root=None):
    path = collection_cache_path(workspace)
    required_output = Path(required_output)
    if not path.exists() or not required_output.exists() or required_output.stat().st_size == 0:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task_signature") != _task_signature(scf_filter_root):
        return None
    return tuple(payload["result"])
