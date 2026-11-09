from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from ase.io import iread, write

from .das.file_conversion import cfg2xyz, xyz2cfg
from .runtime_config import find_runtime_config, load_json_config


def resolve_summary_bundle_dir(run_dir, config):
    run_dir = Path(run_dir).resolve()
    summary = dict((config or {}).get("summary") or {})
    raw_path = Path(summary.get("output_dir", "summary_bundle"))
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (run_dir.parent / raw_path).resolve()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_xyz(path):
    return sum(1 for _ in iread(str(path), index=":"))


class TrainingDatasetStore:
    MANIFEST_VERSION = 1

    def __init__(self, run_dir, config=None):
        self.run_dir = Path(run_dir).resolve()
        runtime_path = self.run_dir / "dcbf.runtime.json"
        if runtime_path.exists():
            config = load_json_config(runtime_path)
        elif config is None:
            config = load_json_config(find_runtime_config(self.run_dir))
        self.config = dict(config)
        parameter = dict(self.config.get("parameter") or {})
        self.elements = list(parameter.get("ele") or [])
        self.element_model = 1 if parameter.get("sort_ele", True) else 2
        self.bundle_dir = resolve_summary_bundle_dir(self.run_dir, self.config)
        self.history_dir = self.bundle_dir / "datasets" / "training_history"
        self.xyz_dir = self.history_dir / "xyz"
        self.manifest_path = self.history_dir / "manifest.json"
        self.runtime_dir = self.run_dir / ".dcbf_runtime" / "training"
        self.global_cfg = self.runtime_dir / "train.cfg"

    @classmethod
    def from_generation(cls, generation_dir):
        runtime_path = find_runtime_config(generation_dir)
        return cls(runtime_path.parent, load_json_config(runtime_path))

    def _empty_manifest(self):
        return {
            "version": self.MANIFEST_VERSION,
            "status": "in_progress",
            "run_dir": os.path.relpath(self.run_dir, self.run_dir),
            "global_cfg": os.path.relpath(self.global_cfg, self.run_dir),
            "elements": self.elements,
            "entries": [],
        }

    def _portable_path(self, path):
        path = Path(path).resolve()
        try:
            path.relative_to(self.run_dir.parent)
        except ValueError:
            return str(path)
        try:
            return os.path.relpath(path, self.run_dir)
        except ValueError:
            return str(path)

    def _resolve_manifest_path(self, raw_path, expected_sha256=None):
        path = Path(raw_path)
        if not path.is_absolute():
            candidate = self.run_dir / path
            if candidate.exists():
                return candidate
            return candidate
        if path.exists():
            return path

        parts = path.parts
        for anchor, base in (("summary_bundle", self.bundle_dir), (self.run_dir.name, self.run_dir)):
            if anchor in parts:
                candidate = base.joinpath(*parts[parts.index(anchor) + 1:])
                if candidate.exists():
                    return candidate

        if expected_sha256:
            for root in (self.run_dir, self.run_dir.parent):
                if not root.exists():
                    continue
                for candidate in root.rglob(path.name):
                    if candidate.is_file():
                        try:
                            if _sha256(candidate) == expected_sha256:
                                return candidate
                        except OSError:
                            continue
        return path

    def load_manifest(self):
        if not self.manifest_path.exists():
            return self._empty_manifest()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != self.MANIFEST_VERSION:
            raise RuntimeError(f"Unsupported training-history manifest: {self.manifest_path}")
        if list(manifest.get("elements") or []) != self.elements:
            raise RuntimeError("Training-history element order does not match the runtime configuration")
        return manifest

    def _save_manifest(self, manifest):
        _atomic_json(self.manifest_path, manifest)

    @staticmethod
    def _entry_key(main_index, generation_index):
        if main_index is None:
            return "initial"
        return f"main_{int(main_index):03d}_gen_{int(generation_index):03d}"

    def _entry(self, manifest, key):
        for entry in manifest.get("entries", []):
            if entry.get("key") == key:
                return entry
        return None

    def _annotate_cfg_as_xyz(self, cfg_path, destination, main_value):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw_path = destination.with_name(destination.name + ".raw.tmp")
        output_path = destination.with_name(destination.name + ".tmp")
        for path in (raw_path, output_path):
            if path.exists():
                path.unlink()
        cfg2xyz(self.elements, self.element_model, str(cfg_path), str(raw_path))
        atoms_list = list(iread(str(raw_path), index=":", format="extxyz"))
        label = dict(self.config.get("dataset") or {}).get("all_label")
        for atoms in atoms_list:
            atoms.info["main"] = int(main_value)
            if label:
                atoms.info["label"] = label
            elif main_value == -1:
                atoms.info.pop("label", None)
        write(str(output_path), atoms_list, format="extxyz")
        raw_path.unlink()
        os.replace(output_path, destination)
        return len(atoms_list)

    def _new_cfg(self, xyz_path, destination):
        destination = Path(destination)
        if destination.exists():
            destination.unlink()
        xyz2cfg(self.elements, self.element_model, str(xyz_path), str(destination))
        if not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError(f"Failed to create training CFG from {xyz_path}")

    def ensure_initial(self, xyz_path):
        xyz_path = Path(xyz_path).resolve()
        if not xyz_path.exists():
            raise FileNotFoundError(xyz_path)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.xyz_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.load_manifest()
        entry = self._entry(manifest, "initial")
        shard = self.xyz_dir / "000000_initial.xyz"
        if entry and entry.get("status") == "complete" and shard.exists():
            complete_entries = [
                item for item in manifest.get("entries", [])
                if item.get("status") == "complete"
            ]
            expected_size = max(
                (int(item.get("cfg_size_after", 0)) for item in complete_entries),
                default=0,
            )
            if self.global_cfg.exists() and self.global_cfg.stat().st_size >= expected_size:
                return self.global_cfg
            return self.rebuild_global_cfg()

        cfg_tmp = self.runtime_dir / "train.cfg.initial.tmp"
        self._new_cfg(xyz_path, cfg_tmp)
        frame_count = self._annotate_cfg_as_xyz(cfg_tmp, shard, -1)
        os.replace(cfg_tmp, self.global_cfg)
        entry_payload = {
            "key": "initial",
            "status": "complete",
            "main": -1,
            "generation": None,
            "source": self._portable_path(xyz_path),
            "path": self._portable_path(shard),
            "frames": frame_count,
            "sha256": _sha256(shard),
            "cfg_size_before": 0,
            "cfg_size_after": self.global_cfg.stat().st_size,
        }
        manifest["entries"] = [item for item in manifest.get("entries", []) if item.get("key") != "initial"]
        manifest["entries"].insert(0, entry_payload)
        manifest["status"] = "in_progress"
        self._save_manifest(manifest)
        return self.global_cfg

    def ensure_generation_link(self, generation_dir):
        generation_dir = Path(generation_dir).resolve()
        link = generation_dir / "train_mlp" / "train.cfg"
        return self.ensure_cfg_link(link)

    def ensure_cfg_link(self, link):
        link = Path(link).resolve() if Path(link).exists() else Path(link).absolute()
        link.parent.mkdir(parents=True, exist_ok=True)
        if not self.global_cfg.exists():
            dataset = dict(self.config.get("dataset") or {})
            initial = dataset.get("xyz_input")
            if not initial:
                raise RuntimeError("Cannot initialize the global train.cfg: dataset.xyz_input is missing")
            initial = Path(initial)
            if not initial.is_absolute():
                initial = self.run_dir / initial
            self.ensure_initial(initial)
        if link.exists() or link.is_symlink():
            try:
                if link.is_symlink() and link.resolve() == self.global_cfg.resolve():
                    return link
                if link.exists() and os.path.samefile(link, self.global_cfg):
                    return link
            except OSError:
                pass
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            link.symlink_to(self.global_cfg)
        except OSError:
            os.link(self.global_cfg, link)
        return link

    def _append_file(self, source, destination):
        with open(source, "rb") as reader, open(destination, "ab") as writer:
            if writer.tell() > 0:
                writer.write(b"\n")
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())

    def _segment_sha256(self, offset, size):
        digest = hashlib.sha256()
        with open(self.global_cfg, "rb") as handle:
            handle.seek(offset)
            remaining = size
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
        return digest.hexdigest() if remaining == 0 else None

    def append_generation(self, generation_dir, scf_xyz):
        generation_dir = Path(generation_dir).resolve()
        scf_xyz = Path(scf_xyz).resolve()
        main_index = int(generation_dir.parent.name.replace("main_", ""))
        generation_index = int(generation_dir.name.replace("gen_", ""))
        key = self._entry_key(main_index, generation_index)
        shard = self.xyz_dir / f"{key}.xyz"
        self.ensure_generation_link(generation_dir)

        manifest = self.load_manifest()
        entry = self._entry(manifest, key)
        if entry and entry.get("status") == "complete" and shard.exists():
            if self.global_cfg.stat().st_size < int(entry["cfg_size_after"]):
                self.rebuild_global_cfg()
            return shard

        append_cfg = self.runtime_dir / f"{key}.cfg.tmp"
        self._new_cfg(scf_xyz, append_cfg)
        frame_count = self._annotate_cfg_as_xyz(append_cfg, shard, main_index)
        append_bytes = append_cfg.read_bytes()
        separator = b"\n" if self.global_cfg.stat().st_size > 0 else b""
        payload = separator + append_bytes
        payload_hash = hashlib.sha256(payload).hexdigest()
        size_before = self.global_cfg.stat().st_size

        if entry and entry.get("status") == "pending":
            old_before = int(entry.get("cfg_size_before", size_before))
            old_size = int(entry.get("cfg_append_size", 0))
            current_size = self.global_cfg.stat().st_size
            if current_size == old_before + old_size and self._segment_sha256(old_before, old_size) == entry.get("cfg_append_sha256"):
                entry["status"] = "complete"
                entry["cfg_size_after"] = current_size
                entry["path"] = self._portable_path(shard)
                entry["sha256"] = _sha256(shard)
                self._save_manifest(manifest)
                append_cfg.unlink(missing_ok=True)
                return shard
            with open(self.global_cfg, "r+b") as handle:
                handle.truncate(old_before)
            size_before = old_before

        pending = {
            "key": key,
            "status": "pending",
            "main": main_index,
            "generation": generation_index,
            "source": self._portable_path(scf_xyz),
            "path": self._portable_path(shard),
            "frames": frame_count,
            "sha256": _sha256(shard),
            "cfg_size_before": size_before,
            "cfg_append_size": len(payload),
            "cfg_append_sha256": payload_hash,
        }
        manifest["entries"] = [item for item in manifest.get("entries", []) if item.get("key") != key]
        manifest["entries"].append(pending)
        self._save_manifest(manifest)

        with open(self.global_cfg, "ab") as writer:
            writer.write(payload)
            writer.flush()
            os.fsync(writer.fileno())
        pending["status"] = "complete"
        pending["cfg_size_after"] = self.global_cfg.stat().st_size
        self._save_manifest(manifest)
        append_cfg.unlink(missing_ok=True)
        return shard

    def rebuild_global_cfg(self):
        manifest = self.load_manifest()
        entries = [entry for entry in manifest.get("entries", []) if entry.get("status") == "complete"]
        if not entries:
            raise RuntimeError("Cannot rebuild train.cfg: no complete XYZ history entries exist")
        temporary = self.runtime_dir / "train.cfg.rebuild.tmp"
        temporary.unlink(missing_ok=True)
        offset = 0
        for entry in entries:
            part = self.runtime_dir / (entry["key"] + ".rebuild.cfg")
            source = self._resolve_manifest_path(entry["path"], entry.get("sha256"))
            self._new_cfg(source, part)
            self._append_file(part, temporary)
            part.unlink()
            entry["path"] = self._portable_path(source)
            entry["cfg_size_before"] = offset
            offset = temporary.stat().st_size
            entry["cfg_size_after"] = offset
        os.replace(temporary, self.global_cfg)
        self._save_manifest(manifest)
        return self.global_cfg

    def export_final_xyz(self, output_path):
        manifest = self.load_manifest()
        entries = [entry for entry in manifest.get("entries", []) if entry.get("status") == "complete"]
        if not entries:
            raise RuntimeError("Cannot export final XYZ: training history is empty")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        with open(temporary, "wb") as writer:
            for entry in entries:
                source = self._resolve_manifest_path(entry["path"], entry.get("sha256"))
                if not source.exists() or _sha256(source) != entry.get("sha256"):
                    raise RuntimeError(f"Training-history shard is missing or corrupt: {source}")
                with open(source, "rb") as reader:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, output_path)
        return output_path

    def mark_complete_and_cleanup(self):
        manifest = self.load_manifest()
        manifest["status"] = "complete"
        manifest["structure_count"] = sum(
            int(entry.get("frames", 0))
            for entry in manifest.get("entries", [])
            if entry.get("status") == "complete"
        )
        self._save_manifest(manifest)
        links = [path for path in self.run_dir.rglob("train.cfg") if path != self.global_cfg]
        for link in links:
            try:
                linked = link.is_symlink() and link.resolve() == self.global_cfg.resolve()
                linked = linked or (link.exists() and os.path.samefile(link, self.global_cfg))
            except OSError:
                linked = False
            if linked:
                link.unlink()
        self.global_cfg.unlink(missing_ok=True)
