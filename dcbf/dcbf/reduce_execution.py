"""Reuse sampling's scheduler, SCF preparation/collection and SUS2 continuation."""
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import re
import shlex
import shutil

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import iread, write
import yaml

from .labelled_data import labels, write_training_cfg
from .reduce_state import ScheduledJob, fault, read_json, save_json, sha
from .runtime_config import build_scheduler_spec
from .sus2_training import check_complete_checkpoint, continuation_document, load_model_yaml, write_run_yaml


@contextmanager
def chdir(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def model_kind(path):
    text = Path(path).read_text(encoding='utf-8')
    def header(key):
        match = re.search(rf'(?m)^{re.escape(key)}\s*=\s*(.*)$', text)
        return match.group(1).strip().lower() if match else ''
    # Actual parameter-mode fields, not the presence of gate metadata alone.
    mode = header('outer_radial_mode')
    if 'adaptive' in mode:
        raise ValueError('coordination-adaptive continuation is not supported')
    effect = header('two_layer_gate_enabled') in ('true', '1')
    return 'coordination-effect' if effect else 'static-radial'


class ReduceExecutor:
    def __init__(self, config, config_dir, workspace, elements, token):
        self.config = config
        self.base = Path(config_dir)
        self.workspace = Path(workspace)
        self.elements = list(elements)
        self.token = token
        grouped_scheduler = dict(config.get('scheduler') or {})
        train = dict(grouped_scheduler.pop('train', {}) or {})
        lmp = dict(grouped_scheduler.pop('lmp', {}) or {})
        scf = dict(grouped_scheduler.pop('scf', {}) or {})
        self.raw_scheduler = dict(grouped_scheduler)
        self.raw_scheduler.update({
            'sus2_mlp_exe': train.get('sus2_mlp_exe', self.raw_scheduler.get('sus2_mlp_exe')),
            'train_env': train.get('train_env', self.raw_scheduler.get('train_env')),
            'original_command': train.get('original_command', self.raw_scheduler.get('original_command')),
            'subsequent_command': train.get('subsequent_command', self.raw_scheduler.get('subsequent_command')),
            'train_sus_queue': train.get('queue', train.get('train_sus_queue', self.raw_scheduler.get('train_sus_queue'))),
            'train_sus_cores': train.get('cores', train.get('train_sus_cores', self.raw_scheduler.get('train_sus_cores'))),
            'train_sus_ptile': train.get('ptile', train.get('train_sus_ptile', self.raw_scheduler.get('train_sus_ptile'))),
            'lmp_exe': lmp.get('lmp_exe', self.raw_scheduler.get('lmp_exe')),
            'lmp_env': lmp.get('lmp_env', self.raw_scheduler.get('lmp_env')),
            'lmp_queue': lmp.get('queue', lmp.get('lmp_queue', self.raw_scheduler.get('lmp_queue'))),
            'lmp_cores': lmp.get('cores', lmp.get('lmp_cores', self.raw_scheduler.get('lmp_cores'))),
            'lmp_ptile': lmp.get('ptile', lmp.get('lmp_ptile', self.raw_scheduler.get('lmp_ptile'))),
            'scf_queue': scf.get('queue', scf.get('scf_queue', self.raw_scheduler.get('scf_queue'))),
            'scf_cores': scf.get('cores', scf.get('scf_cores', self.raw_scheduler.get('scf_cores'))),
            'scf_ptile': scf.get('ptile', scf.get('scf_ptile', self.raw_scheduler.get('scf_ptile'))),
            'scf_cal_engine': scf.get('cal_engine', scf.get('scf_cal_engine', self.raw_scheduler.get('scf_cal_engine'))),
            'dft_clean_dcbf_environment': scf.get('clean_dcbf_environment', scf.get('dft_clean_dcbf_environment', self.raw_scheduler.get('dft_clean_dcbf_environment'))),
            'dft_env': scf.get('dft_env', self.raw_scheduler.get('dft_env')),
            'dft_command': scf.get('dft_command', self.raw_scheduler.get('dft_command')),
        })
        self.raw_scheduler = {key: value for key, value in self.raw_scheduler.items() if value is not None}
        self.scheduler = build_scheduler_spec({'scf_cal_engine': 'vasp', **self.raw_scheduler})
        self.training = dict(config.get('training') or {})
        self.dft = dict(config.get('dft') or {})
        self._submit_false_warned = False

    def _timeout_hours(self, *candidates):
        """Positive hour limit, or None to preserve the historical wait forever."""
        for value in candidates:
            if value is None:
                continue
            try:
                hours = float(value)
            except (TypeError, ValueError):
                continue
            if hours > 0:
                return hours
        return None

    def training_timeout_hours(self):
        return self._timeout_hours(self.training.get('timeout_hours'))

    def scf_timeout_hours(self):
        return self._timeout_hours(
            self.dft.get('scf_timeout_hours'),
            (self.config.get('workflow') or {}).get('scf_timeout_hours'),
        )

    def preflight_training(self, model):
        check_complete_checkpoint(model)
        from .builtin_models import check_continuation_allowed
        check_continuation_allowed(model)
        raw = (self.config.get('potential') or {}).get('model_yaml')
        if raw is None:
            raise ValueError('potential.model_yaml is required when updating/training the Reduce model')
        path = Path(raw)
        if not path.is_absolute():
            path = self.base / path
        document, kind = load_model_yaml(path)
        if kind != model_kind(model):
            raise ValueError('potential.model_yaml training profile does not match checkpoint static/effect kind')
        self.model_document = continuation_document(document, model)
        self.model_yaml = path.resolve()
        for key in ('train_sus_cores', 'train_sus_ptile'):
            value = getattr(self.scheduler, key)
            if value is None:
                # Unset means "let the scheduler pick"; runtime_config omits -n.
                continue
            if int(value) <= 0:
                raise ValueError(f'scheduler.{key} must be positive')

    def prepare_templates(self):
        raw = self.dft.get('template_dir')
        if not raw:
            raise ValueError('dft.template_dir is required when reduce.run_scf_labeling=true')
        source = Path(raw)
        if not source.is_absolute():
            source = self.base / source
        source = source.resolve()
        if not source.is_dir():
            raise ValueError(f'DFT template directory does not exist: {source}')
        engine = self.scheduler.scf_cal_engine
        required = {'vasp': 'INCAR', 'abacus': 'INPUT', 'qe': 'qe.in', 'cp2k': 'cp2k.inp'}
        if engine not in required or not (source / required[engine]).is_file():
            raise ValueError(f'Missing {required.get(engine, engine)} in DFT template directory')
        if engine == 'vasp':
            incar = (source/'INCAR').read_text()
            match = re.search(r'(?im)^\s*NSW\s*=\s*(\d+)', incar)
            if match and int(match.group(1)) != 0:
                raise ValueError('Reduce SCF requires NSW=0 (fixed geometry)')
        # ``main_calc`` and the VASP collector walk up from
        # ``round/dft/scf/filter`` to the Reduce work root and read its
        # ``init`` directory.  ``self.workspace`` is ``<work>/updates``, so
        # copy to ``<work>/init`` rather than ``<work>/updates/init``.
        target = self.workspace.parent / 'init'
        if not target.exists():
            shutil.copytree(source, target, symlinks=False)
        self._prepare_potcar_directory(target)
        return {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}

    def _prepare_potcar_directory(self, target):
        if self.scheduler.scf_cal_engine != 'vasp' or (target/'POTCAR_dir').is_dir():
            return
        combined = target/'POTCAR'
        if not combined.is_file():
            if shutil.which('vaspkit') is None:
                raise ValueError('VASP templates require POTCAR_dir, a POTCAR, or configured vaspkit')
            return
        data = combined.read_bytes()
        entries = re.findall(rb'.*?End of Dataset[^\n]*(?:\n|$)', data, flags=re.S)
        if not entries or b''.join(entries).strip() != data.strip():
            raise ValueError('Cannot safely split template POTCAR into species records')
        for entry in entries:
            title = re.search(rb'TITEL\s*=\s*\S+\s+([A-Z][a-z]?)(?=[_\s])', entry)
            if title is None:
                raise ValueError('POTCAR record has no identifiable TITEL element')
            folder = target/'POTCAR_dir'/title.group(1).decode()
            folder.mkdir(parents=True, exist_ok=True)
            (folder/'POTCAR').write_bytes(entry)

    def _job_name(self, batch, suffix):
        return f'dcbfr-{self.token[:12]}-{Path(batch).name}-{suffix}'

    def train(self, directory, dataset, model, iterations, output_name='current.sus2'):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = directory/'training.json'
        signature = {'input_model': str(model), 'model_sha256': sha(model),
                     'dataset_sha256': sha(dataset), 'iterations': int(iterations),
                     'elements': self.elements, 'model_yaml_sha256': sha(self.model_yaml),
                     'radial_parameterization': self.model_document.get('training', {}).get('radial_parameterization', 'physical')}
        if meta_path.exists():
            meta = read_json(meta_path)
            if meta['signature'] != signature:
                raise RuntimeError(f'Cannot reuse training with changed inputs: {directory}')
            if meta['status'] == 'complete':
                if sha(meta['output_model']) != meta['output_sha256']:
                    raise RuntimeError('Completed training checkpoint was changed')
                return Path(meta['output_model'])
        stats = write_training_cfg(dataset, directory/'train.cfg', self.elements,
                                   self.training.get('labels', 'auto'))
        document = deepcopy(self.model_document)
        if stats['effective'] == 'ef':
            document.setdefault('loss', {})['virial_weight'] = 0
        write_run_yaml(directory/'run.yaml', document, train_cfg=directory/'train.cfg',
                       output_dir=directory, iterations=int(iterations), continuation_model=model,
                       output_name='current.sus2',
                       best_name='_best.sus2' if output_name == 'best.sus2' else 'best.sus2',
                       final_name=output_name if output_name != 'current.sus2' else None,
                       elements=self.elements)
        meta = {'signature': signature, 'status': 'prepared', 'labels': stats}
        save_json(meta_path, meta)
        prefix = str(self.scheduler.subsequent_COMMAND or '').strip()
        if prefix:
            command = prefix.replace('{sus2_mlp_exe}', shlex.quote(self.scheduler.sus2_mlp_exe))
            command = command.replace('{initial_yaml}', 'run.yaml')
            command = command.replace('{continuation_yaml}', 'run.yaml')
        else:
            command = ('mpirun -np "${NP:-1}" ' +
                       shlex.quote(self.scheduler.sus2_mlp_exe) + ' train --config run.yaml')
        name = self._job_name(directory.parent, directory.name)
        script = ('#!/bin/bash\n' + self.scheduler.bsub_script_train_sus_job_name + name + '\n' +
                  self.scheduler.bsub_script_train_sus + '\nset -u\n' + self.scheduler.train_env + '\n' +
                  f'{command} > logout 2>&1\nrc=$?\n'
                  'if [ "$rc" -eq 0 ]; then touch __ok__; else touch __failed__; fi\nexit "$rc"\n')
        script_path = directory/'bsub.lsf'
        if not (directory/'__ok__').exists():
            if self.training.get('submit') is False and not self._submit_false_warned:
                print('[reduce] training.submit=false is ignored; reduce training is always '
                      'submitted to the scheduler', flush=True)
                self._submit_false_warned = True
            script_path.write_text(script, encoding='utf-8')
            job = ScheduledJob(self.scheduler.submission_backend, directory, name, script_path)
            job.submit()
            job.wait(lambda: (directory/'__ok__').exists(),
                     lambda: (directory/'__failed__').exists(),
                     timeout_hours=self.training_timeout_hours())
        output = check_complete_checkpoint(directory/output_name)
        if model_kind(output) != model_kind(model):
            raise RuntimeError('Continuation changed static/effect model kind')
        from .encode.model_metadata import read_sus2_descriptor_metadata
        before = read_sus2_descriptor_metadata(model)
        after = read_sus2_descriptor_metadata(output)
        if (before.species_count, before.alpha_moment_mapping, before.sh_scalar_info) != \
           (after.species_count, after.alpha_moment_mapping, after.sh_scalar_info):
            raise RuntimeError('Continuation changed checkpoint species/descriptor topology')
        meta.update(status='complete', output_model=str(output), output_sha256=sha(output))
        save_json(meta_path, meta)
        save_json(Path(str(output)+'.elements.json'), {'elements': self.elements, 'model_sha256': sha(output)})
        fault('after_training')
        return output

    def scf(self, batch, source_xyz, source_ids):
        from .das.main_calc import main_calc
        from .das.scf_filter_sources import load_scf_filter_sources
        batch = Path(batch)
        atoms = list(iread(str(source_xyz), format='extxyz'))
        scf = batch/'dft/scf'
        scf.mkdir(parents=True, exist_ok=True)
        prepared = batch/'scf_prepared.json'
        count = int(self.dft.get('calc_dir_num', 5))
        count = len(atoms) if count <= 0 else min(count, len(atoms))
        if not prepared.exists():
            # Sampling's engine preparation expects workspace/main_0/gen_N.
            (batch/'parameter.yaml').write_text(yaml.safe_dump({'dft': {**self.dft, 'calc_dir_num': count}}))
            with chdir(scf):
                main_calc(atoms, count, str(batch), self.scheduler, resume_tasks=True)
            save_json(prepared, {'source_sha256': sha(source_xyz), 'source_ids': source_ids, 'groups': count})
        else:
            record = read_json(prepared)
            if record['source_sha256'] != sha(source_xyz) or record['source_ids'] != source_ids:
                raise RuntimeError('SCF source changed after preparation')
            count = record['groups']
        jobs = []
        for group in range(1, count+1):
            group_dir = scf/'filter'/f'dir_{group}'
            tasks = sorted(p for p in group_dir.iterdir() if p.is_dir())
            if all((p/'__ok__').exists() for p in tasks):
                continue
            script = scf/'jobs_script'/f'bsub_{group}.lsf'
            name = self._job_name(batch, f'scf{group}')
            text = script.read_text()
            text = re.sub(r'(?m)^#(?:BSUB -J|SBATCH -J) .*$',
                          ('#BSUB -J ' if self.scheduler.submission_backend == 'bsub' else '#SBATCH -J ') + name, text)
            script.write_text(text, encoding='utf-8')
            job = ScheduledJob(self.scheduler.submission_backend, script.parent, name, script)
            job.submit()
            jobs.append((job, tasks))
        for job, tasks in jobs:
            job.wait(lambda ts=tasks: all((p/'__ok__').exists() for p in ts),
                     lambda ts=tasks: any((p/'__failed__').exists() for p in ts),
                     self.dft.get('pending_warning_hours'),
                     timeout_hours=self.scf_timeout_hours())
        fault('after_scf_jobs')
        collected = batch/'dft/scf_filter.xyz'
        cache = batch/'scf_collection.json'
        if not cache.exists():
            modules = {'vasp': ('vasp_main_xyz', 'vasp_main_xyz'),
                       'abacus': ('abacus_main_xyz', 'abacus_main_xyz'),
                       'cp2k': ('cp2k_main_xyz', 'cp2k_main_xyz'), 'qe': ('qe_main_xyz', 'qe_main_xyz')}
            import importlib
            mod, symbol = modules[self.scheduler.scf_cal_engine]
            collector = getattr(importlib.import_module('.das.'+mod, __package__), symbol)
            result = collector(str(scf/'filter'), str(collected), str(batch/'dft/original_scf.xyz'),
                               float(self.dft.get('force_threshold', 20)))
            sources = load_scf_filter_sources(collected)
            if not sources['frames']:
                raise RuntimeError('SCF completed without any usable labels')
            save_json(cache, {'result': result, 'sources': sources, 'output_sha256': sha(collected)})
        record = read_json(cache)
        if record['output_sha256'] != sha(collected):
            raise RuntimeError('Collected SCF labels changed')
        output_atoms = list(iread(str(collected), format='extxyz'))
        accepted = []
        for atom, source in zip(output_atoms, record['sources']['frames']):
            index = int(Path(source['source_task']).name)-1
            original = atoms[index]
            if self.scheduler.scf_cal_engine == 'vasp':
                ordered = sorted(range(len(original)), key=lambda i: original[i].symbol)
                inverse = np.argsort(ordered)
                reordered = atom[inverse]
                energy, forces, virial = labels(atom)
                reordered.calc = SinglePointCalculator(reordered, energy=energy, forces=forces[inverse])
                if virial is not None:
                    reordered.info['virial'] = virial
                atom = reordered
            if (atom.get_chemical_symbols() != original.get_chemical_symbols() or
                not np.allclose(atom.cell.array, original.cell.array, atol=1e-5, rtol=0) or
                not np.allclose(atom.positions, original.positions, atol=1e-5, rtol=0)):
                raise RuntimeError('SCF changed geometry or atom identity/order')
            labels(atom)
            energy, forces, virial = labels(atom)
            result_atom = original.copy()
            for key in ('forces', 'force'):
                result_atom.arrays.pop(key, None)
            for key in ('energy', 'free_energy', 'stress', 'virial'):
                result_atom.info.pop(key, None)
            result_atom.calc = SinglePointCalculator(result_atom, energy=energy, forces=forces)
            if virial is not None:
                result_atom.info['virial'] = virial
            result_atom.info['dcbf_source_frame'] = int(source_ids[index])
            result_atom.info['dcbf_label_source'] = 'SCF'
            accepted.append((source_ids[index], result_atom, source))
        accepted.sort(key=lambda x: x[0])
        out = batch/'labeled_new.xyz'
        write(str(out), [a for _, a, _ in accepted], format='extxyz')
        save_json(batch/'label_sources.json', {'frames': [{'source_index': i, **r} for i, _, r in accepted],
                                             'collection': record})
        fault('after_scf_collection')
        return out, [i for i, _, _ in accepted]
