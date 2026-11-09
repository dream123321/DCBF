"""Bounded descriptor work and parent-side generation failure supervision."""
from __future__ import annotations

import functools
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

MIB = 1024 ** 2
_active = None


class DescriptorMemoryError(MemoryError):
    pass


def process_memory(pid):
    for source in ('smaps_rollup', 'smaps'):
        try:
            values = {}
            with open(f'/proc/{pid}/{source}') as handle:
                for line in handle:
                    parts = line.split()
                    if parts and parts[0] in {'Pss:', 'SwapPss:', 'Swap:'}:
                        values[parts[0]] = values.get(parts[0], 0) + int(parts[1]) * 1024
            if 'Pss:' in values:
                return values['Pss:'] + values.get('SwapPss:', values.get('Swap:', 0))
        except (OSError, ValueError):
            pass
    try:
        values = {}
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            parts = line.split()
            if parts and parts[0] in {'VmRSS:', 'VmSwap:'}:
                values[parts[0]] = int(parts[1]) * 1024
        return sum(values.values())
    except (OSError, ValueError):
        return 0


def process_tree(pid):
    pending, seen = [int(pid)], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            for task in Path(f'/proc/{current}/task').iterdir():
                pending.extend(map(int, (task / 'children').read_text().split()))
        except (OSError, ValueError):
            pass
    return seen


def tree_memory(pid):
    return sum(process_memory(child) for child in process_tree(pid))


def _cgroup_limits(pid=None):
    """Read finite limits from both cgroup versions and their ancestors."""
    pid = os.getpid() if pid is None else pid
    limits = []
    try:
        entries = Path(f'/proc/{pid}/cgroup').read_text().splitlines()
    except OSError:
        return limits
    for entry in entries:
        _, controllers, suffix = entry.split(':', 2)
        if controllers == '':
            root, limit_name, usage_name = Path('/sys/fs/cgroup'), 'memory.max', 'memory.current'
        elif 'memory' in controllers.split(','):
            root, limit_name, usage_name = Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes', 'memory.usage_in_bytes'
        else:
            continue
        path = root / suffix.lstrip('/')
        while path == root or root in path.parents:
            try:
                limit = int((path / limit_name).read_text().strip())
                used = int((path / usage_name).read_text().strip())
                if 0 < limit < 2 ** 60:
                    limits.append(max(0, limit - used))
            except (OSError, ValueError):
                pass
            if path == root:
                break
            path = path.parent
    return limits


def available_memory():
    limits = _cgroup_limits()
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                limits.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError):
        pass
    value = os.environ.get('SLURM_MEM_PER_NODE')
    if not value and os.environ.get('SLURM_MEM_PER_CPU'):
        cores = os.environ.get('SLURM_CPUS_ON_NODE', '1').split('(', 1)[0]
        try:
            value = str(int(os.environ['SLURM_MEM_PER_CPU']) * int(cores))
        except ValueError:
            pass
    if value:
        try:
            limits.append(max(0, int(value) * MIB - tree_memory(os.getpid())))
        except ValueError:
            pass
    return min(limits) if limits else 512 * MIB


def atomic_json(path, payload, durable=True):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        if durable:
            os.fsync(handle.fileno())
    os.replace(temporary, path)


def record_failure(workspace, payload):
    workspace = Path(workspace)
    payload = dict(payload, failed_at=time.time())
    text = ('Descriptor processing failed: ' + str(payload.get('error', 'unknown failure'))
            + f" | stage={payload.get('stage')} pid={payload.get('pid')}"
            + f" memory_used={payload.get('memory_used_bytes')} budget={payload.get('memory_limit_bytes')}")
    failures = []
    try:
        atomic_json(workspace / 'memory_failure.json', payload)
    except OSError as exc:
        failures.append(str(exc))
    try:
        (workspace / '__error__').touch()
    except OSError as exc:
        failures.append(str(exc))
    if payload.get('task_type') == 'reduce':
        log_paths = (workspace / 'reduce.log',)
    else:
        run_root = workspace.parent.parent
        log_paths = (run_root / 'app.log', run_root / 'logout')
    for path in log_paths:
        try:
            with path.open('a', encoding='utf-8') as handle:
                date_format = '%Y-%m-%d %H:%M' if path.name == 'app.log' else '%Y-%m-%d %H:%M:%S'
                handle.write(time.strftime(date_format) + ' ERROR: ' + text + '\n')
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            failures.append(str(exc))
    logging.getLogger('logger').error(text)
    if failures:
        print(text + ' | failure report could not be fully written: ' + '; '.join(failures), file=sys.stderr, flush=True)


class DescriptorGuard:
    def __init__(self, workspace, logger=None, budget_bytes=None):
        self.workspace = Path(workspace)
        self.logger = logger or logging.getLogger('logger')
        self.pid = os.getpid()
        self.baseline = tree_memory(self.pid)
        self.budget = int(0.8 * available_memory()) if budget_bytes is None else int(budget_bytes)
        self.limit = self.baseline + self.budget
        self.stage = 'descriptor_encoding'
        self.progress = 0
        self.input = None
        self.workers = None
        self.last_log = time.monotonic()
        self.last_check = 0.0
        self.remaining = 0

    def payload(self, active=True):
        return dict(pid=self.pid, active=active, task_type=getattr(self, 'task_type', 'descriptor'),
                    stage=self.stage, input=self.input,
                    processed=self.progress, memory_limit_bytes=self.limit,
                    additional_memory_budget_bytes=self.budget, workers=self.workers, updated_at=time.time())

    def publish(self, stage=None, progress=None, input_path=None):
        if stage is not None:
            self.stage = stage
        if progress is not None:
            self.progress = int(progress)
        if input_path is not None:
            self.input = str(input_path)
        atomic_json(self.workspace / '.descriptor_stage.json', self.payload(), durable=False)
        now = time.monotonic()
        if now - self.last_log >= 60:
            self.logger.info('Descriptor progress: stage=%s processed=%s memory=%.3f GiB',
                             self.stage, self.progress, tree_memory(self.pid) / 1024 ** 3)
            self.last_log = now

    def check(self, additional=0):
        additional = int(additional)
        now = time.monotonic()
        # Amortize expensive PSS reads while conservatively reserving allocations.
        # The independent parent still measures actual usage every two seconds.
        if now - self.last_check < 2 and additional <= self.remaining // 4:
            self.remaining -= additional
            return self.remaining
        used = tree_memory(self.pid)
        remaining = min(self.limit - used, int(0.8 * available_memory()))
        if additional > remaining:
            raise DescriptorMemoryError(
                f'memory budget exceeded in {self.stage}; used={used} requested={additional} limit={self.limit}')
        self.last_check = now
        self.remaining = remaining - additional
        return self.remaining


def current_guard():
    return _active


def work_memory():
    return _active.check() if _active is not None else int(0.8 * available_memory())


def require_memory(amount):
    if _active is not None:
        _active.check(int(amount))
    elif int(amount) > work_memory():
        raise DescriptorMemoryError(f'Allocation of {amount} bytes exceeds available descriptor memory')


def stage_progress(stage, count=0, input_path=None, workers=None):
    if _active is not None:
        if workers is not None:
            _active.workers = int(workers)
        _active.publish(stage, count, input_path)
        _active.check()


def descriptor_stage(function):
    @functools.wraps(function)
    def wrapped(pwd, *args, **kwargs):
        global _active
        previous = _active
        guard = DescriptorGuard(pwd)
        _active = guard
        try:
            guard.publish()
            result = function(pwd, *args, **kwargs)
            atomic_json(guard.workspace / '.descriptor_stage.json', guard.payload(active=False), durable=False)
            return result
        except Exception as exc:
            payload = guard.payload()
            payload.update(error=f'{type(exc).__name__}: {exc}', memory_used_bytes=tree_memory(guard.pid))
            record_failure(pwd, payload)
            raise
        finally:
            _active = previous
    return wrapped


@contextmanager
def descriptor_guard(workspace, logger=None, budget_bytes=None, task_type='descriptor'):
    global _active
    previous = _active
    guard = DescriptorGuard(workspace, logger=logger, budget_bytes=budget_bytes)
    guard.task_type = str(task_type)
    _active = guard
    try:
        guard.publish(stage=f'{task_type}_start')
        yield guard
        atomic_json(guard.workspace / '.descriptor_stage.json', guard.payload(active=False), durable=False)
    except Exception as exc:
        payload = guard.payload()
        payload.update(
            task_type=str(task_type),
            error=f'{type(exc).__name__}: {exc}',
            memory_used_bytes=tree_memory(guard.pid),
        )
        record_failure(workspace, payload)
        raise
    finally:
        _active = previous


def terminate_owned_group(process, grace=5):
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return True
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue
        # Children can outlive the generation leader after it raises.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return True
    return False


def run_monitored_generation(command, workspace, poll_seconds=2, env=None):
    workspace = Path(workspace)
    started = time.time()
    process = subprocess.Popen(command, start_new_session=True, env=env)
    strikes = 0
    active_payload = None
    while process.poll() is None:
        try:
            payload = json.loads((workspace / '.descriptor_stage.json').read_text())
        except (OSError, ValueError):
            payload = {}
        fresh = payload.get('pid') == process.pid and payload.get('updated_at', 0) >= started
        if fresh and not payload.get('active'):
            active_payload = None
            strikes = 0
        if fresh and payload.get('active'):
            active_payload = payload
            used = tree_memory(process.pid)
            strikes = strikes + 1 if used > payload['memory_limit_bytes'] else 0
            try:
                failure = json.loads((workspace / 'memory_failure.json').read_text())
            except (OSError, ValueError):
                failure = {}
            if strikes >= 2 or failure.get('failed_at', 0) >= started:
                payload.update(error=failure.get('error', 'memory budget exceeded'), memory_used_bytes=used)
                record_failure(workspace, payload)
                ended = terminate_owned_group(process)
                if not ended:
                    payload['termination_pending'] = True
                    record_failure(workspace, payload)
                raise DescriptorMemoryError(payload['error'])
        try:
            process.wait(timeout=poll_seconds)
        except subprocess.TimeoutExpired:
            pass
    if process.returncode:
        # A short-lived failure may happen between polling intervals.
        try:
            last = json.loads((workspace / '.descriptor_stage.json').read_text())
            if last.get('pid') == process.pid and last.get('updated_at', 0) >= started:
                active_payload = last if last.get('active') else None
        except (OSError, ValueError):
            pass
        if active_payload is not None:
            task_label = active_payload.get('task_type', 'generation')
            active_payload['error'] = (
                f'{task_label} worker exited with return code {process.returncode}; '
                'OOM is not confirmed'
            )
            try:
                failure = json.loads((workspace / 'memory_failure.json').read_text())
            except (OSError, ValueError):
                failure = {}
            if failure.get('failed_at', 0) < started:
                record_failure(workspace, active_payload)
            terminate_owned_group(process)
        raise subprocess.CalledProcessError(process.returncode, command)
