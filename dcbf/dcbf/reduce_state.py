"""Durable state and scheduler transactions for Reduce updates."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fault(point):
    """Opt-in acceptance-test interruption, never active in ordinary runs."""
    if os.environ.get('DCBF_REDUCE_TEST_INTERRUPT') == point:
        raise RuntimeError(f'DCBF Reduce test interruption: {point}')


@contextmanager
def work_lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another Reduce updater owns {path}') from exc
        stream.seek(0)
        stream.truncate()
        json.dump({'host': socket.gethostname(), 'pid': os.getpid(), 'job': os.environ.get('LSB_JOBID')}, stream)
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class ScheduledJob:
    """Submit once; a lost acknowledgement is reconciled by unique job name."""
    def __init__(self, backend, directory, name, script):
        self.backend = backend
        self.directory = Path(directory)
        self.name = name
        self.script = Path(script)
        self.record = self.script.with_suffix('.job.json')
        self.last_status_output = ''

    def _find_submitted(self):
        if self.backend == 'bsub':
            cmd = ['bjobs', '-a', '-J', self.name, '-noheader', '-o', 'jobid stat']
        else:
            cmd = ['squeue', '--name', self.name, '--noheader', '--format=%i %T']
        result = subprocess.run(cmd, text=True, capture_output=True)
        rows = [line.split() for line in result.stdout.splitlines() if line.split() and line.split()[0].isdigit()]
        if len(rows) != 1:
            raise RuntimeError(f'Cannot safely reconcile submission {self.name}; inspect scheduler before retrying')
        return rows[0][0]

    def submit(self):
        previous = read_json(self.record) if self.record.exists() else None
        if previous and previous.get('job_id'):
            return previous['job_id']
        if previous and previous.get('status') == 'submitting':
            job_id = self._find_submitted()
        else:
            save_json(self.record, {'status': 'submitting', 'name': self.name, 'backend': self.backend})
            if self.backend == 'bsub':
                result = subprocess.run(['bsub'], input=self.script.read_text(), cwd=self.directory,
                                        text=True, capture_output=True)
                match = re.search(r'Job <(\d+)>', result.stdout)
            else:
                result = subprocess.run(['sbatch', '--parsable', str(self.script)], cwd=self.directory,
                                        text=True, capture_output=True)
                match = re.match(r'(\d+)', result.stdout)
            if result.returncode or match is None:
                raise RuntimeError(f'Scheduler submission needs reconciliation: {result.stdout} {result.stderr}')
            job_id = match.group(1)
        save_json(self.record, {'status': 'submitted', 'name': self.name, 'backend': self.backend, 'job_id': job_id})
        fault('after_submit')
        return job_id

    def status(self):
        """Last non-empty status line, or UNKNOWN when the query cannot answer.

        ``bjobs``/``squeue`` may print a header plus one row per array element;
        the newest state is the last line.  A failed query must never be read
        as a valid scheduler state, otherwise the wait loop would trust an
        arbitrary error string.
        """
        self.last_status_output = ''
        record = read_json(self.record)
        if self.backend == 'bsub':
            cmd = ['bjobs', '-a', '-noheader', '-o', 'stat', str(record['job_id'])]
        else:
            cmd = ['squeue', '--noheader', '--format=%T', '--jobs', str(record['job_id'])]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.last_status_output = ((result.stdout or '') + (result.stderr or ''))[-2000:]
        if result.returncode != 0:
            return 'UNKNOWN'
        lines = [line.strip() for line in (result.stdout or '').splitlines() if line.strip()]
        if not lines:
            return 'UNKNOWN'
        return lines[-1].split()[0]

    def _status_detail(self):
        return f'status_query={self.last_status_output!r}'

    def wait(self, finished, failed, pending_warning_hours=None, timeout_hours=None,
             unknown_limit=5):
        started = time.monotonic()
        warned = False
        unknown_streak = 0
        status = 'UNKNOWN'
        while True:
            if failed():
                raise RuntimeError(f'Job failed; preserve results and inspect {self.directory}')
            if finished():
                return
            status = self.status()
            if status in ('EXIT', 'DONE', 'FAILED', 'CANCELLED', 'COMPLETED', 'TIMEOUT', 'OUT_OF_MEMORY'):
                # Allow the shared filesystem a short propagation window.
                time.sleep(1)
                if not finished():
                    raise RuntimeError(f'Job {status} without complete outputs: {self.directory}')
                return
            if status == 'UNKNOWN':
                unknown_streak += 1
                if unknown_streak >= int(unknown_limit):
                    raise RuntimeError(
                        f'Job status stayed UNKNOWN for {unknown_streak} consecutive polls '
                        f'while no completion marker appeared: {self.directory}; '
                        f'job_id={read_json(self.record).get("job_id")}; '
                        f'{self._status_detail()}'
                    )
            else:
                unknown_streak = 0
            if pending_warning_hours is not None and not warned and time.monotonic()-started > float(pending_warning_hours)*3600:
                print(f'[reduce.scf] Pending task warning: {self.directory}', flush=True)
                warned = True
            if timeout_hours is not None and time.monotonic()-started >= float(timeout_hours)*3600:
                raise RuntimeError(
                    f'Timed out after {float(timeout_hours):.2f} h waiting for job '
                    f'{self.name} in {self.directory}; last status={status}; '
                    f'{self._status_detail()}'
                )
            time.sleep(2)
