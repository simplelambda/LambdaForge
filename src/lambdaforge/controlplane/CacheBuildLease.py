"""One cache-build/GC handshake for managed environments and Python runtimes."""

from __future__ import annotations

import json
import math
import os
import socket
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import PurePosixPath
from threading import Event, Thread
from typing import Any

import psutil

from lambdaforge.controlplane.Transport import Transport


class CacheBuildLease:
    """Publish ownership atomically with collection; unknown remote owners stay protected."""

    @staticmethod
    def acquire(
        transport: Transport,
        lock: PurePosixPath,
        *,
        python: str,
        timeout: float,
        references: Mapping[str, Any] | None = None,
    ) -> None:
        runtime = lock.name.startswith(".python-runtime-")
        owner = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "process_birth": psutil.Process().create_time(),
            "heartbeat": time.time(),
            "created_at": time.time(),
            "environment_id": lock.name.removeprefix(".environment-build-")
            .removeprefix(".python-runtime-")
            .removesuffix(".lock"),
            "operation": "runtime-build" if runtime else "environment-build",
            "job_id": os.environ.get("LAMBDAFORGE_JOB_ID"),
            **({"references": dict(references)} if references is not None else {}),
        }
        code = (
            "# lambdaforge-environment-build-lease\n"
            "import fcntl,os,pathlib,stat,sys\n"
            "lock=pathlib.Path(sys.argv[1]); root=lock.parent\n"
            "if root==pathlib.Path(root.anchor) or any(p.is_symlink() for p in "
            "(root,*root.parents,lock)):\n"
            " raise SystemExit('Unsafe managed cache/build root')\n"
            "fd=os.open(root/'.gc.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)\n"
            "try:\n"
            " if not stat.S_ISREG(os.fstat(fd).st_mode):\n"
            "  raise SystemExit('Unsafe cache GC lock metadata')\n"
            " try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
            " except BlockingIOError: raise SystemExit(75)\n"
            " try: lock.mkdir()\n"
            " except FileExistsError: raise SystemExit(75)\n"
            " try:\n"
            "  with (lock/'owner.json').open('x') as handle:\n"
            "   handle.write(sys.argv[2]); handle.flush(); os.fsync(handle.fileno())\n"
            " except BaseException:\n"
            "  (lock/'owner.json').unlink(missing_ok=True); lock.rmdir(); raise\n"
            "finally: os.close(fd)\n"
        )
        deadline = time.monotonic() + timeout
        next_orphan_probe = 0.0
        while True:
            result = transport.run((python, "-c", code, str(lock), json.dumps(owner)), timeout=15)
            if result.returncode == 0:
                return
            if result.returncode != 75:
                raise RuntimeError(
                    f"Could not persist cache build ownership: {result.stderr.strip()}"
                )
            if time.monotonic() >= next_orphan_probe:
                if CacheBuildLease._reclaim_dead_controller(transport, lock, python=python):
                    continue
                next_orphan_probe = time.monotonic() + 20
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for managed cache build lock {lock}.")
            time.sleep(0.2)

    @staticmethod
    @contextmanager
    def preparation(
        transport: Transport,
        *,
        cache_root: PurePosixPath,
        environment: str | None,
        bundle_id: str,
        job_id: str,
        work_dir: str,
        python: str,
        runtime_id: str | None = None,
    ) -> Iterator[None]:
        """Keep exact references through staging → provider ownership hand-off, not just pip."""
        marker = cache_root / f".environment-build-{environment or bundle_id}.use-{job_id}.lock"
        created = transport.run(("mkdir", "-p", str(cache_root)))
        if created.returncode:
            raise RuntimeError("Cannot create the managed preparation reference root.")
        paths = [str(cache_root / "bundles" / bundle_id), work_dir]
        if environment:
            paths.extend(
                [
                    str(cache_root / "environments" / environment),
                    str(cache_root / "environments" / f".{environment}.tmp-"),
                ]
            )
        CacheBuildLease.acquire(
            transport,
            marker,
            python=python,
            timeout=3600,
            references={
                "environments": [environment] if environment else [],
                "bundles": [bundle_id],
                "runtimes": [runtime_id] if runtime_id else [],
                "paths": paths,
            },
        )
        with CacheBuildLease.maintain(transport, marker, python=python):
            yield

    @staticmethod
    def _controller_dead(owner: dict[str, object]) -> bool:
        """Only this controller host can attest its process lifetime; time is not proof."""
        if owner.get("host") != socket.gethostname():
            return False
        try:
            pid, heartbeat = owner["pid"], float(owner["heartbeat"])  # type: ignore[arg-type]
            if type(pid) is not int or pid <= 0 or not math.isfinite(heartbeat):
                return False
            if time.time() - heartbeat < 60:
                return False
            process = psutil.Process(pid)
            birth = owner.get("process_birth", owner.get("created_at"))
            if birth is not None and process.create_time() > float(birth) + 1:  # type: ignore[arg-type]
                return True
            return process.status() in {psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE}
        except psutil.NoSuchProcess:
            return True
        except (KeyError, TypeError, ValueError, psutil.AccessDenied, OSError):
            return False

    @staticmethod
    def _reclaim_dead_controller(transport: Transport, lock: PurePosixPath, *, python: str) -> bool:
        """CAS orphan metadata only, under GC lock, after local death and remote-child proof.

        No environment/temp contents are removed. Inaccessible processes, changed heartbeats,
        foreign controllers and unsupported inspection remain protected. The script is stdlib
        only because bootstrap Python need not have LambdaForge/psutil installed yet.
        """
        inspect = """# lambdaforge-cache-build-inspect
import pathlib,sys
p=pathlib.Path(sys.argv[1]); f=p/'owner.json'
if any(v.is_symlink() for v in (p,*p.parents,f)) or not f.is_file(): raise SystemExit(75)
if f.stat().st_size>8192: raise SystemExit(75)
print(f.read_text(),end='')
"""
        result = transport.run((python, "-c", inspect, str(lock)), timeout=15)
        try:
            raw = result.stdout
            owner = json.loads(raw) if result.returncode == 0 and len(raw) <= 8192 else None
        except (TypeError, ValueError):
            return False
        if not isinstance(owner, dict) or not CacheBuildLease._controller_dead(owner):
            return False
        reclaim = """# lambdaforge-cache-build-reclaim
import fcntl,json,os,pathlib,stat,sys
p=pathlib.Path(sys.argv[1]); f=p/'owner.json'; root=p.parent
if root==pathlib.Path(root.anchor) or any(v.is_symlink() for v in (p,*p.parents,f)):
 raise SystemExit(75)
name=p.name
if not (name.startswith('.environment-build-') or name.startswith('.python-runtime-')):
 raise SystemExit(75)
fd=os.open(root/'.gc.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
try:
 if not stat.S_ISREG(os.fstat(fd).st_mode): raise SystemExit(75)
 try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError: raise SystemExit(75)
 if not f.is_file() or f.stat().st_size>8192 or f.read_text()!=sys.argv[2]: raise SystemExit(75)
 if {v.name for v in p.iterdir()}!={'owner.json'}: raise SystemExit(75)
 if name.startswith('.environment-build-'):
  identity=name[len('.environment-build-'):-len('.lock')]
  needles=(str(root/'environments'/('.'+identity+'.tmp-')),str(root/'environments'/identity)+'/')
 else: needles=(str(root/'runtimes')+'/',)
 references=json.loads(sys.argv[2]).get('references',{})
 needles=(*needles,*(str(value) for value in references.get('paths',())))
 proc=pathlib.Path('/proc')
 if not proc.is_dir(): raise SystemExit(75)
 for entry in proc.iterdir():
  if not entry.name.isdigit() or int(entry.name)==os.getpid(): continue
  try: argv=(entry/'cmdline').read_bytes().split(b'\\0')
  except (FileNotFoundError,ProcessLookupError): continue
  except OSError: raise SystemExit(75)
  if any(n.encode() in arg for n in needles for arg in argv): raise SystemExit(75)
 # Revalidate the snapshot after inspection; no heartbeat or unrelated files may be lost.
 if f.read_text()!=sys.argv[2] or {v.name for v in p.iterdir()}!={'owner.json'}:
  raise SystemExit(75)
 f.unlink(); p.rmdir()
finally: os.close(fd)
"""
        return transport.run((python, "-c", reclaim, str(lock), raw), timeout=15).returncode == 0

    @staticmethod
    @contextmanager
    def maintain(transport: Transport, lock: PurePosixPath, *, python: str) -> Iterator[None]:
        """Keep liveness bounded; never let a failed heartbeat delete an unrelated lease."""
        stopped = Event()
        identity = json.dumps({"pid": os.getpid(), "host": socket.gethostname()})
        code = (
            "# lambdaforge-cache-build-heartbeat\n"
            "import json,pathlib,sys,time\n"
            "p=pathlib.Path(sys.argv[1]); owner=json.loads(sys.argv[2])\n"
            "if p.is_symlink() or p.parent.is_symlink(): raise SystemExit(1)\n"
            "d=json.loads(p.read_text())\n"
            "if any(d.get(k)!=v for k,v in owner.items()): raise SystemExit(1)\n"
            "d['heartbeat']=time.time()\n"
            "t=p.with_suffix('.tmp'); t.write_text(json.dumps(d)); t.replace(p)\n"
        )

        def heartbeat() -> None:
            while not stopped.wait(20):
                try:
                    result = transport.run(
                        (python, "-c", code, str(lock / "owner.json"), identity), timeout=10
                    )
                except Exception:
                    # Transport loss is not evidence that the builder died. GC preserves it.
                    continue
                if result.returncode:
                    return

        thread = Thread(target=heartbeat, daemon=True, name="cache-build-lease")
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join(timeout=12)
            release = (
                "# lambdaforge-cache-build-release\n"
                "import json,pathlib,sys\n"
                "p=pathlib.Path(sys.argv[1]); owner=json.loads(sys.argv[2])\n"
                "if p.is_symlink() or p.parent.is_symlink(): raise SystemExit(1)\n"
                "if p.is_file():\n"
                " d=json.loads(p.read_text())\n"
                " if all(d.get(k)==v for k,v in owner.items()):\n"
                "  p.unlink(); p.parent.rmdir()\n"
            )
            try:
                transport.run(
                    (python, "-c", release, str(lock / "owner.json"), identity), timeout=10
                )
            except Exception:
                # Keep uncertain ownership instead of masking the original installation error.
                pass
