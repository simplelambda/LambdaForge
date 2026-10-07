"""One cache-build/GC handshake for managed environments and Python runtimes."""

from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import PurePosixPath
from threading import Event, Thread

from lambdaforge.controlplane.Transport import Transport


class CacheBuildLease:
    """Publish ownership atomically with collection; unknown remote owners stay protected."""

    @staticmethod
    def acquire(
        transport: Transport, lock: PurePosixPath, *, python: str, timeout: float
    ) -> None:
        runtime = lock.name.startswith(".python-runtime-")
        owner = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "heartbeat": time.time(),
            "created_at": time.time(),
            "environment_id": lock.name.removeprefix(".environment-build-")
            .removeprefix(".python-runtime-").removesuffix(".lock"),
            "operation": "runtime-build" if runtime else "environment-build",
            "job_id": os.environ.get("LAMBDAFORGE_JOB_ID"),
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
        while True:
            result = transport.run((python, "-c", code, str(lock), json.dumps(owner)), timeout=15)
            if result.returncode == 0:
                return
            if result.returncode != 75:
                raise RuntimeError(
                    f"Could not persist cache build ownership: {result.stderr.strip()}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for managed cache build lock {lock}.")
            time.sleep(0.2)

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
