"""Stable-prefix installation with an atomic, verified readiness receipt.

Conda prefixes and Python environments are not relocatable directory trees. Installers
must see the lifetime prefix; only the readiness receipt is atomically published.
Existing prefixes, including incomplete builds, are never repaired or removed here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.SecretRedactor import SecretRedactor
from lambdaforge.controlplane.Transport import Transport

PUBLICATION_CONTRACT = "stable-prefix-v1"


class EnvironmentPublication:
    """Used under the existing per-identity build/GC lease, never instead of it."""

    @staticmethod
    @contextmanager
    def build(transport: Transport, python: str, target: PurePosixPath) -> Iterator[None]:
        """Reserve a fresh final path; preserve failed builds for reference-aware GC."""
        token = uuid4().hex
        claim = """import json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]); token=sys.argv[2]; contract=sys.argv[3]
if not p.name.startswith(('env-','deps-')) or any(v.is_symlink() for v in (p,*p.parents)):
 raise SystemExit('Unsafe managed installation prefix: '+str(p))
p.parent.mkdir(parents=True,exist_ok=True)
try: p.mkdir()
except FileExistsError:
 raise SystemExit('Managed prefix already exists without a reusable readiness receipt: '+str(p)+
                  '. Preserved; inspect publication diagnostics and preview lf clean '
                  '--environments-only. Never delete a live/recovery-referenced prefix.')
owner=p.parent.parent/('.environment-build-'+p.name+'.lock')/'publication.json'
if not owner.parent.is_dir() or owner.parent.is_symlink():
 raise SystemExit('Missing managed publication build lease')
with owner.open('x') as f:
 json.dump({'contract':contract,'token':token,'prefix':str(p),'state':'building'},f)
 f.flush(); os.fsync(f.fileno())
"""
        result = transport.run((python, "-c", claim, str(target), token, PUBLICATION_CONTRACT))
        if result.returncode:
            raise RuntimeError(
                f"Cannot reserve managed installation prefix: {result.stderr.strip()}"
            )
        try:
            yield
        except BaseException as error:
            # An owned failed build is not ready, active or reusable. Do not replace it,
            # steal lifetime leases, or remove dependencies retained for recovery.
            failure = """import json,os,pathlib,sys
p=pathlib.Path(sys.argv[1])
owner=p.parent.parent/('.environment-build-'+p.name+'.lock')/'publication.json'
if any(v.is_symlink() for v in (p,*p.parents,owner)):
 raise SystemExit('Unsafe publication failure record')
record=json.loads(owner.read_text())
if record['token']!=sys.argv[2] or (p/'.lambdaforge-environment.json').exists():
 raise SystemExit('Publication owner changed or prefix already sealed; preserved')
record.update(state='failed',error=sys.argv[3])
receipt=p/'.lambdaforge-publication.json'
tmp=receipt.with_suffix('.'+sys.argv[2]+'.tmp')
with tmp.open('x') as f:
 json.dump(record,f); f.flush(); os.fsync(f.fileno())
os.replace(tmp,receipt)
"""
            # Keep the original failure authoritative even if recording is unavailable.
            try:
                recorded = transport.run(
                    (
                        python,
                        "-c",
                        failure,
                        str(target),
                        token,
                        SecretRedactor.redact(f"{type(error).__name__}: {error}")[:4000],
                    )
                )
                if recorded.returncode:
                    logging.getLogger(__name__).warning(
                        "Could not record failed environment publication: %s",
                        SecretRedactor.redact(recorded.stderr)[:2000],
                    )
            except Exception as recording_error:
                logging.getLogger(__name__).warning(
                    "Could not record failed environment publication: %s",
                    SecretRedactor.redact(recording_error)[:2000],
                )
            raise
        finally:
            release = """import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
owner=p.parent.parent/('.environment-build-'+p.name+'.lock')/'publication.json'
if not owner.is_symlink() and owner.is_file():
 if json.loads(owner.read_text()).get('token')==sys.argv[2]: owner.unlink()
"""
            try:
                transport.run((python, "-c", release, str(target), token))
            except Exception as release_error:
                logging.getLogger(__name__).warning(
                    "Could not release managed publication reservation: %s",
                    SecretRedactor.redact(release_error)[:2000],
                )

    @staticmethod
    def seal(
        transport: Transport, python: str, target: PurePosixPath, payload: Mapping[str, Any]
    ) -> None:
        """Publish readiness last, after all probes ran at this exact lifetime prefix."""
        code = """import json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]); payload=json.loads(sys.argv[2]); contract=sys.argv[3]
marker=p/'.lambdaforge-environment.json'
owner=p.parent.parent/('.environment-build-'+p.name+'.lock')/'publication.json'
if any(v.is_symlink() for v in (p,*p.parents,owner,marker)):
 raise SystemExit('Unsafe managed publication path')
record=json.loads(owner.read_text())
if (record.get('state')!='building' or record.get('contract')!=contract
    or record.get('prefix')!=str(p)):
 raise SystemExit('Managed publication ownership/contract mismatch')
if marker.exists(): raise SystemExit('Refusing to replace an existing readiness receipt')
payload.update(publication_contract=contract,installation_prefix=str(p))
tmp=marker.with_suffix('.'+record['token']+'.tmp')
with tmp.open('x') as f:
 json.dump(payload,f,sort_keys=True); f.flush(); os.fsync(f.fileno())
os.replace(tmp,marker)
fd=os.open(p,os.O_RDONLY)
try: os.fsync(fd)
finally: os.close(fd)
"""
        result = transport.run(
            (python, "-c", code, str(target), json.dumps(payload), PUBLICATION_CONTRACT)
        )
        if result.returncode:
            raise RuntimeError(f"Managed environment publication failed: {result.stderr.strip()}")
