"""Host certificate-trust discovery for LambdaForge-managed Python runtimes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lambdaforge.controlplane.Transport import Transport

if TYPE_CHECKING:
    from lambdaforge.controlplane.ClusterProfile import ClusterProfile


@dataclass(frozen=True, slots=True)
class TlsTrust:
    """Describe a validated host CA bundle without copying or modifying trust roots."""

    ca_file: str
    source: str = "host-python-default"

    def __post_init__(self) -> None:
        if not self.ca_file.startswith("/") or "\n" in self.ca_file:
            raise ValueError("A TLS CA bundle must be one absolute remote path.")

    def assignments(self) -> tuple[str, ...]:
        """Return variables understood by Python, Requests, pip and libcurl clients."""
        return (
            f"SSL_CERT_FILE={self.ca_file}",
            f"REQUESTS_CA_BUNDLE={self.ca_file}",
            f"PIP_CERT={self.ca_file}",
            f"CURL_CA_BUNDLE={self.ca_file}",
        )

    def prefix(self) -> tuple[str, ...]:
        """Return an argv-safe ``env`` prefix for one managed subprocess."""
        return ("env", *self.assignments())

    def to_dict(self) -> dict[str, str]:
        """Return identity-safe runtime metadata; certificate contents are never serialized."""
        return {"ca_file": self.ca_file, "source": self.source}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> TlsTrust | None:
        """Restore optional runtime trust metadata."""
        if not value:
            return None
        return cls(str(value["ca_file"]), str(value.get("source", "host-python-default")))


class TlsTrustResolver:
    """Select a host CA file already accepted by system Python using a local-only probe."""

    _PROBE = (
        "import json,os,ssl;"
        "p=ssl.get_default_verify_paths();"
        "values=[p.cafile,p.openssl_cafile," 
        "'/etc/ssl/certs/ca-certificates.crt','/etc/pki/tls/certs/ca-bundle.crt',"
        "'/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem','/etc/ssl/cert.pem'];"
        "valid=[];"
        "\nfor value in values:\n"
        " if value and value not in valid and os.path.isfile(value) and os.access(value,os.R_OK):\n"
        "  try:\n"
        "   if ssl.create_default_context(cafile=value).get_ca_certs(): valid.append(value)\n"
        "  except (OSError,ssl.SSLError): pass\n"
        "print(json.dumps({'ca_file':valid[0] if valid else None,'candidates':valid}))"
    )

    def resolve(
        self,
        profile: ClusterProfile,
        transport: Transport,
        *,
        host_python: str | None = None,
    ) -> TlsTrust:
        """Resolve and validate system trust without network access or host mutation."""
        result = transport.run(
            (
                *profile.command_prefix,
                host_python or profile.runtime_policy.executable,
                "-c",
                self._PROBE,
            )
        )
        if result.returncode:
            raise RuntimeError(
                "Could not inspect the host Python certificate trust configuration: "
                f"{result.stderr.strip()}"
            )
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
            ca_file = payload.get("ca_file") if isinstance(payload, Mapping) else None
        except (json.JSONDecodeError, IndexError):
            ca_file = None
        if not isinstance(ca_file, str) or not ca_file.startswith("/"):
            raise RuntimeError(
                "The host Python has no readable PEM CA bundle that LambdaForge can safely "
                "reuse. Configure a working system Python trust store before provisioning a "
                "managed runtime. Certificate verification was not disabled."
            )
        return TlsTrust(ca_file)

    @staticmethod
    def probe_command(python: str, trust: TlsTrust | None = None) -> tuple[str, ...]:
        """Build a deterministic local trust-store validation command."""
        code = (
            "import json,ssl;"
            "p=ssl.get_default_verify_paths();"
            "c=ssl.create_default_context();"
            "n=len(c.get_ca_certs());"
            "print(json.dumps({'ca_count':n,'cafile':p.cafile,'capath':p.capath}));"
            "raise SystemExit(0 if n else 2)"
        )
        return (*(() if trust is None else trust.prefix()), python, "-c", code)
