# Security policy

## Supported versions

LambdaForge is pre-1.0. Security fixes are applied to the current development branch and the most
recent published minor release when a compatible backport is practical. Older minors receive no
guarantee.

## Reporting a vulnerability

Do not disclose a suspected vulnerability in a public issue. Use GitHub's private vulnerability
reporting/security-advisory channel for this repository when available, or contact the repository
owner privately through a verified channel listed on the SimpleLambda GitHub profile. Include the
affected version/commit, platform, minimal reproduction, impact and whether untrusted input is
required. Do not include real credentials or private datasets.

## Security model

- YAML `target`, `ref`, plugins, pickle payloads and consumer Python are trusted code and may execute
  arbitrary Python. LambdaForge is not a sandbox.
- Configuration interpolation supports references and environment lookup, never expression
  evaluation. Secrets are redacted from ordinary materialization; callers must explicitly request
  their value. Persisted workflow structure rejects secrets.
- Inputs, task outputs, store keys, cache entries, archives and retention operations validate path
  containment and symbolic-link boundaries at their owning layer.
- Checksums detect accidental or malicious modification but do not authenticate a producer. Use
  HMAC where supported, restrict store permissions and obtain artifacts over authenticated channels.
- Local and SLURM backends never interpolate a command through a local shell. Generated batch
  scripts quote arguments and submission/cancellation remains explicit.
- Cluster YAML stores only authentication mode and optional `keyring:`/`env:` reference. Password
  values are never CLI arguments, serialized state, bundles, fingerprints or logs. OpenSSH remains
  preferred; optional Paramiko password transport rejects unknown host keys and uses bounded
  timeouts. Environment-backed secrets inherit the exposure risks of the calling process/CI.
- CLI failures write a local diagnostic record containing the full traceback and sanitized command.
  Before terminal, JSON or file output, LambdaForge redacts explicit secret fields, common
  password/token/API-key assignments, bearer headers, credential URLs and private-key blocks.
  Records are stored below the user state directory with owner-only permissions where supported;
  users must still review records before sharing them because project exceptions can contain
  arbitrary scientific data that no generic redactor can recognize.
- Scheduler command/resource placeholders are allowlisted and rendered to argv. Profile
  prologue/epilogue lines are trusted shell code and must not interpolate secrets or accept
  unreviewed experiment values.
- Managed Python provisioning is unprivileged and confined to the configured cache root. The pinned
  micromamba fallback is downloaded over HTTPS on the controller and SHA-256 verified before and
  after transfer; bootstrap never edits shell profiles, system Python, drivers or system CUDA.
- A LambdaForge-managed runtime reuses only a readable CA bundle already selected and locally
  validated through the host Python trust configuration. Its path is propagated to provisioning,
  pip/Requests and scientific jobs. LambdaForge never disables TLS verification, downloads
  arbitrary trust roots, writes `/etc` or modifies the system trust store.
- Tracking and S3-compatible providers expand the trust boundary to their SDK, credentials, network
  and service. They are optional and loaded only when configured.

The detailed trust boundaries and data flow are documented in the
[canonical manual](docs/MANUAL.md#27-security-model).
