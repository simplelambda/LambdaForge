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
- Tracking and S3-compatible providers expand the trust boundary to their SDK, credentials, network
  and service. They are optional and loaded only when configured.

The detailed trust boundaries and data flow are documented in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
