# Security policy

[Español](SECURITY.es.md) · English

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

- A YAML `run` import names trusted consumer Python and may execute arbitrary code. LambdaForge is
  not a sandbox; configuration must come from the researcher or another trusted source. YAML has no
  expression evaluation, recursive construction, arbitrary function target or secret interpolation.
- Inputs, Work outputs, cache/checkpoint entries and cleanup operations validate path
  containment and symbolic-link boundaries at their owning layer. Managed dataset deletion also
  revalidates the exact immutable manifest identity inside the configured dataset root immediately
  before removing bytes; stale, conflicting or unreachable index state fails closed.
- A cluster `project_root` is researcher-owned persistent storage, outside LambdaForge cleanup.
  Large typed inputs map there only by project-relative path and must match local kind, byte count
  and SHA-256 both before submission and again in the worker. Relative remote output publication
  requires this explicit mirror and cannot escape it; an absolute output path is a deliberate
  trusted-code escape. Checksums provide consistency, not producer authentication or filesystem
  isolation against a concurrently malicious account.
- Work-owned runtime services retain that boundary: `self.outputs.artifact` accepts only verified
  paths and owns an external artifact by safe copy; `self.outputs.dataset` accepts only run-owned
  local assets or explicit URIs, rejects
  traversal/symlinks, hashes assets, verifies staging and atomically publishes before registration.
  Ordinary YAML strings are never guessed to be paths; only explicit `file`/`dataset` markers
  authorize resolution and staging.
- Managed cache/checkpoint files reject absolute/traversing keys and symbolic links, publish only
  after validation, `fsync`, SHA-256 and atomic replacement, and expose a read-only path-like
  handle. `Work.map` persists their logical key/content evidence rather than controller paths.
  Work cache GC takes an exclusive lock while active Work executions retain a shared lease.
- `Work.tools.run` accepts argv only and never invokes a shell. Thread controls and environment
  overrides are scoped to the child process; captured output may contain project data and therefore
  remains subject to the same log-sharing caution as exception diagnostics.
- `lf delete WORK` is preview-first, refuses active attempts and removes only an exact tracked job-ID
  child of the configured job root. Published datasets, shared caches/environments and other Work
  are outside this operation; dataset deletion remains a separate manifest-checked command. A
  minimal completion receipt contains no scientific outputs and makes a repeated deletion safe.
- Interactive history deletion requires a second explicit key, never removes active Jobs and runs
  the same exact-root storage operation as the CLI. Per-Job absolute local roots prevent a later
  invocation from deleting below a different current directory. Whole-history cleanup retains the
  local record whenever its owned workspace could not be removed safely.
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
- Project-native environments are data-only declarations: paths stay inside the project, YAML
  variables/prefixes/nested pip sections and install hooks are rejected, and package names are
  passed as argv without a shell. Online channels and Conda packages remain trusted supply-chain
  inputs and can contain package-manager metadata/scripts; review and pin them accordingly. Offline
  locks reject credentials/queries, require one supported platform plus SHA-256 for every URL, and
  verify matching regular cache bytes before transfer. Temporary prefixes and package caches are
  content-addressed/locked; required executable ownership and version are verified before atomic
  publication. LambdaForge never treats `tools.require()` as authority to install software.
- A LambdaForge-managed runtime reuses only a readable CA bundle already selected and locally
  validated through the host Python trust configuration. Its path is propagated to provisioning,
  pip/Requests and scientific jobs. LambdaForge never disables TLS verification, downloads
  arbitrary trust roots, writes `/etc` or modifies the system trust store.
- Tracking and S3-compatible providers expand the trust boundary to their SDK, credentials, network
  and service. They are optional and loaded only when configured.

The detailed ownership and control-plane boundaries are documented in the
[canonical manual](docs/MANUAL.md#11-cleanup-and-safety).
