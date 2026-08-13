[English](SECURITY.md) | [Español](SECURITY.es.md)

# Cluster credential and scheduler threat model

This guide complements the repository [security policy](../SECURITY.md).

## Boundaries

`ClusterProfile` is durable, shareable execution configuration. It may contain host/user/account,
paths, scheduler policy and a `keyring:`/`env:` identifier; it must never contain the secret value.
`CredentialProvider` is the only secret source. `Transport` receives a password only in process
memory for authentication. Jobs, bundles, scientific/execution fingerprints and exported profiles
receive no value. `SecretRedactor` removes a known in-memory secret from provider-facing errors;
third-party providers must uphold that boundary too.

OpenSSH is preferred because its mature configuration owns keys, agent, host certificates,
`known_hosts` and ProxyJump. Password mode deliberately uses Paramiko rather than building an SSH
protocol: unknown hosts are rejected, keys/agent are disabled for that mode, timeouts are bounded,
and transfers use SFTP. A wrong password and a wrong host key are both failures, never prompts to
weaken verification.

The OS keyring is an external trusted provider. `env:` is a deliberate ephemeral integration and
may be exposed by CI/process inspection outside LambdaForge. Interactive input uses hidden
`getpass`. There is no CLI password option or custom encrypted file; custom crypto would create key
management without improving the system boundary.

Managed CUDA auto-selection trusts the remote `nvidia-smi` binary, configured Python/pip and TLS
connection to the official PyTorch index. It validates driver family, compute capability and remote
wheel availability, records the exact plan in environment identity and fails closed. It never
installs privileged drivers, a system toolkit or NVIDIA forward-compatibility packages. Offline
clusters replace network trust with a reviewed, target-compatible wheelhouse whose bytes are hashed.

Scheduler resource and command templates use fixed placeholders and argv. Static directive names
and rendered values reject newlines. Generated batch scripts necessarily execute on a shell;
`job_script.prologue` and `epilogue` are therefore trusted cluster-profile code and receive no
credential interpolation. Store personal credentials in user scope and review project-scoped
profiles before commit.

## Operational checks

Use `clusters inspect` to confirm source/auth status, `clusters export` to inspect a shareable copy,
`doctor --on` to verify authentication/host key/workspace/commands/mapping/partition without a job,
and `run --dry-run` to review script/directives/submit argv. Never paste a real password into YAML,
an issue, a command line, a scheduler directive or a prologue.
