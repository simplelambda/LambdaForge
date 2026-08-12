[English](CLUSTERS.md) | [Español](CLUSTERS.es.md)

# Secure clusters and scheduler customization

## Contents

1. [Mental model](#1-mental-model)
2. [Catalog scopes and precedence](#2-catalog-scopes-and-precedence)
3. [SSH authentication](#3-ssh-authentication)
4. [Per-cluster SLURM dialect](#4-per-cluster-slurm-dialect)
5. [Register, inspect and diagnose](#5-register-inspect-and-diagnose)
6. [Submit and reconnect](#6-submit-and-reconnect)
7. [Security and intentional limits](#7-security-and-intentional-limits)

## 1. Mental model

LambdaForge remains a local control plane: it materializes the same experiment/task used locally,
builds a bounded content-addressed bundle, selects a named `ClusterProfile`, reaches the cluster
through a `Transport`, and asks its `Scheduler` to run the ordinary
`python -m lambdaforge run ...` command. Cluster access/resources are execution identity; passwords
and physical paths never become scientific identity.

A profile owns four independent choices: connection, remote workspace, Python environment and
scheduler dialect. OpenSSH plus standard SLURM remains the zero-customization default.

## 2. Catalog scopes and precedence

Profiles are merged in this documented order (last wins per profile name):

1. user: `$XDG_CONFIG_HOME/lambdaforge/clusters.yaml`, normally
   `~/.config/lambdaforge/clusters.yaml`;
2. project: the nearest `PROJECT/lambdaforge.clusters.yaml` (an existing catalog, otherwise the
   nearest `pyproject.toml` root);
3. explicit: `--clusters-file PATH`/`--clusters PATH`, or `LAMBDAFORGE_CLUSTERS` when no flag is
   supplied.

New remote profiles are user-scoped by default, which avoids accidentally committing personal
hosts. Choose `clusters add ... --scope project` for a reviewed, shareable descriptor. An explicit
file has highest precedence but does not hide unrelated user/project profiles. Inspect the winning
source and every overridden source with:

```bash
lambdaforge clusters list
lambdaforge clusters inspect atlas
lambdaforge clusters --clusters-file ./team-clusters.yaml inspect atlas
lambdaforge clusters export atlas --output portable-clusters.yaml
```

Export includes a credential *reference* when present, never its value. Review project catalogs
before committing them; the profile may contain host names, usernames, accounts and trusted job
script lines even though it contains no password.

## 3. SSH authentication

### 3.1 OpenSSH (recommended/default)

```yaml
clusters:
  atlas:
    transport: ssh
    host: atlas-login                  # alias from ~/.ssh/config is supported
    user: research                     # optional; alias may already provide it
    port: 22
    auth: {mode: openssh}
    ssh_options: [-o, BatchMode=yes]   # argv items, not a shell string
    scheduler: slurm
    workspace: /scratch/research/lambdaforge
```

`SshTransport` invokes `ssh`/`scp` as argv. Native SSH config continues to own keys, agent,
`known_hosts`, certificates and ProxyJump. LambdaForge never disables host-key verification. Prefer
this route because it reuses the mature access policy already used by `ssh atlas-login`.

### 3.2 Password mode (optional)

Install the optional audited providers and choose one secret source:

```bash
python -m pip install 'lambdaforge[cluster-password]'

# Prompt on every new CLI process; nothing is stored.
lambdaforge clusters add legacy --host login.example.org --user me \
  --workspace /scratch/me/lambdaforge --auth password

# Prompt once and store in the OS keyring; there is deliberately no --password flag.
lambdaforge clusters add legacy --host login.example.org --user me \
  --workspace /scratch/me/lambdaforge --auth password --store-password

lambdaforge clusters credentials set legacy
lambdaforge clusters credentials delete legacy
```

Password profiles use Paramiko SFTP/SSH with `RejectPolicy`, system `known_hosts`, bounded connect/
auth/command timeouts, and no key or agent fallback. Set `known_hosts: /reviewed/file` and
`ssh_timeout: 20` when required. Paramiko connects to a concrete host; OpenSSH aliases/ProxyJump are
an OpenSSH feature, so keep the default transport for those cases.

The only persisted authentication forms are:

```yaml
auth: {mode: password}                                  # hidden interactive prompt
auth: {mode: password, credential: keyring:cluster/legacy/me@login.example.org}
auth: {mode: password, credential: env:LEGACY_SSH_PASSWORD}
```

`keyring:` uses the OS credential store through the optional `keyring` package. If it is missing or
has no usable backend, LambdaForge explains how to install it; use interactive or `env:` as an
explicit fallback. `env:` reads the named variable at connection time and never persists its value.
Environment variables are convenient for an ephemeral CI secret but can leak through the process/
CI environment, so prefer OpenSSH or a system keyring on workstations.

## 4. Per-cluster SLURM dialect

`ResourceRequest` stays portable. One `SlurmResourceMapping` translates it exactly once for the
selected cluster; static directives cannot duplicate translated resource options. Defaults are
`--ntasks`, `--cpus-per-task`, `--mem=...M`, `--gpus` and `--time`.

```yaml
clusters:
  atlas:
    transport: ssh
    host: atlas-login
    scheduler: slurm
    workspace: /scratch/me/lambdaforge
    resource_mapping:
      processes: {option: ntasks, value: "{processes}"}
      cpu: {option: cpus-per-task, value: "{cpu_per_process}"}
      memory: {option: mem, value: "{memory_gib}G"}
      gpu: {option: gres, value: "gpu:a100:{gpus}"}  # --gres=gpu:a100:N
      time: {option: time, value: "{minutes}"}
    scheduler_directives:
      partition: accelerated
      account: project123
      exclusive: true                  # --exclusive
      constraint: [nvlink, ssd]        # repeated directives
    scheduler_commands:
      submit:
        command: site-sbatch
        args: [--parsable, "{script}"]
        job_id_pattern: "^(\\d+)(?:;.*)?$"
      queue: {command: site-squeue, args: [-h, -j, "{job_id}", -o, "%T"]}
      accounting: {command: site-sacct, args: [-n, -X, -j, "{job_id}", -o, State]}
      cancel: {command: site-scancel, args: ["{job_id}"]}
    job_script:
      shell: /bin/bash
      prologue: [module load cuda/13.0]
      epilogue: [echo job-finished]
```

GPU variants are expressed as mappings: standard `option: gpus`, generic GRES
`{option: gres, value: "gpu:{gpus}"}`, or typed GRES as above. Set a rule to `omit` only when a site
wrapper supplies the resource; dry-run and doctor then produce a strong warning because SLURM will
not enforce the portable request itself.

Templates allow only documented numeric fields (`cpu_cores`, `cpu_per_process`, `processes`,
`memory_bytes/mib/gib`, `gpus`, `seconds/minutes/hours`). Scheduler commands allow only `{job_id}`,
`{script}` and `{work_dir}` where relevant. They are rendered into argv and never passed to local
`shell=True`/`eval`. `job_id_pattern` must contain a capture group. Prologue/epilogue are intentionally
trusted profile shell lines: they receive no secret interpolation and should be reviewed like code.
Legacy `scheduler_options` remains compatible and is treated as static directives.

## 5. Register, inspect and diagnose

```bash
lambdaforge clusters add atlas --host atlas-login --workspace /scratch/me/lambdaforge \
  --scheduler slurm --environment managed
lambdaforge clusters inspect atlas
lambdaforge doctor --on atlas
lambdaforge clusters bootstrap atlas
```

`doctor` is read-only. It checks transport/authentication, workspace, selected Python, exact
LambdaForge/consumer import, PyTorch/CUDA view, every configured scheduler executable, resource
mapping and a configured partition. It never submits a real job. `managed` builds exact local
LambdaForge/consumer wheels and creates a user venv below
`WORKSPACE/.lambdaforge/environments`; `existing` installs nothing. Offline bootstrap requires an
explicit target-compatible wheelhouse. No mode installs drivers, system CUDA or cuDNN.

## 6. Submit and reconnect

```bash
lambdaforge run experiment.yaml --on atlas --dry-run
lambdaforge run experiment.yaml --on atlas --cpus 8 --memory 32GiB \
  --resource-gpus 1 --time 4h
lambdaforge status --on atlas --state running
lambdaforge logs JOB --follow
lambdaforge cancel JOB
lambdaforge retry JOB --dry-run
```

Dry-run returns the cluster, portable resources, generated script path, effective directives,
warnings and exact scheduler submit argv without contacting the scheduler. Real jobs persist local
and scheduler IDs, bundle/environment/scientific/execution identities and remote paths, so another
process can reconnect. Small result evidence can be synchronized explicitly; heavy artifacts need
`artifact fetch`.

## 7. Security and intentional limits

No password is accepted on argv, serialized, logged, bundled, fingerprinted or placed in YAML.
Known in-memory secrets are redacted from transport errors; plugin transports/providers must apply
the same rule. Passwords exist only long enough to authenticate and are then left to the encrypted
SSH session/provider. Scheduler customizations are trusted administrator/user configuration, not
untrusted experiment input.

LambdaForge still does not provide automatic cluster placement, a resident daemon, distributed
workflow recovery, implicit large-data/checkpoint transfer, an SSH bastion implementation separate
from OpenSSH, driver installation or platform wheel synthesis.
