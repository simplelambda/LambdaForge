# Operational health, monitoring and deliberate recovery

[Español](OPERATIONAL_RELIABILITY.es.md) · English

Logical execution state, scientific evidence and operational health are different facts.
An old `running` record is not a current process observation. `UNKNOWN` means ownership or
observation is unresolved, not that science failed or that starting a duplicate is safe.
These capabilities use the existing project JobStore, providers and Work runner.
The [acceptance ledger](OPERATIONAL_RELIABILITY_PROGRESS.md) lists the broader reform still pending.

## Inspect without starting computation

```sh
lf health --json
lf health JOB --refresh --json
lf incidents --json
lf jobs show JOB --json
lf jobs logs JOB --tail 100
lf monitor diagnose --json
```

`health` without `--refresh` reads saved observations only. `--refresh` queries the provider;
the observer's attempt/success times remain separate from the host's heartbeat clock and from
scientific progress. A failed query preserves the last confirmed state/time and increments its
failure count. Reading the cache or rewriting a local record does not advance confirmation.
Health commands return exit code 2 when observations are unknown, stale, degraded, unreachable
or confirm lost processes; that exit code is not a scientific outcome.

Health may be `healthy`, `degraded`, `unreachable`, `stale`, `lost_confirmed` or `unknown`.
Provider errors do not become a successful empty inventory. A missing inventory entry stays
unknown until exact ownership can be verified. Direct observation checks process lifetimes and
same-user inherited Job markers, including reparented children. Inaccessible ownership evidence
fails closed. Observation itself never kills processes or edits host `state.json`.

A corrupt or unreadable individual Job record is isolated within direct-host inventory: healthy
siblings still receive fresh state/health observations. Only that known Job becomes degraded
(a saved terminal outcome remains terminal). Error rows never create new Jobs or certify loss or
recovery eligibility. Root/transport failures still invalidate the host query. A later valid
observation clears the prior query error; no manual history editing or scientific restart is needed.
Shipped read probes select definitions and helper dependencies by name from one source snapshot,
not loaded function line offsets that can become stale after an editable source update. Restart
an already-open Console/monitor after upgrading LF to load the corrected probe implementation;
this does not restart scientific Jobs or require modifying their immutable worker environments.

SLURM retains queue/accounting source, raw state, reason, exit code, node and available times.
An accounting delay is unknown; a query error remains an error unless another query supplies
positive evidence. See the official [sacct reference](https://slurm.schedmd.com/sacct.html) and
[state codes](https://slurm.schedmd.com/job_state_codes.html). `NODE_FAIL` is infrastructure evidence,
`OUT_OF_MEMORY` is resource evidence, not an objective score. Polling remains bounded.
If `squeue` times out, a bounded accounting query may still supply positive evidence. Empty or
failed accounting never legitimizes a cached RUNNING response; previous query details are cleared.

Each health read includes a native `diagnostic` with category, source, confirmation, event time,
impact and recommendation. Scheduler OOM does not by itself identify CUDA OOM; exit 137 does not
prove host OOM. Generic CUDA kernel/dtype errors remain consumer failures, not OOM retry grants.
Malformed data is degraded observation, not proof of an offline host.
An unclassified query exception likewise keeps connectivity unknown even after repeated failures;
typed connection/authentication errors remain distinct in the shared diagnostic authority.
Legacy terminal files without ownership requests remain readable but do not certify ownership
or safe recovery.

## Independent project monitor

```sh
lf monitor config --file monitor.json
lf monitor start
lf monitor status --json
lf monitor once --json
lf monitor stop
```

`start` launches a detached observer of the native project catalog, not a scientific runner.
One cross-process owner lock prevents duplicate monitors. `stop` requests only that exact monitor
to exit; it never cancels Works. Status checks process lifetime and heartbeat separately and
reports repeated observation failures as degraded. Restart preserves observations, incident IDs
and notification delivery records. The monitor does not read epochs/artifact bytes, fit HPO models,
allocate GPUs or change scientific YAML. Config/state/logs are project-owned below the JobStore's
`.operational/monitor/`; incidents are compact records below `.operational/incidents/`.

Example: [monitor.json](../examples/monitor.json). Operational deadlines are seconds:

```json
{
  "interval_seconds": 60,
  "observation": {
    "freshness_seconds": 180,
    "heartbeat_seconds": 45,
    "grace_seconds": 120,
    "progress_seconds": null,
    "unreachable_failures": 2
  },
  "smtp": {"enabled": false}
}
```

Progress expectations default to disabled. Some valid Works emit no metrics for hours; silence
alone must not prove death. Heartbeat grace, provider freshness and scientific-progress expectations
are independent configurable checks. Failed observations immediately show connectivity loss;
repeated failures escalate health without repeated Console toasts.

Closing Textual does not stop the detached monitor. Reboot/logout persistence is a separate OS
service concern. Linux provides a preview-first systemd-user installer:

```sh
lf monitor install --json
lf monitor install --apply
```

The installer refuses unrelated units, enables only the project observer and does not enable user
lingering or change administrative policy. Inspect the unit and `loginctl show-user "$USER"`.
systemd's [user manager lifecycle](https://github.com/systemd/systemd/blob/main/man/org.freedesktop.login1.xml)
depends on session/lingering configuration; a detached process alone does not guarantee survival
of a site's logout cleanup. On macOS/Windows, use `lf monitor run` with launchd/Task Scheduler or
another managed service. These platform integrations and real-site logout/reboot have not been
certified by the local CPU regression tests.

## Optional email, durable incidents and no polling spam

SMTP is off by default. Configure a host, sender, recipient list and TLS mode `starttls` or `ssl`.
Credentials must be `env:NAME` or `keyring:KEY` references; never store a password in configuration.
Authenticated monitoring is noninteractive: the referenced credential must be available to the
service process. A site may use an authenticated relay or an explicitly configured unauthenticated
TLS relay. Verify access and policy with the operator.

```json
{
  "interval_seconds": 60,
  "smtp": {
    "enabled": true,
    "host": "smtp.example.org",
    "port": 587,
    "tls": "starttls",
    "sender": "research@example.org",
    "recipients": ["research@example.org"],
    "username": "research@example.org",
    "credential": "env:LF_SMTP_PASSWORD",
    "cooldown_seconds": 900
  }
}
```

```sh
lf monitor test-alert --json
lf incidents --json
```

A delivery is marked sent only after SMTP accepts it. Transport failure never changes a Job.
Automatic attempts are bounded at three per incident transition, with a persisted cooldown.
Stable incident IDs deduplicate repeated probes; escalation, reopening and resolution remain
auditable in native Job events. Resolution requires positive healthy evidence, not disappearance.
For an audited retry, origin incidents resolve only after fresh scientific progress in the exact
accepted recovery Job, after its request timestamp. Enqueue, preparation, scheduler acknowledgement
and heartbeat alone do not resolve them. The origin outcome stays unchanged; `recovered_by` links
the new Job/request and the observed progress. A failed recovery cannot resolve the origin.
Loading reused Runs establishes a baseline, not fresh progress. Changes to obligation totals,
counter resets and duplicate polls likewise do not resolve an incident; a shared completion/step/
epoch counter must advance beyond its persisted/native or positively observed runtime baseline.
Native previews retain the number of successful logical Runs, so a first observation may certify
a genuinely new completion beyond that count; rereading the same count never refreshes progression.
SMTP runs outside the incident lock; a slow relay cannot block provider commits. If escalation or
resolution occurs during delivery, the earlier acknowledgement never marks that newer transition sent.
SMTP acceptance is not an inbox receipt. Exactly-once delivery cannot be guaranteed across an
interrupted send; stable Message-ID supports receiver deduplication. No blanket mail-all policy
or implicit external messaging is enabled.

## Research Console

Bare `lf` → **Operational health…** (or Ctrl+P → Operational Health System) opens saved health,
incident details and monitor settings. Refresh providers is explicit; Start/Stop affect the
independent observer only. Save settings uses the same validator as `lf monitor config`.
Send test email exposes acceptance or failure inline. While an operation is pending, duplicate
activation is disabled; closing a panel does not cancel scientific work.

Monitor and Console share nonblocking project/provider observation gates. While a same-cluster probe
is in flight, readers retain native last-good records and their original clocks; receipt files are
coalescing metadata, never proof of provider freshness. Separate SLURM Jobs retain separate lanes.
An abruptly terminated observer releases its OS lock; it cannot leave a permanent probe barrier.

Overview/Work/Studies display health separately from scientific state and last confirmation.
Imported snapshots remain historical read-only evidence, with no expectation of a live host.
Do not interpret a previously completed scientific result as invalid merely because its host
cannot now be contacted.

## Recovery and relocated YAML

```sh
lf retry STUDY --dry-run --json
lf retry STUDY --config experiments/moved.yaml --dry-run --json
lf retry STUDY --config experiments/moved.yaml
# Only after explicitly reviewing whether earlier evidence/checkpoints remain valid:
lf retry STUDY --config experiments/moved.yaml --accept-code-change
```

The Console's **Resume Study…** offers the historical path, a YAML picker and Validate recovery
plan. Missing historical YAML is not a reason to replay the full experiment. A selected YAML is
compared against original persisted scientific identity after restoring the original seed namespace.
Formatting/comments/key order do not define scientific identity. Parameters, input bytes, dataset
identity, seeds, objectives, design and publication contracts must still match. Relative inputs
are resolved from the selected YAML: move it only with correct exact bindings. Code-change
acknowledgement is not an identity-check waiver.

The selected file hash is pinned from preview to preparation; editing it invalidates the plan.
Preview creates no Job, consumes no seed/budget, reserves no GPUs and downloads no checkpoint bytes.
CLI `--dry-run` also leaves the saved operational audit unchanged. Explicit Console **Validate
recovery plan** records an operational inspection request/outcome, but never changes scientific
evidence or launches computation.
Apply rechecks provider ownership under the native recovery lock; unreachable/ambiguous live
ownership blocks duplication. A fresh, positively verified lost process can be recovered without
falsely rewriting its scientific outcome to `failed`.

Accepted recovery receives a `recovery_request_id`, linked from the origin's append-only events
and new Job metadata/events. `lf jobs show NEW_JOB --json` exposes `recovery.phase`: accepted,
preparing, scheduler acknowledged, runtime observed, scientific progress observed or terminal
outcome. Admission is not progress; epoch zero is not training progress. Preparation failures retain
their phase/category and controller logs. The Console shows the request ID and observed phase.

### Saved recovery history, including rejected requests

```sh
lf jobs recovery JOB --json
lf jobs recovery JOB --offset 100 --limit 100 --json
```

This read-only view uses native saved Jobs and append-only events, without contacting the execution
host or reading checkpoint/artifact bytes. It remains available during an outage. It links exact
retry/dependency/scientific identities, never display names, and shows missing dependencies rather
than fabricating history. Events are paged newest-first; `next_offset` selects older events.

Study workspace → **Recovery history** opens the same paged view, with linked Jobs, request IDs,
phases and event details. Apply and explicit Console inspection allocate a request ID before
validation: rejected ownership, missing state or incompatible configuration remains auditable even
when no new Job exists. Rejection leaves scientific records and completed Runs untouched.
An audit-write failure is reported rather than falsely claiming the request was durably recorded.

On recovery failure the Console keeps a persistent dialog with the phase, request ID, bounded
diagnostic and Logs / Saved history / Back to plan controls. It does not repeatedly notify on polls,
and closing the dialog is not another retry. A successful preview is not admission; only apply may
launch preparation, and it revalidates the exact plan under the existing ownership lock.

Fixed/repeated designs use owned design/Run/Attempt records; adaptive Studies require their genuine
controller state. Valid Runs remain valid, failed/interrupted Runs create new Attempts, spent
budgets remain spent, and original Execution/Attempts are never silently overwritten. Recovery of
a Study is not proof of epoch continuation: the Work must implement compatible checkpoint loading.
Managed checkpoint helpers, explicit retention pins and exact recovery transport remain the native
checkpoint authority. Native Fleet recovery remains pending and rejects single-host replay.

## Managed checkpoint policy and evidence

At an application-safe boundary, a Work may configure advisory cadence and terminal retention:

```python
self.checkpoints.configure(
    interval_steps=5,
    interval_seconds=1800,
    retention_seconds=7 * 86400,
    storage_durability="shared",
)
if self.checkpoints.should_save(step=epoch):
    self.checkpoints.save_json("state.json", algorithm_state, step=epoch)
```

No default write interval is imposed and arbitrary algorithms are never automatically serialized.
Either interval may make a save due. Only a committed write advances cadence, not reads/polling.
JSON saves preserve exact checksums. Named `checkpoints.file(..., build=..., step=...)` files also
register native evidence; version binary names or supply semantic validation when rebuilding,
because `file` still reuses valid existing content. Advanced `path()` writes remain unregistered.

New records include execution/Run/Attempt/scientific owner, write time and optional generic step.
`inventory()` pages metadata; `inventory(verify=True)` verifies bytes on the owning host, not
application compatibility. Corrupt registered JSON refuses loading; legacy unregistered JSON
remains readable without a fabricated certificate.
Registered Execution/Run bindings are checked before managed state is loaded or replaced; a foreign
owner or corrupt registration is rejected without changing payloads. Older Attempts of the same
logical Run remain usable. Code/scientific compatibility still requires native recovery validation
and explicit acknowledgement where applicable; this owner check does not waive it.
Recovery preview/Console show the last registered descriptor with **not checked** integrity,
without downloading payload or promising
epoch continuation. Historical `resumed_from_checkpoint` indicates availability at Attempt start,
not proof of application restoration; subsequent progress is separate.

Durability (`unspecified`, `ephemeral`, `persistent`, `shared`) is an operator/project declaration,
not a filesystem guarantee. Node-local bytes do not protect against node loss; verify the actual
storage policy. Configured terminal retention integrates with preview-first cleanup; failed/
interrupted evidence and pins remain protected. Increasing retention after preview is rechecked
before deletion. Live rotation, application-specific deserialization and cross-host/Fleet transfer
acceptance are not certified by this cadence/metadata API.

## Emergency procedure

1. Preserve the exact Job ID. Inspect `health --refresh --json`, Job details, bounded logs and
   incidents. Do not delete history, checkpoints or processes to make a stale row disappear.
2. Separate connectivity loss from proved process loss, scheduler failure and consumer failure.
   An inaccessible host may still be computing. Retry is not permission to duplicate it.
3. Restore connectivity, inspect/reconcile native evidence and preview deliberate recovery.
   Select moved YAML if necessary; inspect reuse/retry/pending Runs and code compatibility.
4. Confirm apply only when ownership and scientific identity are valid. Track the **new** Job's
   preparation, admission, heartbeat and subsequent scientific progress, retaining the origin ID.
5. Preserve diagnostic/incident/recovery events. Notify site support for infrastructure failures.
   Historical successes are reusable; arbitrary consumer exceptions are not automatically retried.

Local regression evidence is listed in the acceptance ledger. It does not certify a real SMTP
server, CUDA failure, SLURM site or systemd activation. The native local two-Study/Textual recovery
fixture passes with simulated provider/admission faults; real detached preparation and the full
multi-host chaos scenario remain outside that demonstrated acceptance.
