# Operational reliability implementation ledger

Baseline: `56c48957b448a6f631e606188687e4b57b6bf4d3`, LambdaForge 0.18.0.
This is an acceptance ledger, not a declaration that unfinished capabilities exist.
The supplied specification has 34 sections. Keep each requirement mapped to native authorities;
never add a second JobStore, runner or scientific coordinator.

| Sections | Native authority / demonstrated gap | Implementation and acceptance evidence | Status |
|---|---|---|---|
| 0 | JobService, JobStore, JobObservation, OverviewService, ResearchWork | Audit real HEAD and contracts; retain this ledger | Audited |
| 1–2 | get/list/reconcile swallowed provider errors or refreshed stored state files | OperationalHealth and locked ordered JobStore observations; failed/empty inventories; provider fault tests | Implemented core |
| 3 | Work/Study/Overview presentation used state alone | Shared health/freshness cells/details and OperationalWorkspace incident table; imported read-only distinction | Implemented core |
| 4 | Supervisor heartbeat and scientific progress are independent signals | Configurable grace/freshness/progress policy; hysteresis, clock-skew and no metric-free death inference tests | Implemented core |
| 5 | Detached per-Job supervisors existed, no independent catalog observer | MonitorService start/stop/status/config/diagnose; real local detach/restart/duplicate tests; Linux installation preview | Partial: OS logout/reboot and non-Linux service validation pending |
| 6 | CredentialService and diagnostics redaction exist | Durable deduplicated incidents, TLS SMTP with references, bounded retries/resolution events; Console settings/test | Partial: real SMTP/TLS server and full duration/escalation matrix pending |
| 7 | SLURM discarded errors/reasons | Structured queue/accounting evidence; NODE_FAIL/OOM/TIMEOUT/PREEMPTED/delay/query-error fixtures | Implemented core; real site pending |
| 8 | ProcessScheduler read state.json as liveness | Read-only host lifetime/marker observation; legacy immutable runtime bridge uses same implementation; live-child ambiguity refuses recovery | Implemented core |
| 9 | diagnostics/failure is the classification authority | Shared native operational diagnostics for scheduler/source facts, generic CUDA versus OOM and unknown causes; incidents carry recommendations | Implemented core; real site/complete source matrix pending |
| 10–12 | Native recovery preview/apply and lock already exist | Native request IDs before apply/explicit inspection, durable rejected requests, derived admission/runtime/progress phases and exact locked revalidation; CLI dry-run remains read-only | Implemented single-host core; complete operational reconciliation matrix pending |
| 13–15 | Retry recorded original source path, no public alternate YAML | --config CLI/API/Console picker; semantic origin/seed checks; preview/apply pin and preparation guard; moved-YAML reuse 9/retry 1 test | Implemented single-host core |
| 16–17 | Managed checkpoints and recovery preserve Runs | Opt-in configure/should_save and terminal retention; exact JSON/managed-file hashes, owner/step/write time, paged host verification, metadata-only recovery table | Partial: live rotation, application restoration acknowledgement and cross-host transfer pending |
| 18 | Fleet coordinator v2 fencing exists; public recovery gated | Native coordinated recovery and verified compatible placement; never replay as single-host Work | Pending |
| 19–20 | Native Job events existed, no incident workbench | OperationalWorkspace, Resume Study picker and paged offline Recovery history; persistent preview/config/apply failure dialogs with exact request/phase/log/history/plan controls; no toast polling | Implemented single-host core; full checkpoint/compatibility workbench pending |
| 21 | Existing jobs/overview/retry parser and domain services | Implemented health/monitor/incidents, jobs recovery audit and retry --config; shared CLI/Console domain services and JSON | Implemented core |
| 22 | Work Results 0.18.0 already implemented | Retained Work Results/import/export/product suites; internal checkpoint metadata excluded by the shared payload authority | Verified retained local core; complete compatibility/fault matrix pending |
| 23 | Last-good UI snapshots and bounded workers already exist | Nonblocking cross-process project/provider probe gates, separate SLURM lanes, receipt ordering and abrupt observer-death tests; cached clocks unchanged | Implemented probe core; complete UI race/platform matrix pending |
| 24 | runner owns scientific runtime | Incremental control-plane observability extraction, no fingerprint/runner changes for display | In progress |
| 25 | Job/Attempt/lease identity, locks, safety already exist | Corruption/late response/unknown-owner tests; never duplicate ambiguous computation | Pending verification |
| 26–27 | Existing real local process and fixed/adaptive retry tests | Two native CPU Studies through a faulty simulated provider and Textual YAML picker; confirmed loss, reuse 9/retry 1 each, durable incidents/audits and Console reopen | Partial: real provider preparation and full A–BW/Fleet chaos matrix pending |
| 28 | CI lint/types/tests/wheel exists | Explicit operational/fixed/adaptive reliability gate; packaged-guide and installed-wheel import/parser smoke | Partial: full chaos/platform acceptance pending |
| 29 | Manuals, READMEs, AGENTS and changelogs bilingual | Operational reliability guide, emergency steps, monitor example and public references; packaged EN/ES resources | Implemented for delivered core |
| 30 | Existing schemas and immutable evidence | Additive metadata, legacy reads, no imported active Jobs; retained suites | Pending verification |
| 31–33 | Acceptance and final report | Report actual tests/platform boundaries; do not claim complete before end-to-end acceptance | Pending |

## Demonstrated defects (not hypothetical claims about the real incident)

- `JobService.list` and `OverviewService._jobs` returned prior records after exceptions without
  recording that a provider observation failed.
- `JobService.reconcile` turned an exception into an empty inventory and refreshed state-file facts
  without checking the recorded process lifetime.
- `ProcessScheduler.state` read a durable state file, not a current process check.
- SLURM queue/accounting failures collapsed to `UNKNOWN` without preserving the query failure.
- Native retry had no alternate configuration parameter and accepted a source path before checking
  whether it still exists. Recovery does correctly protect scientific identity and original Runs.
- A lost-confirmed UNKNOWN origin counted as active and blocked its own retry. The exact origin is
  now excluded only on the native eligible recovery path, with provider revalidation under lock;
  unrelated active Jobs still block duplicate execution.
- Checkpoint registration introduced internal files that Work Results wrongly listed as outputs.
  Runtime and result views now share the payload filter, including the standalone host reader.
- SMTP originally held the incident lock through network delivery. Delivery now reserves an attempt
  then releases the lock; an old acknowledgement cannot commit a superseding transition.
- Recovery rejected before enqueue lacked a request ID/history. Apply and explicit Console
  inspection now persist an operational request before validation; native dry-run is still read-only.
- Loading nine reused Runs could be mistaken for new recovery progress and resolve an incident.
  A baseline and subsequent positive runtime advancement now distinguish new computation from reuse,
  obligation totals, counter resets and observed process loss.
- An `squeue` transport timeout skipped accounting, and cached query details could survive failure.
  A bounded accounting fallback now preserves actual evidence and clears stale query details.

No productive cluster or WISDOM evidence has been inspected or modified during this audit.

## Verification and remaining acceptance

### Specification fault matrix (A–BW)

This maps requirements to evidence, not to a blanket HPC acceptance claim. **Local** means native
local implementation with the test's stated faults; **partial** means only the listed boundary is
covered. Transport fixtures do not certify real SSH, SMTP, SLURM, logout or reboot behavior.
All file references below are beneath `tests/`.

| Cases | Local evidence / remaining gap | Boundary |
|---|---|---|
| A, H, X | `controlplane/test_operational_health.py`: failed probe retains confirmation/science then recovers; `test_operational_providers.py`: failed inventory is not empty success | Local provider faults |
| B, C, E | `test_operational_providers.py`: timeout inventory and queue/accounting transport failures | Local faults; actual SSH host outage pending |
| D | `test_operational_health.py`: typed authentication failure separate from science and unknown query errors | Local typed failure; real SSH auth pending |
| F | `test_operational_providers.py`: successful empty inventory becomes unknown, not process-loss proof | Local |
| G | `test_operational_diagnostics.py`: malformed response is not unreachable; corrupt direct records fail closed | Partial: full malformed inventory/source matrix pending |
| I | `test_operational_health.py`: late observation cannot erase a newer fact or cancellation | Local locked observations |
| J, K, M | `test_operational_health.py`: real short-lived process terminates, retained state stays running but ownership reports lost; ambiguous live-child markers prevent loss inference | Local process + marker fixture; supervisor/child outage combinations partial |
| L, AF, AG | `test_operational_health.py`: heartbeat freshness distinct from optional progress expectations | Local clocks/policy; no inference of death from metric silence |
| N | `controlplane/test_process_scheduler.py`: reused/mismatched PID identity rejected | Local identity checks |
| O, P | `test_operational_providers.py`: live queue and positive terminal accounting, same-read details | Local SLURM fixtures |
| Q, W | `test_operational_providers.py`: delayed/empty accounting unknown; query failure remains error and clears old details | Local SLURM fixtures |
| R, S, T, U, V | `test_operational_providers.py` and `test_operational_diagnostics.py`: exact scheduler reasons, OOM not inferred CUDA OOM, preemption distinct from user cancellation | Local SLURM fixtures; real site pending |
| Y, Z | `test_operational_monitor.py`: actual disposable detached monitor, duplicate start, stop and restart | Local process; OS logout/reboot acceptance pending |
| AA, AB, AC, AD, AE | `test_operational_monitor.py`: durable dedup/escalation/resolution, bounded delivery, rejection/acceptance and superseding transition | Local SMTP transport doubles; real TLS/mail server pending |
| AH | `controlplane/test_study_retry.py`, `work/test_study_recovery.py`, `work/test_fixed_study_recovery.py` | Local native fixed/adaptive recovery |
| AI | `RecoveryObservation.eligible` and shared terminal-state recovery path | Partial: explicit native TIMEOUT preview/apply integration pending |
| AJ, AK | `test_operational_providers.py`: live/unresolved ownership rejection; `tui/test_operational_recovery_e2e.py`: confirmed loss + exact retry | Local; provider admission simulated in E2E |
| AL | `work/test_fixed_study_recovery.py`, `work/test_study_recovery.py`: existing controller lock prevents duplicate execution | Partial: full concurrent public request/admission race pending |
| AM | `test_operational_providers.py`: apply-time ownership recheck; `tui/test_recovery_history_console.py`: rejected apply remains visible | Partial: complete mid-operation connectivity matrix pending |
| AN | Request IDs/rejected enqueue auditing use native error boundary | Pending ambiguous transport/admission acknowledgement acceptance |
| AO | `controlplane/test_submission_tls_monitor.py`: worker persists pre-scheduler failure | Local worker boundary |
| AP | `test_recovery_observation.py`: accepted/preparing/scheduler phases are not progress | Partial: sustained never-admitted preparation integration pending |
| AQ | `test_recovery_incidents.py`, `tui/test_operational_recovery_e2e.py`: baseline reuse is not new progress; exact audited progress resolves origin | Native CPU execution; provider/admission simulated |
| AR, AS | `test_recovery_audit.py`, `tui/test_recovery_history_console.py`: durable rejected requests, reopened offline history, persistent preview/config/apply errors | Local native audit + Textual |
| AT, AU, BA | `test_relocated_recovery_config.py`, `tui/test_operational_recovery_e2e.py`: original snapshot, moved YAML/comments, source missing, original seed namespace | Local |
| AV | Existing input semantic identity checks and worker binding | Partial: dedicated relocated large-input equivalent-path integration pending |
| AW | `test_study_retry.py`, `work/test_fixed_study_recovery.py`, `work/test_study_recovery.py`: explicit code acknowledgement preserves revisions, never overrides science | Local |
| AX, AY, AZ | Native fixed/adaptive changed-configuration/identity guards and relocated seed-policy tests | Partial: individual Dataset/architecture source matrix pending |
| BB | `test_relocated_recovery_config.py`: corrupt original identity; native owned JSON validation | Local |
| BC, BD, BE | `work/test_checkpoint_evidence.py`, `work/test_fixed_study_recovery.py`: host hashes, corrupt JSON refusal, missing checkpoint restarts only failed Run | Local; no application restoration certificate |
| BF | `test_study_retry.py`: foreign execution preview refusal; `work/test_checkpoint_evidence.py`: foreign Execution/Run owner rejects load/replace/reference/inventory without byte mutation | Local managed identity; cross-host/application compatibility pending |
| BG, BH, BI, BJ | Fixed/adaptive suites: nine successes reused, partial/pending exact cells, missing adaptive state rejection and spent budgets preserved | Local native recovery, including spawned CPU workers |
| BK, BL, BM, BN, BO, BP | Fleet retry still refuses independent single-host replay; coordinator fencing is not distributed recovery/transfer acceptance | Pending native Fleet recovery/compatible materialization |
| BQ | `tui/test_recovery_history_console.py`: slow worker + close/unmount is responsive; stale callbacks do not mutate removed screen | Partial: abrupt Console process-death matrix pending |
| BR | `test_observation_gate.py`: concurrent service probes, receipt ordering, distinct SLURM lanes and abrupt observer death | Native locks/processes + provider doubles |
| BS, BT, BU, BV | Retained full suites plus `tui/test_work_evidence_console.py`, `work/test_work_results.py`, `work/test_study_import.py`, products suites and outside-checkout wheel smoke | Full-suite outcome recorded separately; no imported live-host inference |
| BW | Legacy record/checkpoint reads and retained compatibility suites | Partial: complete historical-version snapshot matrix pending |

Executed during this implementation:

- Native fixed/adaptive recovery suites: 60 passed before later operational presentation changes.
- Combined provider/monitor/direct-process/recovery/relocated-YAML/Console regression batch:
  58 passed; targeted health/provider/request/Console batch: 22 passed.
- Relocated YAML preview/apply plus native recovery: 5 passed, including reuse of nine successes,
  new Attempt only for seed 54 and immutable original Execution/Attempt results.
- Broad control-plane/products/recovery/Console batch: 585 passed and one legacy host-reader failure,
  subsequently fixed and covered by 47 passing targeted tests.
- Current operational/fixed/adaptive/checkpoint/Console batch: 159 passed. Further source/preemption,
  incident-resolution and unsafe-path regressions were added afterward and run separately.
- Complete pytest invocation: 1,816 passed, one Work Results checkpoint-listing regression failed
  (1,218.35 seconds; four Lightning CPU warnings). The regression was fixed; the complete Work
  Results/Console evidence/checkpoint batch then passed 38 tests. This is **not** a claim of a second
  all-green full-suite invocation on the final revision.
- Provider/native recovery/probe/Work Results/checkpoint batch: 62 passed.
- Recovery audit/native retry/configuration/Console batch: 33 passed. A broader health/incident/
  audit/Console batch passed 40 tests but exposed an unmount callback warning; message-based
  delivery then passed the three history tests with thread-exception warnings treated as errors.
- SLURM fallback and native recovery/audit regressions: 38 passed. Subsequent positive-progress/
  loss, incident, audit and history regressions: 32 passed; per-test-clock incident batch: 12 passed.
  The final UI batch/full-suite results are recorded after completion, not inferred from these.
- Latest audit/monitor/Console/Work Results/checkpoint batch: 80 passed; checkpoint/native fixed/
  adaptive recovery/retention batch: 87 passed. Shared diagnosis/provider/incident batch: 51 passed.
- Positive-runtime baseline/native preview/audit/Console/structure batch: 43 passed. The full suite
  exposed the pytest module-alias clock fixture defect: patching a namespace import did not update
  the collected module's clock. The fixture now patches its own globals, and an injected six-hour
  collection delay is verified independently, without sleeping or weakening freshness policy.
- After the fixture alias correction, the combined incident/health/audit/Console batch passed
  36 tests and the six-hour-delay injection passed all 16 incident tests.
- Complete follow-up pytest invocation: 1,850 passed, three failures (two collected-clock failures
  and missing Message class docstrings), four Lightning CPU warnings, 1,212.09 seconds. Those defects
  are corrected. A subsequent invocation was interrupted after 1,221 passes when the remaining
  clock-alias defect was reproduced; neither run is presented as final all-green acceptance.
- **Final delivered-core revision:** complete pytest passed **1,866 tests**, four existing Lightning
  CPU smoke warnings, **1,125.33 seconds**. The JUnit report was generated in the temporary
  validation directory. This invocation includes the corrected clock fixture, managed checkpoint
  owner/registration guards, native recovery baseline, saved audit and persistent Console errors.
  A passing local suite is not acceptance of the unimplemented Fleet or real-site boundaries.
- Two-Study Textual recovery plus incident/core tests: seven passed. The provider/submit
  acknowledgement is simulated; actual CPU WorkRunner execution, YAML validation, locks, Attempts,
  original success bytes, incident resolution and Console reopen are native local implementations.
- Ruff and mypy passed (603 source files). No remote CI run was triggered.
- Source/wheel distributions built. Installed-wheel smoke passed outside the checkout with the
  existing dependency environment explicitly shared; package code/metadata were verified to come
  from the temporary wheel installation. CI retains a fresh dependency installation gate.
  The final wheel is rebuilt after recording this outcome; no version/release publication occurred.

CUDA is detected on the local machine but CUDA fault/site tests have not been executed in this
phase. SMTP delivery tests inject transport acceptance/failure; no real mail was sent. systemd
tests preview the unit; no service was installed on this workstation. Existing monitor lifecycle
tests launch/stop only disposable project observers.

Do not mark the complete supplied specification accepted. Remaining work includes native Fleet
fencing/recovery and compatible transfer, live checkpoint rotation/application restoration receipts,
the complete checkpoint/compatibility workbench, real OS service/mail/site validation and
the full A–BW fault matrix. The local two-Study recovery fixture does not certify real SSH, scheduler
admission or a complete native detached background preparation cycle.
Existing single-host recovery checks, products and Work Results remain the native authority.

## Inventory follow-up — 2026-10-11

Saved WISDOM controller observations (read locally, without contacting execution hosts) exposed
two distinct causes of host-wide UNKNOWN/degraded displays: a corrupt historical direct Job
aborted the inventory, and a long-lived observer shipped the wrong helper definitions after
editable source line offsets changed. Neither observation establishes scientific failure.

Native direct inventory now returns bounded, validated per-Job error rows without adopting them,
mutating host evidence or authorizing recovery. Healthy siblings receive fresh facts and clear
prior query errors. Terminal outcomes remain preserved; root/transport failures still fail closed.
Observer construction selects named definitions and their helper closure from current source
snapshots rather than stale loaded function offsets. The Study overview helper follows the same
rule. Console/monitor sessions open before this correction must be restarted to load it, not Jobs.

Validation: **480 tests passed** in **414.79 seconds**, covering all control-plane tests, operational
Console/recovery E2E and repository structure; the new inventory file contains 12 regressions.
An earlier broad run reproduced three missing-helper failures after source edits (475 other tests
passed), prompting the source-snapshot fix; it is not reported as green. Ruff check and mypy
(603 source files) passed. A newly built wheel's observer/CLI smoke passed outside the checkout,
with existing dependencies explicitly shared. No real cluster, WISDOM code, dataset, scientific
record or training process was modified. Whole-repository formatting still reports findings
outside this focused fix; changed implementation/test files are formatted separately.
