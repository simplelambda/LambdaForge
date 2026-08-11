# LambdaForge configuration

[Root guide](../../../README.md) · [Español](README.es.md)

## 0. Contents

- [1. Mental model](#1-mental-model)
- [2. Friendly authoring](#2-friendly-authoring)
- [3. Materialization and validation](#3-materialization-and-validation)
- [4. Composition](#4-composition)
- [5. Compatibility and safety](#5-compatibility-and-safety)

## 1. Mental model

Users write an `AuthoringConfig`; LambdaForge compiles it to a strict `MaterializedConfig`, then
the existing task, experiment or workflow validator/runner takes over. This is progressive
disclosure, not a second execution system:

```text
authoring YAML -> normalization -> strict task/experiment/workflow IR -> validation -> execution
```

`ConfigurationKind` is inferred only from unambiguous structure. `kind: task` and
`kind: workflow` remain explicit in strict documents; the historical experiment shape is detected
by its `experiment` block.

## 2. Friendly authoring

The concise preprocessing form names inputs and outputs once:

```yaml
name: prepare-data
inputs: {raw: data/raw.jsonl}
outputs: {processed: processed}
preprocess:
  function: my_project.preprocessing.normalize_record
  input: raw
  output: processed
  workers: 4
  workload: io
resources: {cpus: 4, memory: 8GiB, time: 30m}
```

Only known object slots accept a target string, for example
`model: my_project.models.ProjectModel`. Use an explicit `{target, params}`, `{ref, params}` or
`{plugin, params}` specification when parameters or import behavior matter. Arbitrary strings are
never guessed as Python imports.

## 3. Materialization and validation

Use `lambdaforge inspect CONFIG --resolved` to see the exact strict mapping. Use
`lambdaforge validate CONFIG` to check its packaged Schema, references, imports and constructor
contract. Materialization does not import the configured callable or write a run directory.

`AuthoringConfig.load(path).materialize()` is the Python equivalent. `AuthoringSchemaCatalog`
exposes the permissive input-shape Schema; the selected strict Schema remains authoritative for
execution. `LambdaForge.materialize(path)` returns the same `MaterializedConfig` used by the CLI.

## 4. Composition

`ConfigurationComposer` resolves `extends`, then `include`, then the leaf and explicit overrides.
Mappings merge recursively, lists replace and `{$delete: true}` removes a value. Interpolation is
limited to `${config:path}`, `${env:NAME}` and full-value `${secret:NAME}`. `compose` displays
redacted values and provenance; `diff` compares semantic leaves.

## 5. Compatibility and safety

Strict task Schema 1.0, workflow Schema 1.0 and experiment Schema 1.1 documents remain supported.
Experiment migrations run before authoring defaults so an old document retains its source version
and migration steps. YAML targets/plugins are trusted Python configuration. Secret interpolation
does not turn YAML into a sandbox; never execute an untrusted configuration.
