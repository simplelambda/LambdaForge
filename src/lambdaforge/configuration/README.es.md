# Configuración de LambdaForge

[Guía principal](../../../README.es.md) · [English](README.md)

## 0. Contenidos

- [1. Modelo mental](#1-modelo-mental)
- [2. Autoría sencilla](#2-autoría-sencilla)
- [3. Materialización y validación](#3-materialización-y-validación)
- [4. Composición](#4-composición)
- [5. Compatibilidad y seguridad](#5-compatibilidad-y-seguridad)

## 1. Modelo mental

El usuario escribe un `AuthoringConfig`; LambdaForge lo compila a un `MaterializedConfig` estricto
y entrega ese objeto al validador/runner existente. No hay dos motores:

```text
YAML de autoría -> normalización -> IR estricto -> validación -> ejecución
```

`ConfigurationKind` sólo se deduce de estructura inequívoca. Un experimento histórico se reconoce
por `experiment`; las formas estrictas de tarea/workflow conservan `kind`.

## 2. Autoría sencilla

El preprocesado conciso nombra inputs y outputs una vez:

```yaml
name: preparar-datos
inputs: {raw: data/raw.jsonl}
outputs: {processed: processed}
preprocess:
  function: mi_proyecto.preprocessing.normalizar
  input: raw
  output: processed
  workers: 4
  workload: io
resources: {cpus: 4, memory: 8GiB, time: 30m}
```

Sólo posiciones conocidas aceptan un target como string. Usa `{target, params}`, `{ref, params}` o
`{plugin, params}` cuando necesites parámetros o semántica explícita; no se adivinan imports a
partir de strings arbitrarios.

## 3. Materialización y validación

`lambdaforge inspect CONFIG --resolved` muestra el mapa estricto exacto y `validate` comprueba
Schema, referencias, imports y constructor. Ninguno ejecuta la tarea ni crea el directorio del run.
En Python usa `AuthoringConfig.load(path).materialize()` o `LambdaForge.materialize(path)`.

## 4. Composición

`ConfigurationComposer` resuelve `extends`, `include`, hoja y overrides en ese orden. Los mapas se
fusionan, las listas sustituyen y `{$delete: true}` borra. Sólo existen `${config:path}`,
`${env:NAME}` y `${secret:NAME}` como valor completo. `compose` muestra valores redactados y
procedencia; `diff` compara hojas semánticas.

## 5. Compatibilidad y seguridad

Siguen válidos tarea 1.0, workflow 1.0 y experimento 1.1 estrictos. Las migraciones de experimento se
aplican antes de defaults para conservar versión y pasos históricos. Los targets/plugins ejecutan
Python confiable: YAML no es un sandbox y no debe aceptarse de una fuente no confiable.
