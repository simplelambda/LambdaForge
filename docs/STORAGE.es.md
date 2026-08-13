# Almacenamiento interno y GC conservador

Español | [English](STORAGE.md) | [Guía raíz](../README.es.md)

## 1. Categorías

`storage status` muestra raíces exactas, bytes y ficheros para estado, bundles, entornos, caché de
paquetes, workspaces, temporales y datasets.

| Categoría | Significado | GC por defecto |
|---|---|---|
| state | Registros/punteros pequeños | nunca |
| bundles | Entrada inmutable reconstruible | stale/incompleto sin referencias |
| environments | venv verificado por contenido | stale/incompleto sin referencias |
| package cache | Descargas pip reutilizables | se informa; política del proveedor |
| job workspaces | Estado, logs y trabajo científico | nunca automático |
| temporary | Trabajo de caché incompleto | elegible |
| datasets | Datos científicos | nunca |

Resultados y checkpoints retenidos son evidencia científica, no caché.

```bash
lambdaforge storage status
lambdaforge storage status --on atlas
lambdaforge storage status --all --json
lambdaforge storage gc --on atlas
lambdaforge storage gc --on atlas --apply
lambdaforge environments list --on atlas
lambdaforge environments show ENV --on atlas
```

## 2. Seguridad de GC

GC genera primero un plan. Protege bundles/entornos referenciados por jobs activos/en cola, exige
un descendiente exacto de una raíz interna configurada, rechaza symlinks y nunca acepta raíces de
datasets/resultados. Apply elimina sólo candidatos reconstruibles mostrados. Sin política de edad
no hay borrado agresivo.

Los recolectores comparten un lock entre procesos. Además, GC falla cerrado mientras exista un
marcador de construcción de entorno para no competir con `pip install` ni la publicación atómica.
Investiga un marcador sobrante sólo tras demostrar que no hay ningún bootstrap activo.

Un entorno se construye en `.env-ID.tmp-...`, instala wheels exactos usando caché pip compartida,
verifica LambdaForge/Torch/CUDA y sólo entonces se renombra atómicamente a `env-ID`. Los fallos
limpian el temporal exacto cuando es seguro. 0.6 conserva una identidad completa de framework,
proyecto y dependencias: no separa una capa runtime pesada si puede introducir ambigüedad.

## 3. Planificación de cuotas

Pon estado pequeño en home fiable; caché/work pesado reconstruible en scratch; datasets en storage
de proyecto y registra placements. Bytes y número de ficheros importan en HPC. `cache_max_size` y
`cache_max_age` expresan política, pero GC sigue siendo preview-first y no autoriza borrar ciencia.
