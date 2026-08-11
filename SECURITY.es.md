# Política de seguridad

[English](SECURITY.md) | [Español](SECURITY.es.md)

## Versiones soportadas

LambdaForge es pre-1.0. Las correcciones se aplican a la rama actual y, cuando sea viable, a la
minor publicada más reciente. No se garantizan minors anteriores.

## Reportar una vulnerabilidad

No publiques una sospecha en un issue. Usa el canal privado de GitHub Security Advisory cuando esté
disponible o contacta al propietario mediante un canal verificado de SimpleLambda. Incluye
versión/commit, plataforma, reproducción mínima, impacto y si requiere input no confiable. No
incluyas credenciales ni datasets privados.

## Modelo de seguridad

- YAML `target`/`ref`, plugins, pickle y Python consumidor son código confiable; no hay sandbox.
- La interpolación no evalúa expresiones; secrets se redactan y workflows persistidos los rechazan.
- Inputs/outputs/stores/cache/archives/retention validan containment y symlinks en su capa.
- Checksums detectan cambios pero no autentican productor; usa HMAC/permisos/canales autenticados.
- Backends local/SLURM usan argv y scripts quoted, no interpolación de shell local.
- Tracking/S3 amplían confianza a SDK, credenciales, red y servicio y sólo cargan si se configuran.
- NPZ/NPY deshabilita pickle; sync remoto está allowlisted/acotado y fetch no sale del work_dir.
- SSH conserva `known_hosts`/agent/config; bootstrap no instala drivers ni CUDA del sistema.

Los límites completos están en [arquitectura](docs/ARCHITECTURE.es.md).
