[English](GOVERNANCE.md) | [Español](GOVERNANCE.es.md)

# Versionado, deprecación y releases

LambdaForge sigue Semantic Versioning. Antes de 1.0 una minor puede romper API si se documenta; una
patch debe conservar la API pública de su serie. API pública significa re-exports documentados,
Schema YAML y contratos JSON. Las ubicaciones privadas no lo son.

Una API nueva necesita responsabilidad clara, validación, tipos/docstrings, re-export, Schema y
ejemplo si aplican, tests focalizados, docs EN/ES y AGENTS. Una deprecación emite warning y conserva
al menos un ciclo minor salvo problema de seguridad. Dependencias especializadas son extras lazy.

Cada release actualiza versión y CHANGELOG, ejecuta formato/lint/mypy/tests/build/twine y prueba la
wheel instalada fuera del source. El tag y GitHub Release se crean después del commit verificado.
El repositorio aún no elige licencia y no debe inventarse una.

Tras commitear 0.6.0, el owner publica primero el commit y espera toda la matriz CI. Sólo con ese
commit verde crea el tag anotado y la GitHub Release:

```bash
git push origin main
# Esperar a que CI de este commit exacto quede verde.
git tag -a v0.6.0 -m "LambdaForge 0.6.0"
git push origin v0.6.0
gh release create v0.6.0 --verify-tag --generate-notes \
  --title "LambdaForge 0.6.0"
```

Esta preparación no afirma que ya exista un tag/release remoto y esos comandos no deben ejecutarse
antes del CI verde.

La disciplina científica prohíbe seleccionar por mtime. Fingerprint, attempt ID, configuración,
entorno y artifacts elegidos acompañan resultados de publicación. No se afirma paridad sin una
referencia externa fijada y revisada. Seguridad se comunica según `SECURITY.md`.
