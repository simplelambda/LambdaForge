# Política de seguridad

Español · [English](SECURITY.md)

## Versiones soportadas

LambdaForge es pre-1.0. Las correcciones de seguridad se aplican a la rama de desarrollo y a la
minor publicada más reciente cuando un backport compatible es práctico. No se garantizan minors
anteriores.

## Informar de una vulnerabilidad

No publiques una vulnerabilidad sospechada en un issue. Usa el canal privado de vulnerabilidades/
security advisory de GitHub cuando esté disponible o contacta de forma privada al propietario por
un canal verificado del perfil de SimpleLambda. Incluye versión/commit, plataforma, reproducción
mínima, impacto y si requiere entrada no confiable. Nunca adjuntes credenciales ni datasets reales.

## Modelo de seguridad

- Un `run` YAML importa Python consumidor confiable y puede ejecutar código arbitrario. LambdaForge
  no es sandbox. YAML no evalúa expresiones, construye objetos recursivos, ejecuta funciones
  arbitrarias ni interpola secretos.
- Entradas, outputs, cache/checkpoints y limpieza comprueban containment y symlinks en su capa. Un
  dataset se elimina solo tras volver a verificar manifest, identidad y raíz exacta; inconsistencias
  o estado inaccesible fallan de forma cerrada.
- El `project_root` de un clúster es almacenamiento persistente propiedad del investigador y queda
  fuera de la limpieza de LambdaForge. Los inputs grandes solo se mapean por ruta relativa al
  proyecto y deben coincidir en tipo, bytes y SHA-256 antes del envío y otra vez en el worker. Un
  output relativo remoto exige este mirror y no puede escapar de él; una ruta absoluta es un escape
  deliberado de código confiable. El hash aporta consistencia, no autentica al productor ni aísla de
  otra cuenta maliciosa concurrente.
- `outputs.file/directory` asigna storage del Attempt y solo registra un conjunto completamente
  finalizado. `outputs.artifact` copia de forma segura una ruta externa. `outputs.dataset` exige
  assets del Run o URI explícita, rechaza traversal/symlinks, hashea, verifica staging y publica
  atómicamente. Una cadena YAML solo es ruta con un marcador `file` o `dataset` explícito.
- Los ficheros gestionados de cache/checkpoint rechazan claves absolutas o con traversal, symlinks y
  contenido no regular. Publican tras validación, `fsync`, SHA-256 y reemplazo atómico y exponen un
  handle path-like de solo lectura. `Work.map` guarda key/huella/tamaño, nunca rutas del controlador.
  La limpieza de Work cache usa un lock exclusivo mientras Works activos conservan lease compartida.
- `Work.tools.run` acepta argv y nunca shell. Threads y overrides de entorno solo afectan al hijo.
  Stdout/stderr puede contener datos del proyecto y debe revisarse antes de compartir, igual que un
  registro de diagnóstico.
- `lf delete WORK` es preview-first, rechaza Attempts activos y solo elimina una raíz exacta de Job
  o Execution. Datasets, caches/entornos compartidos y otros Works están fuera; un recibo mínimo
  permite repetir la operación de forma segura.
- El borrado interactivo exige una segunda tecla explícita, nunca elimina Jobs activos y usa la
  misma operación de raíz exacta que la CLI. Las raíces locales absolutas por Job evitan borrar
  desde otro directorio actual; la limpieza global conserva el registro si su workspace no pudo
  eliminarse de forma segura.
- Un checksum detecta modificaciones, pero no autentica al productor. Usa HMAC cuando esté
  disponible, permisos restrictivos y canales autenticados.
- Los backends local/SLURM no interpolan comandos en un shell local. Los scripts batch escapan argv.
  Prologue/epilogue configurados son código shell confiable y no reciben valores no revisados.
- El YAML de clúster solo conserva modo y referencia `keyring:`/`env:`. Passwords no entran en argv,
  YAML, bundles, estado, fingerprints ni logs. OpenSSH es preferido; Paramiko opcional rechaza host
  keys desconocidas. Los secretos en variables heredan el riesgo del proceso/CI.
- Los fallos CLI guardan traceback y comando saneado bajo el estado de usuario con permisos
  restrictivos. Se redactan campos comunes de password/token/API key, bearer headers, URLs con
  credenciales y claves privadas, pero una excepción de proyecto puede incluir datos científicos
  que ningún redactor genérico reconoce: revisa antes de compartir.
- Provisioning Python es sin privilegios y se limita a cache. Micromamba se obtiene por HTTPS y se
  verifica por SHA-256. Bootstrap no edita perfiles, Python/CUDA/drivers del sistema. El runtime solo
  reutiliza un bundle CA legible ya elegido por la confianza del host y nunca desactiva TLS, descarga
  roots arbitrarios ni modifica `/etc`.
- Los entornos nativos son declaraciones de datos: rutas dentro del proyecto y sin variables,
  prefix, pip anidado ni hooks; paquetes pasan como argv sin shell. Canales/paquetes online siguen
  siendo supply chain confiable y pueden incluir metadata/scripts propios de Conda, por lo que deben
  revisarse y fijarse. Locks offline rechazan credenciales/queries y exigen plataforma, SHA-256 y
  bytes regulares coincidentes antes del envío. Prefijos temporales y caches se coordinan por
  identidad; ownership/versión de ejecutables se valida antes de publicar. `tools.require()` nunca
  autoriza instalar.
- Tracking y S3 amplían la frontera de confianza a su SDK, credenciales, red y servicio. Son
  opcionales y lazy.

Los límites de ownership y control plane se detallan en el
[manual canónico](docs/MANUAL.es.md#11-limpieza-y-seguridad).
