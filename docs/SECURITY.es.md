[English](SECURITY.md) | [Español](SECURITY.es.md)

# Modelo de amenazas de credenciales y scheduler

Esta guía complementa la [política de seguridad](../SECURITY.es.md).

## Límites

`ClusterProfile` es configuración durable y compartible: puede contener host/user/account, paths,
política y un identificador `keyring:`/`env:`, nunca el secreto. `CredentialProvider` es la única
fuente. `Transport` recibe una contraseña solo en memoria al autenticar. Jobs, bundles, fingerprints
y exports no reciben el valor. `SecretRedactor` elimina secretos conocidos de errores; plugins
deben respetar el mismo límite.

OpenSSH es preferido porque gestiona claves, agente, certificados, `known_hosts` y ProxyJump. El
modo contraseña delega en Paramiko: rechaza hosts desconocidos, deshabilita claves/agente en ese
modo, aplica timeouts y transfiere por SFTP. Contraseña o host key incorrectos fallan sin degradar la
verificación.

El keyring del SO es un proveedor externo confiable. `env:` es integración efímera deliberada y
puede quedar expuesta por CI/inspección de procesos fuera de LambdaForge. El prompt usa `getpass`
oculto. No hay flag de contraseña ni fichero cifrado propio: crypto casera añadiría gestión de
claves sin mejorar el límite real.

Las plantillas de recursos/comandos tienen placeholders fijos y producen argv. Nombres/valores
rechazan saltos de línea. El batch script sí ejecuta shell: `prologue`/`epilogue` son código confiable
del perfil y no reciben secretos. Guarda credenciales personales en ámbito usuario y revisa perfiles
de proyecto antes del commit.

## Comprobaciones

Usa `clusters inspect` para fuente/auth, `clusters export` para copia compartible, `doctor --on` para
auth/host key/workspace/comandos/mapping/partición sin job y `run --dry-run` para revisar script,
directivas y argv. Nunca pegues contraseñas en YAML, issues, línea de comandos, directivas o prologue.
