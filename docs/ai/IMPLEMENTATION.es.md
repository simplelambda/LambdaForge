# Estado técnico de la integración nativa de agentes

[Registro detallado y matriz de requisitos](IMPLEMENTATION.md) ·
[Guía de uso](INTEGRATION.es.md).

La reforma integral **no está terminada**. LambdaForge 0.19.0 empaqueta los fundamentos
implementados y las correcciones operativas, no una promesa de capacidades inexistentes.
El punto de partida y las pruebas históricas de 0.18.0 se conservan como tales; la nueva versión
no cambia los requisitos pendientes ni habilita mutaciones científicas no soportadas.
No se ha modificado WISDOM, iniciado trabajos en clusters ni cambiado evidencia científica.

## Arquitectura conservada

Instrucciones de repositorio y consumidor, Skills procedimentales empaquetadas, fachada Python
fijada al proyecto, MCP oficial opcional y memoria Markdown son capas distintas. Todas las
consultas operativas/científicas delegan en servicios nativos; no existe otro runner ni un backend
que interprete salida humana de CLI. AGENTS inglés breve es canónico; sus contratos detallados
siguen en FRAMEWORK_CONTRACTS.md y la explicación española no define una segunda política.

Las 24 Skills tienen referencias y ejemplos nativos, y se han validado con skills-ref oficial.
Init/sync trabaja con recursos del paquete instalado, mantiene personalizaciones y no cambia
la configuración global del cliente. Sin flags, init conserva el comportamiento científico anterior.

## Permisos y aprobación humana

READ/PLAN es el valor predeterminado. La disponibilidad de una operación no significa que esté
habilitada en la sesión. Las acciones aprobadas son crear/editar/enlazar notas y exportarlas
en ZIP, bajo WRITE_PROJECT. Preview no escribe; request guarda una propuesta sin aprobarla.
Una persona revisa los valores exactos desde CLI TTY o Console y escribe el identificador completo.
Solo después puede el agente aplicar ese plan por separado.

Los registros firmados vinculan proyecto, argumentos, revisiones, hash y caducidad. Claim previo
bajo lock, auditoría y revalidación impiden replay concurrente. Un crash tras el claim no permite
repetir automáticamente. MCP no expone aprobar; texto recuperado y `approved=true` no autorizan.
Esto protege la frontera del protocolo, no un programa con acceso shell independiente al mismo
usuario del sistema operativo. SUBMIT/OPERATE/DESTRUCTIVE científicos siguen pendientes.

## Transporte, dependencias y memoria

SDK oficial 2.3: stdio predeterminado, protocolo limpio y planificación importadora aislada con
timeout/cancelación. HTTP local es opt-in, READ/PLAN, token privado, loopback/Host/Origin exactos,
comprobación por petición y límites. No es OAuth ni permite exposición de red pública.
Dependencias opcionales requieren consentimiento; solo extras declarados en virtualenv/Conda,
con modo wheelhouse sin red. Los fallos no borran archivos/notas.

Markdown de once tipos es canónico; SQLite FTS incremental es reconstruible. Ediciones externas,
revisiones, notas corruptas y duplicados se diagnostican sin convertir interpretaciones en ciencia.
Consultas relacionadas usan enlaces/tags/evidencia exacta. Verificar referencias consulta registros,
no interpreta resultados ni descarga pesos. Obsidian headless funciona sin escritorio o plugins;
búsqueda semántica y vault externo todavía no están implementados.

Exportación de notas ZIP con bytes exactos, manifiesto checksum y revisión del inventario completo:
rechaza corrupción, duplicados, symlinks, cambios y sobrescrituras; nunca incluye ciencia/plugins.
Auditoría nativa de intención/finalización con ID/revisiones, sin títulos/cuerpos y paginada por
bytes; una intención interrumpida no acredita éxito. Trece prompts MCP y recursos de contratos
instalados, índices y resultados/productos exactos respetan toolsets y no autorizan ejecución.
Ayuda Obsidian distingue diagnóstico pasivo de prueba CLI/apertura local explícita, sin instalar
plugins ni alterar preferencias. Headless/ausencia/fallo/timeout se probaron mediante fixtures;
no se acredita una sesión de escritorio real ni configuración de vault externo.

## Verificación realizada

- 67 pruebas focalizadas: protocolo real, HTTP en socket, aprobaciones, aislamiento, pip real
  fallando sin red en venv temporal, CPU Work/evidencia/memoria, Console, estructura y versión.
  Se comprueba también el cambio externo entre validación nativa y escritura antes de commit.
- Codex 0.162.0-alpha.17.2 real descubre herramientas y llama project_info desde otro directorio;
  configuración temporal aislada, sin inferencia ni credenciales y sin tocar ajustes existentes.
- Todos los modelos públicos resuelven declaraciones/docstrings sin importar Torch. Consultas
  seleccionadas de Runs, productos, incidentes y recursos usan los servicios nativos y límites.
- Ruff sobre src/tests y mypy sobre 622 archivos pasan. Wheel 0.18.0 instalado y probado fuera
  del checkout; no depende del MCP para uso normal. CI configurada, no afirmada como ejecutada.
- Pytest completo (iniciado antes de las últimas regresiones): 1930 correctas, 1 fallo y 4 avisos,
  1195,10 segundos. El único fallo era la versión ausente en el enlace español AGENTS; se corrigió
  y pasaron las cuatro pruebas de versión y el gate final de 67 pruebas. No se repitió toda la
  suite tras esa corrección: esta ejecución completa no se presenta como totalmente verde.

## Lo que queda por implementar o acreditar

Operaciones científicas autorizadas; resto de herramientas/recursos/prompts y planificación nativa;
semántica local/híbrida; vaults existentes/externos autorizados y aceptación real de Obsidian;
UX de recuperación del journal ante escrituras interrumpidas; GUI ampliada; un segundo cliente real;
conversaciones completas con ciencia CPU y memoria entre sesiones; gates finales completos.
Descubrimiento MCP real no equivale a una conversación de agente ni a autorización científica.
El registro inglés conserva los siguientes pasos para continuar sin repetir trabajo verificado.
