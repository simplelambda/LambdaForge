# Integración de agentes fijada al proyecto

[English](INTEGRATION.md). **La reforma integral todavía no está terminada**:
consulta el [registro de aceptación](IMPLEMENTATION.md). No cambia ciencia, identidades, HPO o GPU.

## Inicio rápido

```bash
lf init mi-investigacion --agents portable --knowledge markdown
lf ai --project /ruta/absoluta init --agents codex --knowledge obsidian
lf ai --project /ruta/absoluta init --agents codex --knowledge obsidian --apply
lf ai --project /ruta/absoluta doctor
lf ai skills list
lf ai capabilities --json
```

Sin flags, init sigue creando únicamente el proyecto científico. `--interactive` pregunta en una
TTY; no instala dependencias. **Agents / knowledge…** muestra estado, Skills, búsqueda y preparación
con confirmación mediante el mismo backend. La guía consumidora enseña APIs públicas, no internals.
AGENTS raíz inglés es canónico; los contratos detallados siguen en FRAMEWORK_CONTRACTS.md.

## MCP, clientes y permisos

Instala expresamente `python -m pip install "lambdaforge[agents]"` en el entorno deseado.
Arranca `python -m lambdaforge ai --project /ruta/absoluta mcp serve`. SDK oficial 2.3, stdio;
stdout pertenece al protocolo. `mcp test` usa un cliente real en memoria; las pruebas además usan
un proceso stdio desde otro directorio. Codex 0.162.0-alpha.17.2 ya ha descubierto herramientas y
realizado llamadas reales con configuración temporal aislada y sin credenciales de modelo.
Eso no acredita una conversación LLM ni un segundo cliente de proveedor, todavía pendientes.

Codex genera `.lambdaforge/agents/codex-config.example.toml`; revisa y combina la entrada con
la configuración existente, usando el Python absoluto correcto. No modifica archivos globales.
Toolsets core/work/study/results/products/datasets/clusters/operations/knowledge, por defecto
core/work/knowledge. El catálogo distingue disponible, habilitado en sesión y no soportado.
Los detalles remotos se piden expresamente; listar no descarga todos los artifacts/epochs.

READ/PLAN no autoriza enviar, cancelar, borrar ni escribir como agente. Las mutaciones científicas
siguen deshabilitadas. Ediciones aprobadas de notas y HTTP local emparejado son opt-in explícito.
Validar importa Python del consumidor y requiere `--allow-project-code` del operador. MCP aísla
planificación en un hijo con timeout/cancelación y sin prints en stdout del protocolo. **No es un
sandbox** contra código confiado o agentes con shell propio. Texto recuperado no concede permisos.
Las respuestas son tipadas, con origen/fecha de consulta/error; consultar ahora no demuestra
observación fresca del proveedor. No ofrece shell ni lectura arbitraria de archivos.

## Aprobación humana y dependencias

El operador stdio puede habilitar `--permissions READ,PLAN,WRITE_PROJECT`. `agent_actions` descubre
`knowledge.create/update/link/export`: export crea un ZIP de notas, nunca exportación científica.
`preview_agent_action` y `request_agent_action` no editan notas ni crean archivos ZIP.
Una persona revisa valores exactos en **Agents / knowledge… → Review agent requests…**, o usa:

```bash
lf ai approvals list
lf ai approvals show approval-ID_EXACTO
lf ai approvals approve approval-ID_EXACTO --plan-hash HASH_EXACTO
lf ai dependencies check
lf ai dependencies install                          # solo preview
lf ai dependencies install --apply                  # virtualenv/Conda actual
lf ai dependencies install --wheelhouse wheels --apply # sin red
lf ai init --agents portable --apply --install-deps
```

CLI exige TTY local y escribir el ID completo; Console muestra todos los argumentos y exige
confirmación equivalente. Aprobar no ejecuta: el agente debe aplicar el plan por separado antes
de caducar (máximo 15 minutos). Proyecto, argumentos, revisiones y hash se revalidan; registros
firmados, claim bajo lock y auditoría rechazan falsificación, cambios y replay/concurrencia.
Un crash tras el claim no permite repetir. MCP no ofrece aprobar, ni admite `approved=true`.
Esto no aísla agentes que dispongan de shell independiente bajo el mismo usuario.

Solo se instala el extra declarado `agents`, con consentimiento y en el intérprete del operador;
nunca Python de sistema, paquetes arbitrarios, aplicaciones de escritorio ni clusters.
Fallos conservan archivos/notas, muestran diagnóstico acotado y exit code de error.

## HTTP local opcional

`lf ai mcp token --apply` prepara un token privado y no muestra su contenido.
`lf ai mcp serve --transport http --port 8765` usa el mismo SDK/backend, solo READ/PLAN y
`127.0.0.1`. El cliente local autorizado lee `.lambdaforge/agents/http/token` y usa el header
`Authorization: Bearer TOKEN`, nunca una URL. Navegadores requieren además un `--origin` exacto.
Se verifica token/proyecto en cada petición, Host exacto, peer loopback, límites y Origin;
no se confía en headers de proxy. Es emparejamiento local, **no un servidor OAuth** ni exposición
de red. Stdio sigue siendo el transporte recomendado. Rotar el token invalida el anterior.

## Memoria y sincronización

```bash
lf ai knowledge init --apply
lf ai knowledge create hypothesis "¿Mejora width el score?" --author-type ai --apply
lf ai knowledge search width --kind hypothesis
lf ai knowledge read ID_EXACTO
lf ai knowledge reindex --force
lf ai knowledge reindex --rebuild
lf ai knowledge audit --limit 20  # --cursor usa el next_cursor exacto, en bytes
lf ai knowledge export notas.zip          # preview sin crear el ZIP
lf ai knowledge export notas.zip --apply  # acción explícita del operador local
lf ai knowledge obsidian                  # solo indicios, no arranca la app
lf ai knowledge obsidian --probe-cli --apply # prueba explícita, puede arrancar Obsidian
lf ai knowledge open --apply              # solicita abrir el README propio, sin cambiar ajustes
lf ai sync
lf ai sync --apply
```

Markdown canónico con once tipos de nota, autor humano/IA, fecha, tags, estado, enlaces y proyecto.
Hipótesis IA son propuestas, no ciencia verificada. Enlaza Execution exacta o contenido SHA-256;
un nombre ambiguo no sirve. Comprobar referencia no verifica interpretación. SQLite FTS es un
índice reconstruible que reconoce ediciones externas; revisiones hash rechazan conflictos nativos.
El editor externo no participa en el lock. `knowledge link` enlaza notas existentes con revisión
actual. Un índice corrupto permite reconstrucción explícita y conserva su copia antigua sin tocar
Markdown. IDs duplicados/corrupción de notas generan avisos sin borrarlas.
Las consultas de notas relacionadas usan enlaces/tags/evidencia exacta, no confianza científica.
Hipótesis, decisiones y preguntas abiertas tienen vistas acotadas; verificar referencias consulta
registros nativos locales, sin descargar artifacts ni verificar la interpretación.
Obsidian es opcional, usa los mismos Markdown y preserva preferencias: no necesita escritorio/plugins.
La auditoría nativa guarda intención/finalización, ID y revisiones, no títulos/cuerpos. Intención
interrumpida no significa éxito; la paginación por cursor no lee todo el historial. Ediciones de
Obsidian no se atribuyen automáticamente a una persona ni se convierten en ciencia verificada.
El journal `.lambdaforge/agents/knowledge-audit.jsonl` es independiente del índice FTS desechable:
reindex no borra historia. Un journal corrupto bloquea nuevas ediciones hasta inspección explícita,
sin eliminarlo ni fingir que su historial sigue siendo fiable.
Export comprime solo notas canónicas y un manifiesto con checksums exactos: rechaza corrupción,
duplicados, symlinks, planes obsoletos y sobrescrituras. Conserva los bytes originales; no incluye
datos científicos, índices, plugins ni archivos ajenos. La ruta debe estar dentro del proyecto,
fuera del vault y tener un directorio padre existente. No es `lf export STUDY` ni recovery.
MCP `knowledge_export` solo revisa; aplicar usa `knowledge.export` con aprobación humana separada
y hash del inventario completo. Vaults externos y búsqueda semántica siguen pendientes.

Diagnóstico Obsidian consulta PATH/configuración/indicios de escritorio sin asumir instalación,
versión ni aplicación abierta. La prueba explícita de la [CLI oficial](https://obsidian.md/help/cli)
puede arrancar la aplicación, tiene timeout y salida acotada. Open utiliza la
[URI oficial codificada](https://obsidian.md/help/uri) del README propio: antes hay que registrar
manualmente esa carpeta como vault. Éxito significa `open_requested`, no ventana verificada;
entorno headless, CLI ausente/deshabilitada y fallos dan indicaciones. No instala aplicaciones/
plugins ni cambia ajustes, y MCP no expone abrir escritorio. Se probaron fixtures, no una sesión
real de Obsidian instalada.

MCP ofrece recursos acotados de contratos Work/Study instalados, estructura, recetas, resultados/
productos exactos e índice de notas. Se registran según toolset, nunca para leer rutas arbitrarias.
Trece prompts cortos orientan creación, modelos, checkpointing, comparación, HTML, publicación,
diagnóstico, recovery, revisión de evidencia, decisiones y próximos experimentos, sin autorizarlos.

Las 24 Skills proceden del paquete instalado. Sync preserva AGENTS/Skills personalizados y notas,
usando manifiesto de hashes; rechaza planes modificados. Aplicar desde CLI como operador no sustituye
el futuro protocolo de aprobación del agente. No guarda conversaciones automáticamente.

## Python y GUI

`lambdaforge.agents.AgentServices` da respuestas tipadas fijadas al proyecto; IntegrationService
es el mismo backend de CLI/Console. `examples/agent_gui.py` muestra un cliente Tk opcional sin LLM.
Se prueban protocolo real, aislamiento, sesiones/ediciones externas, conflictos, assets y Work CPU
determinista con métricas, checkpoint, producto y HTML. La explicación inglesa detalla contratos,
fuentes oficiales y límites. WISDOM, clusters y evidencia científica existente permanecen intactos.
Los modelos exponen declaraciones y docstrings reales sin importar Torch; no se inventan shapes.
Contratos/miembros/provenance/consumidores de productos son metadata. Runs/curvas/logs seleccionados
y recursos de un cluster son consultas explícitas nativas; preview de texto verifica ownership e
integridad con límite de 64 KiB. Aprobaciones científicas, semántica, vault externo, cobertura total
de APIs y conversaciones completas siguen en el registro de pendientes.
