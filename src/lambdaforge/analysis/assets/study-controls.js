/* A single anchored, searchable selector. No modal and no scientific state. */
window.LambdaForgeStudyControls = function () {
  const panel = document.createElement('div');
  panel.id = 'study-dropdown'; panel.className = 'semantic-dropdown';
  panel.setAttribute('popover', 'auto'); panel.setAttribute('role', 'group');
  const query = document.createElement('input'); query.type = 'search';
  query.id = 'study-dropdown-query'; query.setAttribute('aria-label', 'Search options');
  query.placeholder = 'Search names, aliases or categories…';
  const status = document.createElement('div'); status.className = 'dropdown-status muted';
  status.setAttribute('aria-live', 'polite');
  const list = document.createElement('div'); list.id = 'study-dropdown-options';
  const close = document.createElement('button'); close.textContent = 'Done';
  close.className = 'dropdown-done'; close.onclick = () => panel.hidePopover();
  panel.append(query, status, list, close); document.body.append(panel);
  let current, anchor;
  const text = message => window.LambdaForgeStudyLocale?.t(message) || message;
  const position = () => {
    if (!anchor || !panel.matches(':popover-open')) return;
    const r = anchor.getBoundingClientRect(), width = Math.min(520, innerWidth - 24);
    panel.style.width = width + 'px'; panel.style.maxHeight = Math.max(180, innerHeight - 32) + 'px';
    panel.style.left = Math.max(12, Math.min(r.left, innerWidth - width - 12)) + 'px';
    panel.style.top = Math.max(12, Math.min(r.bottom + 8, innerHeight - panel.offsetHeight - 12)) + 'px';
  };
  function render() {
    if (!current) return;
    const words = query.value.toLowerCase().split(/\s+/).filter(Boolean);
    const all = current.items().filter(item => words.every(word =>
      [item.value, item.label, item.description, ...(item.aliases || [])].join(' ').toLowerCase().includes(word)));
    const selected = new Set(current.selected()); list.replaceChildren();
    status.textContent = `${selected.size} ${text('selected')} · ${all.length} ${text('matches')}`
      + (all.length > 80 ? ' · ' + text('Type to narrow the results') : '');
    for (const item of all.slice(0, 80)) {
      const row = document.createElement('label'); row.className = 'dropdown-option';
      const checkbox = document.createElement('input'); checkbox.type = 'checkbox';
      checkbox.checked = selected.has(item.value); checkbox.dataset.choice = item.value;
      const description = document.createElement('span'), title = document.createElement('strong');
      title.textContent = item.label; if(item.authored)title.dataset.authored='true';const note = document.createElement('small');
      note.textContent = item.description || item.value; description.append(title, note); row.append(checkbox, description);
      checkbox.onchange = () => {
        current.change(item.value, checkbox.checked);
        // Do not replace the activated checkbox while its click is still dispatching.
        // This also preserves keyboard focus while a multi-selection remains open.
        if(current.multiple){const values=new Set(current.selected());
          list.querySelectorAll('input').forEach(input=>input.checked=values.has(input.dataset.choice));
          status.textContent=`${values.size} ${text('selected')} · ${all.length} ${text('matches')}`;}
        if (!current.multiple) panel.hidePopover();
      };
      list.append(row);
    }
    if (!all.length) {const empty = document.createElement('p'); empty.textContent = text('No matching options.'); list.append(empty);}
    position();
  }
  query.oninput = render;
  query.onkeydown = event => {
    if (['ArrowDown', 'Enter'].includes(event.key)) {event.preventDefault();
      const first = list.querySelector('input'); if (event.key === 'Enter') first?.click(); else first?.focus();}
  };
  list.onkeydown = event => {
    if (!['ArrowDown', 'ArrowUp'].includes(event.key)) return;
    event.preventDefault(); const inputs = [...list.querySelectorAll('input')], index = inputs.indexOf(document.activeElement);
    if (index === 0 && event.key === 'ArrowUp') query.focus();
    else inputs[Math.max(0, Math.min(inputs.length - 1, index + (event.key === 'ArrowDown' ? 1 : -1)))]?.focus();
  };
  panel.addEventListener('toggle', () => {
    anchor?.setAttribute('aria-expanded', String(panel.matches(':popover-open')));
  });
  window.addEventListener('resize', position);
  document.addEventListener('scroll', position, true);
  return {open(button, options) {
    if (panel.matches(':popover-open')) panel.hidePopover();
    anchor?.setAttribute('aria-expanded', 'false'); anchor = button; current = options;
    button.setAttribute('aria-controls', panel.id); button.setAttribute('aria-haspopup', 'true');
    panel.setAttribute('aria-label', options.title || 'Choose evidence');
    query.value = ''; panel.showPopover(); render(); query.focus();
  }};
};

/* Scoped presentation styles; the Run/epoch dashboard is deliberately unaffected. */
(() => {
  const style = document.createElement('style'); style.textContent = `
input:not([type=hidden]):not([type=checkbox]):not([type=file]),textarea{
  min-height:46px;min-width:240px;padding:12px 14px;font:inherit;background:var(--bg);
  color:var(--text);border:1px solid var(--line);border-radius:9px}
input[type=search]{width:min(100%,440px)}button,select{min-height:42px}
button:focus-visible,select:focus-visible,input:focus-visible,summary:focus-visible{
  outline:2px solid var(--accent);outline-offset:3px}
.semantic-dropdown{position:fixed;inset:auto;margin:0;padding:14px;background:var(--panel);
  color:var(--text);border:1px solid #527396;border-radius:13px;box-shadow:0 18px 70px #000a;overflow:auto;z-index:20}
.semantic-dropdown::backdrop{background:transparent}.semantic-dropdown input[type=search]{width:100%;min-width:0}
.dropdown-status{font-size:.8rem;padding:10px 2px}.dropdown-option{display:flex!important;align-items:flex-start!important;
  flex-direction:row!important;gap:12px;padding:12px 8px;border-radius:8px;cursor:pointer}
.dropdown-option:hover,.dropdown-option:focus-within{background:#58a6ff17}.dropdown-option input{margin-top:5px;accent-color:var(--accent);width:18px;height:18px}
.dropdown-option strong,.dropdown-option small{display:block;overflow-wrap:anywhere;white-space:normal}
.dropdown-option small{color:var(--muted);font-size:.8rem;margin-top:3px}
#study-dropdown-options{max-height:320px;overflow:auto}.dropdown-done{width:100%;margin-top:10px}
#research-questions{display:block}.dropdown-option input[type=checkbox]{min-width:18px;flex:none}
.card-carousel{display:flex;gap:14px;overflow-x:auto;scroll-snap-type:x mandatory;padding:12px 16px 20px}
.card-carousel>.finding{flex:0 0 min(360px,85vw);scroll-snap-align:start;margin:0;display:flex;flex-direction:column}
.card-carousel>.finding button{margin-top:auto;align-self:flex-start}.carousel-controls{display:flex;align-items:center;gap:12px;padding:12px 16px}
.carousel-controls small{color:var(--muted)}.info-button{border-radius:50%;width:42px;padding:5px}
.overview-summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin:16px 0}
.overview-summary .card strong{font-size:1rem}.overview-summary .card p{margin:8px 0 0;color:var(--muted);font-size:.85rem}
#overview-states{min-height:350px}.overview-warning{padding:14px 18px;border-left:3px solid #ffa657;background:#ffa6570a;white-space:normal}
#overview-conditional-table td{white-space:normal;text-align:left}#overview-conditional-table th{text-align:left}
.suggestion-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px;padding:16px}
.suggestion-grid .finding{margin:0}.suggestion-grid p{color:var(--muted)}.language-control{display:flex;gap:8px;align-items:center}
.panel:is(details){height:auto!important;resize:none;min-height:0}.panel:is(details)>summary{cursor:pointer}
#study-overview .plot{min-height:350px}
.panel:has(.semantic-dropdown){overflow:visible}.study-local-help{max-width:75ch;line-height:1.7}
.finding-pagination{display:flex;gap:12px;align-items:center;padding:14px 0}
@media(max-width:650px){.shell{padding:12px}.tools{padding:12px}input[type=search]{min-width:0;width:100%}
  .semantic-dropdown{max-width:calc(100vw - 24px)}.cards{grid-template-columns:1fr 1fr}.tab{white-space:nowrap}}
`; document.head.append(style);
})();
