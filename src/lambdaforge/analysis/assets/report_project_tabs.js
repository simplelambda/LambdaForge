/* Native parent owns only scoped asset transport; no consumer script enters this realm. */
(() => {
 const groups = JSON.parse(document.getElementById('lf-project-sections').textContent);
 const key = 'lambdaforge:project-sections:' + JSON.parse(document.getElementById('lf-study-data').textContent).report_id;
 let preferences = {}; try { preferences = JSON.parse(localStorage.getItem(key) || '{}'); } catch (_) {}
 const policy = "default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' data: blob:; style-src 'unsafe-inline' data: blob:; img-src data: blob:; font-src data: blob:; media-src data: blob:; worker-src blob:; connect-src 'none'; form-action 'none'; base-uri 'none';";
 const active = new Map();
 function dispose(index) {
   const state = active.get(index); if (!state) return;
   for (const controller of state.requests.values()) controller.abort();
   state.requests.clear(); state.port?.close(); active.delete(index);
   state.frame.removeAttribute('srcdoc'); delete state.frame.dataset.document;
 }
 function load(index) {
  const panel = document.getElementById('project-section-' + index);
  if (panel.hidden) { dispose(index); return; }
  const selected = panel.querySelector('select').value, frame = panel.querySelector('iframe');
  if (frame.dataset.document === selected) return;
  dispose(index);
  const doc = groups[index].documents[Number(selected)];
  const status = panel.querySelector('.lf-report-asset-status');
  status.textContent = doc.error ? 'Presentation unavailable: ' + doc.error :
    (doc.bundle && location.protocol === 'file:' ? 'Full-resolution assets need the native offline viewer: lf reports serve /path/to/report.html. No data was omitted.' : '');
  if (doc.error) return;
  const state = {frame, requests: new Map(), port: null}; active.set(index, state);
  frame.onload = () => {
    if (active.get(index) !== state || !doc.bundle) return;
    const channel = new MessageChannel(); state.port = channel.port1;
    channel.port1.onmessage = async event => {
      const message = event.data;
      if (!message || typeof message !== 'object') return;
      if (Number.isInteger(message.cancel)) { state.requests.get(message.cancel)?.abort(); return; }
      const request = message.request;
      if (!Number.isSafeInteger(request) || request <= 0 || state.requests.has(request)) return;
      const row = doc.bundle.assets.find(a => a.id === message.asset);
      const fail = (code, text) => channel.port1.postMessage({request, error: {code, message: text}});
      if (!row) { fail('UNDECLARED_ASSET', 'Only assets declared by this bundle can be loaded.'); return; }
      if (!/^https?:$/.test(location.protocol)) {
        const text = 'Lazy assets require the native local viewer: lf reports serve /path/to/report.html. file:// fetch is not supported.';
        status.textContent = text; fail('VIEWER_REQUIRED', text); return;
      }
      if (state.requests.size >= 2) { fail('VIEW_BUSY', 'Two assets are loading; await or cancel a load first.'); return; }
      const controller = new AbortController(); state.requests.set(request, controller);
      status.textContent = 'Loading ' + row.id + ' (' + (row.size_bytes / 1048576).toFixed(1) + ' MiB)…';
      try {
        const url = new URL('.assets/' + doc.bundle.index + '/' + encodeURIComponent(row.id), location.href);
        const response = await fetch(url, {signal: controller.signal, credentials: 'omit', headers: {'X-LambdaForge-Report': location.pathname.split('/')[1]}});
        if (!response.ok) {
          const error = await response.json(); throw Object.assign(new Error(error.message), {code: error.code});
        }
        const buffer = await response.arrayBuffer();
        if (buffer.byteLength !== row.size_bytes) throw Object.assign(new Error('Asset byte count differs from its manifest.'), {code: 'CORRUPT_ASSET'});
        const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', buffer))].map(b => b.toString(16).padStart(2, '0')).join('');
        if (hash !== row.content_sha256) throw Object.assign(new Error('Asset checksum differs from finalized content.'), {code: 'CORRUPT_ASSET'});
        if (active.get(index) === state && !controller.signal.aborted) {
          channel.port1.postMessage({request, buffer}, [buffer]); status.textContent = 'Loaded ' + row.id + ' · exact finalized bytes';
        }
      } catch (error) {
        if (active.get(index) === state) {
          const text = error.name === 'AbortError' ? 'Asset loading cancelled.' : error.message;
          status.textContent = text; fail(error.code || error.name || 'ASSET_UNAVAILABLE', text);
        }
      } finally { state.requests.delete(request); }
    };
    // The iframe has an opaque origin; the channel is transferred only to this exact Window.
    frame.contentWindow.postMessage({type: 'lf-report-port'}, '*', [channel.port2]);
  };
  const raw = Uint8Array.from(atob(doc.content), c => c.charCodeAt(0));
  frame.srcdoc = '<!doctype html><meta http-equiv="Content-Security-Policy" content="' + policy + '">' + new TextDecoder().decode(raw);
  frame.dataset.document = selected;
 }
 groups.forEach((group, index) => {
  const panel = document.getElementById('project-section-' + index), select = panel.querySelector('select');
  if ([...select.options].some(option => option.value === preferences[index])) {
    select.value = preferences[index]; select.dispatchEvent(new Event('change'));
  }
  select.addEventListener('change', () => {
   preferences[index] = select.value;
   try { localStorage.setItem(key, JSON.stringify(preferences)); } catch (_) {}
   load(index);
  });
  new MutationObserver(() => load(index)).observe(panel, {attributes:true, attributeFilter:['hidden']});
  load(index);
 });
 window.addEventListener('pagehide', () => [...active.keys()].forEach(dispose));
})();
