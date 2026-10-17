/* Public managed report asset API. Runs inside the opaque-origin project iframe. */
(() => {
  let port, sequence = 0, disposed = false;
  const pending = new Map();
  let connect;
  const ready = new Promise(resolve => { connect = resolve; });
  function receive(event) {
    if (disposed) return;
    if (event.source !== parent || event.data?.type !== 'lf-report-port' || event.ports.length !== 1 || port) return;
    port = event.ports[0];
    port.onmessage = event => {
      const row = pending.get(event.data?.request);
      if (!row) return;
      pending.delete(event.data.request); row.cleanup();
      if (event.data.error) {
        const error = new Error(event.data.error.message); error.code = event.data.error.code;
        row.reject(error);
      } else row.resolve(event.data.buffer);
    };
    connect();
  }
  window.addEventListener('message', receive);
  function load(id, {signal} = {}) {
    return new Promise((resolve, reject) => {
      if (disposed || signal?.aborted) { reject(new DOMException('Asset loading cancelled', 'AbortError')); return; }
      if (pending.size >= 2) {
        reject(Object.assign(new Error('Two assets are loading; await/cancel one first.'), {code: 'VIEW_BUSY'})); return;
      }
      const request = ++sequence;
      const abort = () => {
        if (!pending.delete(request)) return;
        port?.postMessage({cancel: request}); cleanup();
        reject(new DOMException('Asset loading cancelled', 'AbortError'));
      };
      const cleanup = () => signal?.removeEventListener('abort', abort);
      pending.set(request, {resolve, reject, cleanup});
      signal?.addEventListener('abort', abort, {once: true});
      ready.then(() => { if (pending.has(request)) port.postMessage({request, asset: id}); });
    });
  }
  const dispose = () => {
    disposed = true; window.removeEventListener('message', receive);
    for (const [request, row] of pending) {
      port?.postMessage({cancel: request}); row.cleanup();
      row.reject(new DOMException('Report view disposed', 'AbortError'));
    }
    pending.clear(); port?.close();
  };
  window.addEventListener('pagehide', dispose);
  window.lambdaforgeReport = Object.freeze({version: 1, ready, assets: Object.freeze(__LF_ASSETS__), load, dispose});
})();
