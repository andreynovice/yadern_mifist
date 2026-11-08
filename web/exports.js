(function () {
  function renderExports() {
    const state = window.trackingState;
    if (!state || !state.job || !state.report) { window.setTimeout(renderExports, 500); return; }
    let bar = document.querySelector('.export-bar');
    if (!bar) { bar = document.createElement('div'); bar.className = 'export-bar'; document.querySelector('#dashboard').prepend(bar); }
    bar.innerHTML = '<span class="muted">Экспорт:</span> <a href="/api/jobs/' + state.job.id + '/export/tracks_projection.json" download>tracks_projection.json</a> <a href="/api/jobs/' + state.job.id + '/export/embeddings.json" download>embeddings.json</a>';
    window.setTimeout(renderExports, 1000);
  }
  renderExports();
}());


