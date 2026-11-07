(function () {
  function renderExports() {
    const state = window.trackingState;
    if (!state || !state.job || !state.report) {
      window.setTimeout(renderExports, 500);
      return;
    }
    let bar = document.querySelector('.export-bar');
    if (!bar) {
      bar = document.createElement('section');
      bar.className = 'panel export-bar';
      const dashboard = document.querySelector('#dashboard');
      if (dashboard) dashboard.insertBefore(bar, dashboard.firstChild);
    }
    bar.innerHTML = '<strong>Экспорт</strong> ' +
      '<a href="/api/jobs/' + state.job.id + '/export/report.json" download>report.json</a> ' +
      '<a href="/api/jobs/' + state.job.id + '/export/tracks.csv" download>tracks.csv</a> ' +
      '<a href="/api/jobs/' + state.job.id + '/export/embeddings.json" download>embeddings.json</a>';
    window.setTimeout(renderExports, 1000);
  }
  renderExports();
})();
