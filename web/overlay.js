// Draw report boxes on the image area of each video.
(function () {
  function draw() {
    const shared = window.trackingState;
    if (!shared || !shared.report) { requestAnimationFrame(draw); return; }
    document.querySelectorAll('video[data-camera]').forEach((video) => {
      const card = video.closest('.video-card');
      let canvas = card.querySelector('canvas');
      if (!canvas) { canvas = document.createElement('canvas'); card.appendChild(canvas); }
      const width = video.videoWidth || 1;
      const height = video.videoHeight || 1;
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
      const videoRect = video.getBoundingClientRect();
      const cardRect = card.getBoundingClientRect();
      const scale = Math.min(videoRect.width / width, videoRect.height / height);
      const drawnWidth = width * scale;
      const drawnHeight = height * scale;
      const offsetX = (videoRect.left - cardRect.left) + (videoRect.width - drawnWidth) / 2;
      const offsetY = (videoRect.top - cardRect.top) + (videoRect.height - drawnHeight) / 2;
      canvas.style.left = offsetX + 'px'; canvas.style.top = offsetY + 'px';
      canvas.style.width = drawnWidth + 'px'; canvas.style.height = drawnHeight + 'px';
      const context = canvas.getContext('2d'); context.setTransform(1, 0, 0, 1, 0, 0); context.clearRect(0, 0, width, height);
      const camera = Number(video.dataset.camera);
      const metadata = (shared.report.cameras || []).find((item) => String(item.camera) === String(camera)) || {};
      const frame = Math.round(video.currentTime * (metadata.fps || 25));
      (shared.report.tracks || []).forEach((track) => {
        if (shared.selectedTrack && Number(track.global_id) !== Number(shared.selectedTrack.global_id)) return;
        const observations = (track.boxes || []).filter((item) => String(item.camera) === String(camera)).sort((a, b) => a.frame - b.frame);
        let previous = null; let next = null;
        for (const item of observations) { if (item.frame <= frame) previous = item; if (item.frame >= frame) { next = item; break; } }
        if (!previous || !next || frame - previous.frame > 8 || next.frame - frame > 8) return;
        let box = previous.box;
        if (next !== previous && next.frame !== previous.frame) { const ratio = (frame - previous.frame) / (next.frame - previous.frame); box = previous.box.map((value, index) => value + (next.box[index] - value) * ratio); }
        const [x1, y1, x2, y2] = box; const selected = shared.selectedTrack && Number(track.global_id) === Number(shared.selectedTrack.global_id);
        context.strokeStyle = selected ? '#ffd166' : '#28df85'; context.lineWidth = selected ? 5 : 3; context.strokeRect(x1, y1, x2 - x1, y2 - y1);
        context.fillStyle = context.strokeStyle; context.font = 'bold 66px sans-serif'; context.fillText('ID ' + track.global_id, x1, Math.max(70, y1 - 20));
      });
    });
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
})();
