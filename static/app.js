// Loading state on submit
document.querySelectorAll('form[data-loading]').forEach(form => form.addEventListener('submit', () => {
  const btn = form.querySelector('button[type="submit"]');
  if (btn) { btn.disabled = true; btn.textContent = form.dataset.loading || 'Working…'; }
}));

// URL page: example chips fill the URL box
document.querySelectorAll('[data-fill]').forEach(b => b.addEventListener('click', () => {
  const box = document.getElementById('url');
  if (box) { box.value = b.dataset.fill; box.focus(); }
}));

// Feature page: preset buttons fill all 15 inputs
document.querySelectorAll('[data-preset]').forEach(b => b.addEventListener('click', () => {
  const kind = b.dataset.preset;
  const preset = (window.PHISH_PRESETS || {})[kind];
  document.querySelectorAll('form input[data-default], form select[data-default]').forEach(el => {
    if (kind === 'default') { el.value = el.dataset.default; return; }
    if (preset && Object.prototype.hasOwnProperty.call(preset, el.name)) el.value = String(preset[el.name]);
  });
}));

// After a scan, jump to the result
(() => {
  const target = document.getElementById('results');
  if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
})();
