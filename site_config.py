"""Loads site_config.json - the single place to edit navbar, brand, footer, page text and options.

* Edits show on the next page refresh (the file's modified-time is checked on every request).
* Missing keys never crash a page: they render as empty text.
* If the file is saved with invalid JSON, the last good version keeps working and the problem is
  shown as a red banner at the top of the page (see `error`).
"""
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / 'site_config.json'

# Friendly page names usable in site_config.json  ->  Flask endpoint names
PAGE_ENDPOINTS = {
    'home': 'home', 'login': 'login', 'register': 'registration', 'logout': 'logout',
    'dashboard': 'userhome', 'scan_url': 'prediction', 'feature_scan': 'predict_value',
    'models': 'model_page', 'chart': 'graph', 'history': 'history',
    'csv_upload': 'load_data', 'csv_view': 'view_data',
}


class Cfg(dict):
    """dict that also allows site.pages.login.title and returns '' for anything missing."""

    def __getattr__(self, key):
        if key.startswith('_'):
            raise AttributeError(key)
        return self[key] if key in self else Cfg()

    def __str__(self):
        return '' if not self else dict.__str__(self)


def _wrap(value):
    if isinstance(value, dict):
        return Cfg({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


class SiteConfig:
    def __init__(self, path=CONFIG_PATH):
        self.path = Path(path)
        self._mtime = None
        self._data = Cfg()
        self.error = None

    def get(self):
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self.error = f'{self.path.name} was not found - page text will be empty.'
            return self._data
        if mtime != self._mtime:
            self._mtime = mtime
            try:
                raw = json.loads(self.path.read_text(encoding='utf-8'))
                if not isinstance(raw, dict):
                    raise ValueError('the top level must be a { ... } object')
                self._data = _wrap(raw)
                self.error = None
            except Exception as exc:   # keep the last good config running
                self.error = f'{self.path.name} has a problem and was not applied: {exc}'
                print('[site_config]', self.error)
        return self._data


def safe_href(value):
    """Allow only http(s), mailto and site-relative links in config-defined URLs."""
    v = (value or '').strip()
    return v if v.startswith(('http://', 'https://', 'mailto:', '/')) else None
