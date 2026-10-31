"""LIME + SHAP for the full RF + XGBoost + Deep + meta-fusion probability.

* Always runs (the page no longer depends on an opt-in checkbox).
* Real training rows are used as reference when models/xai_background.csv exists;
  otherwise a clearly-labelled synthetic reference built from the training medians.
* If LIME or SHAP fails for any reason, a feature-substitution ("occlusion") explanation is
  shown in that panel, labelled as a fallback, so the section is never blank.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from feature_info import label_of, URL_MEASURABLE

CAN_BE_NEGATIVE = {'domain_age', 'domain_registration_length'}


def build_background(features, medians, models_dir):
    """Returns (background ndarray, mode text, warning or None)."""
    ref = Path(models_dir) / 'xai_background.csv'
    warning = None
    if ref.exists():
        try:
            back = pd.read_csv(ref)
            absent = [c for c in features if c not in back.columns]
            if absent:
                warning = ('models/xai_background.csv is missing columns (' + ', '.join(absent[:5]) +
                           '); using an approximate reference instead.')
            else:
                bg = back[features].apply(pd.to_numeric, errors='coerce').dropna()
                if len(bg) >= 8:
                    if len(bg) > 200:
                        bg = bg.sample(200, random_state=23)
                    return bg.to_numpy(dtype=float), 'real training rows', None
                warning = 'models/xai_background.csv has fewer than 8 usable rows; using an approximate reference.'
        except Exception as exc:  # unreadable CSV must not break scanning
            warning = f'Could not read models/xai_background.csv ({type(exc).__name__}); using an approximate reference.'
    med = np.array([float(medians.get(f, 0.0)) for f in features], dtype=float)
    rng = np.random.default_rng(23)
    scale = np.maximum(np.abs(med) * 0.6, 1.0)
    bg = np.vstack((med, med + rng.normal(0, scale, size=(79, len(features)))))
    for j, f in enumerate(features):
        if f not in CAN_BE_NEGATIVE:
            bg[:, j] = np.clip(bg[:, j], 0, None)
    warning = warning or ('Approximate reference: models/xai_background.csv was not found, so explanations use '
                          'a synthetic reference built from training medians. Add real training rows for sharper results.')
    return bg, 'approximate reference', warning


def _items(names, values):
    pairs = sorted(zip(names, values), key=lambda x: abs(x[1]), reverse=True)
    return [{'name': n, 'value': round(float(v), 5)} for n, v in pairs]


def _occlusion(predict_fn, point, bg_median, features):
    """Fallback: change in phishing probability when ONE feature is replaced by its reference median."""
    p0 = float(predict_fn(point.reshape(1, -1))[0])
    batch = np.tile(point, (len(features), 1))
    for i in range(len(features)):
        batch[i, i] = bg_median[i]
    deltas = p0 - np.asarray(predict_fn(batch), dtype=float)
    return _items(features, deltas)


def explain_instance(predict_fn, point, features, background, n_lime=300, n_shap=150):
    point = np.asarray(point, dtype=float).reshape(-1)
    out = {'lime': [], 'shap': [], 'base_value': None, 'warning': None, 'error': None,
           'notes': {'lime': '', 'shap': ''}, 'seconds': 0.0}
    t0 = time.time()
    errors = []

    # ---------------- LIME
    try:
        from lime.lime_tabular import LimeTabularExplainer

        def proba(a):
            p = np.asarray(predict_fn(np.asarray(a, dtype=float)), dtype=float)
            return np.column_stack((1 - p, p))

        lime_exp = LimeTabularExplainer(background, feature_names=list(features),
                                        class_names=['Legitimate', 'Phishing'], mode='classification',
                                        discretize_continuous=False, sample_around_instance=True,
                                        random_state=23)
        res = lime_exp.explain_instance(point, proba, num_features=len(features), num_samples=n_lime,
                                        labels=(1,))
        # LIME weights are slopes on standardised features. Convert to a contribution relative to the
        # average reference row (weight x standardised value) so the sign means the same as in SHAP:
        # positive = pushes toward phishing, negative = pushes toward legitimate.
        mean = np.asarray(lime_exp.scaler.mean_, dtype=float)
        scale = np.asarray(lime_exp.scaler.scale_, dtype=float)
        scale = np.where(scale == 0, 1.0, scale)
        contrib = np.zeros(len(features))
        for idx, w in res.local_exp[1]:
            contrib[idx] = w * (point[idx] - mean[idx]) / scale[idx]
        out['lime'] = _items(features, contrib)
        out['notes']['lime'] = 'LIME'
    except Exception as exc:
        errors.append('LIME: ' + type(exc).__name__ + ' - ' + str(exc)[:120])

    # ---------------- SHAP
    try:
        import shap
        k = min(10, len(background))
        explainer = shap.KernelExplainer(lambda a: np.asarray(predict_fn(np.asarray(a, dtype=float)), dtype=float),
                                         shap.kmeans(background, k))
        vals = explainer.shap_values(point.reshape(1, -1), nsamples=n_shap, silent=True, l1_reg=0)
        if isinstance(vals, list):
            vals = vals[-1]
        arr = np.asarray(vals)
        if arr.ndim == 3:
            arr = arr[..., -1]
        arr = arr.reshape(-1)
        if arr.size != len(features):
            raise ValueError(f'unexpected SHAP shape {arr.shape}')
        out['shap'] = _items(features, arr)
        out['notes']['shap'] = 'SHAP'
        ev = np.asarray(explainer.expected_value).reshape(-1)
        out['base_value'] = float(ev[-1])
    except Exception as exc:
        errors.append('SHAP: ' + type(exc).__name__ + ' - ' + str(exc)[:120])

    # ---------------- fallback so neither panel is ever empty
    if not out['lime'] or not out['shap']:
        try:
            fb = _occlusion(predict_fn, point, np.median(background, axis=0), features)
            if not out['lime']:
                out['lime'], out['notes']['lime'] = fb, 'fallback'
            if not out['shap']:
                out['shap'], out['notes']['shap'] = fb, 'fallback'
        except Exception as exc:
            errors.append('Fallback explanation failed: ' + type(exc).__name__)
    if errors:
        out['error'] = ' | '.join(errors)
    out['seconds'] = round(time.time() - t0, 1)
    return out


def decorate(items, feature_values, mode, top=10):
    """Add plain-English label, actual value, bar size and direction for the template."""
    items = items[:top]
    peak = max([abs(i['value']) for i in items] + [1e-9])
    rows = []
    for i in items:
        n = i['name']
        v = feature_values.get(n)
        rows.append({
            'name': n, 'label': label_of(n), 'value': i['value'],
            'actual': ('%g' % v) if isinstance(v, (int, float)) else '',
            'estimated': mode == 'url' and n not in URL_MEASURABLE,
            'pct': round(abs(i['value']) / peak * 100, 1),
            'dir': 'phishing' if i['value'] >= 0 else 'legit',
        })
    return rows
