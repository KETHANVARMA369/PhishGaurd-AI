"""Flask UI for the dataset_B_05_2020 notebook's exported models (15-feature model).
No retraining, no network requests to submitted URLs.

Result pipeline
  URL scan     : 6/15 features measured from the link  +  domain reputation  +  URL warning signs  ->  verdict
  Manual scan  : the 15 trained features typed by the user                                         ->  model verdict
  Both         : LIME + SHAP (always), plain-English summary (always), optional Groq text.
"""
import math
import os
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error as MySQLError

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from feature_extraction import url_to_feature_vector, extract_lexical_features
from feature_info import CANONICAL_15, URL_MEASURABLE, label_of, manual_fields, presets
from hybrid_engine import (normalize_url, feature_coverage, google_safe_browsing,
                           calibrated_probability, final_verdict, recommendations)
from url_trust import trust_assessment, url_red_flags
from xai_engine import build_background, explain_instance, decorate
from summary import build_summary
from groq_advice import generate_groq_advice
from site_config import SiteConfig, PAGE_ENDPOINTS, safe_href

# ============================================================================
# App / config
# ============================================================================
ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-only-change-me-before-deploy')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.getenv('COOKIE_SECURE', '0') == '1'

SITE = SiteConfig()

MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

# ============================================================================
# Model state
# ============================================================================
art = None
model = None
load_error = None
FEATURES = []
MEDIANS = {}
feature_warning = None
_bg_cache = {}


def set_models(artifact, keras_model, error=None):
    """Install loaded artifacts (also used by the tests with stand-in models)."""
    global art, model, load_error, FEATURES, MEDIANS, feature_warning
    art, model, load_error = artifact, keras_model, error
    FEATURES = list(artifact['selected_features']) if artifact is not None else []
    MEDIANS = dict(artifact['train_medians']) if artifact is not None else {}
    feature_warning = None
    if FEATURES and set(FEATURES) != set(CANONICAL_15):
        feature_warning = (
            f'The model files in models/ were trained on {len(FEATURES)} features, '
            f'but the notebook "15 Features" exports exactly {len(CANONICAL_15)}. '
            'The manual form shows the 15 notebook features; any other feature the old '
            'model needs is filled with its training median. Re-export '
            'msra_novelty_highest.joblib and the .keras file from the 15-feature notebook.'
        )
    _bg_cache.clear()


def load_models():
    try:
        import tensorflow as tf
        bundle_path = MODEL_DIR / 'msra_novelty_highest.joblib'
        if not bundle_path.exists():
            raise FileNotFoundError(
                'Missing models/msra_novelty_highest.joblib. Export it from the notebook.'
            )
        artifact = joblib.load(bundle_path)
        model_path = next(
            (p for p in (MODEL_DIR / 'msra_bilstm_afg_deep.keras',
                         MODEL_DIR / 'msra_bilstm_afg_deep.h5')
             if p.exists()),
            None,
        )
        if model_path is None:
            raise FileNotFoundError('Missing models/msra_bilstm_afg_deep.keras (or .h5).')
        keras_model = tf.keras.models.load_model(str(model_path), compile=False)
        for key in ('rf', 'xgb', 'meta_fusion', 'scaler', 'selected_features', 'train_medians'):
            if key not in artifact:
                raise ValueError('Model bundle missing required key: ' + key)
        set_models(artifact, keras_model)
    except Exception as exc:
        set_models(None, None, str(exc))
        print('[Model load]', load_error)


load_models()


def model_ready():
    return art is not None and model is not None


def predict_matrix(matrix):
    """Returns phishing probabilities for original-scale selected-feature rows."""
    if not model_ready():
        raise RuntimeError(load_error or 'Model not loaded')
    X = pd.DataFrame(np.asarray(matrix, dtype=float), columns=FEATURES)
    p_rf = np.asarray(art['rf'].predict_proba(X)[:, 1], dtype=float)
    p_xgb = np.asarray(art['xgb'].predict_proba(X)[:, 1], dtype=float)
    scaled = art['scaler'].transform(X).astype(np.float32)
    seq = scaled.reshape(len(X), len(FEATURES), 1)
    p_deep = np.asarray(model.predict(seq, verbose=0, batch_size=512)).reshape(-1).astype(float)
    meta = np.column_stack((p_rf, p_xgb, p_deep))
    p_final = np.asarray(art['meta_fusion'].predict_proba(meta)[:, 1], dtype=float)
    return np.clip(p_final, 0, 1)


def infer(X):
    p = float(predict_matrix(X.to_numpy(dtype=float))[0])
    return {'label': int(p >= 0.5), 'prob': p, 'confidence': p if p >= 0.5 else 1 - p}


def validate_url(value):
    return normalize_url(value)


def manual_feature_names():
    """The inputs shown on the manual page: exactly the notebook's 15."""
    return [f for f in CANONICAL_15 if not FEATURES or f in FEATURES]


def parse_manual_form(form):
    """Read the 15 trained features from the form. Raises ValueError with a friendly message."""
    values = {}
    for f in manual_feature_names():
        raw = (form.get(f) or '').strip()
        if raw == '':
            raise ValueError(f"Please enter a value for '{label_of(f)}'.")
        try:
            v = float(raw)
        except ValueError:
            raise ValueError(f"'{label_of(f)}' must be a number (got '{raw[:30]}').")
        if not math.isfinite(v):
            raise ValueError(f"'{label_of(f)}' must be a finite number.")
        values[f] = v
    return values


def get_background():
    ref = MODEL_DIR / 'xai_background.csv'
    key = (ref.stat().st_mtime if ref.exists() else None, tuple(FEATURES))
    if _bg_cache.get('key') != key:
        _bg_cache['key'] = key
        _bg_cache['val'] = build_background(FEATURES, MEDIANS, MODEL_DIR)
    return _bg_cache['val']


def run_xai(X, mode):
    """LIME + SHAP for every result. Never raises: problems are reported inside the returned dict."""
    out = {'lime_rows': [], 'shap_rows': [], 'warning': None, 'error': None, 'notes': {},
           'reference': '', 'base_value': None, 'seconds': 0}
    if not model_ready():
        out['error'] = 'Model not loaded, so no explanation could be computed.'
        return out
    try:
        bg, ref_mode, warn = get_background()
        res = explain_instance(predict_matrix, X[FEATURES].to_numpy(dtype=float)[0], FEATURES, bg)
        values = X.iloc[0].to_dict()
        out.update(
            lime_rows=decorate(res['lime'], values, mode),
            shap_rows=decorate(res['shap'], values, mode),
            warning=warn, error=res['error'], notes=res['notes'],
            reference=ref_mode, base_value=res['base_value'], seconds=res['seconds'],
        )
    except Exception as exc:
        out['error'] = 'Explanation could not be generated: ' + type(exc).__name__ + ' - ' + str(exc)[:160]
    return out


def verdict_to_history_label(kind):
    return {'danger': 1, 'unknown': 2}.get(kind, 0)


# ============================================================================
# Database layer
# ============================================================================
def _db_config():
    return {
        'host':       os.getenv('DB_HOST', 'localhost'),
        'user':       os.getenv('DB_USER', 'root'),
        'password':   os.getenv('DB_PASSWORD', 'root'),
        'database':   os.getenv('DB_NAME', 'db'),
        'port':       int(os.getenv('DB_PORT', '3306')),
        'charset':    'utf8mb4',
        'autocommit': False,
    }


def db_connect():
    return mysql.connector.connect(**_db_config())


def query_one(sql, args=()):
    conn = db_connect()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, args)
        return cur.fetchone()
    finally:
        conn.close()


def execute(sql, args=()):
    """INSERT/UPDATE/DELETE with commit. Raises mysql.connector.Error on failure."""
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, args)
        conn.commit()
        return cur.lastrowid
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def auth_required():
    return redirect(url_for('login')) if not session.get('user_email') else None


def save_history(source, identifier, pred):
    try:
        execute(
            'INSERT INTO history (user_email, source, identifier, prediction_label, probability, created_at) '
            'VALUES (%s, %s, %s, %s, %s, %s)',
            (session['user_email'], source, identifier,
             pred['history_label'], round(pred['raw_prob'], 4), datetime.now()),
        )
    except Exception as exc:
        log.warning('[History save failed] %s', exc)


def _friendly_db_error(exc):
    """Map a mysql.connector.Error to a friendly user message."""
    if not isinstance(exc, MySQLError):
        return 'Unexpected error. Please try again.'
    if exc.errno == 1062:
        return 'An account with that email already exists.'
    if exc.errno in (1048, 1364):
        return 'A required field was missing. Please check the form.'
    if exc.errno == 1406:
        return ('A value was too long for a database column. '
                'Check that user3.password is VARCHAR(255).')
    if exc.errno == 1146:
        return 'Database tables are missing. Run schema.sql to create them.'
    if exc.errno == 1045:
        return 'Database rejected the credentials. Check DB_USER / DB_PASSWORD.'
    if exc.errno in (2003, 2005, 2013):
        return 'Database is unavailable. Please try again later.'
    return 'Database error. Please try again.'


def page_url(name):
    """Friendly page name -> URL. Unknown names give '#', never an error."""
    endpoint = PAGE_ENDPOINTS.get(name)
    try:
        return url_for(endpoint) if endpoint else '#'
    except Exception:
        return '#'


def build_nav(site):
    key = 'logged_in' if session.get('user_email') else 'logged_out'
    items = []
    for it in site.get('nav', {}).get(key, []) or []:
        href = safe_href(it.get('url')) if it.get('url') else page_url(it.get('page', ''))
        if href and href != '#' and it.get('label'):
            items.append({'label': it['label'], 'href': href, 'style': it.get('style', '')})
    return items


@app.context_processor
def globals_for_templates():
    site = SITE.get()
    return {
        'model_ready':     model_ready(),
        'load_error':      load_error,
        'feature_names':   FEATURES,
        'feature_warning': feature_warning,
        'manual_groups':   manual_fields(MEDIANS),
        'manual_names':    manual_feature_names(),
        'example_presets': presets(),
        'site':            site,
        'site_error':      SITE.error,
        'nav_items':       build_nav(site),
        'page_url':        page_url,
    }


# ============================================================================
# Routes — auth
# ============================================================================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/registration', methods=['GET', 'POST'])
def registration():
    if request.method == 'POST':
        name     = (request.form.get('name') or '').strip()
        email    = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''
        confirm  = request.form.get('conformpassword') or ''

        problems = []
        if not name:
            problems.append('Please enter your name.')
        elif len(name) > 50:
            problems.append('Name must be 50 characters or fewer.')

        if not email or '@' not in email or '.' not in email.split('@')[-1]:
            problems.append('Please enter a valid email address.')
        elif len(email) > 100:
            problems.append('Email must be 100 characters or fewer.')

        if len(password) < 8:
            problems.append('Password must be at least 8 characters.')
        elif password != confirm:
            problems.append('Passwords do not match.')

        if problems:
            for p in problems:
                flash(p, 'error')
            return render_template('registration.html', form={'name': name, 'email': email})

        try:
            password_hash = generate_password_hash(password)
            execute(
                'INSERT INTO user3 (name, email, password) VALUES (%s, %s, %s)',
                (name, email, password_hash),
            )
            flash('Account created. Please sign in.', 'success')
            return redirect(url_for('login'))
        except MySQLError as exc:
            log.exception('Registration DB error: errno=%s msg=%s', exc.errno, exc.msg)
            flash(_friendly_db_error(exc), 'error')
        except Exception:
            log.exception('Registration unexpected error')
            flash('Registration failed. Please try again.', 'error')

        return render_template('registration.html', form={'name': name, 'email': email})

    return render_template('registration.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''

        if not email or not password:
            flash('Please enter both email and password.', 'error')
            return render_template('login.html', form={'email': email})

        try:
            user = query_one(
                'SELECT name, email, password FROM user3 WHERE email=%s', (email,)
            )
            ok = False
            if user:
                stored = user.get('password') or ''
                if stored.startswith(('scrypt:', 'pbkdf2:', 'argon2')):
                    ok = check_password_hash(stored, password)
                else:
                    ok = stored == password  # legacy plaintext rows only
            if ok:
                session.clear()
                session['user_email'] = user['email']
                session['user_name']  = user['name']
                return redirect(url_for('userhome'))
            flash('Invalid email or password.', 'error')
        except MySQLError as exc:
            log.exception('Login DB error: errno=%s msg=%s', exc.errno, exc.msg)
            flash(_friendly_db_error(exc), 'error')
        except Exception:
            log.exception('Login unexpected error')
            flash('Login failed. Please try again.', 'error')

        return render_template('login.html', form={'email': email})

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


@app.route('/userhome')
def userhome():
    guard = auth_required()
    if guard:
        return guard
    return render_template('userhome.html')


# ============================================================================
# Routes — prediction
# ============================================================================
def render_result(template, source='url'):
    guard = auth_required()
    if guard:
        return guard
    if request.method != 'POST':
        return render_template(template)
    if not model_ready():
        return render_template(template, error=load_error or 'Missing model files.')

    mode = 'url' if source == 'url' else 'features'
    try:
        coverage = None
        trust = None
        flags = []
        reputation = {'status': 'not_checked',
                      'detail': 'Manual features do not identify a submitted URL.'}

        if mode == 'url':
            identifier = validate_url(request.form.get('a', ''))
            X = url_to_feature_vector(identifier, FEATURES, MEDIANS)
            coverage = feature_coverage(X, extract_lexical_features(identifier), FEATURES, MEDIANS)
            reputation = google_safe_browsing(identifier)
            trust = trust_assessment(identifier)
            flags = url_red_flags(identifier, trust['trusted'])
        else:
            identifier = (request.form.get('domain', 'Manual feature input').strip()[:250]
                          or 'Manual feature input')
            values = parse_manual_form(request.form)
            X = pd.DataFrame(
                [{f: values.get(f, float(MEDIANS.get(f, 0.0))) for f in FEATURES}],
                columns=FEATURES,
            )

        raw = infer(X)
        try:
            prob, calibration_available = calibrated_probability(raw['prob'])
        except Exception as cal_exc:
            prob, calibration_available = raw['prob'], False
            print('[Calibration unavailable]', cal_exc)

        verdict = final_verdict(prob, mode, reputation, trust, flags)
        xai = run_xai(X, mode)
        drivers = xai['shap_rows'] or xai['lime_rows']
        measured = len([f for f in FEATURES if f in URL_MEASURABLE]) or 6
        summary = build_summary(verdict, mode, prob, trust, flags, drivers,
                                measured=measured, total=len(FEATURES))
        advice = recommendations(verdict['kind'], reputation)

        prediction = {
            'label':        verdict['label'],
            'kind':         verdict['kind'],
            'reason':       verdict['reason'],
            'basis':        verdict['basis'],
            'prob':         prob,
            'raw_prob':     raw['prob'],
            'calibrated':   calibration_available,
            'history_label': verdict_to_history_label(verdict['kind']),
            'model_says':   'Phishing' if prob >= 0.5 else 'Legitimate',
        }

        groq_report = None
        if request.form.get('groq') == '1':
            groq_report = generate_groq_advice(
                identifier, verdict['label'], prob, reputation,
                coverage or {}, advice,
            )

        save_history(source, identifier, prediction)

        feature_rows = [
            {'label':     label_of(f),
             'name':      f,
             'value':     '%g' % float(X.iloc[0][f]),
             'estimated': mode == 'url' and f not in URL_MEASURABLE}
            for f in FEATURES
        ]

        return render_template(
            template,
            prediction=prediction, identifier=identifier, mode=mode, xai=xai,
            summary=summary, advice=advice, groq_report=groq_report,
            feature_rows=feature_rows, coverage=coverage,
            reputation=reputation, trust=trust, flags=flags,
        )
    except Exception as exc:
        log.exception('Prediction failed')
        return render_template(template, error=str(exc)[:350])


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    return render_result('prediction.html', 'url')


@app.route('/predict_value', methods=['GET', 'POST'])
def predict_value():
    return render_result('predict_value.html', 'manual_features')


# ============================================================================
# Routes — model comparison / graph
# ============================================================================
def _pct(v):
    """0.9654 -> 96.54 ; 96.54 -> 96.54"""
    v = float(v)
    return v * 100 if v <= 1 else v


def load_comparison():
    """Numbers for the Models page and chart, controlled by site_config.json > models_page."""
    cfg = SITE.get().get('models_page', {})
    source = str(cfg.get('source') or 'csv').lower()
    source = source if source in ('manual', 'csv') else 'csv'
    csv_name = Path(str(cfg.get('csv_file') or 'final_comparison.csv')).name
    wanted = [str(c) for c in (cfg.get('columns') or ['Accuracy'])]
    try:
        decimals = max(0, min(4, int(cfg.get('decimals', 2))))
    except (TypeError, ValueError):
        decimals = 2
    hidden = {str(h).strip().lower() for h in (cfg.get('hide_models') or [])}
    keep_order = str(cfg.get('sort') or 'accuracy_desc').lower() == 'as_listed'

    rows, updated, err = [], None, None
    columns = wanted

    def add(name, acc, values):
        rows.append({
            'name':          name,
            'accuracy':      round(acc, 2),
            'accuracy_text': f'{acc:.{decimals}f}%',
            'cells':         [f'{v:.{decimals}f}%' if v is not None else '-' for v in values],
        })

    if source == 'manual':
        problems = []
        for i, item in enumerate(cfg.get('manual') or [], start=1):
            try:
                name = str(item.get('name', '')).strip()
                if not name:
                    raise ValueError('missing name')
                acc = _pct(item.get('accuracy'))
                if not 0 <= acc <= 100:
                    raise ValueError('accuracy must be between 0 and 100')
                values = []
                for c in wanted:
                    v = item.get(c.lower())
                    values.append(_pct(v) if v not in (None, '') else None)
            except (TypeError, ValueError, AttributeError) as exc:
                problems.append(f'row {i} ({exc})')
                continue
            if name.lower() not in hidden:
                add(name, acc, values)
        if problems:
            err = ('site_config.json > models_page > manual has a problem in '
                   + '; '.join(problems) + '. Those rows were skipped.')
    else:
        path = MODEL_DIR / csv_name
        if path.exists():
            try:
                df = pd.read_csv(path, index_col=0)
                lookup = {str(c).strip().lower(): c for c in df.columns}
                if 'accuracy' not in lookup:
                    raise ValueError("the CSV has no 'Accuracy' column")
                use = [lookup[c.lower()] for c in wanted if c.lower() in lookup] or [lookup['accuracy']]
                columns = [str(c) for c in use]
                for name, row in df.iterrows():
                    if str(name).strip().lower() not in hidden:
                        add(str(name), _pct(row[lookup['accuracy']]),
                            [_pct(row[c]) for c in use])
                updated = datetime.fromtimestamp(path.stat().st_mtime).strftime('%d %b %Y, %H:%M')
            except Exception as exc:
                rows, err = [], 'Could not load comparison results: ' + str(exc)

    if not keep_order:
        rows.sort(key=lambda r: r['accuracy'], reverse=True)

    return rows, updated, err, columns, csv_name, source


@app.route('/model')
def model_page():
    guard = auth_required()
    if guard:
        return guard
    results, updated, err, columns, csv_name, source = load_comparison()
    if err:
        flash(err, 'error')
    return render_template('model.html', results=results, updated=updated,
                           columns=columns, csv_name=csv_name, source=source)


@app.route('/graph')
def graph():
    guard = auth_required()
    if guard:
        return guard
    results, updated, err, columns, csv_name, source = load_comparison()
    if err:
        flash(err, 'error')
    return render_template('graph.html', results=results, updated=updated, source=source)


# ============================================================================
# Routes — history & uploads
# ============================================================================
@app.route('/history')
def history():
    guard = auth_required()
    if guard:
        return guard
    rows = []
    try:
        conn = db_connect()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute(
                'SELECT source, identifier, prediction_label, probability, created_at '
                'FROM history WHERE user_email=%s ORDER BY created_at DESC LIMIT 200',
                (session['user_email'],),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as exc:
        log.exception('History fetch failed')
        flash('History unavailable: ' + str(exc), 'error')
    return render_template('history.html', rows=rows)


@app.route('/load_data', methods=['GET', 'POST'])
def load_data():
    guard = auth_required()
    if guard:
        return guard
    if request.method == 'POST':
        f = request.files.get('file')
        if not f or not f.filename.lower().endswith('.csv'):
            flash('Upload a CSV file.', 'error')
        else:
            folder = ROOT / 'uploads'
            folder.mkdir(exist_ok=True)
            filename = secure_filename(f.filename)
            path = folder / filename
            f.save(path)
            session['last_uploaded_csv'] = filename
            flash('CSV uploaded. Preview below.', 'success')

    filename = session.get('last_uploaded_csv', '')
    path = ROOT / 'uploads' / secure_filename(filename)
    preview = None
    if filename and path.exists():
        try:
            preview = pd.read_csv(path, nrows=25).fillna('').to_dict(orient='records')
        except Exception:
            log.exception('CSV preview failed')
    return render_template('load_data.html', preview=preview)


@app.route('/view_data')
def view_data():
    guard = auth_required()
    if guard:
        return guard
    filename = session.get('last_uploaded_csv', '')
    path = ROOT / 'uploads' / secure_filename(filename)
    preview = None
    if filename and path.exists():
        try:
            preview = pd.read_csv(path, nrows=100).fillna('').to_dict(orient='records')
        except Exception:
            log.exception('CSV view failed')
    return render_template('view_data.html', preview=preview)


# ============================================================================
# Startup: warn loudly in the console about DB / schema issues
# ============================================================================
def _startup_db_check():
    try:
        conn = db_connect()
        try:
            cur = conn.cursor()
            cur.execute('SELECT 1')
            cur.fetchone()
            cur.execute("SHOW COLUMNS FROM user3 LIKE 'password'")
            col = cur.fetchone()
            if col:
                col_type = str(col[1]).lower()
                digits = ''.join(ch for ch in col_type if ch.isdigit())
                if col_type.startswith('varchar') and digits and int(digits) < 200:
                    print(
                        f'[Startup WARNING] user3.password is {col_type} — werkzeug '
                        f'hashes are ~162 chars. Run: '
                        f'ALTER TABLE user3 MODIFY password VARCHAR(255) NOT NULL;'
                    )
        finally:
            conn.close()
        print('[Startup] DB OK')
    except MySQLError as exc:
        print(f'[Startup ERROR] DB unreachable: errno={exc.errno} {exc.msg}')
    except Exception as exc:
        print(f'[Startup ERROR] {type(exc).__name__}: {exc}')


_startup_db_check()


if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', '0') == '1')