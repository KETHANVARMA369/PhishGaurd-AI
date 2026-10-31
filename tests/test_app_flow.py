"""End-to-end UI tests with STAND-IN models (no TensorFlow / trained files needed).

The stand-in is deliberately biased the way the real model behaves on bare links:
anything without 'www' (nb_www == 0) scores as phishing. This reproduces the grok.com false positive
so the tests prove the reputation layer fixes it. Run:  python -m unittest discover -s tests -v
"""
import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from feature_info import CANONICAL_15


class FakeKeras:
    def __init__(self, w): self.w = w
    def predict(self, seq, verbose=0, batch_size=32):
        z = seq.reshape(len(seq), -1) @ self.w
        return (1 / (1 + np.exp(-z))).reshape(-1, 1)


def build_stub():
    rng = np.random.default_rng(1)
    n = 3000
    F = CANONICAL_15
    X = np.column_stack([rng.normal(50, 25, n).clip(5), rng.integers(0, 2, n), rng.integers(0, 20, n),
                         rng.integers(0, 4, n), rng.integers(0, 200, n), rng.uniform(0, 100, n),
                         rng.uniform(0, 100, n), rng.normal(300, 200, n), rng.normal(3000, 2500, n),
                         rng.integers(0, 1000000, n), rng.integers(0, 2, n), rng.integers(0, 10, n),
                         rng.integers(0, 5, n), rng.normal(200, 100, n).clip(0), rng.normal(1e6, 2e6, n).clip(0)])
    y = ((X[:, 1] == 0) ^ (rng.random(n) < 0.05)).astype(int)       # "no www" => phishing
    import pandas as pd
    df = pd.DataFrame(X, columns=F)
    rf = RandomForestClassifier(100, random_state=0).fit(df, y)
    xgb = GradientBoostingClassifier(random_state=0).fit(df, y)
    sc = StandardScaler().fit(df)
    w = np.zeros(15); w[1] = -2.0
    deep = FakeKeras(w)
    meta_X = np.column_stack([rf.predict_proba(df)[:, 1], xgb.predict_proba(df)[:, 1],
                              deep.predict(sc.transform(df).reshape(n, 15, 1)).ravel()])
    meta = LogisticRegression().fit(meta_X, y)
    art = {'rf': rf, 'xgb': xgb, 'meta_fusion': meta, 'scaler': sc, 'selected_features': list(F),
           'train_medians': df.median().to_dict()}
    return art, deep


class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as appmod
        cls.appmod = appmod
        art, deep = build_stub()
        appmod.set_models(art, deep)
        appmod.app.config['TESTING'] = True
        cls.client = appmod.app.test_client()

    def setUp(self):
        with self.client.session_transaction() as s:
            s['user_email'] = 'tester@example.com'; s['user_name'] = 'Tester'

    def scan(self, url, **extra):
        return self.client.post('/prediction', data={'a': url, **extra}).get_data(as_text=True)

    def test_stub_reproduces_the_false_positive(self):
        X = self.appmod.url_to_feature_vector('https://grok.com/', self.appmod.FEATURES, self.appmod.MEDIANS)
        self.assertGreaterEqual(self.appmod.infer(X)['prob'], 0.5)   # raw model alone says phishing

    def test_grok_is_legitimate_and_explained(self):
        html = self.scan('https://grok.com/')
        self.assertIn('Legitimate Website', html)
        self.assertNotIn('>Phishing Website<', html)
        self.assertIn('QUICK SUMMARY', html)
        self.assertIn('LIME', html); self.assertIn('SHAP', html)
        self.assertGreaterEqual(html.count('class="drow"'), 20)     # 10 LIME + 10 SHAP rows
        self.assertIn('estimated', html)

    def test_obvious_phishing_is_phishing(self):
        html = self.scan('http://secure-paypal-update.tk/login/verify')
        self.assertIn('Phishing Website', html)
        self.assertIn('QUICK SUMMARY', html); self.assertIn('class="drow"', html)

    def test_embedded_famous_domain_is_not_trusted(self):
        html = self.scan('https://grok.com.evil.tk/login')
        self.assertNotIn('Legitimate Website</h2>', html)

    def test_unknown_clean_domain_is_cautious_not_phishing(self):
        html = self.scan('https://smallshop-example.org/')
        self.assertIn('Suspicious - Verify First', html)

    def test_manual_page_has_exactly_15_inputs(self):
        html = self.client.get('/predict_value').get_data(as_text=True)
        import re
        names = re.findall(r'<(?:input|select)[^>]*name="([^"]+)"', html)
        feats = [n for n in names if n not in ('domain', 'groq')]
        self.assertEqual(sorted(feats), sorted(CANONICAL_15))
        self.assertEqual(len(feats), 15)

    def test_manual_prediction_shows_summary_lime_shap(self):
        from feature_info import presets
        data = {k: v for k, v in presets()['phish'].items()}
        data['domain'] = 'demo'
        html = self.client.post('/predict_value', data=data).get_data(as_text=True)
        self.assertIn('QUICK SUMMARY', html)
        self.assertGreaterEqual(html.count('class="drow"'), 20)
        self.assertIn('FEATURE SCAN', html)

    def test_manual_validation_message(self):
        from feature_info import presets
        data = dict(presets()['legit']); data['domain_age'] = ''
        html = self.client.post('/predict_value', data=data).get_data(as_text=True)
        self.assertIn('Please enter a value for', html)
        data['domain_age'] = 'abc'
        self.assertIn('must be a number', self.client.post('/predict_value', data=data).get_data(as_text=True))
        data['domain_age'] = 'nan'
        self.assertIn('finite', self.client.post('/predict_value', data=data).get_data(as_text=True))


if __name__ == '__main__':
    unittest.main()


class PageTests(unittest.TestCase):
    """Every page is a complete standalone HTML file (no base.html) and the comparison page matches the notebook."""
    EXPECTED = [('MSRA-BiLSTM-AFG (Novelty)', '96.54'), ('Random Forest', '96.40'), ('XGBoost', '96.36'),
                ('ExtraTrees', '95.91'), ('CNN-BiLSTM', '95.07'), ('Bi-LSTM', '94.19'), ('CNN', '92.51')]

    @classmethod
    def setUpClass(cls):
        import app as appmod
        cls.appmod = appmod
        appmod.app.config['TESTING'] = True
        cls.client = appmod.app.test_client()

    def test_no_template_extends_base(self):
        tdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
        self.assertFalse(os.path.exists(os.path.join(tdir, 'base.html')))
        for f in os.listdir(tdir):
            txt = open(os.path.join(tdir, f), encoding='utf-8').read()
            self.assertNotIn('extends', txt, f)
            if not f.startswith('_'):
                self.assertTrue(txt.lstrip().lower().startswith('<!doctype html>'), f)
                self.assertIn("{% include '_navbar.html' %}", txt, f)

    def test_public_and_private_pages_render(self):
        for url in ['/', '/login', '/registration']:
            html = self.client.get(url).get_data(as_text=True)
            self.assertIn('<!doctype html>', html.lower()); self.assertNotIn('{%', html)
        with self.client.session_transaction() as s:
            s['user_email'] = 't@t.com'; s['user_name'] = 'T'
        for url in ['/userhome', '/prediction', '/predict_value', '/model', '/graph', '/load_data', '/view_data']:
            r = self.client.get(url)
            self.assertEqual(r.status_code, 200, url)
            html = r.get_data(as_text=True)
            self.assertIn('</html>', html, url); self.assertNotIn('{%', html, url)

    def test_model_page_and_chart_match_notebook(self):
        with self.client.session_transaction() as s:
            s['user_email'] = 't@t.com'; s['user_name'] = 'T'
        for url in ('/model', '/graph'):
            html = self.client.get(url).get_data(as_text=True)
            for name, acc in self.EXPECTED:
                self.assertIn(name, html, url); self.assertIn(acc + '%', html, url)
            self.assertNotIn('MSRA-BiLSTM-AFG (Deep)', html); self.assertNotIn('79.04', html)
        rows = self.appmod.load_comparison()[0]
        self.assertEqual(rows[0]['name'], 'MSRA-BiLSTM-AFG (Novelty)')   # highest first
