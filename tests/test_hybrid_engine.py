import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid_engine import normalize_url, final_verdict, feature_coverage, recommendations
from url_trust import trust_assessment, url_red_flags, registered_domain
from feature_info import CANONICAL_15, URL_MEASURABLE

NONE = {'status': 'no_match'}


def verdict(url, p, rep=NONE):
    u = normalize_url(url)
    t = trust_assessment(u)
    return final_verdict(p, 'url', rep, t, url_red_flags(u, t['trusted']))


class HybridTests(unittest.TestCase):
    def test_public_url(self):
        self.assertEqual(normalize_url('grok.com'), 'https://grok.com/')

    def test_local_block(self):
        for host in ['localhost', 'http://127.0.0.1/', 'http://192.168.1.1/', 'file:///etc/passwd', 'http://[::1]/']:
            with self.subTest(host=host), self.assertRaises(ValueError): normalize_url(host)

    def test_registered_domain(self):
        self.assertEqual(registered_domain('www.grok.com'), 'grok.com')
        self.assertEqual(registered_domain('a.b.example.co.uk'), 'example.co.uk')

    # ---- the reported bug: well-known site flagged as phishing by the raw model
    def test_grok_high_model_score_is_still_legitimate(self):
        for p in (0.62, 0.88, 0.99):
            self.assertEqual(verdict('https://grok.com/', p)['kind'], 'safe')

    def test_lookalikes_never_borrow_trust(self):
        for u in ['https://grok.com.evil.tk/', 'https://paypal-login.xyz/', 'https://grok-com.evil.org/']:
            self.assertFalse(trust_assessment(normalize_url(u))['trusted'], u)

    def test_shared_hosting_not_trusted(self):
        self.assertFalse(trust_assessment('https://evil.github.io/')['trusted'])
        self.assertFalse(trust_assessment('https://docs.google.com/forms/d/x')['trusted'])
        self.assertTrue(trust_assessment('https://accounts.google.com/')['trusted'])

    def test_phishing_needs_corroboration(self):
        self.assertEqual(verdict('https://smallshop-example.org/', 0.97)['kind'], 'unknown')
        self.assertEqual(verdict('http://secure-paypal-update.tk/login/verify', 0.80)['kind'], 'danger')

    def test_low_score_but_high_risk_flag_is_suspicious(self):
        self.assertEqual(verdict('http://8.8.8.8/login', 0.1)['kind'], 'unknown')

    def test_threat_list_wins_over_everything(self):
        v = verdict('https://grok.com/', 0.01, {'status': 'malicious'})
        self.assertEqual(v['kind'], 'danger')

    def test_manual_mode_is_plain_binary(self):
        self.assertEqual(final_verdict(0.7, 'features', NONE)['label'], 'Phishing Website')
        self.assertEqual(final_verdict(0.2, 'features', NONE)['label'], 'Legitimate Website')

    def test_coverage_counts_six_measurable(self):
        c = feature_coverage(None, {'length_url': 1, 'nb_www': 0, 'longest_word_path': 0, 'phish_hints': 0}, CANONICAL_15, {})
        self.assertEqual(len(URL_MEASURABLE), 6)
        self.assertIn('page_rank', c['missing'])

    def test_exactly_15_features(self):
        self.assertEqual(len(CANONICAL_15), 15); self.assertEqual(len(set(CANONICAL_15)), 15)

    def test_advice_never_empty(self):
        for k in ('danger', 'unknown', 'safe'): self.assertTrue(recommendations(k, NONE))


if __name__ == '__main__':
    unittest.main()
