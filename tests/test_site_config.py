"""site_config.json drives the navbar, text and model table; edits show without a restart."""
import json, os, re, shutil, sys, tempfile, time, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as appmod
        from site_config import SiteConfig
        cls.appmod, cls.SiteConfig = appmod, SiteConfig
        appmod.app.config['TESTING'] = True
        cls.client = appmod.app.test_client()
        cls.orig_site = appmod.SITE
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        cls.appmod.SITE = cls.orig_site
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        with self.client.session_transaction() as s:
            s.clear()

    def use_config(self, mutate=None, raw=None):
        path = os.path.join(self.tmp, 'site_config.json')
        data = json.load(open(os.path.join(ROOT, 'site_config.json'), encoding='utf-8'))
        if mutate: mutate(data)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(raw if raw is not None else json.dumps(data))
        self.appmod.SITE = self.SiteConfig(path)
        return path

    def login(self):
        with self.client.session_transaction() as s:
            s['user_email'] = 't@t.com'; s['user_name'] = 'Tester'

    def nav_labels(self, html):
        nav = re.search(r'<nav class="links">(.*?)</nav>', html, re.S).group(1)
        return re.findall(r'<a [^>]*>([^<]+)</a>', nav)

    def test_navbar_matches_screenshot(self):
        self.appmod.SITE = self.orig_site
        self.login()
        html = self.client.get('/model').get_data(as_text=True)
        self.assertEqual(self.nav_labels(html), ['Home', 'Scan URL', 'Feature scan', 'Models', 'History', 'Logout'])
        self.assertNotIn('Dashboard', html.split('</header>')[0])
        self.client.get('/logout')
        html = self.client.get('/login').get_data(as_text=True)
        self.assertEqual(self.nav_labels(html), ['Home', 'Login', 'Get started'])

    def test_editing_config_changes_pages_live(self):
        path = self.use_config(lambda d: (d['pages']['login'].update(heading='Hello again'),
                                          d['nav']['logged_out'].append({'label': 'Docs', 'url': 'https://example.com/docs'}),
                                          d['footer'].update(text='My custom footer')))
        html = self.client.get('/login').get_data(as_text=True)
        self.assertIn('Hello again', html); self.assertIn('My custom footer', html)
        self.assertIn('href="https://example.com/docs"', html)
        # edit again - the change must appear on the next request without restarting
        time.sleep(0.05)
        data = json.load(open(path, encoding='utf-8')); data['pages']['login']['heading'] = 'Second edit'
        json.dump(data, open(path, 'w', encoding='utf-8')); os.utime(path, (time.time() + 5, time.time() + 5))
        self.assertIn('Second edit', self.client.get('/login').get_data(as_text=True))

    def test_bad_json_keeps_last_good_and_shows_banner(self):
        path = self.use_config(lambda d: d['pages']['login'].update(heading='Good version'))
        self.assertIn('Good version', self.client.get('/login').get_data(as_text=True))
        with open(path, 'w', encoding='utf-8') as fh: fh.write('{ "broken": ')
        os.utime(path, (time.time() + 5, time.time() + 5))
        r = self.client.get('/login'); html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn('Good version', html); self.assertIn('has a problem', html)

    def test_missing_keys_and_unsafe_links_do_not_crash(self):
        def m(d):
            del d['pages']['login']['heading']; del d['pages']['home']
            d['nav']['logged_out'] += [{'label': 'Bad', 'url': 'javascript:alert(1)'}, {'label': 'Ghost', 'page': 'nope'}]
        self.use_config(m)
        for url in ('/', '/login', '/registration'):
            self.assertEqual(self.client.get(url).status_code, 200, url)
        html = self.client.get('/login').get_data(as_text=True)
        self.assertNotIn('javascript:alert', html); self.assertNotIn('>Bad<', html); self.assertNotIn('>Ghost<', html)

    def test_models_table_options_from_config(self):
        self.use_config(lambda d: d['models_page'].update(columns=['Accuracy', 'F1', 'AUC'], decimals=1,
                                                          hide_models=['cnn', 'Bi-LSTM']))
        self.login()
        html = self.client.get('/model').get_data(as_text=True)
        for h in ('<th>Accuracy</th>', '<th>F1</th>', '<th>AUC</th>'): self.assertIn(h, html)
        self.assertIn('96.5%', html)                       # decimals=1
        self.assertNotIn('<td>CNN</td>', html); self.assertNotIn('Bi-LSTM', html)
        self.assertIn('CNN-BiLSTM', html)                  # only exact (case-insensitive) names are hidden

    def test_default_models_table_is_accuracy_only(self):
        self.appmod.SITE = self.orig_site
        self.login()
        html = self.client.get('/model').get_data(as_text=True)
        self.assertEqual(len(re.findall(r'<th>', html)), 2)   # Model + Accuracy
        self.assertIn('96.54%', html)

    # ---------------------------------------------------------------- manual accuracies
    def test_manual_accuracies_are_shown_and_editable(self):
        def m(d):
            d['models_page']['manual'] = [
                {'name': 'Model A', 'accuracy': 91.234, 'f1': 90.1}, {'name': 'Model B', 'accuracy': 0.9876},
                {'name': 'Model C', 'accuracy': '95.5'}]
            d['models_page']['columns'] = ['Accuracy', 'F1']
        self.use_config(m); self.login()
        html = self.client.get('/model').get_data(as_text=True)
        self.assertIn('91.23%', html)                      # rounded to 2 decimals
        self.assertIn('98.76%', html)                      # 0.9876 treated as a fraction
        self.assertIn('95.50%', html)                      # numbers typed as text work
        self.assertIn('90.10%', html)
        self.assertLess(html.index('Model B'), html.index('Model C'))   # sorted best first
        self.assertLess(html.index('Model C'), html.index('Model A'))
        self.assertNotIn('CSV last updated', html); self.assertNotIn('96.54', html)
        chart = self.client.get('/graph').get_data(as_text=True)
        self.assertIn('98.76%', chart)

    def test_manual_order_as_listed(self):
        def m(d):
            d['models_page']['sort'] = 'as_listed'
            d['models_page']['manual'] = [{'name': 'Low', 'accuracy': 50}, {'name': 'High', 'accuracy': 99}]
        self.use_config(m); self.login()
        html = self.client.get('/model').get_data(as_text=True)
        self.assertLess(html.index('>Low<'), html.index('>High<'))

    def test_manual_bad_rows_are_skipped_with_message(self):
        def m(d):
            d['models_page']['manual'] = [{'name': 'Good', 'accuracy': 90}, {'name': 'Bad', 'accuracy': 'abc'},
                                          {'name': 'TooBig', 'accuracy': 250}, {'accuracy': 80}]
        self.use_config(m); self.login()
        r = self.client.get('/model'); html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn('>Good<', html); self.assertNotIn('>Bad<', html); self.assertNotIn('>TooBig<', html)
        self.assertIn('row 2', html); self.assertIn('row 3', html); self.assertIn('row 4', html)

    def test_csv_source_still_works(self):
        csvdir = tempfile.mkdtemp()
        with open(os.path.join(csvdir, 'final_comparison.csv'), 'w') as fh:
            fh.write(',Accuracy,F1\nCsvModel,0.8123,0.8\nOther,0.7,0.7\n')
        old_dir = self.appmod.MODEL_DIR
        try:
            from pathlib import Path
            self.appmod.MODEL_DIR = Path(csvdir)
            self.use_config(lambda d: d['models_page'].update(source='csv')); self.login()
            html = self.client.get('/model').get_data(as_text=True)
            self.assertIn('CsvModel', html); self.assertIn('81.23%', html)
            self.assertIn('CSV last updated', html); self.assertNotIn('96.54', html)
        finally:
            self.appmod.MODEL_DIR = old_dir; shutil.rmtree(csvdir, ignore_errors=True)

    def test_shipped_default_is_manual_with_notebook_numbers(self):
        self.appmod.SITE = self.orig_site; self.login()
        rows, updated, err, cols, csv_name, source = self.appmod.load_comparison()
        self.assertEqual(source, 'manual'); self.assertIsNone(err)
        self.assertEqual([(r['name'], r['accuracy']) for r in rows][:3],
                         [('MSRA-BiLSTM-AFG (Novelty)', 96.54), ('Random Forest', 96.4), ('XGBoost', 96.36)])


if __name__ == '__main__':
    unittest.main()
