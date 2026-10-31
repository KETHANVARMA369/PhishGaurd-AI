# What changed in this update

## 1. LIME + SHAP now show for every result (URL scan and manual scan)
* They run automatically - the old opt-in checkbox (default OFF) is gone, which is why nothing appeared.
* New `xai_engine.py`: LIME and SHAP each run on the full RF + XGBoost + Deep + fusion score in ONE batched call.
* LIME weights are converted to contributions so red/green mean the same as SHAP (red = toward phishing).
* If either library fails, that panel shows a clearly labelled fallback instead of going blank, and the error text.
* Bars are normalised, labelled in plain English, and URL-scan inputs that could not be measured are tagged "estimated".

## 2. Wrong predictions on real sites (e.g. https://grok.com/)
Root cause: only 6 of the 15 trained features can be computed from a URL string (length_url, nb_www,
longest_word_path, phish_hints, Suspicious_Score, Length_x_Dots). The other 9 (domain age, traffic rank, Google index,
page rank, link statistics...) are filled with training medians, and the model over-reacts to cues such as a missing "www".
Fix (new `url_trust.py`, `hybrid_engine.final_verdict`): a URL verdict now combines
  1. the AI model score, 2. domain reputation (exact registered-domain match - `grok.com.evil.tk` or `paypal-login.xyz`
  can never borrow trust; blog/hosting domains are excluded), 3. address warning signs (IP host, brand look-alikes,
  punycode, shorteners, odd TLDs, many hyphens/sub-domains ...).
Phishing from the model now needs a real warning sign to back it up; otherwise the result is "Suspicious - Verify First".
Manual (15-feature) scans stay plain Legitimate / Phishing from the model.

## 3. Short summary
`summary.py` builds a deterministic, plain-English "Quick summary" (headline, one sentence, up to 3-4 reasons, what to do)
for every scan. It needs no Groq key. Groq remains an optional extra.

## 4. Manual input = the 15 trained features
`feature_info.py` is the single list of the 15 features (labels, hints, groups). The form shows only those, grouped, with
typical-value defaults and example presets. If your models/ files were exported from an older notebook (more features),
an orange banner explains it and the other features are filled with training medians.

## Files
New: feature_info.py, url_trust.py, xai_engine.py, summary.py, data/trusted_domains.txt, tests/test_app_flow.py
Changed: app.py, hybrid_engine.py, groq_advice.py, templates/_results.html, prediction.html, predict_value.html,
history.html, userhome.html, static/style.css, static/app.js, db.sql (comment only), requirements.txt,
tests/test_hybrid_engine.py, models/README.txt

## Update 2: model accuracies + no base.html
* `models/final_comparison.csv` now holds the exact results printed by the notebook
  (Novelty 96.54, Random Forest 96.40, XGBoost 96.36, ExtraTrees 95.91, CNN-BiLSTM 95.07, Bi-LSTM 94.19, CNN 92.51).
  The Models page and chart read this file, so COPY IT OVER the old models/final_comparison.csv on your machine
  (or re-run the notebook's comparison cell, which writes the same file). The "(Deep)" row is gone: the notebook
  keeps that model for fusion only.
* Both pages now show "CSV last updated" so a stale file is obvious. One shared loader (`load_comparison` in app.py).
* `templates/base.html` is deleted. Every page (index, login, registration, userhome, prediction, predict_value,
  model, graph, history, load_data, view_data) is a complete standalone HTML file with its own head, nav, footer
  and script tag. `_results.html` and `_table.html` remain as include-fragments only.
* The two notebook files you uploaded have identical code and results; no model-export changes were needed.

## Update 3: navbar + editable frontend
* Navbar now exactly: Home, Scan URL, Feature scan, Models, History, Logout (no Dashboard link). Visitors who are not
  signed in see Home, Login, Get started.
* New `site_config.json` holds the navbar, brand, footer, every page's text, the example links and the model-table
  options. New `site_config.py` loads it (hot reload, bad JSON keeps the last good version and shows a banner).
* Pages no longer have text/links hard-coded. Each page in templates/ is still a complete standalone HTML file
  (no base.html); `_navbar.html` and `_footer.html` are small include-fragments fed by the config.
* Models page: columns (Accuracy / F1 / Precision / Recall / AUC), decimals, hidden models and CSV name are config options.
* TEMPLATES_AUTO_RELOAD is on, so HTML edits also show on refresh. See EDITING_GUIDE.md.
* New tests: tests/test_site_config.py (30 tests in total).

## Update 4: accuracies typed by hand
* `site_config.json` > `models_page` > `source` is now `"manual"`: the Models page and chart show the numbers typed in
  `models_page.manual` (pre-filled with the notebook's results). Set `"source": "csv"` to read models/final_comparison.csv again.
* New options: `sort` (accuracy_desc / as_listed); manual rows accept percentages or fractions; bad rows are skipped with a message.
* Page text is mode-aware, so the manual table never claims it was "loaded from the CSV".
* Tests: 35 passing, including manual mode, bad rows, ordering, and CSV mode.
* Tested with your real models in this session (TensorFlow 2.21): see the notes in the reply about the 58-feature model files.
