# How to change things without touching HTML

Almost everything you see on screen is stored in **`site_config.json`**. Edit it, save, refresh the browser. No restart.
(If you save invalid JSON the site keeps working with your last good version and shows a red message at the top.)

## Navbar (matches the design: Home · Scan URL · Feature scan · Models · History · Logout)
`"nav"` has two lists: `logged_in` and `logged_out`. Each item:

    { "label": "Models", "page": "models" }                       <- link to a page of this app
    { "label": "Docs",   "url": "https://example.com/docs" }      <- link anywhere (http, https, mailto, or /path)
    { "label": "Logout", "page": "logout", "style": "outline" }   <- style: "" | "outline" | "button"

Delete an item to hide it, add one to show it, reorder them to reorder the navbar.
Want the Dashboard link back? Add `{ "label": "Dashboard", "page": "dashboard" }` to `logged_in`.

Allowed `page` names: home, login, register, logout, dashboard, scan_url, feature_scan, models, chart, history, csv_upload, csv_view

## Brand, footer, browser tab titles
`"brand"` (name, accent, logo), `"footer"`, and each page's `"title"` under `"pages"`.

## Page text
`"pages"` -> one block per page (home, login, register, dashboard, scan_url, feature_scan, models, chart, history, csv_upload, csv_view).
Change headings, intro text, button text, placeholders, the example links on the Scan URL page
(`"examples"`, hide them with `"show_examples": false`), the home-page cards and the dashboard tiles.

## Model accuracies (Models page and chart) - type them yourself
Open `site_config.json`, find `"models_page"` -> `"manual"`. One line per model; change the numbers, save, refresh:

    "source": "manual",
    "manual": [
      { "name": "MSRA-BiLSTM-AFG (Novelty)", "accuracy": 96.54, "f1": 96.55, "precision": 96.12, "recall": 96.99, "auc": 99.29 },
      { "name": "Random Forest",             "accuracy": 96.40, ... },
      ...
    ]

* Write values as percentages (96.54). Fractions like 0.9654 also work.
* Add a model = add a line. Remove a model = delete its line (or put its name in `"hide_models"`).
* `"columns": ["Accuracy"]` chooses what the table shows; use any of Accuracy, F1, Precision, Recall, AUC.
  The Models page and the chart use the same numbers.
* `"sort": "accuracy_desc"` lists the best model first; `"as_listed"` keeps the order you typed.
* `"decimals": 2` controls rounding (0-4).
* A row with a missing name or a non-numeric accuracy is skipped and a red message names the row.
* To go back to reading the notebook's file instead, set `"source": "csv"` (file: `models/final_comparison.csv`).
  The page then shows when that file was last changed.
* The page text for each mode is under `"pages"` -> `"models"` (`intro_manual` / `intro`) and `"chart"`.

## Not in site_config.json (edit here instead)
* The 15 feature names / hints on the Feature scan page: `feature_info.py`
* Result panel wording (summary, XAI notes): `templates/_results.html`, `summary.py`
* Colours and layout: top of `static/style.css` (`--mint`, `--bg`, `--panel` ...)
* Page structure: each file in `templates/` is a complete HTML page. There is no base.html. The navbar and footer
  markup live in `templates/_navbar.html` and `templates/_footer.html` (their contents come from the config).
