<<<<<<< HEAD
> **Update (15-feature model):** see `CHANGES_THIS_UPDATE.md`. URL verdicts now combine the AI score, domain reputation and
> address warning signs (the older "always Uncertain" text below no longer applies); LIME/SHAP and a plain-English summary
> appear for every scan; the manual page uses exactly the 15 trained features.

# PhishGuard AI — Option C hybrid implementation

## Scope and limitations

This is a **conservative hybrid prototype**, not a verified production phishing detector. It retains the Dataset-B notebook's RF + XGBoost + MSRA-BiLSTM-AFG + logistic fusion and the previous Flask UI/database. It adds safe URL normalization, *optional* Google Safe Browsing reputation lookup, feature-coverage diagnostics, an independent calibration workflow, uncertain/abstention results, optional LIME/SHAP, and clear user advice. No claim is made that every real-world URL can be reliably classified.

### Important notebook compatibility issue

The uploaded Dataset-B notebook trains on precomputed dataset features but predicts external URLs with hand-written lexical approximations and median substitutions for unavailable features. Features such as web page statistics, DNS, WHOIS or page content cannot be reconstructed solely from URL text. **This package does not pretend the missing features have been measured**: the model score is exploratory and the verdict becomes `Uncertain`. You must reproduce upstream dataset feature definitions, collect required inputs consistently, retrain if they cannot be reproduced, and validate on domain-separated/time-separated datasets before relaxing abstention.

The classifier needs the trusted notebook outputs, **not included in the upload**:

- `models/msra_novelty_highest.joblib` (contains `rf`, `xgb`, `meta_fusion`, `scaler`, `selected_features`, `train_medians`)
- `models/msra_bilstm_afg_deep.keras` (or `.h5`)
- Strongly recommended: `models/xai_background.csv` (rows of the actual selected numeric training features, no labels)
- Optional: `models/final_comparison.csv`

## Install

1. Create a virtual environment and run `pip install -r requirements.txt`.
2. Import `db.sql` into MySQL and configure `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_PORT` environment variables.
3. Set a random `FLASK_SECRET_KEY`; for HTTPS production set `COOKIE_SECURE=1`.
4. Export and copy trusted notebook model artifacts into `models/`.
5. **Optional**: set `GOOGLE_SAFE_BROWSING_API_KEY` to enable live Safe Browsing v4 lookup. Check Google's API availability, permitted use, quotas and terms. Submitted URL text is shared with that provider when configured. A 'no_match' result NEVER means verified legitimate.
6. Start with `python app.py`, then visit `http://127.0.0.1:5000`.
7. Run checks with `python -m unittest discover -s tests -v` and `python -m compileall -q .`.

## Independent calibration

Build an independently labeled calibration dataset **with the exact selected numeric features** as used at training, plus `Label` (0 legitimate, 1 phishing). Keep training, calibration, and final held-out test datasets separate by registered domain and preferably chronology. Run:

```bash
python calibrate.py independent_calibration_features.csv
```

This produces `models/probability_calibrator.joblib` using isotonic regression. The printed Brier score after calibration is *in-sample* and optimistic; report Brier / ECE / classification metrics on a separate held-out test set. A calibrator cannot fix incorrect input features or dataset shift. Example probability cutoffs (.10 and .90) in `hybrid_engine.decision` are **illustrative placeholders**, not validated operating thresholds: derive them from false-positive/false-negative costs and holdout curves before deployment.

## What the outputs mean

- `Known threat`: the configured Safe Browsing lookup returned a positive threat match.
- `Uncertain`: unsupported features, non-validated heuristics, no independent calibration, a reputation error, or a middle-range model probability.
- `Likely phishing` / `Lower risk`: only if input feature definitions pass coverage and calibration is present and score passes selected cutoffs. `Lower risk` is not a safety certification.
- LIME/SHAP describe the model component **only**, not external threat intelligence. The XAI background should come from real training features.

## Remaining steps for a research-grade real-world classifier

1. Audit each Dataset-B feature against the original dataset generation code; document exact formulas, encodings, and missing-value strategies.
2. Reproduce those features at inference safely, using isolated retrieval if absolutely needed. Never make arbitrary user-provided URLs directly fetchable from the production app; SSRF, redirects and DNS rebinding are hazards.
3. Otherwise retrain on a consistent lexical-only feature subset generated identically for training/inference.
4. Collect separately verified legitimate/phishing URLs with timestamps and domain-disjoint splits; check legal/usage terms and duplicate or mislabeled data.
5. Evaluate ROC-AUC, PR-AUC, F1, phishing recall, false positive rate, calibration (Brier/ECE), and abstention coverage on an untouched real-world test set.
6. Choose and test calibrated thresholds; validate false positives on legitimate sites such as `grok.com` without hardcoding a whitelist.
7. To add additional reputation sources, integrate well-defined APIs (e.g. VirusTotal) with explicit privacy/quotas; keep `no match` distinct from `safe`.

## Security caveats

Registration has password hashing; older passwords created by the original backend can still be checked in the compatibility branch. Migrate legacy plaintext passwords for production, disable debug, set HTTPS/session-cookie protections, add CSRF and login throttling, handle credentials via environment variables, and restrict file uploads. Do not trust arbitrary third-party model or pickle files. History stores the **model-only** label and probability, not the hybrid conclusion; update your schema before using history for downstream clinical/security decisions.
=======
# PhishGaurd-AI
>>>>>>> dffa87815dae237feeaacaf0fb63662dc2df1a29
