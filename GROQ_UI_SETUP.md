# Groq + binary results UI integration

This package extends Option C. Replace the previous Option C files with the files from this folder, while **preserving your trained `models/` artifacts and local configuration**.

1. `pip install -r requirements.txt`
2. Configure `.env.example` values as process environment variables (or supply them through a securely configured deployment). Note that this project does **not** automatically load `.env`.
3. Put Dataset-B exported model artifacts in `models/` as documented in `README.md`.
4. Configure MySQL using `db.sql`, and run `python app.py`.
5. Open `/prediction`, tick Groq and LIME/SHAP if desired, and submit a URL.

The primary headline always displays exactly **Legitimate Website** or **Phishing Website**, based on the numerical ML threshold 0.5. A separate security assessment can still be **Uncertain** when missing/approximated features or missing calibration prevent verification. This separation is intentional. Groq never changes your classifier and cannot guarantee accuracy. The backend will only call Groq if checked and configured. Store API keys server-side, never in browser JavaScript.

For accurate real-world predictions, fix the Dataset-B feature mismatch, validate against domain-held-out external test URLs, and train the calibration model; the frontend is not a substitute for these steps.
