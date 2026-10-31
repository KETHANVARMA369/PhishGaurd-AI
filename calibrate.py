"""Fit probability calibration on INDEPENDENT, labeled original-feature validation rows.
Input CSV must have exact selected feature column names plus Label (1=phishing).
Never calibrate with training or final test samples.
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT=Path(__file__).parent
if len(sys.argv)!=2: raise SystemExit('Usage: python calibrate.py /path/to/independent_calibration_features.csv')
from app import FEATURES, predict_matrix, model_ready
if not model_ready(): raise SystemExit('Missing valid model artifacts in models/.')
df=pd.read_csv(sys.argv[1]); required=FEATURES+['Label']
missing=[f for f in required if f not in df.columns]
if missing: raise SystemExit('Missing columns: '+', '.join(missing))
X=df[FEATURES].apply(pd.to_numeric, errors='raise')
if X.isna().any().any(): raise SystemExit('Calibration features contain nulls')
y=df['Label'].astype(int).to_numpy()
if len(df)<100 or len(set(y))!=2: raise SystemExit('Provide >=100 independent samples containing both classes.')
raw=predict_matrix(X.to_numpy())
cal=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip')
cal.fit(raw,y)
print('Brier raw:',round(brier_score_loss(y,raw),4),' calibrated in-sample:',round(brier_score_loss(y,cal.predict(raw)),4))
print('WARNING: in-sample calibrated Brier is optimistic; evaluate calibration on a separate held-out test set.')
joblib.dump(cal,ROOT/'models'/'probability_calibrator.joblib')
print('Saved models/probability_calibrator.joblib')
