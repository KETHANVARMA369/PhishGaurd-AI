"""Hybrid assessment: combines ML score + domain reputation + URL warning signs.
No arbitrary URL network fetching. Google Safe Browsing is opt-in via API key.
"""
import os
import json
import ipaddress
import socket
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
import requests

ROOT=Path(__file__).parent

def normalize_url(value):
    v=(value or '').strip()
    if not v: raise ValueError('Enter a website URL.')
    if '://' not in v: v='https://'+v
    if len(v)>2048 or any(ord(c)<32 for c in v): raise ValueError('Invalid or overly long URL.')
    try: parts=urlsplit(v)
    except ValueError: raise ValueError('Malformed URL.')
    if parts.scheme.lower() not in ('http','https') or not parts.hostname: raise ValueError('Only public HTTP/HTTPS URLs are supported.')
    if parts.username or parts.password: raise ValueError('URLs with embedded credentials are not supported.')
    try: port=parts.port
    except ValueError: raise ValueError('Invalid port.')
    if port not in (None,80,443): raise ValueError('Nonstandard ports are not supported.')
    host=parts.hostname.rstrip('.').lower()
    if host in ('localhost',) or host.endswith(('.localhost','.local','.internal','.test')) or '.' not in host:
        raise ValueError('Enter a public internet domain.')
    try:
        ip=ipaddress.ip_address(host)
        if not ip.is_global: raise ValueError('Private, reserved and local IPs are not supported.')
    except ValueError as exc:
        if 'not supported' in str(exc): raise
        try: host.encode('idna')
        except Exception: raise ValueError('Invalid hostname.')
    # Do not resolve the hostname or GET its content. Reputation API receives URL text only.
    return urlunsplit((parts.scheme.lower(),parts.netloc.lower(),parts.path or '/',parts.query,''))

def feature_coverage(X, lexical, features, medians):
    # Only structural URL features can be reconstructed without visiting websites.
    derived=set(lexical)
    derived.update(('Suspicious_Score','Length_x_Dots','Digits_x_Slash'))
    missing=[f for f in features if f not in derived]
    # Some lexical features have heuristic, rather than reference-dataset-equivalent definitions.
    heuristic={'random_domain','domain_in_brand','brand_in_subdomain','brand_in_path',
               'suspecious_tld','phish_hints','statistical_report','nb_redirection',
               'nb_external_redirection','https_token','nb_subdomains','nb_www'}
    approximated=[f for f in features if f in heuristic]
    return {'missing':missing,'approximated':approximated,'available':len(features)-len(missing),
            'total':len(features),'coverage':round((len(features)-len(missing))/max(len(features),1)*100,1)}

def google_safe_browsing(url):
    """Only definitive matches are flagged. API failures are not clean results."""
    key=os.getenv('GOOGLE_SAFE_BROWSING_API_KEY','').strip()
    if not key: return {'status':'unavailable','detail':'Google Safe Browsing is not configured.'}
    payload={'client':{'clientId':'phishguard-research','clientVersion':'1.0'},'threatInfo':{
        'threatTypes':['MALWARE','SOCIAL_ENGINEERING','UNWANTED_SOFTWARE','POTENTIALLY_HARMFUL_APPLICATION'],
        'platformTypes':['ANY_PLATFORM'],'threatEntryTypes':['URL'],'threatEntries':[{'url':url}]}}
    try:
        response=requests.post('https://safebrowsing.googleapis.com/v4/threatMatches:find',
            params={'key':key},json=payload,timeout=6)
        response.raise_for_status()
        data=response.json()
        matches=data.get('matches',[])
        if matches: return {'status':'malicious','detail':'Matched Google Safe Browsing threat intelligence.',
                            'categories':sorted(set(str(m.get('threatType','UNKNOWN')) for m in matches))}
        return {'status':'no_match','detail':'No current threat match; this does not establish safety.'}
    except (requests.RequestException,ValueError) as exc:
        return {'status':'error','detail':'Reputation lookup unavailable: '+type(exc).__name__}

def calibrated_probability(p):
    """Optional isotonic/sigmoid calibrator fitted on an independent calibration split.
    Calibration joblib must contain a trusted object with predict(array) or predict_proba(array).
    Never label the uncalibrated score calibrated.
    """
    calfile=ROOT/'models'/'probability_calibrator.joblib'
    if not calfile.exists(): return float(p),False
    import joblib
    cal=joblib.load(calfile)  # Only load trusted, locally produced files.
    if hasattr(cal,'predict_proba'): val=float(cal.predict_proba([[p]])[0,1])
    else: val=float(cal.predict([p])[0])
    return max(0.,min(1.,val)),True

PHISHING = 'Phishing Website'
LEGIT = 'Legitimate Website'
SUSPICIOUS = 'Suspicious - Verify First'


def final_verdict(prob, mode, reputation, trust=None, flags=None):
    """Single, explainable decision.

    mode == 'features': the user typed all 15 model inputs -> the model decides (0.5 cut-off).
    mode == 'url'     : only 6 of 15 inputs can be measured from a link, so the ML score alone is
                        not trusted to say "phishing". Evidence is combined:
        1. Safe Browsing match            -> Phishing
        2. well-known registered domain   -> Legitimate (model score kept as secondary info)
        3. model >= 0.5 AND real warning signs (1 high / 2+ any)    -> Phishing
        4. model >= 0.5 without supporting warning signs           -> Suspicious (verify first)
        5. model < 0.5 but a high-risk warning sign               -> Suspicious
        6. otherwise                                              -> Legitimate (unknown domain)
    Returns dict(label, kind, reason, basis). kind in {'danger','safe','unknown'}.
    """
    flags = flags or []
    high = [f for f in flags if f['level'] == 'high']
    pct = round(prob * 100)
    if reputation.get('status') == 'malicious':
        return dict(label=PHISHING, kind='danger', basis='threat_list',
                    reason='Google Safe Browsing lists this address as dangerous.')
    if mode == 'features':
        if prob >= 0.5:
            r = f'The 15 values you entered look like known phishing sites (model score {pct}%).'
            if prob < 0.65: r += ' This is close to the borderline, so double-check the inputs.'
            return dict(label=PHISHING, kind='danger', basis='model', reason=r)
        r = f'The 15 values you entered look like normal legitimate sites (model phishing score {pct}%).'
        if prob > 0.35: r += ' This is close to the borderline, so double-check the inputs.'
        return dict(label=LEGIT, kind='safe', basis='model', reason=r)

    if trust and trust.get('trusted') and not high:
        r = f"{trust['domain']} is a well-known, established domain and the address shows no warning signs."
        if prob >= 0.5:
            r += (f' The AI model alone scored it {pct}%, but a link by itself does not reveal the page '
                  'content or domain history the model relies on, so domain reputation was given priority.')
        return dict(label=LEGIT, kind='safe', basis='trusted_domain', reason=r)
    if prob >= 0.5:
        if high or len(flags) >= 2:
            return dict(label=PHISHING, kind='danger', basis='model+flags',
                        reason=f'The AI model scored {pct}% and the address shows real warning signs.')
        return dict(label=SUSPICIOUS, kind='unknown', basis='model_only',
                    reason=(f'The AI model scored {pct}%, but the address itself shows no clear warning sign '
                            'and the domain is not widely known. Treat it with caution until verified.'))
    if high:
        return dict(label=SUSPICIOUS, kind='unknown', basis='flags_only',
                    reason='The AI model score is low, but the address shows a high-risk warning sign.')
    r = f'The AI model scored only {pct}% and the address shows no clear warning signs.'
    if not (trust and trust.get('trusted')):
        r += ' The domain is not on the well-known list, so this is a probability, not a guarantee.'
    return dict(label=LEGIT, kind='safe', basis='model', reason=r)


def recommendations(kind, reputation):
    common = ['Check the exact web address (spelling of the domain) through an official channel.',
              'Never share passwords, one-time codes or payment details because of an unexpected link.']
    if reputation.get('status') == 'malicious':
        return ['Do not open this link or enter any information.',
                'Report the URL to your security team or a trusted phishing-reporting service.',
                'If you already submitted credentials, change them on the official site and enable MFA.']
    if kind == 'danger':
        return ['Do not enter passwords, OTPs, payment or personal details on this site.',
                "Open the organization's real website by typing its address yourself or using a saved bookmark.",
                'If you already entered a password, change it on the official site and turn on MFA.']
    if kind == 'unknown':
        return ['Do not sign in or pay on this site until you have verified it independently.',
                "Open the organization's service from a bookmark or a manually typed official address."] + common
    return ['No threat was found, but no check can prove a website is 100% safe.'] + common
