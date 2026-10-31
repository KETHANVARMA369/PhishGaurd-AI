import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse, parse_qs

# ============================================================
# Offline lexical feature extraction for ANY external URL
# ============================================================

SHORTENERS = {
    'bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co', 'is.gd', 'buff.ly',
    'adf.ly', 'bit.do', 'mcaf.ee', 'rebrand.ly', 'cutt.ly', 'shorturl.at',
    'tiny.cc', 'lnkd.in', 'db.tt', 'qr.ae', 'cur.lv', 'ity.im', 'q.gs',
    'po.st', 'bc.vc', 'twitthis.com', 'u.to', 'j.mp', 'buzurl.com',
    'vzturl.com', 'tr.im', 'cli.gs', 'rb.gy', 's.id'
}

PHISH_HINTS = [
    'login', 'signin', 'sign-in', 'account', 'secure', 'update', 'verify',
    'banking', 'confirm', 'password', 'paypal', 'appleid', 'ebay', 'wallet',
    'invoice', 'suspend', 'limited', 'unlock', 'security', 'authenticate',
    'webscr', 'cgi-bin', 'wp-admin', 'wp-login'
]

BRANDS = [
    'google', 'facebook', 'apple', 'microsoft', 'amazon', 'paypal', 'ebay',
    'netflix', 'instagram', 'whatsapp', 'linkedin', 'twitter', 'yahoo',
    'outlook', 'office365', 'icloud', 'dropbox', 'adobe', 'chase', 'wellsfargo'
]

SUSPICIOUS_TLDS = {
    'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'click', 'link',
    'info', 'biz', 'online', 'site', 'website', 'space', 'pw', 'cc', 'icu'
}

def _safe_parse(url: str):
    if not isinstance(url, str):
        url = str(url)
    url = url.strip()
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', url):
        url = 'http://' + url
    return urlparse(url)

def _is_ip(host: str) -> int:
    if not host:
        return 0
    # IPv4
    if re.fullmatch(r'(?:\d{1,3}\.){3}\d{1,3}', host):
        return 1
    # IPv6 rough
    if ':' in host and re.fullmatch(r'[0-9a-fA-F:]+', host):
        return 1
    return 0

def _tokenize_words(text: str):
    if not text:
        return []
    return [w for w in re.split(r'[^A-Za-z0-9]+', text) if w]

def extract_lexical_features(url: str) -> dict:
    """Compute lexical / structural features from a raw URL string."""
    parsed = _safe_parse(url)
    full = url if isinstance(url, str) else str(url)
    scheme = (parsed.scheme or '').lower()
    host = (parsed.hostname or '').lower()
    path = parsed.path or ''
    query = parsed.query or ''
    port = parsed.port

    path_query = path + (('?' + query) if query else '')
    words_host = _tokenize_words(host)
    words_path = _tokenize_words(path_query)
    words_all = words_host + words_path

    digits_url = sum(c.isdigit() for c in full)
    digits_host = sum(c.isdigit() for c in host)

    # TLD
    tld = host.split('.')[-1] if '.' in host else ''
    subdomains = host.split('.')[:-2] if host.count('.') >= 2 else []

    # Prefix-suffix: hyphen in domain label
    domain_label = host.split('.')[-2] if host.count('.') >= 1 else host
    prefix_suffix = 1 if '-' in domain_label else 0

    # Random-looking domain (high consonant ratio / digit mix heuristic)
    random_domain = 0
    if domain_label and len(domain_label) >= 8:
        vowels = sum(c in 'aeiou' for c in domain_label)
        if vowels / max(len(domain_label), 1) < 0.2 or sum(c.isdigit() for c in domain_label) > 3:
            random_domain = 1

    shortening = 1 if any(s in host for s in SHORTENERS) else 0

    path_ext = 1 if re.search(r'\.(txt|exe|js)$', path, re.I) else 0

    phish_count = sum(1 for h in PHISH_HINTS if h in full.lower())

    brand_in_path = 1 if any(b in path.lower() for b in BRANDS) else 0
    brand_in_sub = 1 if any(b in '.'.join(subdomains) for b in BRANDS) else 0
    domain_in_brand = 1 if any(b in domain_label for b in BRANDS) else 0

    # Char repeat (max consecutive identical chars)
    char_repeat = 0
    if full:
        cur, best = 1, 1
        for i in range(1, len(full)):
            if full[i] == full[i - 1]:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        char_repeat = best

    def _stats(words):
        if not words:
            return 0, 0, 0.0
        lengths = [len(w) for w in words]
        return min(lengths), max(lengths), float(np.mean(lengths))

    sw_raw, lw_raw, aw_raw = _stats(words_all)
    sw_host, lw_host, aw_host = _stats(words_host)
    sw_path, lw_path, aw_path = _stats(words_path)

    feats = {
        'length_url': len(full),
        'length_hostname': len(host),
        'ip': _is_ip(host),
        'nb_dots': full.count('.'),
        'nb_hyphens': full.count('-'),
        'nb_at': full.count('@'),
        'nb_qm': full.count('?'),
        'nb_and': full.count('&'),
        'nb_or': full.count('|'),
        'nb_eq': full.count('='),
        'nb_underscore': full.count('_'),
        'nb_tilde': full.count('~'),
        'nb_percent': full.count('%'),
        'nb_slash': full.count('/'),
        'nb_star': full.count('*'),
        'nb_colon': full.count(':'),
        'nb_comma': full.count(','),
        'nb_semicolumn': full.count(';'),
        'nb_dollar': full.count('$'),
        'nb_space': full.count(' ') + full.count('%20'),
        'nb_www': 1 if 'www.' in host else (host.count('www')),
        'nb_com': full.lower().count('.com'),
        'nb_dslash': full.count('//'),
        'http_in_path': 1 if 'http' in path.lower() else 0,
        'https_token': 1 if scheme == 'http' else 0,  # dataset: 1 means NOT https (legacy encoding)
        'ratio_digits_url': digits_url / max(len(full), 1),
        'ratio_digits_host': digits_host / max(len(host), 1),
        'punycode': 1 if 'xn--' in host else 0,
        'port': 1 if port is not None else 0,
        'tld_in_path': 1 if tld and tld in path.lower() else 0,
        'tld_in_subdomain': 1 if any(tld == s for s in subdomains) else 0,
        'abnormal_subdomain': 1 if len(subdomains) >= 3 else 0,
        'nb_subdomains': max(host.count('.'), 0),
        'prefix_suffix': prefix_suffix,
        'random_domain': random_domain,
        'shortening_service': shortening,
        'path_extension': path_ext,
        'nb_redirection': full.count('//') - 1 if full.count('//') > 1 else 0,
        'nb_external_redirection': 0,
        'length_words_raw': len(words_all),
        'char_repeat': char_repeat,
        'shortest_words_raw': sw_raw,
        'shortest_word_host': sw_host,
        'shortest_word_path': sw_path,
        'longest_words_raw': lw_raw,
        'longest_word_host': lw_host,
        'longest_word_path': lw_path,
        'avg_words_raw': aw_raw,
        'avg_word_host': aw_host,
        'avg_word_path': aw_path,
        'phish_hints': phish_count,
        'domain_in_brand': domain_in_brand,
        'brand_in_subdomain': brand_in_sub,
        'brand_in_path': brand_in_path,
        'suspecious_tld': 1 if tld in SUSPICIOUS_TLDS else 0,
        'statistical_report': 0,
    }
    return feats

def url_to_feature_vector(url: str, selected_features, train_medians) -> pd.DataFrame:
    """
    Build a 1-row DataFrame aligned with selected_features.
    Lexical features are extracted; missing content/WHOIS features use train medians.
    """
    lex = extract_lexical_features(url)

    # Engineered features used in training
    lex['Suspicious_Score'] = (
        lex.get('ip', 0) + lex.get('nb_at', 0) + lex.get('prefix_suffix', 0) +
        lex.get('shortening_service', 0) + lex.get('phish_hints', 0) +
        lex.get('suspecious_tld', 0) + lex.get('abnormal_subdomain', 0)
    )
    lex['Length_x_Dots'] = lex['length_url'] * lex['nb_dots']
    lex['Digits_x_Slash'] = lex['ratio_digits_url'] * lex['nb_slash']
    lex['Age_x_Reg'] = train_medians.get('Age_x_Reg', 0)

    row = {}
    for f in selected_features:
        if f in lex:
            row[f] = lex[f]
        else:
            row[f] = train_medians.get(f, 0.0)
    return pd.DataFrame([row], columns=selected_features)

def predict_url(url: str, return_details: bool = True):
    """
    Score a single external URL with RF, XGB, Deep MSRA and Novelty fusion.
    Returns a dict with probabilities and final label.
    """
    X_df = url_to_feature_vector(url, selected_features, TRAIN_MEDIANS)

    p_rf  = float(rf.predict_proba(X_df)[0, 1])
    p_xgb = float(xgb.predict_proba(X_df)[0, 1])

    X_scaled = scaler.transform(X_df).astype(np.float32)
    X_seq = X_scaled.reshape(-1, X_scaled.shape[1], 1)
    p_deep = float(msra.predict(X_seq, verbose=0).ravel()[0])

    meta_vec = np.array([[p_rf, p_xgb, p_deep]])
    p_nov = float(meta.predict_proba(meta_vec)[0, 1])
    label = int(p_nov >= 0.5)

    out = {
        'url': url,
        'probability_phishing': round(p_nov, 4),
        'prediction': 'phishing' if label == 1 else 'legitimate',
        'components': {
            'RandomForest': round(p_rf, 4),
            'XGBoost': round(p_xgb, 4),
            'MSRA_Deep': round(p_deep, 4),
            'Novelty_Fusion': round(p_nov, 4),
        }
    }
    if return_details:
        out['features_used'] = list(selected_features)
    return out

