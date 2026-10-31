"""Domain reputation + URL warning signs. No network access, no page fetching.

Why this exists
---------------
The trained model needs 15 inputs but only 6 can be measured from a URL string.
The other 9 (domain age, traffic rank, Google index, page rank, link statistics...)
are filled with training medians, which makes well-known sites such as grok.com look
"average" - and the model then over-reacts to small things such as a missing 'www'.

So a URL scan combines THREE independent signals:
  1. the ML model score,
  2. domain reputation (this file: popularity list match on the *registered domain*),
  3. URL warning signs (IP host, brand look-alikes, shorteners, odd TLDs...).

Trust is granted ONLY on an exact registered-domain match, never on substrings, so
`grok.com.evil.tk` or `paypal-login.xyz` can never borrow trust from a famous name.
"""
import os
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).parent

# ---- public-suffix handling (small built-in list; enough for registered-domain extraction)
MULTI_PART_SUFFIXES = {
    'co.uk', 'org.uk', 'ac.uk', 'gov.uk', 'me.uk', 'com.au', 'net.au', 'org.au', 'edu.au', 'gov.au',
    'co.in', 'net.in', 'org.in', 'gov.in', 'nic.in', 'ac.in', 'edu.in', 'res.in', 'co.jp', 'ne.jp', 'or.jp',
    'ac.jp', 'com.br', 'com.cn', 'com.hk', 'com.tw', 'com.sg', 'com.my', 'com.ph', 'com.vn', 'com.pk',
    'com.bd', 'com.tr', 'com.mx', 'com.ar', 'com.co', 'com.ua', 'com.eg', 'com.ng', 'co.za', 'co.nz',
    'co.kr', 'co.id', 'co.il', 'co.th', 'org.nz', 'gov.sg', 'edu.sg', 'gov.br', 'edu.cn', 'gov.cn',
}

# Registrable domains whose sub-domains / paths anybody can create (blogs, hosting, forms).
# For these, trust is NEVER extended to sub-domains.
MULTI_TENANT = {
    'blogspot.com', 'wordpress.com', 'medium.com', 'tumblr.com', 'substack.com', 'github.io',
    'githubusercontent.com', 'netlify.app', 'vercel.app', 'pages.dev', 'workers.dev', 'web.app',
    'firebaseapp.com', 'herokuapp.com', 'azurewebsites.net', 'amazonaws.com', 'cloudfront.net',
    'weebly.com', 'wixsite.com', 'wix.com', 'squarespace.com', 'myshopify.com', 'notion.site',
    'googleapis.com', 'appspot.com', 'onrender.com', 'glitch.me', 'repl.co', 'ngrok.io',
    'ngrok-free.app', 'trycloudflare.com', '000webhostapp.com', 'godaddysites.com', 'webflow.io',
    'carrd.co', 'sharepoint.com', 'onmicrosoft.com', 'azureedge.net', 'yolasite.com', 'jimdosite.com',
    'canva.site', 'typeform.com',
}

# Individual hosts that are routinely abused for phishing even though the parent is famous.
RISKY_HOSTS = {
    'sites.google.com', 'docs.google.com', 'forms.google.com', 'drive.google.com', 'script.google.com',
    'storage.googleapis.com', 'forms.gle', 'firebasestorage.googleapis.com',
}

# Restricted TLDs: only governments / accredited institutions can register these.
TRUSTED_SUFFIXES = ('.gov', '.edu', '.mil', '.gov.in', '.nic.in', '.ac.in', '.gov.uk', '.ac.uk',
                    '.gov.au', '.edu.au', '.gov.sg', '.edu.sg')

SHORTENERS = {
    'bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co', 'is.gd', 'buff.ly', 'adf.ly', 'bit.do',
    'rebrand.ly', 'cutt.ly', 'shorturl.at', 'tiny.cc', 'lnkd.in', 'qr.ae', 'j.mp', 'rb.gy', 's.id',
    'tr.im', 'u.to', 'cli.gs', 'mcaf.ee', 'db.tt', 'cur.lv', 'ity.im', 'q.gs', 'po.st', 'bc.vc',
}

SUSPICIOUS_TLDS = {
    'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'click', 'link', 'info', 'biz', 'online',
    'site', 'website', 'space', 'pw', 'cc', 'icu', 'rest', 'monster', 'buzz', 'zip', 'mov',
}

BRANDS = [
    'google', 'facebook', 'apple', 'microsoft', 'amazon', 'paypal', 'ebay', 'netflix', 'instagram',
    'whatsapp', 'linkedin', 'twitter', 'yahoo', 'outlook', 'office365', 'icloud', 'dropbox', 'adobe',
    'chase', 'wellsfargo', 'gmail', 'binance', 'coinbase', 'metamask', 'paytm', 'phonepe', 'hdfcbank',
    'icicibank', 'axisbank', 'onlinesbi',
]

HOST_KEYWORDS = ['login', 'signin', 'secure', 'verify', 'update', 'account', 'banking', 'confirm',
                 'wallet', 'authenticate', 'unlock', 'suspend', 'webscr']


# --------------------------------------------------------------------------- helpers
def registered_domain(host):
    host = (host or '').lower().strip('.')
    if _is_ip(host):
        return host
    parts = host.split('.')
    if len(parts) <= 2:
        return host
    if '.'.join(parts[-2:]) in MULTI_PART_SUFFIXES and len(parts) >= 3:
        return '.'.join(parts[-3:])
    return '.'.join(parts[-2:])


def _is_ip(host):
    return bool(re.fullmatch(r'(?:\d{1,3}\.){3}\d{1,3}', host or '')) or (':' in (host or ''))


@lru_cache(maxsize=1)
def _load_trusted():
    """Returns {registered_domain: rank_or_0}. Sources (merged):
       1) data/trusted_domains.txt  - bundled starter list (one domain per line)
       2) models/top_domains.csv    - OPTIONAL Tranco/Alexa-style list ('rank,domain' or 'domain')
          or the path in env TRUST_LIST_PATH. Only the first TRUST_TOP_N rows are used."""
    trusted = {}
    bundled = ROOT / 'data' / 'trusted_domains.txt'
    if bundled.exists():
        for line in bundled.read_text(encoding='utf-8', errors='ignore').splitlines():
            d = line.split('#')[0].strip().lower()
            if d:
                trusted[d] = 0
    extra = Path(os.getenv('TRUST_LIST_PATH', '') or (ROOT / 'models' / 'top_domains.csv'))
    if extra.exists():
        limit = int(os.getenv('TRUST_TOP_N', '100000'))
        try:
            with open(extra, encoding='utf-8', errors='ignore') as fh:
                for i, line in enumerate(fh):
                    if i >= limit:
                        break
                    cells = [c.strip().strip('"').lower() for c in line.strip().split(',')]
                    if not cells or not cells[0]:
                        continue
                    if len(cells) >= 2 and cells[0].isdigit():
                        trusted.setdefault(cells[1], int(cells[0]))
                    elif '.' in cells[0] and not cells[0].isdigit():
                        trusted.setdefault(cells[0], 0)
        except OSError:
            pass
    return trusted


def reload_trusted():
    _load_trusted.cache_clear()
    return len(_load_trusted())


def trust_assessment(url):
    """Is the registered domain of `url` a well-known one? Conservative by design."""
    parts = urlsplit(url)
    host = (parts.hostname or '').lower().rstrip('.')
    reg = registered_domain(host)
    out = {'trusted': False, 'domain': reg, 'rank': None, 'source': None, 'note': ''}
    if not host or _is_ip(host) or 'xn--' in host:
        out['note'] = 'IP-address or look-alike (punycode) hosts are never trusted.'
        return out
    if host in RISKY_HOSTS:
        out['note'] = f'{host} lets anyone publish pages, so the famous parent domain is not a guarantee.'
        return out
    if reg in MULTI_TENANT and host not in (reg, 'www.' + reg):
        out['note'] = f'{reg} hosts content created by anyone, so sub-domains are not trusted.'
        return out
    # brand token in a sub-domain that is not the domain's own name (paypal.com.google.com ...)
    sub = host[:-len(reg)].strip('.') if host.endswith(reg) else ''
    own = reg.split('.')[0]
    sub_tokens = set(re.split(r'[^a-z0-9]+', sub))
    if any(b in sub_tokens and b != own for b in BRANDS):
        out['note'] = 'A different brand name appears in the sub-domain.'
        return out
    trusted = _load_trusted()
    if reg in trusted:
        out.update(trusted=True, rank=trusted[reg] or None,
                   source='popularity list' if trusted[reg] else 'built-in well-known domain list')
        return out
    if host.endswith(TRUSTED_SUFFIXES):
        out.update(trusted=True, source='restricted government/education domain')
    return out


# --------------------------------------------------------------------------- red flags
def url_red_flags(url, trusted=False):
    """Plain-English warning signs. level: 'high' | 'medium'. Independent of the ML model."""
    parts = urlsplit(url)
    host = (parts.hostname or '').lower().rstrip('.')
    full = url.lower()
    reg = registered_domain(host)
    label = reg.split('.')[0] if reg else ''
    tld = host.split('.')[-1] if '.' in host else ''
    flags = []

    def add(code, level, text):
        flags.append({'code': code, 'level': level, 'text': text})

    if _is_ip(host):
        add('ip_host', 'high', 'The address is a raw IP number instead of a normal website name.')
    if 'xn--' in host:
        add('punycode', 'high', 'The domain uses look-alike (punycode) characters that can imitate a real name.')
    if not trusted:
        tokens = [t for t in re.split(r'[^a-z0-9]+', host) if t]
        for b in BRANDS:
            if b in tokens and label != b:
                add('brand_lookalike', 'high', f"The address mentions '{b}' but it is not the official {b} website.")
                break
    if not trusted:
        # a famous registered domain written inside someone else's host, e.g. grok.com.evil.tk
        labels = host.split('.')
        known = _load_trusted()
        for i in range(len(labels) - 1):
            cand = '.'.join(labels[i:i + 2])
            if cand in known and cand != reg and not host.endswith('.' + cand) and host != cand:
                add('embedded_domain', 'high', f"The address contains the name of a well-known site ({cand}) "
                                                f"but actually belongs to {reg}.")
                break
    if host in SHORTENERS or reg in SHORTENERS:
        add('shortener', 'medium', 'This is a shortened link, so the real destination is hidden.')
    if host.count('-') >= 3:
        add('many_hyphens', 'medium', 'The domain name contains many hyphens, a common trick in fake sites.')
    if not trusted and any(k in host for k in HOST_KEYWORDS):
        add('host_keywords', 'medium', "The domain name contains words like 'login', 'secure' or 'verify'.")
    if not trusted and tld in SUSPICIOUS_TLDS:
        add('odd_tld', 'medium', f"The domain ends in '.{tld}', an ending that is frequently abused.")
    if host.count('.') >= 4:
        add('many_subdomains', 'medium', 'The address has an unusually large number of sub-domain levels.')
    if len(url) >= 100:
        add('long_url', 'medium', 'The URL is unusually long.')
    if not trusted and len(label) >= 8 and (sum(c in 'aeiou' for c in label) / len(label) < 0.2
                                            or sum(c.isdigit() for c in label) > 3):
        add('random_name', 'medium', 'The domain name looks randomly generated.')
    if parts.scheme == 'http':
        add('no_https', 'medium', 'The connection is not encrypted (http, not https).')
    return flags
