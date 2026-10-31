"""Short, plain-English result summary. Deterministic (no LLM), so it ALWAYS displays."""

HEADLINES = {
    'danger': 'This looks like a phishing website.',
    'safe': 'This looks like a legitimate website.',
    'unknown': "We can't be sure - treat this website with caution.",
}


def _driver_phrase(item):
    val = f" ({item['actual']})" if item.get('actual') != '' and item.get('actual') is not None else ''
    if item['dir'] == 'phishing':
        return f"{item['label']}{val} pushes the result toward phishing."
    return f"{item['label']}{val} points toward a legitimate site."


def build_summary(verdict, mode, prob, trust=None, flags=None, drivers=None, measured=6, total=15):
    kind = verdict['kind']
    flags = sorted(flags or [], key=lambda f: f['level'] != 'high')
    reasons = []

    if mode == 'features':
        for d in (drivers or [])[:3]:
            reasons.append(_driver_phrase(d))
        if not reasons:
            reasons.append(f'The model gave these 15 values a phishing score of {round(prob * 100)}%.')
    else:
        if trust and trust.get('trusted'):
            src = trust.get('source') or 'well-known domain list'
            reasons.append(f"{trust['domain']} is a recognised, established domain ({src}).")
        if flags:
            reasons += [f['text'] for f in flags[:2]]
        elif not (trust and trust.get('trusted')):
            reasons.append('The web address shows no obvious tricks (no IP number, look-alike brand names or odd endings).')
        else:
            reasons.append('The web address itself shows no warning signs.')
        reasons.append(f'AI model score: {round(prob * 100)}% phishing. It can measure only {measured} of its {total} '
                       'signals from a link alone, so domain reputation and address checks were used too.')
    reasons = reasons[:3] if mode == 'features' else reasons[:4]

    if kind == 'danger':
        action = ("Don't enter passwords or personal details. Go to the official website by typing its address "
                  'yourself.')
    elif kind == 'unknown':
        action = "Don't log in or pay until you have verified the site through an official channel."
    elif trust and trust.get('trusted'):
        action = 'It is fine to continue, but always check the address bar before you log in.'
    else:
        action = 'Looks fine, but double-check the spelling of the address before you sign in.'

    return {'kind': kind, 'headline': HEADLINES[kind], 'reason': verdict['reason'],
            'reasons': reasons, 'action': action}
