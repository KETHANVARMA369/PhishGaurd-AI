"""The 15 features the notebook's final model was trained on.

Single source of truth for: the manual-input form, the plain-English names shown
in LIME/SHAP, and the "can this be measured from a URL alone?" flag.

Order below == `selected_features` printed by the notebook (original column order).
"""

CANONICAL_15 = [
    'length_url', 'nb_www', 'longest_word_path', 'phish_hints', 'nb_hyperlinks',
    'links_in_tags', 'safe_anchor', 'domain_registration_length', 'domain_age',
    'web_traffic', 'google_index', 'page_rank', 'Suspicious_Score',
    'Length_x_Dots', 'Age_x_Reg',
]

# Features that can really be computed from the URL text itself.
# Everything else needs page content, WHOIS/DNS or ranking services.
URL_MEASURABLE = {
    'length_url', 'nb_www', 'longest_word_path', 'phish_hints',
    'Suspicious_Score', 'Length_x_Dots',
}

GROUPS = [
    ('URL structure', 'Can be read straight from the web address.',
     ['length_url', 'nb_www', 'longest_word_path', 'phish_hints', 'Suspicious_Score', 'Length_x_Dots']),
    ('Page content', 'Measured from the web page itself.',
     ['nb_hyperlinks', 'links_in_tags', 'safe_anchor']),
    ('Domain & reputation', 'From WHOIS / ranking / search-index data.',
     ['domain_registration_length', 'domain_age', 'Age_x_Reg', 'web_traffic', 'google_index', 'page_rank']),
]

# label = plain-English name; help = one-line explanation; step/min/max for the <input>
FEATURE_INFO = {
    'length_url': dict(label='URL length', help='Total number of characters in the full URL.',
                       step='1', min='0', example_ok=22, example_bad=85),
    'nb_www': dict(label="'www' count", help="How many times 'www' appears in the URL.",
                   step='1', min='0', example_ok=1, example_bad=0),
    'longest_word_path': dict(label='Longest word in path', help='Length of the longest word in the URL path/query.',
                              step='1', min='0', example_ok=0, example_bad=18),
    'phish_hints': dict(label='Suspicious keywords', help='Count of words like login, verify, secure, update, account.',
                        step='1', min='0', example_ok=0, example_bad=3),
    'nb_hyperlinks': dict(label='Links on the page', help='Total number of hyperlinks found on the page.',
                          step='1', min='0', example_ok=120, example_bad=3),
    'links_in_tags': dict(label='Links in meta/script/link tags (%)', help='Percentage of links placed inside <meta>, <script> and <link> tags.',
                          step='any', min='0', max='100', example_ok=40, example_bad=0),
    'safe_anchor': dict(label='Unsafe anchor links (%)', help="Percentage of anchors that point nowhere safe (e.g. '#', javascript:void).",
                        step='any', min='0', max='100', example_ok=8, example_bad=70),
    'domain_registration_length': dict(label='Domain registration length (days)', help='Days until the domain registration expires.',
                                       step='any', example_ok=400, example_bad=20),
    'domain_age': dict(label='Domain age (days)', help='How many days ago the domain was first registered.',
                       step='any', example_ok=7000, example_bad=30),
    'web_traffic': dict(label='Website traffic rank', help='Traffic-rank number used in the dataset (Alexa-style). A smaller number means a more popular site.',
                        step='any', min='0', example_ok=500, example_bad=0),
    'google_index': dict(label='Google index status', help='Dataset convention: 0 = page is indexed by Google, 1 = not indexed.',
                         step='1', min='0', max='1', kind='binary', example_ok=0, example_bad=1),
    'page_rank': dict(label='PageRank score', help='Reputation score from 0 to 10. Higher means more reputable.',
                      step='any', min='0', max='10', example_ok=6, example_bad=0),
    'Suspicious_Score': dict(label='Combined suspicious score', help='Sum of warning flags (IP host, @, hyphen, shortener, keywords, odd TLD, many subdomains).',
                             step='any', min='0', example_ok=0, example_bad=4),
    'Length_x_Dots': dict(label='URL length × dots', help='URL length multiplied by the number of dots in the URL.',
                          step='any', min='0', example_ok=44, example_bad=340),
    'Age_x_Reg': dict(label='Domain age × registration length', help='Domain age multiplied by registration length (both in days).',
                      step='any', min='0', example_ok=2800000, example_bad=600),
}


def label_of(name):
    return FEATURE_INFO.get(name, {}).get('label', name.replace('_', ' ').capitalize())


def manual_fields(medians=None):
    """Grouped form-field definitions for the manual 15-feature page."""
    medians = medians or {}
    groups = []
    for title, blurb, names in GROUPS:
        fields = []
        for n in names:
            info = dict(FEATURE_INFO[n])
            info['name'] = n
            med = medians.get(n)
            try:
                info['default'] = ('%g' % float(med)) if med is not None else ''
            except (TypeError, ValueError):
                info['default'] = ''
            fields.append(info)
        groups.append({'title': title, 'blurb': blurb, 'fields': fields})
    return groups


def presets():
    return {
        'legit': {n: FEATURE_INFO[n]['example_ok'] for n in CANONICAL_15},
        'phish': {n: FEATURE_INFO[n]['example_bad'] for n in CANONICAL_15},
    }
