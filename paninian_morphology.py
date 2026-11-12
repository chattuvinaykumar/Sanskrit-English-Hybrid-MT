"""
Paninian Morphological Feature Extraction Engine.
Extracts roots (dhatu), affixes (pratyaya), nominal case inflections (vibhakti),
gender (linga), number (vachana), and verbal tense/mood (lakara).
Grounded in Paninian Ashtadhyayi paradigms.
"""

from typing import Dict, Any, List

class PaninianMorphologicalEngine:
    def __init__(self):
        # Explicit priority: Longest suffixes checked first to prevent substring collisions
        self.raw_vibhakti = [
            ('ebhyah', ('dative/ablative', 'plural', 'chaturthi/panchami')),
            ('bhyam',  ('inst/dat/abl', 'dual', 'tritiya/chaturthi/panchami')),
            ('anam',   ('genitive', 'plural', 'shashthi')),
            ('asya',   ('genitive', 'singular', 'shashthi')),
            ('eshu',   ('locative', 'plural', 'saptami')),
            ('ena',    ('instrumental', 'singular', 'tritiya')),
            ('aya',    ('dative', 'singular', 'chaturthi')),
            ('aih',    ('instrumental', 'plural', 'tritiya')),
            ('at',     ('ablative', 'singular', 'panchami')),
            ('am',     ('accusative', 'singular', 'dvitiya')),
            ('an',     ('accusative', 'plural', 'dvitiya')),
            ('ah',     ('nominative', 'singular', 'prathama')),
            ('au',     ('nominative/accusative', 'dual', 'prathama/dvitiya')),
            ('e',      ('locative', 'singular', 'saptami')),
        ]
        self.vibhakti_patterns = sorted(self.raw_vibhakti, key=lambda x: len(x[0]), reverse=True)

        self.raw_lakara = [
            ('ante', ('present', '3rd', 'plural', 'lat', 'atmanepada')),
            ('anti', ('present', '3rd', 'plural', 'lat', 'parasmaipada')),
            ('nti',  ('present', '3rd', 'plural', 'lat', 'parasmaipada')),
            ('tah',  ('present', '3rd', 'dual', 'lat', 'parasmaipada')),
            ('mah',  ('present', '1st', 'plural', 'lat', 'parasmaipada')),
            ('ti',   ('present', '3rd', 'singular', 'lat', 'parasmaipada')),
            ('si',   ('present', '2nd', 'singular', 'lat', 'parasmaipada')),
            ('mi',   ('present', '1st', 'singular', 'lat', 'parasmaipada')),
        ]
        self.lakara_patterns = sorted(self.raw_lakara, key=lambda x: len(x[0]), reverse=True)

        self.known_roots = {
            'ram': 'rama', 'ravan': 'ravana', 'ban': 'bana',
            'han': 'han', 'gam': 'gam', 'path': 'path',
            'krid': 'krid', 'pach': 'pach', 'raksh': 'raksh',
            'ish': 'isha', 'dharma': 'dharma', 'vidya': 'vidya',
            'balak': 'balaka', 'shikshak': 'shikshaka'
        }

    def analyze_token(self, token: str) -> Dict[str, Any]:
        t_clean = token.lower().strip(",.|;:!?\"'।॥")
        
        # Verb matching
        for suffix, details in self.lakara_patterns:
            if t_clean.endswith(suffix) and len(t_clean) > len(suffix):
                stem = t_clean[:-len(suffix)]
                root = self.known_roots.get(stem, stem)
                return {
                    'token': token,
                    'pos': 'VERB',
                    'root': root,
                    'tense': details[0],
                    'person': details[1],
                    'number': details[2],
                    'lakara': details[3],
                    'vibhakti': None,
                    'morph_tag': f"Verb|{details[3]}|{details[0]}|{details[1]}|{details[2]}"
                }

        # Noun matching (Greedy longest suffix first)
        for suffix, details in self.vibhakti_patterns:
            if t_clean.endswith(suffix) and len(t_clean) > len(suffix):
                stem = t_clean[:-len(suffix)]
                root = self.known_roots.get(stem, stem)
                return {
                    'token': token,
                    'pos': 'NOUN',
                    'root': root,
                    'case': details[0],
                    'number': details[1],
                    'vibhakti': details[2],
                    'morph_tag': f"Noun|{details[2]}|{details[0]}|{details[1]}"
                }

        return {
            'token': token,
            'pos': 'NOUN_STEM',
            'root': t_clean,
            'case': 'prathama',
            'number': 'singular',
            'vibhakti': 'prathama',
            'morph_tag': 'STEM|prathama'
        }

    def parse_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        return [self.analyze_token(t) for t in sentence.split()]
