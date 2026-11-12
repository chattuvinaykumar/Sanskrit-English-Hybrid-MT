"""
Paninian Karaka Semantic Role Labeling (SRL) Module.
Maps morphological vibhakti tags to syntactic-semantic Karaka dependencies:
- Kartri (Agent)
- Karma (Patient / Theme)
- Karana (Instrument)
- Sampradana (Recipient)
- Apadana (Ablative / Separation source)
- Adhikarana (Locus / Substrate)
- Sambandha (Genitive relation)
"""

from typing import List, Dict, Any

class KarakaAnalyzer:
    def __init__(self):
        self.case_to_karaka = {
            'prathama': 'Kartri',       # Nominative -> Agent
            'dvitiya': 'Karma',         # Accusative -> Patient / Object
            'tritiya': 'Karana',        # Instrumental -> Instrument
            'chaturthi': 'Sampradana',  # Dative -> Recipient
            'panchami': 'Apadana',      # Ablative -> Source
            'shashthi': 'Sambandha',    # Genitive -> Relation
            'saptami': 'Adhikarana'     # Locative -> Location / Time
        }

    def assign_karaka_roles(self, parsed_tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        annotated = []
        for token in parsed_tokens:
            tok_copy = dict(token)
            pos = tok_copy.get('pos')
            vibhakti = tok_copy.get('vibhakti')

            if pos == 'VERB':
                tok_copy['karaka_role'] = 'KRIYA'
            elif vibhakti in self.case_to_karaka:
                tok_copy['karaka_role'] = self.case_to_karaka[vibhakti]
            else:
                tok_copy['karaka_role'] = 'NONE'

            annotated.append(tok_copy)

        return annotated
