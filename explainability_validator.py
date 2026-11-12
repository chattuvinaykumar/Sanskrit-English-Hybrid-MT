"""
Post-Hoc Explainability Validation Layer.
Validates structural correspondence between source Karaka semantic roles
and the generated English hypothesis using bilingual lemma alignment.
"""

from typing import Dict, Any, List

class ExplainabilityValidator:
    def __init__(self):
        # Bilingual lemma dictionary to evaluate English lexicalizations
        self.lemma_dict = {
            'rama': ['rama'],
            'ravana': ['ravana'],
            'bana': ['arrow', 'dart'],
            'balaka': ['boy', 'child'],
            'vidyalaya': ['school', 'academy'],
            'pustaka': ['book'],
            'shikshaka': ['teacher', 'guru'],
            'dharma': ['dharma', 'righteousness', 'duty'],
            'satya': ['truth'],
            'bhojana': ['food', 'meal'],
            'vriksha': ['tree'],
            'himalaya': ['himalaya']
        }

    def validate_alignment(
        self,
        source_karaka_list: List[Dict[str, Any]],
        target_hypothesis: str
    ) -> Dict[str, Any]:
        hyp_lower = target_hypothesis.lower().strip()
        tokens = hyp_lower.split()

        kartri_roots = [t['root'].lower() for t in source_karaka_list if t.get('karaka_role') == 'Kartri']
        karma_roots = [t['root'].lower() for t in source_karaka_list if t.get('karaka_role') == 'Karma']
        karana_roots = [t['root'].lower() for t in source_karaka_list if t.get('karaka_role') == 'Karana']

        report = {
            'status': 'PASSED',
            'explanation': [],
            'soft_flags': [],
            'preserved_roles': []
        }

        # Check Kartri (Agent)
        if kartri_roots:
            agent_root = kartri_roots[0]
            expected_words = self.lemma_dict.get(agent_root, [agent_root])
            if any(w in hyp_lower for w in expected_words):
                if ' by ' in hyp_lower:
                    report['explanation'].append("Kartri (Agent) realized via passive voice construction (Approximated).")
                    report['preserved_roles'].append('Kartri (Passive)')
                else:
                    report['explanation'].append(f"Kartri (Agent: '{agent_root}') is preserved as principal subject.")
                    report['preserved_roles'].append('Kartri')
            else:
                report['soft_flags'].append(f"Kartri root '{agent_root}' not identified as subject in target.")

        # Check Karma (Patient)
        if karma_roots:
            patient_root = karma_roots[0]
            expected_words = self.lemma_dict.get(patient_root, [patient_root])
            if any(w in hyp_lower for w in expected_words):
                report['explanation'].append(f"Karma (Patient: '{patient_root}') preserved as direct argument.")
                report['preserved_roles'].append('Karma')
            else:
                report['soft_flags'].append(f"Karma root '{patient_root}' realization ambiguous.")

        # Check Karana (Instrument)
        if karana_roots:
            if 'with' in tokens or 'by' in tokens or 'using' in tokens:
                report['explanation'].append("Karana (Instrument) realized via appropriate prepositional modifier.")
                report['preserved_roles'].append('Karana')

        if report['soft_flags']:
            report['status'] = 'FLAGGED'

        return report
