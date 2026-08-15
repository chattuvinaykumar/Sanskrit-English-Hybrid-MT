from dataclasses import dataclass
from typing import List, Dict

@dataclass
class LinguisticToken:
    token: str
    root: str = ""
    suffix: str = ""
    case: str = ""
    number: str = ""
    gender: str = ""
    karaka: str = ""

def build_linguistic_features(tokens: List[LinguisticToken]) -> List[Dict[str, str]]:
    """Convert verified linguistic annotations into model-ready records."""
    return [token.__dict__.copy() for token in tokens]

def validate_role_consistency(source_roles: List[str], predicted_roles: List[str]) -> bool:
    """Basic role-consistency hook; replace with the project's actual validator."""
    return source_roles == predicted_roles
