from typing import Callable, Dict, Optional
from .score100 import parse_score100_first, parse_score100_last
from .score5 import parse_score5_last, parse_score5_first

ParseFn = Callable[[str], Optional[float]]

PARSER_REGISTRY: Dict[str, ParseFn] = {
    "score100_first": parse_score100_first,
    "score100_last": parse_score100_last,
    "score5_last": parse_score5_last,
    "score5_first": parse_score5_first,
}

def get_parser(name: str) -> ParseFn:
    if name not in PARSER_REGISTRY:
        raise KeyError(f"Unknown parser '{name}'. Available: {sorted(PARSER_REGISTRY.keys())}")
    return PARSER_REGISTRY[name]
