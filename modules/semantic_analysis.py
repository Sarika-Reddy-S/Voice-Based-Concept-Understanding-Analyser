"""
semantic_analysis.py
---------------------
BACKWARD-COMPATIBILITY shim — delegates to semantic_eval.py.

This file is preserved so existing imports of `modules.semantic_analysis`
continue to work. All new code should import from `modules.semantic_eval`.
"""

from .semantic_eval import (
    SemanticResult,
    SemanticEvaluator,
    get_semantic_evaluator,
    semantic_similarity,
)

__all__ = [
    "SemanticResult",
    "SemanticEvaluator",
    "get_semantic_evaluator",
    "semantic_similarity",
]
