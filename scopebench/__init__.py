"""ScopeBench framework: task contracts, scope predicate, retrieval, ScopeGuard, scoring.

Implements EXPERIMENT_COMPLETION_PLAN.md §4. The package is deliberately
dependency-light: numpy/pandas only where analysis needs them, the dense
retriever and LLM backends are imported lazily.
"""

__version__ = "0.1.0"
