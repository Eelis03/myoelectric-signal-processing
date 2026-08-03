"""Analysis layer: detector metrics, fatigue statistics, delay budgets, tables, figures.

This layer consumes traces from the pipeline layer and produces numbers and figures. It
runs no simulation of its own, so a reported number can always be traced back to the
pipeline call that produced it.
"""

from myoelectric.analysis.delay_budget import (
    DelayBudget,
    DelayBudgetExceededError,
    DelayStage,
    assemble_budget,
    detector_stage,
    enforce,
    envelope_stage,
    filter_stage,
    fixed_stage,
    format_budget_table,
)
from myoelectric.analysis.detector_metrics import (
    DetectorMetrics,
    format_metrics_table,
    summarise_sweep,
)
from myoelectric.analysis.fatigue_stats import (
    FatigueTrend,
    analyse_fatigue,
    format_fatigue_summary,
)
from myoelectric.analysis.reporting import (
    format_filter_response_table,
    format_frequency_feature_table,
    format_latency_table,
    format_time_feature_table,
)

__all__ = [
    "DelayBudget",
    "DelayBudgetExceededError",
    "DelayStage",
    "DetectorMetrics",
    "FatigueTrend",
    "analyse_fatigue",
    "assemble_budget",
    "detector_stage",
    "enforce",
    "envelope_stage",
    "filter_stage",
    "fixed_stage",
    "format_budget_table",
    "format_fatigue_summary",
    "format_filter_response_table",
    "format_frequency_feature_table",
    "format_latency_table",
    "format_metrics_table",
    "format_time_feature_table",
    "summarise_sweep",
]
