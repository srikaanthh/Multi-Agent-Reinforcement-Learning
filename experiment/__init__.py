from .runner import ExperimentRunner, compute_summary_metrics, load_completed_task_ids


def run_ablation_from_config(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Lazy import so `import experiment` stays lightweight (YAML / ablation deps)."""
    from .ablation import run_ablation_from_config as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "ExperimentRunner",
    "compute_summary_metrics",
    "load_completed_task_ids",
    "run_ablation_from_config",
]
