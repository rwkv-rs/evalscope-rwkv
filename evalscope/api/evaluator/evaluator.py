import abc
from typing import TYPE_CHECKING, Dict

from evalscope.report import Report

if TYPE_CHECKING:
    from evalscope.api.benchmark import DataAdapter
    from evalscope.api.model import Model
    from evalscope.config import TaskConfig
    from evalscope.utils.io_utils import OutputsStructure


class Evaluator(abc.ABC):
    """
    Abstract base class for evaluators.

    Args:
        benchmark (DataAdapter): The data adapter for the benchmark.
        model (Model): The model to evaluate.
        outputs (OutputsStructure, optional): The output structure for results.
        task_config (TaskConfig, optional): The task configuration.
    """

    def __init__(
        self,
        benchmark: 'DataAdapter',
        model: 'Model',
        outputs: 'OutputsStructure' = None,
        task_config: 'TaskConfig' = None,
    ):
        self.benchmark = benchmark
        self.model = model
        self.outputs = outputs
        self.task_config = task_config
        # Retry state managed by ``evalscope.run`` (``--eval-retries``).
        self.has_retryable_errors = False
        """Set when an attempt left failures that ``--eval-retries`` may recover."""
        self.keep_review_reruns_on_failure = False
        """Keep staged review reruns of a failed attempt so a retry can resume them."""
        self.resume_review_reruns: Dict[str, str] = {}
        """Staged review reruns kept by a failed earlier attempt, keyed by review cache path."""
        self.suspended_review_reruns: Dict[str, str] = {}
        """Staged review reruns this attempt kept after failing."""

    @abc.abstractmethod
    def eval(self, *args, **kwargs) -> Report:
        """Run the evaluation process."""
        pass

    @abc.abstractmethod
    def get_report(self, *args, **kwargs) -> Report:
        """Get the evaluation report."""
        pass

    @abc.abstractmethod
    def finalize(self, *args, **kwargs) -> None:
        """Finalize the evaluation process."""
        pass
