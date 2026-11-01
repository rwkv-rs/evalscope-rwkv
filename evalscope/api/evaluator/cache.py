import copy
import hashlib
import json
import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from evalscope.api.agent import AgentTrace
from evalscope.api.dataset import Dataset
from evalscope.api.messages import ChatMessage, messages_pretty_str, messages_to_markdown
from evalscope.api.metric import SampleScore
from evalscope.api.model import ModelOutput
from evalscope.constants import ScoreStatus
from evalscope.utils.io_utils import JsonlWriter, OutputsStructure, convert_normal_types
from evalscope.utils.logger import get_logger

from .state import TaskState

logger = get_logger()


_UNSTABLE_MESSAGE_FIELDS = {'id', 'metadata', 'internal', 'perf_metrics'}


class CacheIdentityError(RuntimeError):
    """A cached prediction does not belong to the sample now at its index."""


def sample_fingerprint(sample: Any) -> str:
    """Hash the sample content that determines the request and the grading target.

    Message ids are random, so only stable message fields are included. The positional
    ``id`` is excluded on purpose: it is what the fingerprint verifies.
    """
    if isinstance(sample.input, str):
        model_input: Any = sample.input
    else:
        model_input = [message.model_dump(mode='json', exclude=_UNSTABLE_MESSAGE_FIELDS) for message in sample.input]
    payload = {
        'input': model_input,
        'target': sample.target,
        'choices': sample.choices,
        'tools': [tool.model_dump(mode='json') for tool in sample.tools] if sample.tools else None,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def review_needs_retry(sample_score: SampleScore) -> bool:
    """Retry unavailable transport results, never real zeroes or rejected judge replies."""
    score = sample_score.score
    if score.status.is_usable:
        return False
    return score.status is ScoreStatus.TRANSPORT_ERROR or bool(
        score.judge_summary and score.judge_summary.failures.get(ScoreStatus.TRANSPORT_ERROR.value, 0)
    )


def _review_matches_prediction(review: 'ReviewResult', state: TaskState) -> bool:
    """Whether ``review`` graded exactly the prediction restored in ``state``."""
    if state.prediction_id is not None:
        return review.prediction_id == state.prediction_id
    # Legacy prediction without an id: link only when no regeneration could have happened.
    return review.prediction_id is None and state.legacy_unique_prediction is not False


class CacheManager:
    """
    Manage model results and review results for evaluation caching.

    This class handles the caching mechanism for evaluation results, allowing
    the system to resume evaluations from previously computed results and
    avoid redundant computations.
    """

    def __init__(
        self,
        outputs: OutputsStructure,
        model_name: str,
        benchmark_name: str,
    ):
        """
        Initialize the cache manager.

        Args:
            outputs: Output directory structure for storing cache files
            model_name: Name of the model being evaluated
            benchmark_name: Name of the benchmark being used
        """
        self.outputs = outputs
        self.model_name = model_name
        self.benchmark_name = benchmark_name
        self._writers: Dict[str, JsonlWriter] = {}
        self._review_reruns: Dict[str, str] = {}
        self._write_lock = threading.RLock()

    def _get_writer(self, cache_file: str) -> JsonlWriter:
        """Return a persistent writer for *cache_file*, opening on first use.

        Keeping a single writer open per file avoids the rapid open/close
        cycle that triggers ``PermissionError`` on Windows due to file-lock
        release latency.
        """
        cache_file = os.path.expanduser(cache_file)
        if cache_file not in self._writers:
            self._writers[cache_file] = JsonlWriter(cache_file, durable=True)
        return self._writers[cache_file]

    def _read_cache(self, cache_file: str) -> List[Dict[str, Any]]:
        """Read cached records, skipping malformed rows and repairing an interrupted final append.

        The file is reopened for writing only when its last line lacks a newline, so complete
        read-only caches remain usable. A torn final record is removed so the next append
        cannot merge with it.
        """
        records: List[Dict[str, Any]] = []
        line, torn = b'', False
        with self._write_lock:
            with open(cache_file, 'rb') as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        if not line.endswith(b'\n'):
                            torn = True
                        logger.warning(f'Skipping invalid JSONL row {line_number} in {cache_file}: {e}')
                        continue
                    if not isinstance(record, dict):
                        logger.warning(f'Skipping invalid JSONL row {line_number} in {cache_file}: not an object')
                        continue
                    records.append(record)
            if line and not line.endswith(b'\n'):
                self._repair_tail(cache_file, line, torn)
        return records

    @staticmethod
    def _repair_tail(cache_file: str, tail: bytes, torn: bool) -> None:
        with open(cache_file, 'rb+') as stream:
            end = stream.seek(0, os.SEEK_END)
            if torn:
                stream.truncate(end - len(tail))
                logger.warning(f'Removed interrupted final cache record from {cache_file}.')
            else:
                # A complete record without its delimiter must not merge with the next append.
                stream.write(b'\n')
            stream.flush()
            os.fsync(stream.fileno())

    def close(self) -> None:
        """Close all open writers. Idempotent — safe to call multiple times."""
        with self._write_lock:
            for writer in self._writers.values():
                writer.close()
            self._writers.clear()

    def __del__(self) -> None:
        """Best-effort cleanup if :meth:`close` was not called explicitly."""
        try:
            self.close()
        except Exception:  # noqa: BLE01 - never propagate from __del__
            pass

    def filter_prediction_cache(self, subset: str, dataset: Dataset) -> Tuple[List[TaskState], Dataset]:
        """
        Load cached prediction results and filter them from the dataset.

        This method checks for existing prediction cache files and loads any
        previously computed results. It then filters these samples from the
        input dataset to avoid recomputation.

        Args:
            subset: Name of the dataset subset
            dataset: The dataset to filter

        Returns:
            Tuple of (cached task states, filtered dataset with remaining samples)
        """
        cache_file = self.get_prediction_cache_path(subset)
        if not os.path.exists(cache_file):
            # No cache file exists, return empty cache and full dataset
            return [], dataset

        cached_by_sample_id = {}
        duplicate_rows = 0
        cache_items = self._read_cache(cache_file)
        rows_per_index: Dict[Any, int] = {}
        for cache_item in cache_items:
            rows_per_index[cache_item.get('index')] = rows_per_index.get(cache_item.get('index'), 0) + 1

        # Process each cached item
        for cache_item in cache_items:
            # Deserialize the cached model result
            try:
                cached_model_result = ModelResult.model_validate(cache_item)
            except ValidationError as e:
                logger.warning(f'Skipping invalid prediction cache row in {cache_file}: {e}')
                continue
            if cached_model_result.model_output is None:
                logger.warning(f'Skipping prediction cache row without model output in {cache_file}.')
                continue
            if cached_model_result.model_output.error:
                # A failed later attempt supersedes earlier rows; the sample is regenerated.
                cached_by_sample_id.pop(cached_model_result.index, None)
                continue
            # Convert to task state for further processing
            try:
                cached_state = cached_model_result.to_task_state(dataset=dataset)
            except ValidationError as e:
                logger.warning(f'Skipping invalid prediction cache row in {cache_file}: {e}')
                continue

            if cached_state is None:
                continue
            if cached_model_result.prediction_id is None:
                cached_state.legacy_unique_prediction = rows_per_index.get(cached_model_result.index) == 1
            if cached_state.sample_id in cached_by_sample_id:
                duplicate_rows += 1
            cached_by_sample_id[cached_state.sample_id] = cached_state

        cached_task_states = list(cached_by_sample_id.values())
        if duplicate_rows:
            logger.warning(f'Dropped {duplicate_rows} duplicate rows from prediction cache: {cache_file}')

        # Remove cached samples from the dataset to avoid reprocessing
        cached_sample_ids = set(cached_by_sample_id)
        filtered_dataset = dataset.filter(lambda sample: sample.id not in cached_sample_ids)

        logger.info(
            f'Reusing predictions from {cache_file}, got {len(cached_task_states)} predictions, '
            f'remaining {len(filtered_dataset)} samples'
        )
        return cached_task_states, filtered_dataset

    def get_prediction_cache_path(self, subset: str) -> str:
        """
        Get the file path for prediction cache storage.

        Args:
            subset: Name of the dataset subset

        Returns:
            Path to the prediction cache file
        """
        file_path = os.path.join(self.outputs.predictions_dir, self.model_name, f'{self.benchmark_name}_{subset}.jsonl')
        # Ensure the directory exists
        if self.outputs.is_make:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    def save_prediction_cache(self, subset: str, task_state: TaskState, save_metadata: bool = True) -> 'ModelResult':
        """
        Save a prediction result to the cache.

        Args:
            subset: Name of the dataset subset
            task_state: The task state containing prediction results

        Returns:
            The saved model result object
        """
        cache_file = self.get_prediction_cache_path(subset)
        # Every persisted generation gets a new identity, so a review written for an earlier
        # generation of the same sample can never be applied to this one.
        task_state.prediction_id = uuid.uuid4().hex
        # Convert task state to serializable model result
        model_result = ModelResult.from_task_state(task_state, save_metadata)
        # Serialize to dictionary, convert non-JSON types (numpy, datetime), append.
        model_result_dict = convert_normal_types(model_result.model_dump())
        with self._write_lock:
            self._get_writer(cache_file).write(model_result_dict)
        return model_result

    def filter_review_cache(
        self, subset: str, task_states: List[TaskState]
    ) -> Tuple[List[SampleScore], List[TaskState]]:
        """
        Load cached review results and filter corresponding task states.

        This method loads previously computed review scores and removes
        the corresponding task states from further review processing.

        Args:
            subset: Name of the dataset subset
            task_states: List of task states to potentially review

        Returns:
            Tuple of (cached sample scores, filtered task states for remaining reviews)
        """
        cache_file = self.get_review_cache_path(subset)
        if not os.path.exists(cache_file):
            # No review cache exists, return empty scores and all task states
            return [], task_states

        cached_by_sample_id = {}
        states_by_id = {state.sample_id: state for state in task_states}
        valid_sample_ids = set(states_by_id)
        orphan_rows = 0
        duplicate_rows = 0
        stale_rows = 0
        cache_items = self._read_cache(cache_file)

        # Process each cached review result
        for cache_item in cache_items:
            # Deserialize the cached review result
            try:
                cached_review_result = ReviewResult.from_cache_item(cache_item)
            except ValidationError as e:
                logger.warning(f'Skipping invalid review cache row in {cache_file}: {e}')
                continue
            sample_score = cached_review_result.to_sample_score()
            if sample_score.sample_id not in valid_sample_ids:
                orphan_rows += 1
                continue
            if not _review_matches_prediction(cached_review_result, states_by_id[sample_score.sample_id]):
                stale_rows += 1
                continue
            if sample_score.sample_id in cached_by_sample_id:
                duplicate_rows += 1
            if review_needs_retry(sample_score):
                cached_by_sample_id.pop(sample_score.sample_id, None)
                continue
            cached_by_sample_id[sample_score.sample_id] = sample_score

        cached_sample_scores: List[SampleScore] = list(cached_by_sample_id.values())
        if orphan_rows or duplicate_rows:
            logger.warning(
                f'Dropped {orphan_rows} orphan and {duplicate_rows} duplicate rows from review cache: {cache_file}'
            )
        if stale_rows:
            logger.warning(f'Dropped {stale_rows} reviews of earlier generations from review cache: {cache_file}')

        # Filter out task states that already have review scores
        cached_sample_ids = {review.sample_id for review in cached_sample_scores}
        filtered_task_states = [state for state in task_states if state.sample_id not in cached_sample_ids]

        logger.info(f'Reusing reviews from {cache_file}, got {len(cached_sample_scores)} reviews')
        return cached_sample_scores, filtered_task_states

    def get_review_cache_path(self, subset: str) -> str:
        """
        Get the file path for review cache storage.

        Args:
            subset: Name of the dataset subset

        Returns:
            Path to the review cache file
        """
        file_path = os.path.join(self.outputs.reviews_dir, self.model_name, f'{self.benchmark_name}_{subset}.jsonl')
        # Ensure the directory exists
        if self.outputs.is_make:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    def delete_review_cache(self, subset: str):
        """Start a transactional review rerun without touching the previous review file."""
        file_path = self.get_review_cache_path(subset)
        temporary = f'{file_path}.rerun-{uuid.uuid4().hex}'
        # A tolerated scoring failure must publish an empty cache, not retain old scores.
        with open(temporary, 'x', encoding='utf-8'):
            pass
        self._review_reruns[file_path] = temporary
        logger.debug(f'Rescoring reviews into temporary cache: {temporary}')

    def commit_review_reruns(self) -> None:
        """Atomically publish every fully written review rerun."""
        for file_path, temporary in list(self._review_reruns.items()):
            writer = self._writers.pop(temporary, None)
            if writer is not None:
                writer.close()
            os.replace(temporary, file_path)
            del self._review_reruns[file_path]

    def discard_review_reruns(self) -> None:
        """Close and remove incomplete transactional review files after a failed rerun."""
        for file_path, temporary in list(self._review_reruns.items()):
            writer = self._writers.pop(temporary, None)
            if writer is not None:
                writer.close()
            if os.path.exists(temporary):
                os.unlink(temporary)
            del self._review_reruns[file_path]

    def save_review_cache(
        self,
        subset: str,
        task_state: TaskState,
        sample_score: SampleScore,
        save_metadata: bool = True,
    ) -> 'ReviewResult':
        """
        Save a review result to the cache.

        Args:
            subset: Name of the dataset subset
            task_state: The task state that was reviewed
            sample_score: The computed score for the sample

        Returns:
            The saved review result object
        """
        cache_file = self.get_review_cache_path(subset)
        cache_file = self._review_reruns.get(cache_file, cache_file)
        # Convert score and state to serializable review result
        review_result = ReviewResult.from_score_state(sample_score, task_state, save_metadata)
        # Serialize to dictionary, convert non-JSON types (numpy, datetime), append.
        review_result_dict = convert_normal_types(review_result.model_dump())
        with self._write_lock:
            self._get_writer(cache_file).write(review_result_dict)
        return review_result

    def get_report_path(self) -> str:
        """
        Get the directory path for report storage.

        Returns:
            Path to the reports directory for this model
        """
        report_path = os.path.join(self.outputs.reports_dir, self.model_name)
        # Ensure the directory exists
        if self.outputs.is_make:
            os.makedirs(report_path, exist_ok=True)
        return report_path

    def get_report_file(self) -> str:
        """
        Get the report file path for the benchmark.

        The report file is named as '{benchmark_name}.json' and contains
        the final evaluation results for the benchmark.

        Returns:
            Full path to the benchmark report file
        """
        return os.path.join(self.get_report_path(), f'{self.benchmark_name}.json')


class ModelResult(BaseModel):
    """
    Serializable container for model prediction results.

    This class represents a single model prediction that can be cached
    and restored later to avoid recomputation.
    """

    index: int
    """Index of the sample in the dataset that was processed."""

    model: str = ''
    """Name of the model that generated this prediction."""

    model_output: Optional[ModelOutput] = None
    """The actual prediction/output generated by the model."""

    messages: List[ChatMessage] = []
    """Chat messages exchanged during evaluation (for conversational models)."""

    agent_trace: Optional[AgentTrace] = None
    """Structured agent trajectory, when prediction used an agent runner."""

    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata associated with the model result."""

    prediction_id: Optional[str] = None
    """Unique identity of this generation; ``None`` for legacy rows."""

    sample_fingerprint: Optional[str] = None
    """Content hash of the sample this prediction answered; ``None`` for legacy rows."""

    @model_validator(mode='after')
    def _sync_perf_to_messages(self) -> 'ModelResult':
        """Best-effort sync of assistant perf_metrics onto ``messages[-1]``."""
        if self.model_output is None or not self.model_output.choices or not self.messages:
            return self
        out_msg = self.model_output.message
        if out_msg is None or out_msg.perf_metrics is None:
            return self
        last = self.messages[-1]
        if getattr(last, 'perf_metrics', None) is None:
            try:
                last.perf_metrics = out_msg.perf_metrics
            except Exception:  # noqa: BLE001 - defensive, must not break loading
                logger.debug('Failed to sync perf_metrics onto messages[-1]; skipping.')
        return self

    @classmethod
    def from_task_state(cls, task_state: TaskState, save_metadata: bool = True) -> 'ModelResult':
        """
        Create a ModelResult from a TaskState for caching.

        Args:
            task_state: The completed task state to serialize

        Returns:
            ModelResult object ready for caching
        """
        return cls(
            model=task_state.model,
            index=task_state.sample_id,
            messages=task_state.messages,
            agent_trace=task_state.agent_trace,
            model_output=task_state.output,
            metadata=task_state.metadata if save_metadata else {},
            prediction_id=task_state.prediction_id,
            sample_fingerprint=task_state.sample_fingerprint,
        )

    def to_task_state(self, dataset: Dataset) -> TaskState:
        """
        Restore a TaskState from cached ModelResult.

        Args:
            dataset: The dataset to retrieve the original sample from

        Returns:
            Reconstructed TaskState with cached results

        Raises:
            ValueError: If the sample index is not found in the dataset
        """
        try:
            sample = dataset[self.index]
        except IndexError:
            logger.warning(f'Sample index {self.index} not found in dataset during cache restoration.')
            return None

        # Verify identity before merging cached metadata into the freshly loaded sample;
        # otherwise a reordered dataset would yield a hybrid of two different samples.
        fingerprint = sample_fingerprint(sample)
        if self.sample_fingerprint is not None and self.sample_fingerprint != fingerprint:
            raise CacheIdentityError(
                f'Cached prediction {self.index} answered a different sample than the one now at that index. '
                'The dataset order or content changed since the cache was written (for example shuffling '
                'without a fixed seed); resume from a fresh output directory.'
            )
        if self.sample_fingerprint is None and getattr(dataset, 'shuffled', False):
            raise CacheIdentityError(
                f'Cached prediction {self.index} has no sample fingerprint and the dataset is shuffled, so its '
                'sample cannot be verified; resume from a fresh output directory or disable shuffling.'
            )

        # update metadata if exists
        if self.metadata:
            sample.metadata.update(self.metadata)

        task_state = TaskState(
            model=self.model,
            sample=sample,
            messages=self.messages,
            output=ModelOutput.model_validate(self.model_output),
            completed=True,  # Mark as completed since it was cached
        )
        task_state.agent_trace = self.agent_trace
        task_state.sample_fingerprint = fingerprint
        task_state.prediction_id = self.prediction_id
        return task_state

    def pretty_print(self) -> str:
        """
        Generate a pretty-printed string representation of the model result.

        Returns:
            A string representation of the model result
        """
        return self.model_dump_json(indent=2)


class ReviewResult(BaseModel):
    """
    Serializable container for review/scoring results.

    This class represents the result of reviewing a model's prediction,
    including the computed score and relevant context.
    """

    model_config = ConfigDict(extra='ignore')

    index: int
    """Index of the sample that was reviewed."""

    target: Optional[List[str]] = None
    """Accepted target answers for the sample, if available."""

    messages: List[ChatMessage] = Field(default_factory=list)
    """Full chat message history exchanged during evaluation."""

    agent_trace: Optional[AgentTrace] = None
    """Structured agent trajectory (only populated for agent-mode runs)."""

    sample_score: SampleScore
    """The computed evaluation score for this sample."""

    prediction_id: Optional[str] = None
    """Identity of the prediction this review graded; ``None`` for legacy rows."""

    @model_validator(mode='before')
    @classmethod
    def _migrate_legacy_input(cls, data: Any) -> Any:
        """Migrate legacy review cache rows into the structured contract."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        target = data.get('target')
        if isinstance(target, str):
            target = target.strip()
            data['target'] = [target] if target else None
        elif isinstance(target, list):
            data['target'] = list(dict.fromkeys(value.strip() for value in target if value.strip()))
        legacy_input = data.pop('input', None)
        if legacy_input and not data.get('messages'):
            data['messages'] = [
                {
                    'role': 'user',
                    'content': legacy_input,
                }
            ]
        # Drop obsolete TrajectoryStep list (the new ``agent_trace`` has a
        # different shape; legacy values are not useful and would fail
        # validation).
        data.pop('trajectory', None)
        return data

    @field_validator('target')
    @classmethod
    def _validate_target(cls, target: Optional[List[str]]) -> Optional[List[str]]:
        if target is not None and not target:
            raise ValueError('ReviewResult.target must not be an empty list.')
        return target

    @classmethod
    def from_cache_item(cls, data: Any) -> 'ReviewResult':
        """Load a review result from an on-disk cache row."""
        return cls.model_validate(data)

    @property
    def messages_markdown(self) -> str:
        """Render ``messages`` as markdown (multi-modal aware)."""
        return messages_to_markdown(self.messages)

    @property
    def messages_text(self) -> str:
        """Render ``messages`` as plain text."""
        return messages_pretty_str(self.messages)

    @classmethod
    def from_score_state(
        cls, sample_score: SampleScore, state: TaskState, save_metadata: bool = True
    ) -> 'ReviewResult':
        """
        Create a ReviewResult from a score and task state for caching.

        Args:
            sample_score: The computed score for the sample
            state: The task state containing sample information

        Returns:
            ReviewResult object ready for caching
        """
        if not save_metadata:
            sample_score = copy.deepcopy(sample_score)
            sample_score.sample_metadata = None

        return cls(
            index=state.sample_id,
            target=list(state.target_reference.values) or None,
            messages=state.messages,
            agent_trace=state.agent_trace,
            sample_score=sample_score,
            prediction_id=state.prediction_id,
        )

    def to_sample_score(self) -> SampleScore:
        """
        Extract the sample score from the cached review result.

        Returns:
            The sample score object
        """
        return self.sample_score

    def pretty_print(self) -> str:
        """
        Generate a pretty-printed string representation of the review result.

        Returns:
            A string representation of the review result
        """
        output = [
            f'Review Result for Sample {self.index}:',
            f'Target: {self.target or []}',
            f'Score: {self.sample_score.model_dump_json(indent=2)}',
        ]
        return '\n'.join(output)
