"""
workflow:
    document
        -> (input_encoding, target_encoding) -> task_encoding
            -> model_encoding -> model_output
        -> task_output
    -> document
"""

import dataclasses
import logging
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
from pie_modules.annotations import Label
from pie_modules.documents import TextBasedDocument
from pytorch_ie import AnnotationLayer, TaskEncoding, TaskModule, annotation_field
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TestDocumentWithLabel(TextBasedDocument):
    label: AnnotationLayer[Label] = annotation_field()


class TaskOutput(TypedDict, total=False):
    label: str
    probability: float


# Define task specific input and output types
DocumentType: TypeAlias = TestDocumentWithLabel
InputEncodingType: TypeAlias = List[int]
TargetEncodingType: TypeAlias = int
ModelInputType = List[List[int]]
ModelTargetType = List[int]
ModelEncodingType: TypeAlias = Tuple[
    ModelInputType,
    Optional[ModelTargetType],
]
ModelOutputType = Dict[str, List[List[float]]]
TaskOutputType: TypeAlias = TaskOutput

# This should be the same for all taskmodules
TaskEncodingType: TypeAlias = TaskEncoding[DocumentType, InputEncodingType, TargetEncodingType]
TaskModuleType: TypeAlias = TaskModule[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
    ModelEncodingType,
    ModelOutputType,
    TaskOutputType,
]


def softmax(scores: List[float]) -> List[float]:
    """Compute the softmax of a list of scores."""
    max_score = max(scores)
    exp_scores = [np.exp(score - max_score) for score in scores]
    sum_exp_scores = sum(exp_scores)
    return [score / sum_exp_scores for score in exp_scores]


def argmax(scores: List[float]) -> int:
    """Get the index of the maximum score."""
    max_index = 0
    max_value = scores[0]
    for i, score in enumerate(scores):
        if score > max_value:
            max_value = score
            max_index = i
    return max_index


@TaskModule.register()
class TestTaskModule(TaskModuleType):
    # If these attributes are set, the taskmodule is considered as prepared. They should be calculated
    # within _prepare() and are dumped automatically when saving the taskmodule with save_pretrained().
    PREPARED_ATTRIBUTES = ["labels"]
    DOCUMENT_TYPE = TestDocumentWithLabel

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        # Important: Remaining keyword arguments need to be passed to super.
        super().__init__(**kwargs)
        # Save all passed arguments. They will be available via self._config().
        self.save_hyperparameters()

        self.labels = labels
        self.token2id = {"PAD": 0}
        self.id2token = {0: "PAD"}

    def _prepare(self, documents: Sequence[DocumentType]) -> None:
        """Prepare the task module with training documents, e.g. collect all possible labels.

        This method needs to set all attributes listed in PREPARED_ATTRIBUTES.
        """

        # create the label-to-id mapping
        labels = set()
        for document in documents:
            # all annotations of a document are hold in list like containers,
            # so we have to take its first element
            label_annotation = document.label[0]
            labels.add(label_annotation.label)

        self.labels = sorted(labels)

    def _post_prepare(self):
        """Any further preparation logic that requires the result of _prepare().

        But its result is not serialized with the taskmodule.
        """
        # create the mapping, but spare the first index for the "O" (outside) class
        self.label_to_id = {label: i + 1 for i, label in enumerate(self.labels)}
        self.label_to_id["O"] = 0
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def tokenize(self, text: str) -> List[int]:
        """Tokenize the input text using the tokenizer."""
        # Tokenize the input text via whitespace
        tokens = text.split(" ")
        ids = []
        for token in tokens:
            # If the token is not already in the vocabulary, add it
            if token not in self.token2id:
                self.token2id[token] = len(self.token2id)
            ids.append(self.token2id[token])
        return ids

    def token_ids2tokens(self, ids: List[int]) -> List[str]:
        """Convert token ids back to tokens."""
        if len(self.id2token) != len(self.token2id):
            self.id2token = {v: k for k, v in self.token2id.items()}

        return [self.id2token[id] for id in ids]

    def encode_input(
        self,
        document: DocumentType,
    ) -> TaskEncodingType:
        """Create one or multiple task encodings for the given document."""

        # tokenize the input text, this will be the input
        inputs = self.tokenize(document.text)

        return TaskEncoding(
            document=document,
            inputs=inputs,
        )

    def encode_target(
        self,
        task_encoding: TaskEncodingType,
    ) -> TargetEncodingType:
        """Create a target for a task encoding.

        This may use any annotations of the underlying document.
        """

        # as above, all annotations are hold in lists, so we have to take its first element
        label_annotation = task_encoding.document.label[0]
        # translate the textual label to the target id
        if self.label_to_id is None:
            raise ValueError(
                "Task module is not prepared. Call prepare() or post_prepare() first."
            )
        return self.label_to_id[label_annotation.label]

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> ModelEncodingType:
        """Convert a list of task encodings to a batch that will be passed to the model."""
        # get the inputs from the task encodings
        inputs = [task_encoding.inputs for task_encoding in task_encodings]

        if task_encodings[0].has_targets:
            # get the targets (label ids) from the task encodings
            targets = [task_encoding.targets for task_encoding in task_encodings]
        else:
            # during inference, we do not have any targets
            targets = None

        return inputs, targets

    def unbatch_output(self, model_output: ModelOutputType) -> Sequence[TaskOutputType]:
        """Convert one model output batch to a sequence of taskmodule outputs."""

        # get the logits from the model output
        logits = model_output["logits"]

        # convert the logits to "probabilities"
        probabilities = [softmax(scores) for scores in logits]

        # get the max class index per example
        max_label_ids = [argmax(probs) for probs in probabilities]

        outputs = []
        for idx, label_id in enumerate(max_label_ids):
            # translate the label id back to the label text
            label = self.id_to_label[label_id]
            # get the probability and convert from tensor value to python float
            prob = round(float(probabilities[idx][label_id]), 4)
            # we create TransformerTextClassificationTaskOutput primarily for typing purposes,
            # a simple dict would also work
            result: TaskOutput = {
                "label": label,
                "probability": prob,
            }
            outputs.append(result)

        return outputs

    def create_annotations_from_output(
        self,
        task_encodings: TaskEncodingType,
        task_outputs: TaskOutputType,
    ) -> Iterator[Tuple[str, Label]]:
        """Convert a task output to annotations.

        The method has to yield tuples (annotation_name, annotation).
        """

        # just yield a single annotation (other tasks may need multiple annotations per task output)
        yield "label", Label(label=task_outputs["label"], score=task_outputs["probability"])
