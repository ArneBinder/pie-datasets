from dataclasses import dataclass

import datasets
from pie_documents.annotations import LabeledSpan
from pie_core import AnnotationLayer, annotation_field
from pie_documents.documents import TextBasedDocument

from pie_datasets import GeneratorBasedBuilder
from tests import FIXTURES_ROOT


class ExampleConfig(datasets.BuilderConfig):
    """BuilderConfig for CoNLL2002."""

    def __init__(self, parameter: str, **kwargs):
        """BuilderConfig for CoNLL2002.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)
        self.parameter = parameter


@dataclass
class ExampleDocument(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


class Example(GeneratorBasedBuilder):
    DOCUMENT_TYPE = ExampleDocument

    BASE_DATASET_PATH = str(FIXTURES_ROOT / "builder" / "datasets" / "base_multi_config")

    BASE_CONFIG_KWARGS_DICT = {
        "nl": {"version": datasets.Version("0.0.0"), "description": "new description"},
    }

    BUILDER_CONFIGS = [
        ExampleConfig(
            name="es",
            version=datasets.Version("1.0.0"),
            description="CoNLL2002 Spanish dataset",
            parameter="test",
        ),
        ExampleConfig(
            name="nl",
            version=datasets.Version("1.0.0"),
            description="CoNLL2002 Dutch dataset",
            parameter="test",
        ),
    ]

    def _generate_document_kwargs(self, dataset):
        pass

    def _generate_document(self, example, int_to_str):
        pass
