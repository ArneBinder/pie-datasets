from dataclasses import dataclass
from typing import Type

import datasets
from pie_modules.annotations import LabeledSpan
from pie_core import AnnotationLayer, annotation_field
from pie_modules.documents import TextBasedDocument

from pie_datasets import GeneratorBasedBuilder
from tests import FIXTURES_ROOT


class ExampleConfig(datasets.BuilderConfig):
    """BuilderConfig for CoNLL2003."""

    def __init__(self, parameter: str, **kwargs):
        """BuilderConfig for CoNLL2003.

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

    BASE_DATASET_PATH = str(FIXTURES_ROOT / "builder" / "datasets" / "base_single_config")

    BUILDER_CONFIGS = [
        ExampleConfig(
            name="conll2003",
            version=datasets.Version("1.0.0"),
            description="Example dataset",
            parameter="test",
        ),
    ]

    # required to create config from scratch via kwargs
    BUILDER_CONFIG_CLASS: Type[datasets.BuilderConfig] = ExampleConfig

    def _generate_document_kwargs(self, dataset):
        pass

    def _generate_document(self, example, int_to_str):
        pass
