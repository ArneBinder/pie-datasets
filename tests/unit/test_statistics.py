import dataclasses

import pytest
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument

from pie_datasets import DatasetDict
from pie_datasets.statistics import SpanLengthCollector
from tests import FIXTURES_ROOT


@pytest.fixture
def dataset():
    @dataclasses.dataclass
    class Conll2003Document(TextBasedDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")

    return DatasetDict.from_json(
        data_dir=FIXTURES_ROOT / "dataset_dict" / "conll2003_extract",
        document_type=Conll2003Document,
    )


def test_statistics(dataset):
    statistic = SpanLengthCollector(layer="entities")
    values = statistic(dataset)
    assert values == {
        "train": {"len": 5, "mean": 7.6, "std": 4.223742416388575, "min": 2, "max": 15},
        "validation": {
            "len": 6,
            "mean": 10.833333333333334,
            "std": 2.9674156357941426,
            "min": 6,
            "max": 14,
        },
        "test": {"len": 5, "mean": 9.4, "std": 5.748043145279966, "min": 5, "max": 20},
    }

    statistic = SpanLengthCollector(layer="entities", labels="INFERRED")
    values = statistic(dataset)
    assert values == {
        "train": {
            "ORG": {"max": 2, "len": 1},
            "MISC": {"max": 7, "len": 2},
            "PER": {"max": 15, "len": 1},
            "LOC": {"max": 8, "len": 1},
        },
        "test": {
            "LOC": {
                "max": 20,
                "len": 3,
            },
            "PER": {"max": 11, "len": 2},
        },
        "validation": {
            "ORG": {"max": 14, "len": 3},
            "LOC": {"max": 6, "len": 1},
            "MISC": {"max": 11, "len": 1},
            "PER": {"max": 12, "len": 1},
        },
    }


def test_statistics_with_tokenize(dataset):
    @dataclasses.dataclass
    class TokenDocumentWithLabeledEntities(TokenBasedDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="tokens")

    statistic = SpanLengthCollector(
        layer="entities",
        tokenize=True,
        tokenizer="bert-base-uncased",
        tokenized_document_type=TokenDocumentWithLabeledEntities,
    )
    values = statistic(dataset)
    assert values == {
        "test": {"len": 5, "max": 4, "mean": 2.4, "min": 1, "std": 1.2000000000000002},
        "train": {"len": 5, "max": 2, "mean": 1.2, "min": 1, "std": 0.4},
        "validation": {
            "len": 6,
            "max": 2,
            "mean": 1.3333333333333333,
            "min": 1,
            "std": 0.4714045207910317,
        },
    }
