import dataclasses
import glob
import os
from pathlib import Path

import pkg_resources
import pytest
from datasets import load_dataset
from pie_core import AnnotationLayer, annotation_field
from pie_modules.annotations import BinaryRelation, LabeledSpan, Span
from pie_modules.documents import TextBasedDocument

from pie_datasets import DatasetDict
from tests import FIXTURES_ROOT

SRC_ROOTS = [Path("src")]

# import all src files to include them in the coverage score (and report)
# this is necessary because we calculate coverage by calling "pytest --cov"
for src_root in SRC_ROOTS:
    for file in glob.glob(f"{src_root}/**/*.py", recursive=True):
        # get the base file path without the extension
        import_base_path = os.path.splitext(file)[0]
        import_path = import_base_path.replace(os.sep, ".")
        __import__(import_path)


_TABULATE_AVAILABLE = "tabulate" in {pkg.key for pkg in pkg_resources.working_set}

CREATE_FIXTURE_DATA = False


# just ensure that this never happens on CI
def test_dont_create_fixture_data():
    assert not CREATE_FIXTURE_DATA


@pytest.fixture
def documents(dataset):
    return list(dataset["train"])


@dataclasses.dataclass
class TestDocument(TextBasedDocument):
    sentences: AnnotationLayer[Span] = annotation_field(target="text")
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


def example_to_doc_dict(example):
    doc = TestDocument(text=example["text"], id=example["id"])
    doc.metadata = dict(example["metadata"])
    sentences = [Span.fromdict(dct) for dct in example["sentences"]]
    entities = [LabeledSpan.fromdict(dct) for dct in example["entities"]]
    relations = [
        BinaryRelation(head=entities[rel["head"]], tail=entities[rel["tail"]], label=rel["label"])
        for rel in example["relations"]
    ]
    for sentence in sentences:
        doc.sentences.append(sentence)

    for entity in entities:
        doc.entities.append(entity)

    for relation in relations:
        doc.relations.append(relation)

    return doc.asdict()


SPLIT_SIZES = {"train": 6, "validation": 2, "test": 2}


@pytest.fixture(scope="session")
def hf_dataset():
    result = load_dataset(
        "json",
        field="data",
        data_dir=str(FIXTURES_ROOT / "hf_datasets" / "json"),
    )

    return result


def test_hf_dataset(hf_dataset):
    assert hf_dataset is not None
    assert set(hf_dataset) == set(SPLIT_SIZES)
    for split in hf_dataset:
        assert len(hf_dataset[split]) == SPLIT_SIZES[split]


@pytest.fixture()
def dataset(hf_dataset):
    mapped_dataset = hf_dataset.map(example_to_doc_dict)
    dataset = DatasetDict.from_hf(hf_dataset=mapped_dataset, document_type=TestDocument)
    return dataset


def test_dataset(dataset):
    assert dataset is not None
    assert set(dataset) == set(SPLIT_SIZES)
    for split in dataset:
        assert len(dataset[split]) == SPLIT_SIZES[split]
    # try getting a split
    d_train = dataset["train"]
    assert d_train is not None
    # try getting a document
    doc0 = d_train[0]
    assert doc0 is not None
    assert isinstance(doc0, TestDocument)


@pytest.fixture(scope="session")
def iterable_hf_dataset():
    result = load_dataset(
        "json",
        field="data",
        data_dir=str(FIXTURES_ROOT / "hf_datasets" / "json"),
        streaming=True,
    )

    return result


@pytest.fixture()
def iterable_dataset(iterable_hf_dataset):
    mapped_dataset = iterable_hf_dataset.map(example_to_doc_dict)
    dataset = DatasetDict.from_hf(hf_dataset=mapped_dataset, document_type=TestDocument)
    return dataset


def test_iterable_dataset(iterable_dataset):
    assert iterable_dataset is not None
    assert set(iterable_dataset) == set(SPLIT_SIZES)


@pytest.fixture(params=["dataset", "iterable_dataset"])
def maybe_iterable_dataset(request):
    return request.getfixturevalue(request.param)
