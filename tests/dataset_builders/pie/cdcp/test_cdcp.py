import dataclasses

import pytest
from datasets import disable_caching, load_dataset
from pie_core import AnnotationLayer, Document, annotation_field
from pie_documents.annotations import LabeledSpan
from pie_documents.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
)

from dataset_builders.pie.cdcp.cdcp import (
    CDCP,
    CDCPDocument,
    convert_to_text_document_with_labeled_spans_and_binary_relations,
    document_to_example,
    example_to_document,
)
from pie_datasets import DatasetDict
from tests import FIXTURES_ROOT
from tests.dataset_builders.common import PIE_BASE_PATH, _deep_compare

disable_caching()

DATASET_NAME = "cdcp"
BUILDER_CLASS = CDCP
SPLIT_SIZES = {"train": 580, "test": 150}
HF_DATASET_PATH = CDCP.BASE_DATASET_PATH
HF_DATASET_REVISION = CDCP.BASE_DATASET_REVISION
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
DATA_PATH = FIXTURES_ROOT / "dataset_builders" / "cdcp_acl17.zip"

HF_EXAMPLE_00195 = {
    "id": "00195",
    "text": "State and local court rules sometimes make default judgments much more likely. For example, "
    "when a person who allegedly owes a debt is told to come to court on a work day, they may be "
    "forced to choose between a default judgment and their job. I urge the CFPB to find practices "
    "that involve scheduling hearings at inconvenient times unfair, deceptive, and abusive, or "
    "inconsistent with 1692i.",
    "propositions": {
        "start": [0, 78, 242],
        "end": [78, 242, 391],
        "label": [4, 4, 1],
        "url": ["", "", ""],
    },
    "relations": {"head": [0, 2], "tail": [1, 0], "label": [1, 1]},
}


HF_EXAMPLE_00194 = {
    "id": "00194",
    "text": "Recently, courts have held that debt collectors can escape 1692i's venue provisions entirely "
    "by pursuing debt collection through arbitration instead. As the NAF studies reflect, arbitration "
    "has not proven a satisfactory alternative. I urge the CFPB to include in a rule language "
    "interpreting 1692i as requiring debt collectors to proceed in court, not through "
    "largely-unregulated arbitral forums.",
    "propositions": {
        "start": [0, 149, 232],
        "end": [149, 232, 396],
        "label": [0, 4, 1],
        "url": ["", "", ""],
    },
    "relations": {"head": [2], "tail": [1], "label": [1]},
}


@pytest.fixture(scope="module", params=["train", "test"])
def split(request):
    return request.param


@pytest.fixture(scope="module")
def hf_dataset():
    return load_dataset(
        str(HF_DATASET_PATH),
        data_dir=DATA_PATH,
        revision=HF_DATASET_REVISION,
        **BUILDER_CLASS.BASE_BUILDER_KWARGS_DICT[None],
    )


def test_hf_dataset(hf_dataset):
    assert hf_dataset is not None
    assert {name: len(ds) for name, ds in hf_dataset.items()} == SPLIT_SIZES


@pytest.fixture(scope="module")
def hf_example(hf_dataset, split):
    return hf_dataset[split][0]


def test_hf_example(hf_example, split):
    assert hf_example is not None
    if split == "train":
        assert hf_example == HF_EXAMPLE_00195
    elif split == "test":
        assert hf_example == HF_EXAMPLE_00194
    else:
        raise ValueError(f"Unknown split: {split}")


@pytest.fixture(scope="module")
def generate_document_kwargs(hf_dataset, split):
    return BUILDER_CLASS()._generate_document_kwargs(hf_dataset[split])


@pytest.fixture(scope="module")
def generated_document(hf_example, generate_document_kwargs):
    return BUILDER_CLASS()._generate_document(hf_example, **generate_document_kwargs)


def test_generated_document(generated_document, split):
    assert isinstance(generated_document, CDCPDocument)
    if split == "train":
        assert generated_document.text == HF_EXAMPLE_00195["text"]
        assert len(generated_document.propositions) == 3
        assert len(generated_document.relations) == 2
    elif split == "test":
        assert generated_document.text == HF_EXAMPLE_00194["text"]
        assert len(generated_document.propositions) == 3
        assert len(generated_document.relations) == 1
    else:
        raise ValueError(f"Unknown split: {split}")


@pytest.fixture(scope="module")
def hf_example_back(generated_document, generate_document_kwargs):
    return document_to_example(generated_document, **generate_document_kwargs)


def test_example_to_document_and_back(hf_example, hf_example_back):
    _deep_compare(
        obj=hf_example_back,
        obj_expected=hf_example,
    )


def test_example_to_document_and_back_all(hf_dataset, generate_document_kwargs, split):
    for hf_ex in hf_dataset[split]:
        doc = example_to_document(hf_ex, **generate_document_kwargs)
        _assert_no_span_overlap(document=doc, text_field="text", span_layer="propositions")
        hf_example_back = document_to_example(doc, **generate_document_kwargs)
        _deep_compare(
            obj=hf_example_back,
            obj_expected=hf_ex,
        )


@pytest.fixture(scope="module")
def dataset() -> DatasetDict:
    return DatasetDict.load_dataset(str(PIE_DATASET_PATH), data_dir=DATA_PATH)


def test_pie_dataset(dataset):
    assert dataset is not None
    assert {name: len(ds) for name, ds in dataset.items()} == SPLIT_SIZES


@pytest.fixture(scope="module")
def document(dataset, split) -> CDCPDocument:
    result = dataset[split][0]
    # we can not assert the real document type because it may come from a dataset loading script
    # downloaded to a temporary directory and thus have a different type object, although it is
    # semantically the same
    assert isinstance(result, Document)
    return result


def test_compare_document_and_generated_document(document, generated_document):
    assert document.text == generated_document.text
    assert document.relations == generated_document.relations
    assert document.metadata == generated_document.metadata


def _assert_no_span_overlap(document: Document, text_field: str, span_layer: str):
    spans = document[span_layer]
    text = getattr(document, text_field)
    seq = [None] * len(text)
    for span in spans:
        assert seq[span.start : span.end] == [None] * len(text[span.start : span.end])
        seq[span.start : span.end] = text[span.start : span.end]


def test_assert_no_span_overlap():
    @dataclasses.dataclass
    class TextDocumentWithEntities(TextBasedDocument):
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")

    doc0 = TextDocumentWithEntities(text="abcdefghijklmnop")
    doc0.entities.append(LabeledSpan(start=0, end=4, label="A"))
    doc0.entities.append(LabeledSpan(start=4, end=6, label="B"))

    # this should work
    _assert_no_span_overlap(document=doc0, text_field="text", span_layer="entities")

    doc1 = TextDocumentWithEntities(text="abcdefghijklmnop")
    doc1.entities.append(LabeledSpan(start=0, end=4, label="A"))
    doc1.entities.append(LabeledSpan(start=2, end=6, label="B"))

    # this should fail
    with pytest.raises(AssertionError):
        _assert_no_span_overlap(document=doc1, text_field="text", span_layer="entities")


@pytest.fixture(scope="module")
def dataset_of_text_documents_with_labeled_spans_and_binary_relations(
    dataset,
) -> DatasetDict:
    converted_dataset = dataset.to_document_type(TextDocumentWithLabeledSpansAndBinaryRelations)
    return converted_dataset


def test_dataset_of_text_documents_with_labeled_spans_and_binary_relations(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations, split
):
    assert dataset_of_text_documents_with_labeled_spans_and_binary_relations is not None
    # get a document to check
    document = dataset_of_text_documents_with_labeled_spans_and_binary_relations[split][0]
    assert isinstance(document, TextDocumentWithLabeledSpansAndBinaryRelations)
    if split == "train":
        assert document.id == "00195"
        # check entities
        assert len(document.labeled_spans) == 3
        entity_tuples = [(str(ent), ent.label) for ent in document.labeled_spans]
        assert entity_tuples[0] == (
            "State and local court rules sometimes make default judgments much more likely.",
            "value",
        )
        assert entity_tuples[1] == (
            "For example, when a person who allegedly owes a debt is told to come to court on a work day, "
            "they may be forced to choose between a default judgment and their job.",
            "value",
        )
        assert entity_tuples[2] == (
            "I urge the CFPB to find practices that involve scheduling hearings at inconvenient times unfair, "
            "deceptive, and abusive, or inconsistent with 1692i.",
            "policy",
        )

        # check relations
        assert len(document.binary_relations) == 2
        relation_tuples = [
            (str(rel.head), rel.label, str(rel.tail)) for rel in document.binary_relations
        ]
        assert relation_tuples[0] == (
            "State and local court rules sometimes make default judgments much more likely.",
            "reason",
            "For example, when a person who allegedly owes a debt is told to come to court on a work day, "
            "they may be forced to choose between a default judgment and their job.",
        )
        assert relation_tuples[1] == (
            "I urge the CFPB to find practices that involve scheduling hearings at inconvenient times unfair, "
            "deceptive, and abusive, or inconsistent with 1692i.",
            "reason",
            "State and local court rules sometimes make default judgments much more likely.",
        )

    elif split == "test":
        assert document.id == "00194"
        # check entities
        assert len(document.labeled_spans) == 3
        entity_tuples = [(str(ent), ent.label) for ent in document.labeled_spans]
        assert entity_tuples[0] == (
            "Recently, courts have held that debt collectors can escape 1692i's venue provisions entirely "
            "by pursuing debt collection through arbitration instead.",
            "fact",
        )
        assert entity_tuples[1] == (
            "As the NAF studies reflect, arbitration has not proven a satisfactory alternative.",
            "value",
        )
        assert entity_tuples[2] == (
            "I urge the CFPB to include in a rule language interpreting 1692i as requiring debt collectors to proceed "
            "in court, not through largely-unregulated arbitral forums.",
            "policy",
        )

        # check relations
        assert len(document.binary_relations) == 1
        relation_tuples = [
            (str(rel.head), rel.label, str(rel.tail)) for rel in document.binary_relations
        ]
        assert relation_tuples[0] == (
            "I urge the CFPB to include in a rule language interpreting 1692i as requiring debt collectors to proceed "
            "in court, not through largely-unregulated arbitral forums.",
            "reason",
            "As the NAF studies reflect, arbitration has not proven a satisfactory alternative.",
        )
    else:
        raise ValueError(f"Unknown Split {split}")


def test_convert_to_textdocument_with_entities_and_relations(
    document, dataset_of_text_documents_with_labeled_spans_and_binary_relations, split
):
    # just check that we get the same as in the converted dataset when explicitly calling the conversion method
    converted_doc = convert_to_text_document_with_labeled_spans_and_binary_relations(document)
    doc_from_converted_dataset = dataset_of_text_documents_with_labeled_spans_and_binary_relations[
        split
    ][0]
    assert converted_doc == doc_from_converted_dataset
