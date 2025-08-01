from typing import List

import pytest
from datasets import disable_caching
from pie_core import Document
from pie_modules.document.processing import tokenize_document
from pie_modules.documents import TextDocumentWithLabeledSpansAndBinaryRelations
from transformers import AutoTokenizer, PreTrainedTokenizer

from dataset_builders.pie.abstrct.abstrct import AbstRCT
from pie_datasets import DatasetDict
from pie_datasets.builders.brat import BratDocumentWithMergedSpans
from tests.dataset_builders.common import (
    PIE_BASE_PATH,
    TestTokenDocumentWithLabeledSpansAndBinaryRelations,
)

disable_caching()

DATASET_NAME = "abstrct"
BUILDER_CLASS = AbstRCT
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
SPLIT_SIZES = {
    "glaucoma_test": 100,
    "mixed_test": 100,
    "neoplasm_dev": 50,
    "neoplasm_test": 100,
    "neoplasm_train": 350,
}
SPLIT = "neoplasm_train"


@pytest.fixture(scope="module", params=[config.name for config in BUILDER_CLASS.BUILDER_CONFIGS])
def dataset_variant(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def dataset(dataset_variant) -> DatasetDict:
    return DatasetDict.load_dataset(str(PIE_DATASET_PATH), name=dataset_variant)


def test_dataset(dataset):
    assert dataset is not None
    assert {name: len(ds) for name, ds in dataset.items()} == SPLIT_SIZES


@pytest.fixture(scope="module")
def builder(dataset_variant) -> BUILDER_CLASS:
    return BUILDER_CLASS(config_name=dataset_variant)


def test_builder(builder, dataset_variant):
    assert builder is not None
    assert builder.config_id == dataset_variant
    assert builder.dataset_name == DATASET_NAME
    assert builder.document_type == BratDocumentWithMergedSpans


@pytest.fixture(scope="module")
def document(dataset) -> BratDocumentWithMergedSpans:
    result = dataset[SPLIT][0]
    # we can not assert the real document type because it may come from a dataset loading script
    # downloaded to a temporary directory and thus have a different type object, although it is
    # semantically the same
    assert isinstance(result, Document)
    return result


def test_document(document, dataset_variant):
    assert document is not None
    assert document.id == "10561201"

    # check the spans
    assert len(document.spans) == 7
    span_texts_labels_tuples = [(str(span), span.label) for span in document.spans]
    assert span_texts_labels_tuples[0] == (
        "A combination of mitoxantrone plus prednisone is preferable to prednisone alone for reduction of pain "
        "in men with metastatic, hormone-resistant, prostate cancer.",
        "MajorClaim",
    )
    assert span_texts_labels_tuples[1] == (
        "At 6 weeks, both groups showed improvement in several HQL domains,",
        "Premise",
    )
    assert span_texts_labels_tuples[2] == (
        "only physical functioning and pain were better in the mitoxantrone-plus-prednisone group than in the "
        "prednisone-alone group.",
        "Premise",
    )
    assert span_texts_labels_tuples[3] == (
        "After 6 weeks, patients taking prednisone showed no improvement in HQL scores, whereas those taking "
        "mitoxantrone plus prednisone showed significant improvements in global quality of life (P =.009), four "
        "functioning domains, and nine symptoms (.001 < P <. 01),",
        "Premise",
    )
    assert span_texts_labels_tuples[4] == (
        "the improvement (> 10 units on a scale of 0 to100) lasted longer than in the prednisone-alone group "
        "(.004 < P <.05).",
        "Premise",
    )
    assert span_texts_labels_tuples[5] == (
        "The addition of mitoxantrone to prednisone after failure of prednisone alone was associated with "
        "improvements in pain, pain impact, pain relief, insomnia, and global quality of life (.001 < P <.003).",
        "Premise",
    )
    assert span_texts_labels_tuples[6] == (
        "Treatment with mitoxantrone plus prednisone was associated with greater and longer-lasting improvement "
        "in several HQL domains and symptoms than treatment with prednisone alone.",
        "Claim",
    )

    # check relations
    assert len(document.relations) == 6
    document.relations[0].label == "Support"
    document.relations[0].head == document.spans[6]
    document.relations[0].tail == document.spans[0]
    document.relations[1].label == "Support"
    document.relations[1].head == document.spans[1]
    document.relations[1].tail == document.spans[6]
    document.relations[2].label == "Support"
    document.relations[2].head == document.spans[2]
    document.relations[2].tail == document.spans[6]
    document.relations[3].label == "Support"
    document.relations[3].head == document.spans[5]
    document.relations[3].tail == document.spans[6]
    document.relations[4].label == "Support"
    document.relations[4].head == document.spans[3]
    document.relations[4].tail == document.spans[6]
    document.relations[5].label == "Support"
    document.relations[5].head == document.spans[5]
    document.relations[5].tail == document.spans[0]


@pytest.fixture(scope="module")
def dataset_of_text_documents_with_labeled_spans_and_binary_relations(
    dataset, dataset_variant
) -> DatasetDict:
    if dataset_variant == "default" or dataset_variant is None:
        converted_dataset = dataset.to_document_type(
            TextDocumentWithLabeledSpansAndBinaryRelations
        )
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")
    return converted_dataset


def test_dataset_of_text_documents_with_labeled_spans_and_binary_relations(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations,
):
    # get a document to check
    converted_doc = dataset_of_text_documents_with_labeled_spans_and_binary_relations[SPLIT][0]
    # check that the conversion is correct and the data makes sense
    assert isinstance(converted_doc, TextDocumentWithLabeledSpansAndBinaryRelations)

    # check the entities
    assert len(converted_doc.labeled_spans) == 7
    entity_tuples = [(str(ent), ent.label) for ent in converted_doc.labeled_spans]
    assert entity_tuples[0] == (
        "A combination of mitoxantrone plus prednisone is preferable to prednisone alone for reduction of pain in men "
        "with metastatic, hormone-resistant, prostate cancer.",
        "MajorClaim",
    )
    assert entity_tuples[1] == (
        "At 6 weeks, both groups showed improvement in several HQL domains,",
        "Premise",
    )
    assert entity_tuples[2] == (
        "only physical functioning and pain were better in the mitoxantrone-plus-prednisone group than in the "
        "prednisone-alone group.",
        "Premise",
    )
    assert entity_tuples[3] == (
        "After 6 weeks, patients taking prednisone showed no improvement in HQL scores, whereas those taking "
        "mitoxantrone plus prednisone showed significant improvements in global quality of life (P =.009), "
        "four functioning domains, and nine symptoms (.001 < P <. 01),",
        "Premise",
    )
    assert entity_tuples[4] == (
        "the improvement (> 10 units on a scale of 0 to100) lasted longer than in the prednisone-alone group "
        "(.004 < P <.05).",
        "Premise",
    )
    assert entity_tuples[5] == (
        "The addition of mitoxantrone to prednisone after failure of prednisone alone was associated with "
        "improvements in pain, pain impact, pain relief, insomnia, and global quality of life (.001 < P <.003).",
        "Premise",
    )
    assert entity_tuples[6] == (
        "Treatment with mitoxantrone plus prednisone was associated with greater and longer-lasting improvement "
        "in several HQL domains and symptoms than treatment with prednisone alone.",
        "Claim",
    )

    # check the relations
    assert len(converted_doc.binary_relations) == 6
    relation_tuples = [
        (str(rel.head), rel.label, str(rel.tail)) for rel in converted_doc.binary_relations
    ]
    assert relation_tuples[0] == (
        "Treatment with mitoxantrone plus prednisone was associated with greater and longer-lasting improvement "
        "in several HQL domains and symptoms than treatment with prednisone alone.",
        "Support",
        "A combination of mitoxantrone plus prednisone is preferable to prednisone alone for reduction of pain "
        "in men with metastatic, hormone-resistant, prostate cancer.",
    )
    assert relation_tuples[1] == (
        "At 6 weeks, both groups showed improvement in several HQL domains,",
        "Support",
        "Treatment with mitoxantrone plus prednisone was associated with greater and longer-lasting improvement "
        "in several HQL domains and symptoms than treatment with prednisone alone.",
    )
    assert relation_tuples[2] == (
        "only physical functioning and pain were better in the mitoxantrone-plus-prednisone group than in the "
        "prednisone-alone group.",
        "Support",
        "Treatment with mitoxantrone plus prednisone was associated with greater and longer-lasting improvement "
        "in several HQL domains and symptoms than treatment with prednisone alone.",
    )
    assert relation_tuples[3] == (
        "the improvement (> 10 units on a scale of 0 to100) lasted longer than in the prednisone-alone group "
        "(.004 < P <.05).",
        "Support",
        "Treatment with mitoxantrone plus prednisone was associated with greater and longer-lasting improvement in "
        "several HQL domains and symptoms than treatment with prednisone alone.",
    )
    assert relation_tuples[4] == (
        "After 6 weeks, patients taking prednisone showed no improvement in HQL scores, whereas those taking "
        "mitoxantrone plus prednisone showed significant improvements in global quality of life (P =.009), four "
        "functioning domains, and nine symptoms (.001 < P <. 01),",
        "Support",
        "Treatment with mitoxantrone plus prednisone was associated with greater and longer-lasting improvement "
        "in several HQL domains and symptoms than treatment with prednisone alone.",
    )
    assert relation_tuples[5] == (
        "The addition of mitoxantrone to prednisone after failure of prednisone alone was associated with "
        "improvements in pain, pain impact, pain relief, insomnia, and global quality of life (.001 < P <.003).",
        "Support",
        "A combination of mitoxantrone plus prednisone is preferable to prednisone alone for reduction of pain in "
        "men with metastatic, hormone-resistant, prostate cancer.",
    )


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def tokenized_documents_with_labeled_spans_and_binary_relations(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations, tokenizer
) -> List[TestTokenDocumentWithLabeledSpansAndBinaryRelations]:
    # get a document to check
    doc = dataset_of_text_documents_with_labeled_spans_and_binary_relations[SPLIT][0]
    # Note, that this is a list of documents, because the document may be split into chunks
    # if the input text is too long.
    tokenized_docs = tokenize_document(
        doc,
        tokenizer=tokenizer,
        return_overflowing_tokens=True,
        result_document_type=TestTokenDocumentWithLabeledSpansAndBinaryRelations,
        strict_span_conversion=True,
        verbose=True,
    )
    return tokenized_docs


def test_tokenized_documents_with_labeled_spans_and_binary_relations(
    tokenized_documents_with_labeled_spans_and_binary_relations,
):
    docs = tokenized_documents_with_labeled_spans_and_binary_relations
    # check that the tokenization was fine
    assert len(docs) == 1
    doc = docs[0]
    assert len(doc.tokens) == 465
    assert len(doc.labeled_spans) == 7
    ent = doc.labeled_spans[0]
    assert (
        str(ent)
        == "('a', 'combination', 'of', 'mit', '##ox', '##ant', '##rone', 'plus', 'pre', '##d', '##nis', '##one', 'is', "
        "'prefer', '##able', 'to', 'pre', '##d', '##nis', '##one', 'alone', 'for', 'reduction', 'of', 'pain', 'in', "
        "'men', 'with', 'meta', '##static', ',', 'hormone', '-', 'resistant', ',', 'prostate', 'cancer', '.')"
    )
    ent = doc.labeled_spans[1]
    assert (
        str(ent)
        == "('at', '6', 'weeks', ',', 'both', 'groups', 'showed', 'improvement', 'in', 'several', 'hq', '##l', "
        "'domains', ',')"
    )
    ent = doc.labeled_spans[2]
    assert (
        str(ent)
        == "('only', 'physical', 'functioning', 'and', 'pain', 'were', 'better', 'in', 'the', 'mit', '##ox', '##ant', "
        "'##rone', '-', 'plus', '-', 'pre', '##d', '##nis', '##one', 'group', 'than', 'in', 'the', 'pre', '##d', "
        "'##nis', '##one', '-', 'alone', 'group', '.')"
    )
    ent = doc.labeled_spans[3]
    assert (
        str(ent)
        == "('after', '6', 'weeks', ',', 'patients', 'taking', 'pre', '##d', '##nis', '##one', 'showed', 'no', "
        "'improvement', 'in', 'hq', '##l', 'scores', ',', 'whereas', 'those', 'taking', 'mit', '##ox', '##ant', "
        "'##rone', 'plus', 'pre', '##d', '##nis', '##one', 'showed', 'significant', 'improvements', 'in', 'global', "
        "'quality', 'of', 'life', '(', 'p', '=', '.', '00', '##9', ')', ',', 'four', 'functioning', 'domains', ',', "
        "'and', 'nine', 'symptoms', '(', '.', '001', '<', 'p', '<', '.', '01', ')', ',')"
    )
    ent = doc.labeled_spans[4]
    assert (
        str(ent)
        == "('the', 'improvement', '(', '>', '10', 'units', 'on', 'a', 'scale', 'of', '0', 'to', '##100', ')', "
        "'lasted', 'longer', 'than', 'in', 'the', 'pre', '##d', '##nis', '##one', '-', 'alone', 'group', '(', '.', "
        "'00', '##4', '<', 'p', '<', '.', '05', ')', '.')"
    )
    ent = doc.labeled_spans[5]
    assert (
        str(ent)
        == "('the', 'addition', 'of', 'mit', '##ox', '##ant', '##rone', 'to', 'pre', '##d', '##nis', '##one', "
        "'after', 'failure', 'of', 'pre', '##d', '##nis', '##one', 'alone', 'was', 'associated', 'with', "
        "'improvements', 'in', 'pain', ',', 'pain', 'impact', ',', 'pain', 'relief', ',', 'ins', '##om', '##nia', "
        "',', 'and', 'global', 'quality', 'of', 'life', '(', '.', '001', '<', 'p', '<', '.', '00', '##3', ')', '.')"
    )
    ent = doc.labeled_spans[6]
    assert (
        str(ent)
        == "('treatment', 'with', 'mit', '##ox', '##ant', '##rone', 'plus', 'pre', '##d', '##nis', '##one', 'was', "
        "'associated', 'with', 'greater', 'and', 'longer', '-', 'lasting', 'improvement', 'in', 'several', "
        "'hq', '##l', 'domains', 'and', 'symptoms', 'than', 'treatment', 'with', 'pre', '##d', '##nis', '##one', "
        "'alone', '.')"
    )


def test_tokenized_documents_with_entities_and_relations_all(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations, tokenizer, dataset_variant
):
    for (
        split,
        docs,
    ) in dataset_of_text_documents_with_labeled_spans_and_binary_relations.items():
        for doc in docs:
            # Note, that this is a list of documents, because the document may be split into chunks
            # if the input text is too long.
            tokenized_docs = tokenize_document(
                doc,
                tokenizer=tokenizer,
                return_overflowing_tokens=True,
                result_document_type=TestTokenDocumentWithLabeledSpansAndBinaryRelations,
                strict_span_conversion=True,
                verbose=True,
            )
            # we just ensure that we get at least one tokenized document
            assert tokenized_docs is not None
            assert len(tokenized_docs) > 0


def test_document_converters(dataset_variant):
    builder = BUILDER_CLASS(config_name=dataset_variant)
    document_converters = builder.document_converters

    if dataset_variant == "default" or dataset_variant is None:
        assert len(document_converters) == 1
        assert set(document_converters) == {
            TextDocumentWithLabeledSpansAndBinaryRelations,
        }
        assert document_converters[TextDocumentWithLabeledSpansAndBinaryRelations] == {
            "spans": "labeled_spans",
            "relations": "binary_relations",
        }
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")
