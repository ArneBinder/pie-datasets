import json
from typing import List

import pytest
from datasets import disable_caching, load_dataset
from pie_modules.document.processing import tokenize_document
from pytorch_ie.core import Document
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndBinaryRelations
from transformers import AutoTokenizer, PreTrainedTokenizer

from dataset_builders.pie.scidtb_argmin.scidtb_argmin import (
    SciDTBArgmin,
    SciDTBArgminDocument,
    convert_to_text_document_with_labeled_spans_and_binary_relations,
    document_to_example,
    example_to_document,
)
from pie_datasets import DatasetDict
from tests import FIXTURES_ROOT
from tests.dataset_builders.common import (
    HF_DS_FIXTURE_DATA_PATH,
    PIE_BASE_PATH,
    TestTokenDocumentWithLabeledSpansAndBinaryRelations,
)

disable_caching()

DATASET_NAME = "scidtb_argmin"
BUILDER_CLASS = SciDTBArgmin
SPLIT_SIZES = {"train": 60}
HF_DATASET_PATH = SciDTBArgmin.BASE_DATASET_PATH
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
DATA_PATH = FIXTURES_ROOT / "dataset_builders" / "scidtb_argmin_annotations.tgz"


@pytest.fixture(scope="module")
def hf_dataset():
    return load_dataset(str(HF_DATASET_PATH), data_dir=DATA_PATH)


def test_hf_dataset(hf_dataset):
    assert hf_dataset is not None
    assert {name: len(ds) for name, ds in hf_dataset.items()} == SPLIT_SIZES


@pytest.fixture(scope="module")
def hf_example(hf_dataset):
    return hf_dataset["train"][0]


def test_hf_example(hf_example):
    assert hf_example is not None
    hf_example_expected = json.load(open(HF_DS_FIXTURE_DATA_PATH / DATASET_NAME / "train.0.json"))
    assert hf_example == hf_example_expected


@pytest.fixture(scope="module")
def generate_document_kwargs(hf_dataset):
    return BUILDER_CLASS()._generate_document_kwargs(hf_dataset["train"])


@pytest.fixture(scope="module")
def generated_document(hf_example, generate_document_kwargs):
    return BUILDER_CLASS()._generate_document(hf_example, **generate_document_kwargs)


def test_generated_document(generated_document):
    assert isinstance(generated_document, SciDTBArgminDocument)
    assert len(generated_document.tokens) == 189
    assert str(generated_document.tokens[:5]) == "('This', 'paper', 'presents', 'a', 'deep')"
    assert len(generated_document.units) == 6
    assert len(generated_document.relations) == 5


@pytest.fixture(scope="module")
def hf_example_back(generated_document, generate_document_kwargs):
    return document_to_example(generated_document, **generate_document_kwargs)


def test_example_to_document_and_back(hf_example, hf_example_back):
    assert hf_example_back == hf_example


def test_example_to_document_and_back_all(hf_dataset, generate_document_kwargs):
    for hf_ex in hf_dataset["train"]:
        doc = example_to_document(hf_ex, **generate_document_kwargs)
        hf_example_back = document_to_example(doc, **generate_document_kwargs)
        assert hf_example_back == hf_ex


@pytest.fixture(scope="module")
def dataset() -> DatasetDict:
    return DatasetDict.load_dataset(str(PIE_DATASET_PATH))


def test_dataset(dataset):
    assert dataset is not None
    assert {name: len(ds) for name, ds in dataset.items()} == SPLIT_SIZES


@pytest.fixture(scope="module")
def document(dataset) -> SciDTBArgminDocument:
    result = dataset["train"][0]
    # we can not assert the real document type because it may come from a dataset loading script
    # downloaded to a temporary directory and thus have a different type object, although it is
    # semantically the same
    assert isinstance(result, Document)
    return result


def test_compare_document_and_generated_document(document, generated_document):
    assert document.id == generated_document.id
    assert document.tokens == generated_document.tokens
    assert document.units == generated_document.units
    assert document.relations == generated_document.relations
    assert document.metadata == generated_document.metadata


def test_pie_dataset(document):
    assert document is not None
    units = document.units
    relations = document.relations

    assert len(units) == 6
    assert (
        str(units[0])
        == "('This', 'paper', 'presents', 'a', 'deep', 'semantic', 'similarity', 'model', '-LRB-', 'DSSM', '-RRB-', "
        "',', 'a', 'special', 'type', 'of', 'deep', 'neural', 'networks', 'designed', 'for', 'text', 'analysis', "
        "',', 'for', 'recommending', 'target', 'documents', 'to', 'be', 'of', 'interest', 'to', 'a', 'user', "
        "'based', 'on', 'a', 'source', 'document', 'that', 'she', 'is', 'reading', '.')"
    )
    assert (
        str(units[1])
        == "('We', 'observe', ',', 'identify', ',', 'and', 'detect', 'naturally', 'occurring', 'signals', 'of', "
        "'interestingness', 'in', 'click', 'transitions', 'on', 'the', 'Web', 'between', 'source', 'and', "
        "'target', 'documents', ',', 'which', 'we', 'collect', 'from', 'commercial', 'Web', 'browser', 'logs', "
        "'.')"
    )
    assert (
        str(units[2])
        == "('The', 'DSSM', 'is', 'trained', 'on', 'millions', 'of', 'Web', 'transitions', ',', 'and', 'maps', "
        "'source-target', 'document', 'pairs', 'to', 'feature', 'vectors', 'in', 'a', 'latent', 'space', 'in', "
        "'such', 'a', 'way', 'that', 'the', 'distance', 'between', 'source', 'documents', 'and', 'their', "
        "'corresponding', 'interesting', 'targets', 'in', 'that', 'space', 'is', 'minimized', '.')"
    )
    assert (
        str(units[3])
        == "('The', 'effectiveness', 'of', 'the', 'DSSM', 'is', 'demonstrated', 'using', 'two', 'interestingness', "
        "'tasks', ':', 'automatic', 'highlighting', 'and', 'contextual', 'entity', 'search', '.')"
    )
    assert (
        str(units[4])
        == "('The', 'results', 'on', 'large-scale', ',', 'real-world', 'datasets', 'show', 'that', 'the', "
        "'semantics', 'of', 'documents', 'are', 'important', 'for', 'modeling', 'interestingness')"
    )
    assert (
        str(units[5])
        == "('and', 'that', 'the', 'DSSM', 'leads', 'to', 'significant', 'quality', 'improvement', 'on', 'both', "
        "'tasks', ',', 'outperforming', 'not', 'only', 'the', 'classic', 'document', 'models', 'that', 'do', "
        "'not', 'use', 'semantics', 'but', 'also', 'state-of-the-art', 'topic', 'models', '.')"
    )

    assert len(relations) == 5
    assert relations[0].label == "detail"
    assert relations[0].head == units[1]
    assert relations[0].tail == units[2]
    assert relations[1].label == "detail"
    assert relations[1].head == units[2]
    assert relations[1].tail == units[0]
    assert relations[2].label == "support"
    assert relations[2].head == units[3]
    assert relations[2].tail == units[0]
    assert relations[3].label == "detail"
    assert relations[3].head == units[4]
    assert relations[3].tail == units[3]
    assert relations[4].label == "detail"
    assert relations[4].head == units[5]
    assert relations[4].tail == units[3]


@pytest.fixture(scope="module")
def dataset_of_text_documents_with_labeled_spans_and_binary_relations(
    dataset,
) -> DatasetDict:
    converted_dataset = dataset.to_document_type(TextDocumentWithLabeledSpansAndBinaryRelations)
    return converted_dataset


def test_convert_to_text_document_with_entities_and_relations(
    document, dataset_of_text_documents_with_labeled_spans_and_binary_relations
):
    # just check that we get the same as in the converted dataset when explicitly calling the conversion method
    converted_doc = convert_to_text_document_with_labeled_spans_and_binary_relations(document)
    doc_from_converted_dataset = dataset_of_text_documents_with_labeled_spans_and_binary_relations[
        "train"
    ][0]
    assert converted_doc == doc_from_converted_dataset


@pytest.mark.slow
def test_dataset_of_text_documents_with_labeled_spans_and_binary_relations(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations,
):
    assert dataset_of_text_documents_with_labeled_spans_and_binary_relations is not None
    # get a document to check
    converted_doc = dataset_of_text_documents_with_labeled_spans_and_binary_relations["train"][0]
    assert isinstance(converted_doc, TextDocumentWithLabeledSpansAndBinaryRelations)

    # check entities
    assert len(converted_doc.labeled_spans) == 6
    entity_tuples = [(str(ent), ent.label) for ent in converted_doc.labeled_spans]
    assert entity_tuples[0] == (
        "This paper presents a deep semantic similarity model -LRB- DSSM -RRB- , a special type of deep neural "
        "networks designed for text analysis , for recommending target documents to be of interest to "
        "a user based on a source document that she is reading .",
        "proposal",
    )
    assert entity_tuples[1] == (
        "We observe , identify , and detect naturally occurring signals of interestingness in click transitions "
        "on the Web between source and target documents , which we collect from commercial Web browser logs .",
        "means",
    )
    assert entity_tuples[2] == (
        "The DSSM is trained on millions of Web transitions , and maps source-target document pairs to feature "
        "vectors in a latent space in such a way that the distance between source documents and their corresponding "
        "interesting targets in that space is minimized .",
        "proposal",
    )
    assert entity_tuples[3] == (
        "The effectiveness of the DSSM is demonstrated using two interestingness tasks : automatic highlighting "
        "and contextual entity search .",
        "result",
    )
    assert entity_tuples[4] == (
        "The results on large-scale , real-world datasets show that the semantics of documents are important for "
        "modeling interestingness",
        "result",
    )
    assert entity_tuples[5] == (
        "and that the DSSM leads to significant quality improvement on both tasks , outperforming not only "
        "the classic document models that do not use semantics but also state-of-the-art topic models .",
        "result",
    )

    # check relations
    relation_tuples = [
        (str(rel.head), rel.label, str(rel.tail)) for rel in converted_doc.binary_relations
    ]
    assert len(relation_tuples) == 5
    assert relation_tuples[0] == (
        "We observe , identify , and detect naturally occurring signals of interestingness in click transitions on "
        "the Web between source and target documents , which we collect from commercial Web browser logs .",
        "detail",
        "The DSSM is trained on millions of Web transitions , and maps source-target document pairs to feature "
        "vectors in a latent space in such a way that the distance between source documents and their corresponding "
        "interesting targets in that space is minimized .",
    )
    assert relation_tuples[1] == (
        "The DSSM is trained on millions of Web transitions , and maps source-target document pairs to feature vectors "
        "in a latent space in such a way that the distance between source documents and their corresponding "
        "interesting targets in that space is minimized .",
        "detail",
        "This paper presents a deep semantic similarity model -LRB- DSSM -RRB- , a special type of deep neural "
        "networks designed for text analysis , for recommending target documents to be of interest to a user "
        "based on a source document that she is reading .",
    )
    assert relation_tuples[2] == (
        "The effectiveness of the DSSM is demonstrated using two interestingness tasks : automatic highlighting "
        "and contextual entity search .",
        "support",
        "This paper presents a deep semantic similarity model -LRB- DSSM -RRB- , a special type of deep neural "
        "networks designed for text analysis , for recommending target documents to be of interest to a "
        "user based on a source document that she is reading .",
    )
    assert relation_tuples[3] == (
        "The results on large-scale , real-world datasets show that the semantics of documents are important "
        "for modeling interestingness",
        "detail",
        "The effectiveness of the DSSM is demonstrated using two interestingness tasks : automatic highlighting "
        "and contextual entity search .",
    )
    assert relation_tuples[4] == (
        "and that the DSSM leads to significant quality improvement on both tasks , outperforming not only "
        "the classic document models that do not use semantics but also state-of-the-art topic models .",
        "detail",
        "The effectiveness of the DSSM is demonstrated using two interestingness tasks : automatic highlighting "
        "and contextual entity search .",
    )


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def tokenized_documents_with_labeled_spans_and_binary_relations(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations, tokenizer
) -> List[TestTokenDocumentWithLabeledSpansAndBinaryRelations]:
    # get a document to check
    doc = dataset_of_text_documents_with_labeled_spans_and_binary_relations["train"][0]
    # Note, that this is a list of documents, because the document may be split into chunks
    # if the input text is too long.
    tokenized_docs = tokenize_document(
        doc,
        tokenizer=tokenizer,
        return_overflowing_tokens=True,
        result_document_type=TestTokenDocumentWithLabeledSpansAndBinaryRelations,
        verbose=True,
    )
    return tokenized_docs


def test_tokenized_documents_with_labeled_spans_and_binary_relations(
    tokenized_documents_with_labeled_spans_and_binary_relations,
):
    docs = tokenized_documents_with_labeled_spans_and_binary_relations
    assert len(docs) == 1
    doc = docs[0]
    assert len(doc.tokens) == 225
    assert len(doc.labeled_spans) == 6
    ent = doc.labeled_spans[0]
    assert (
        str(ent)
        == "('this', 'paper', 'presents', 'a', 'deep', 'semantic', 'similarity', 'model', '-', 'l', '##rb', '-', "
        "'ds', '##sm', '-', 'rr', '##b', '-', ',', 'a', 'special', 'type', 'of', 'deep', 'neural', 'networks', "
        "'designed', 'for', 'text', 'analysis', ',', 'for', 'recommend', '##ing', 'target', 'documents', 'to', "
        "'be', 'of', 'interest', 'to', 'a', 'user', 'based', 'on', 'a', 'source', 'document', 'that', 'she', 'is', "
        "'reading', '.')"
    )
    ent = doc.labeled_spans[1]
    assert (
        str(ent)
        == "('we', 'observe', ',', 'identify', ',', 'and', 'detect', 'naturally', 'occurring', 'signals', 'of', "
        "'interesting', '##ness', 'in', 'click', 'transitions', 'on', 'the', 'web', 'between', 'source', 'and', "
        "'target', 'documents', ',', 'which', 'we', 'collect', 'from', 'commercial', 'web', 'browser', 'logs', '.')"
    )
    ent = doc.labeled_spans[2]
    assert (
        str(ent)
        == "('the', 'ds', '##sm', 'is', 'trained', 'on', 'millions', 'of', 'web', 'transitions', ',', 'and', 'maps', "
        "'source', '-', 'target', 'document', 'pairs', 'to', 'feature', 'vectors', 'in', 'a', 'late', '##nt', "
        "'space', 'in', 'such', 'a', 'way', 'that', 'the', 'distance', 'between', 'source', 'documents', 'and', "
        "'their', 'corresponding', 'interesting', 'targets', 'in', 'that', 'space', 'is', 'minimize', '##d', '.')"
    )
    ent = doc.labeled_spans[3]
    assert (
        str(ent)
        == "('the', 'effectiveness', 'of', 'the', 'ds', '##sm', 'is', 'demonstrated', 'using', 'two', 'interesting', "
        "'##ness', 'tasks', ':', 'automatic', 'highlighting', 'and', 'context', '##ual', 'entity', 'search', '.')"
    )
    ent = doc.labeled_spans[4]
    assert (
        str(ent)
        == "('the', 'results', 'on', 'large', '-', 'scale', ',', 'real', '-', 'world', 'data', '##set', '##s', 'show', "
        "'that', 'the', 'semantics', 'of', 'documents', 'are', 'important', 'for', 'modeling', 'interesting', "
        "'##ness')"
    )
    ent = doc.labeled_spans[5]
    assert (
        str(ent)
        == "('and', 'that', 'the', 'ds', '##sm', 'leads', 'to', 'significant', 'quality', 'improvement', 'on', "
        "'both', 'tasks', ',', 'out', '##per', '##form', '##ing', 'not', 'only', 'the', 'classic', 'document', "
        "'models', 'that', 'do', 'not', 'use', 'semantics', 'but', 'also', 'state', '-', 'of', '-', 'the', "
        "'-', 'art', 'topic', 'models', '.')"
    )


def test_tokenized_documents_with_entities_and_relations_all(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations, tokenizer
):
    for split, docs in dataset_of_text_documents_with_labeled_spans_and_binary_relations.items():
        for doc in docs:
            # Note, that this is a list of documents, because the document may be split into chunks
            # if the input text is too long.
            tokenized_docs = tokenize_document(
                doc,
                tokenizer=tokenizer,
                return_overflowing_tokens=True,
                result_document_type=TestTokenDocumentWithLabeledSpansAndBinaryRelations,
                verbose=True,
            )
            # we just ensure that we get at least one tokenized document
            assert tokenized_docs is not None
            assert len(tokenized_docs) > 0
