import json
from collections import Counter
from typing import List

import pytest
from datasets import disable_caching, load_dataset
from pytorch_ie.core import Document
from pytorch_ie.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
)
from transformers import AutoTokenizer, PreTrainedTokenizer

from dataset_builders.pie.argmicro.argmicro import (
    ArgMicro,
    ArgMicroDocument,
    convert_to_text_document_with_labeled_spans_and_binary_relations,
    document_to_example,
    example_to_document,
)
from pie_datasets import DatasetDict
from pie_datasets.document.conversion import tokenize_document
from pie_datasets.document.types import TokenDocumentWithLabeledSpansAndBinaryRelations
from tests import FIXTURES_ROOT
from tests.dataset_builders.common import HF_DS_FIXTURE_DATA_PATH, PIE_BASE_PATH

disable_caching()

DATASET_NAME = "argmicro"
SPLIT_SIZES = {"train": 112}
DATA_PATH = FIXTURES_ROOT / "dataset_builders" / "arg-microtexts-master.zip"
HF_DATASET_PATH = ArgMicro.BASE_DATASET_PATH
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME


@pytest.fixture(scope="module", params=["en", "de"])
def language(request):
    return request.param


@pytest.fixture(scope="module")
def hf_dataset(language):
    return load_dataset(str(HF_DATASET_PATH), name=language, data_dir=DATA_PATH)


@pytest.fixture(scope="module")
def generate_document_kwargs(hf_dataset, language):
    ds = hf_dataset["train"]
    return ArgMicro(config_name=language)._generate_document_kwargs(ds)


def test_hf_dataset(hf_dataset, language, generate_document_kwargs):
    assert hf_dataset is not None
    assert {name: len(ds) for name, ds in hf_dataset.items()} == SPLIT_SIZES
    ds = hf_dataset["train"]
    # check
    topic_id_values = Counter([ex["topic_id"] for ex in ds])
    stance_values = Counter(
        [generate_document_kwargs["stance_label"].int2str(ex["stance"]) for ex in ds]
    )
    adu_type_values = Counter()
    edge_type_values = Counter()
    for ex in ds:
        adu_type_values.update(
            [generate_document_kwargs["adu_type_label"].int2str(t) for t in ex["adus"]["type"]]
        )
        edge_type_values.update(
            [generate_document_kwargs["edge_type_label"].int2str(t) for t in ex["edges"]["type"]]
        )
    assert dict(topic_id_values) == {
        "UNDEFINED": 23,
        "higher_dog_poo_fines": 9,
        "health_insurance_cover_complementary_medicine": 8,
        "introduce_capital_punishment": 8,
        "allow_shops_to_open_on_holidays_and_sundays": 8,
        "public_broadcasting_fees_on_demand": 7,
        "over_the_counter_morning_after_pill": 6,
        "cap_rent_increases": 6,
        "charge_tuition_fees": 6,
        "keep_retirement_at_63": 6,
        "stricter_regulation_of_intelligence_services": 4,
        "increase_weight_of_BA_thesis_in_final_grade": 4,
        "school_uniforms": 3,
        "make_video_games_olympic": 3,
        "EU_influence_on_political_events_in_Ukraine": 3,
        "TXL_airport_remain_operational_after_BER_opening": 3,
        "buy_tax_evader_data_from_dubious_sources": 2,
        "partial_housing_development_at_Tempelhofer_Feld": 2,
        "waste_separation": 1,
    }
    assert dict(stance_values) == {"pro": 46, "con": 42, "UNDEFINED": 23, "unclear": 1}
    assert dict(adu_type_values) == {"pro": 451, "opp": 125}
    assert dict(edge_type_values) == {
        "seg": 576,
        "sup": 263,
        "reb": 108,
        "und": 63,
        "add": 21,
        "exa": 9,
    }


@pytest.fixture
def hf_example(hf_dataset):
    return hf_dataset["train"][0]


def test_hf_example(hf_example, hf_dataset, language, generate_document_kwargs):
    fixture_path = HF_DS_FIXTURE_DATA_PATH / DATASET_NAME / f"{language}.train.0.json"
    # hf_example_expected = json.loads(fixture_path.read_text())
    hf_example_expected = json.loads(open(fixture_path, encoding="utf-8").read())
    assert hf_example == hf_example_expected
    assert generate_document_kwargs["stance_label"].int2str(hf_example["stance"]) == "pro"
    assert [
        generate_document_kwargs["adu_type_label"].int2str(t) for t in hf_example["adus"]["type"]
    ] == [
        "opp",
        "opp",
        "pro",
        "pro",
        "pro",
    ]
    assert [
        generate_document_kwargs["edge_type_label"].int2str(t) for t in hf_example["edges"]["type"]
    ] == [
        "reb",
        "seg",
        "sup",
        "und",
        "add",
        "seg",
        "seg",
        "seg",
        "seg",
    ]


# need to find bug
@pytest.fixture(scope="module")
def generated_document(hf_dataset, language, generate_document_kwargs):
    return ArgMicro(config_name=language)._generate_document(
        hf_dataset["train"][0], **generate_document_kwargs
    )


def test_generated_document(generated_document):
    assert isinstance(generated_document, ArgMicroDocument)


def test_example_to_document(generated_document, language):
    assert isinstance(generated_document, ArgMicroDocument)
    assert generated_document is not None
    assert generated_document.id == "micro_b001"
    assert generated_document.topic_id == "waste_separation"
    assert generated_document.stance[0].label == "pro"
    assert len(generated_document.edus) == 5
    assert generated_document.metadata["edu_ids"] == ["e1", "e2", "e3", "e4", "e5"]
    edus_dict = {
        edu_id: edu
        for edu_id, edu in zip(generated_document.metadata["edu_ids"], generated_document.edus)
    }
    if language == "en":
        assert (
            str(generated_document.edus[0])
            == "Yes, it's annoying and cumbersome to separate your rubbish properly all the time."
        )
        assert (
            str(generated_document.edus[1])
            == "Three different bin bags stink away in the kitchen and have to be sorted into different wheelie bins."
        )
        assert str(generated_document.edus[2]) == "But still Germany produces way too much rubbish"
        assert (
            str(generated_document.edus[3])
            == "and too many resources are lost when what actually should be separated and recycled is burnt."
        )
        assert (
            str(generated_document.edus[4])
            == "We Berliners should take the chance and become pioneers in waste separation!"
        )
    elif language == "de":
        # we don't check this because spellcheck would complain
        pass
    else:
        raise ValueError(f"Unknown language {language}")

    assert len(generated_document.adus) == 5
    assert generated_document.metadata["adu_ids"] == ["a1", "a2", "a3", "a4", "a5"]
    adus_dict = {
        adu_id: adu
        for adu_id, adu in zip(generated_document.metadata["adu_ids"], generated_document.adus)
    }
    assert generated_document.adus[0].label == "opp"
    assert len(generated_document.adus[0].annotations) == 1
    assert generated_document.adus[0].annotations[0] == generated_document.edus[0]
    assert generated_document.adus[1].label == "opp"
    assert len(generated_document.adus[1].annotations) == 1
    assert generated_document.adus[1].annotations[0] == generated_document.edus[1]
    assert generated_document.adus[2].label == "pro"
    assert len(generated_document.adus[2].annotations) == 1
    assert generated_document.adus[2].annotations[0] == generated_document.edus[2]
    assert generated_document.adus[3].label == "pro"
    assert len(generated_document.adus[3].annotations) == 1
    assert generated_document.adus[3].annotations[0] == generated_document.edus[3]
    assert generated_document.adus[4].label == "pro"
    assert len(generated_document.adus[4].annotations) == 1
    assert generated_document.adus[4].annotations[0] == generated_document.edus[4]

    # sources == heads
    # targets == tails
    assert len(generated_document.relations) == 3
    assert generated_document.metadata["rel_ids"] == ["c1", "c2", "c3"]
    rels_dict = {
        rel_id: rel
        for rel_id, rel in zip(
            generated_document.metadata["rel_ids"], generated_document.relations
        )
    }

    rel = rels_dict["c1"]
    assert rel.label == "reb"  # rebutting attack
    assert len(rel.heads) == 1
    assert rel.heads[0] == adus_dict["a1"]
    assert len(rel.tails) == 1
    assert rel.tails[0] == adus_dict["a5"]

    rel = rels_dict["c2"]
    assert rel.label == "sup"  # support
    assert len(rel.heads) == 1
    assert rel.heads[0] == adus_dict["a2"]
    assert len(rel.tails) == 1
    assert rel.tails[0] == adus_dict["a1"]

    rel = rels_dict["c3"]
    assert rel.label == "und"  # undercutting attack
    assert len(rel.heads) == 2
    assert rel.heads[0] == adus_dict["a3"]
    assert rel.heads[1] == adus_dict["a4"]
    assert len(rel.tails) == 1
    assert rel.tails[0] == adus_dict["a1"]

    assert generated_document.metadata["rel_seg_ids"] == {
        "e1": "c6",
        "e2": "c7",
        "e3": "c8",
        "e4": "c9",
        "e5": "c10",
    }
    assert generated_document.metadata["rel_add_ids"] == {"a4": "c4"}


def test_example_to_document_and_back(hf_example, generated_document, generate_document_kwargs):
    hf_example_back = document_to_example(generated_document, **generate_document_kwargs)
    assert hf_example == hf_example_back


def test_example_to_document_and_back_all(hf_dataset, generate_document_kwargs):
    for ex in hf_dataset["train"]:
        doc = example_to_document(ex, **generate_document_kwargs)
        ex_back = document_to_example(doc, **generate_document_kwargs)
        assert ex == ex_back


@pytest.fixture(scope="module")
def dataset() -> DatasetDict:
    return DatasetDict.load_dataset(path=str(PIE_DATASET_PATH), name="en")


def test_dataset(dataset):
    assert dataset is not None
    assert {name: len(ds) for name, ds in dataset.items()} == SPLIT_SIZES


@pytest.fixture(scope="module")
def document(dataset) -> ArgMicroDocument:
    doc = dataset["train"][0]
    # we can not assert the real document type because it may come from a dataset loading script
    # downloaded to a temporary directory and thus have a different type object, although it is
    # semantically the same
    assert isinstance(doc, Document)
    return doc


def test_compare_document_and_generated_document(document, generated_document, language):
    # We cast the document to the type of the generated document to compare them.
    # This is necessary because the document may come from a dataset loading script
    # downloaded to a temporary directory and thus have a different type object, although it is
    # semantically the same.
    casted_document = document.as_type(type(generated_document))
    if language == "en":
        assert casted_document.id == generated_document.id
        assert casted_document.topic_id == generated_document.topic_id
        assert casted_document.text == generated_document.text
        assert casted_document.edus == generated_document.edus
        assert casted_document.adus == generated_document.adus
        assert casted_document.stance == generated_document.stance
        assert casted_document.relations == generated_document.relations
        # contents of metadata are different: there are additional keys with None values e.g.,
        # 'rel_add_ids': {'a4': 'c4'}' !=  'rel_add_ids': {'a2': None, 'a3': None, 'a4': 'c4', 'a5': None, 'a6': None}'
        # it looks like Huggingface datasets creates a kind of schema from the data and adds all keys it finds
        for k in set(generated_document.metadata) | set(casted_document.metadata):
            v = generated_document.metadata[k]
            v_expected = casted_document.metadata[k]
            if isinstance(v, dict):
                v_expected_without_none = {k: v for k, v in v_expected.items() if v is not None}
                assert v == v_expected_without_none
            else:
                assert v == v_expected
    elif language == "de":
        # we don't check because we only call the dataset in English
        pass
    else:
        raise ValueError(f"Unknown language {language}")


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def dataset_of_text_documents_with_labeled_spans_and_binary_relations(
    dataset,
) -> DatasetDict:
    converted_dataset = dataset.to_document_type(TextDocumentWithLabeledSpansAndBinaryRelations)
    return converted_dataset


def test_convert_to_textdocument_with_entities_and_relations(
    document, dataset_of_text_documents_with_labeled_spans_and_binary_relations
):
    # just check that we get the same as in the converted dataset when explicitly calling the conversion method
    converted_doc = convert_to_text_document_with_labeled_spans_and_binary_relations(document)
    doc_from_converted_dataset = dataset_of_text_documents_with_labeled_spans_and_binary_relations[
        "train"
    ][0]
    assert converted_doc == doc_from_converted_dataset


def test_dataset_of_text_documents_with_labeled_spans_and_binary_relations(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations,
):
    assert dataset_of_text_documents_with_labeled_spans_and_binary_relations is not None
    # get a document to check
    converted_doc = dataset_of_text_documents_with_labeled_spans_and_binary_relations["train"][0]

    assert isinstance(converted_doc, TextDocumentWithLabeledSpansAndBinaryRelations)
    assert len(converted_doc.labeled_spans) == 5

    # check the entities
    entity_tuples = [(str(ent), ent.label) for ent in converted_doc.labeled_spans]
    assert entity_tuples[0] == (
        "Yes, it's annoying and cumbersome to separate your rubbish properly all the time.",
        "opp",
    )
    assert entity_tuples[1] == (
        "Three different bin bags stink away in the kitchen and have to be sorted into different wheelie bins.",
        "opp",
    )
    assert entity_tuples[2] == ("But still Germany produces way too much rubbish", "pro")
    assert entity_tuples[3] == (
        "and too many resources are lost when what actually should be separated and recycled is burnt.",
        "pro",
    )
    assert entity_tuples[4] == (
        "We Berliners should take the chance and become pioneers in waste separation!",
        "pro",
    )

    # check the relations
    relation_tuples = [
        (str(rel.head), rel.label, str(rel.tail)) for rel in converted_doc.binary_relations
    ]
    assert len(relation_tuples) == 6
    assert relation_tuples[0] == (
        "Yes, it's annoying and cumbersome to separate your rubbish properly all the time.",
        "reb",
        "We Berliners should take the chance and become pioneers in waste separation!",
    )
    assert relation_tuples[1] == (
        "Three different bin bags stink away in the kitchen and have to be sorted into different wheelie bins.",
        "sup",
        "Yes, it's annoying and cumbersome to separate your rubbish properly all the time.",
    )
    # Note: originally, the "undercut" relation has a joint head,
    # but we split it into two relations ...
    assert relation_tuples[2] == (
        "But still Germany produces way too much rubbish",
        "und",
        "Yes, it's annoying and cumbersome to separate your rubbish properly all the time.",
    )
    assert relation_tuples[3] == (
        "and too many resources are lost when what actually should be separated and recycled is burnt.",
        "und",
        "Yes, it's annoying and cumbersome to separate your rubbish properly all the time.",
    )
    # ... and link the heads with "joint" relations in a symmetric fashion
    assert relation_tuples[4] == (
        "But still Germany produces way too much rubbish",
        "joint",
        "and too many resources are lost when what actually should be separated and recycled is burnt.",
    )
    assert relation_tuples[5] == (
        "and too many resources are lost when what actually should be separated and recycled is burnt.",
        "joint",
        "But still Germany produces way too much rubbish",
    )


@pytest.fixture(scope="module")
def tokenized_documents_with_labeled_spans_and_binary_relations(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations, tokenizer
) -> List[TokenDocumentWithLabeledSpansAndBinaryRelations]:
    # get a document to check
    doc = dataset_of_text_documents_with_labeled_spans_and_binary_relations["train"][0]
    # Note, that this is a list of documents, because the document may be split into chunks
    # if the input text is too long.
    tokenized_docs = tokenize_document(
        doc,
        tokenizer=tokenizer,
        return_overflowing_tokens=True,
        result_document_type=TokenDocumentWithLabeledSpansAndBinaryRelations,
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
    assert len(doc.tokens) == 81
    assert len(doc.labeled_spans) == 5
    ent = doc.labeled_spans[0]
    assert ent.target[ent.start : ent.end] == (
        "yes",
        ",",
        "it",
        "'",
        "s",
        "annoying",
        "and",
        "cum",
        "##bers",
        "##ome",
        "to",
        "separate",
        "your",
        "rubbish",
        "properly",
        "all",
        "the",
        "time",
        ".",
    )
    ent = doc.labeled_spans[1]
    assert ent.target[ent.start : ent.end] == (
        "three",
        "different",
        "bin",
        "bags",
        "stink",
        "away",
        "in",
        "the",
        "kitchen",
        "and",
        "have",
        "to",
        "be",
        "sorted",
        "into",
        "different",
        "wheel",
        "##ie",
        "bin",
        "##s",
        ".",
    )
    ent = doc.labeled_spans[2]
    assert ent.target[ent.start : ent.end] == (
        "but",
        "still",
        "germany",
        "produces",
        "way",
        "too",
        "much",
        "rubbish",
    )
    ent = doc.labeled_spans[3]
    assert ent.target[ent.start : ent.end] == (
        "and",
        "too",
        "many",
        "resources",
        "are",
        "lost",
        "when",
        "what",
        "actually",
        "should",
        "be",
        "separated",
        "and",
        "recycled",
        "is",
        "burnt",
        ".",
    )
    ent = doc.labeled_spans[4]
    assert ent.target[ent.start : ent.end] == (
        "we",
        "berlin",
        "##ers",
        "should",
        "take",
        "the",
        "chance",
        "and",
        "become",
        "pioneers",
        "in",
        "waste",
        "separation",
        "!",
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
                result_document_type=TokenDocumentWithLabeledSpansAndBinaryRelations,
                verbose=True,
            )
            # we just ensure that we get at least one tokenized document
            assert tokenized_docs is not None
            assert len(tokenized_docs) > 0
