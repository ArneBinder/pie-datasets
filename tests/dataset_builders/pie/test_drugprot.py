from typing import Any, Dict, List, Type, Union

import datasets
import pytest
from pie_modules.document.processing import tokenize_document
from pie_modules.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
    TokenBasedDocument,
    TokenDocumentWithLabeledSpansAndBinaryRelations,
    TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)
from transformers import AutoTokenizer, PreTrainedTokenizer

from dataset_builders.pie.drugprot.drugprot import (
    Drugprot,
    DrugprotBigbioDocument,
    DrugprotDocument,
    example2drugprot,
    example2drugprot_bigbio,
)
from pie_datasets import DatasetDict, load_dataset
from tests.dataset_builders.common import PIE_BASE_PATH

DATASET_NAME = "drugprot"
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
HF_DATASET_PATH = Drugprot.BASE_DATASET_PATH
SPLIT_NAMES = {"train", "validation"}
SPLIT_SIZES = {"train": 3500, "validation": 750}


@pytest.fixture(params=[config.name for config in Drugprot.BUILDER_CONFIGS], scope="module")
def dataset_variant(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def hf_dataset(dataset_variant) -> datasets.DatasetDict:
    return datasets.load_dataset(HF_DATASET_PATH, name=dataset_variant)


def test_hf_dataset(hf_dataset):
    assert set(hf_dataset) == SPLIT_NAMES
    split_sizes = {split_name: len(ds) for split_name, ds in hf_dataset.items()}
    assert split_sizes == SPLIT_SIZES


@pytest.fixture(scope="module")
def hf_example(hf_dataset) -> Dict[str, Any]:
    return hf_dataset["train"][0]


def test_hf_example(hf_example, dataset_variant):
    if dataset_variant == "drugprot_source":
        assert hf_example == {
            "document_id": "17512723",
            "title": "RDH12, a retinol dehydrogenase causing Leber's congenital amaurosis, is also involved in steroid metabolism.",
            "abstract": "Three retinol dehydrogenases (RDHs) were tested for steroid converting abilities: human and murine RDH 12 and human RDH13. RDH12 is involved in retinal degeneration in Leber's congenital amaurosis (LCA). We show that murine Rdh12 and human RDH13 do not reveal activity towards the checked steroids, but that human type 12 RDH reduces dihydrotestosterone to androstanediol, and is thus also involved in steroid metabolism. Furthermore, we analyzed both expression and subcellular localization of these enzymes.",
            "text": "RDH12, a retinol dehydrogenase causing Leber's congenital amaurosis, is also involved in steroid metabolism. Three retinol dehydrogenases (RDHs) were tested for steroid converting abilities: human and murine RDH 12 and human RDH13. RDH12 is involved in retinal degeneration in Leber's congenital amaurosis (LCA). We show that murine Rdh12 and human RDH13 do not reveal activity towards the checked steroids, but that human type 12 RDH reduces dihydrotestosterone to androstanediol, and is thus also involved in steroid metabolism. Furthermore, we analyzed both expression and subcellular localization of these enzymes.",
            "entities": [
                {
                    "id": "17512723_T1",
                    "type": "CHEMICAL",
                    "text": "androstanediol",
                    "offset": [466, 480],
                },
                {
                    "id": "17512723_T2",
                    "type": "CHEMICAL",
                    "text": "retinol",
                    "offset": [115, 122],
                },
                {
                    "id": "17512723_T3",
                    "type": "CHEMICAL",
                    "text": "retinol",
                    "offset": [9, 16],
                },
                {
                    "id": "17512723_T4",
                    "type": "GENE-Y",
                    "text": "human RDH13",
                    "offset": [219, 230],
                },
                {
                    "id": "17512723_T5",
                    "type": "GENE-Y",
                    "text": "RDH12",
                    "offset": [232, 237],
                },
                {
                    "id": "17512723_T6",
                    "type": "GENE-Y",
                    "text": "murine Rdh12",
                    "offset": [326, 338],
                },
                {
                    "id": "17512723_T7",
                    "type": "GENE-Y",
                    "text": "human RDH13",
                    "offset": [343, 354],
                },
                {
                    "id": "17512723_T8",
                    "type": "GENE-N",
                    "text": "RDHs",
                    "offset": [139, 143],
                },
                {
                    "id": "17512723_T9",
                    "type": "GENE-Y",
                    "text": "human type 12 RDH",
                    "offset": [417, 434],
                },
                {
                    "id": "17512723_T10",
                    "type": "GENE-N",
                    "text": "retinol dehydrogenases",
                    "offset": [115, 137],
                },
                {
                    "id": "17512723_T11",
                    "type": "GENE-N",
                    "text": "human and murine RDH 12",
                    "offset": [191, 214],
                },
                {
                    "id": "17512723_T12",
                    "type": "GENE-Y",
                    "text": "RDH12",
                    "offset": [0, 5],
                },
                {
                    "id": "17512723_T13",
                    "type": "GENE-N",
                    "text": "retinol dehydrogenase",
                    "offset": [9, 30],
                },
            ],
            "relations": [
                {
                    "id": "17512723_0",
                    "type": "PRODUCT-OF",
                    "arg1_id": "17512723_T1",
                    "arg2_id": "17512723_T9",
                }
            ],
        }
    elif dataset_variant == "drugprot_bigbio_kb":
        assert hf_example == {
            "id": "17512723",
            "document_id": "17512723",
            "passages": [
                {
                    "id": "17512723_title",
                    "type": "title",
                    "text": [
                        "RDH12, a retinol dehydrogenase causing Leber's congenital amaurosis, is also involved in steroid metabolism."
                    ],
                    "offsets": [[0, 108]],
                },
                {
                    "id": "17512723_abstract",
                    "type": "abstract",
                    "text": [
                        "Three retinol dehydrogenases (RDHs) were tested for steroid converting abilities: human and murine RDH 12 and human RDH13. RDH12 is involved in retinal degeneration in Leber's congenital amaurosis (LCA). We show that murine Rdh12 and human RDH13 do not reveal activity towards the checked steroids, but that human type 12 RDH reduces dihydrotestosterone to androstanediol, and is thus also involved in steroid metabolism. Furthermore, we analyzed both expression and subcellular localization of these enzymes."
                    ],
                    "offsets": [[109, 618]],
                },
            ],
            "entities": [
                {
                    "id": "17512723_T1",
                    "type": "CHEMICAL",
                    "text": ["androstanediol"],
                    "offsets": [[466, 480]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T2",
                    "type": "CHEMICAL",
                    "text": ["retinol"],
                    "offsets": [[115, 122]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T3",
                    "type": "CHEMICAL",
                    "text": ["retinol"],
                    "offsets": [[9, 16]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T4",
                    "type": "GENE-Y",
                    "text": ["human RDH13"],
                    "offsets": [[219, 230]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T5",
                    "type": "GENE-Y",
                    "text": ["RDH12"],
                    "offsets": [[232, 237]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T6",
                    "type": "GENE-Y",
                    "text": ["murine Rdh12"],
                    "offsets": [[326, 338]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T7",
                    "type": "GENE-Y",
                    "text": ["human RDH13"],
                    "offsets": [[343, 354]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T8",
                    "type": "GENE-N",
                    "text": ["RDHs"],
                    "offsets": [[139, 143]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T9",
                    "type": "GENE-Y",
                    "text": ["human type 12 RDH"],
                    "offsets": [[417, 434]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T10",
                    "type": "GENE-N",
                    "text": ["retinol dehydrogenases"],
                    "offsets": [[115, 137]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T11",
                    "type": "GENE-N",
                    "text": ["human and murine RDH 12"],
                    "offsets": [[191, 214]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T12",
                    "type": "GENE-Y",
                    "text": ["RDH12"],
                    "offsets": [[0, 5]],
                    "normalized": [],
                },
                {
                    "id": "17512723_T13",
                    "type": "GENE-N",
                    "text": ["retinol dehydrogenase"],
                    "offsets": [[9, 30]],
                    "normalized": [],
                },
            ],
            "events": [],
            "coreferences": [],
            "relations": [
                {
                    "id": "17512723_0",
                    "type": "PRODUCT-OF",
                    "arg1_id": "17512723_T1",
                    "arg2_id": "17512723_T9",
                    "normalized": [],
                }
            ],
        }
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")


@pytest.fixture(scope="module")
def document(hf_example, dataset_variant) -> Union[DrugprotDocument, DrugprotBigbioDocument]:
    if dataset_variant == "drugprot_source":
        return example2drugprot(hf_example)
    elif dataset_variant == "drugprot_bigbio_kb":
        return example2drugprot_bigbio(hf_example)
    else:
        raise ValueError(f"unknown dataset variant: {dataset_variant}")


def test_document(document, dataset_variant):
    if dataset_variant == "drugprot_source":
        assert isinstance(document, DrugprotDocument)
        assert (
            document.title
            == "RDH12, a retinol dehydrogenase causing Leber's congenital amaurosis, is also involved in steroid metabolism."
        )
        assert (
            document.abstract
            == "Three retinol dehydrogenases (RDHs) were tested for steroid converting abilities: human and murine RDH 12 and human RDH13. RDH12 is involved in retinal degeneration in Leber's congenital amaurosis (LCA). We show that murine Rdh12 and human RDH13 do not reveal activity towards the checked steroids, but that human type 12 RDH reduces dihydrotestosterone to androstanediol, and is thus also involved in steroid metabolism. Furthermore, we analyzed both expression and subcellular localization of these enzymes."
        )
    elif dataset_variant == "drugprot_bigbio_kb":
        assert isinstance(document, DrugprotBigbioDocument)
        passages = list(document.passages)
        assert len(passages)
        assert (
            str(passages[0])
            == "RDH12, a retinol dehydrogenase causing Leber's congenital amaurosis, is also involved in steroid metabolism."
        )
        assert (
            str(passages[1])
            == "Three retinol dehydrogenases (RDHs) were tested for steroid converting abilities: human and murine RDH 12 and human RDH13. RDH12 is involved in retinal degeneration in Leber's congenital amaurosis (LCA). We show that murine Rdh12 and human RDH13 do not reveal activity towards the checked steroids, but that human type 12 RDH reduces dihydrotestosterone to androstanediol, and is thus also involved in steroid metabolism. Furthermore, we analyzed both expression and subcellular localization of these enzymes."
        )
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")

    assert (
        document.text
        == "RDH12, a retinol dehydrogenase causing Leber's congenital amaurosis, is also involved in steroid metabolism. Three retinol dehydrogenases (RDHs) were tested for steroid converting abilities: human and murine RDH 12 and human RDH13. RDH12 is involved in retinal degeneration in Leber's congenital amaurosis (LCA). We show that murine Rdh12 and human RDH13 do not reveal activity towards the checked steroids, but that human type 12 RDH reduces dihydrotestosterone to androstanediol, and is thus also involved in steroid metabolism. Furthermore, we analyzed both expression and subcellular localization of these enzymes."
    )
    entities = list(document.entities)
    assert len(entities) == 13
    assert str(entities[0]) == "RDH12"
    assert str(entities[1]) == "retinol"
    assert str(entities[-1]) == "androstanediol"
    assert document.metadata["entity_ids"][0] == "17512723_T12"
    assert document.metadata["entity_ids"][1] == "17512723_T3"
    assert document.metadata["entity_ids"][-1] == "17512723_T1"

    relations = list(document.relations)
    assert len(relations) == 1
    assert relations[0].label == "PRODUCT-OF"
    assert str(relations[0].head) == "androstanediol"


@pytest.fixture(scope="module")
def pie_dataset(dataset_variant) -> DatasetDict:
    return load_dataset(str(PIE_DATASET_PATH), name=dataset_variant)


def test_pie_dataset(pie_dataset):
    assert set(pie_dataset) == SPLIT_NAMES
    split_sizes = {split_name: len(ds) for split_name, ds in pie_dataset.items()}
    assert split_sizes == SPLIT_SIZES


@pytest.fixture(scope="module")
def converted_document_type(dataset_variant) -> Type[TextBasedDocument]:
    if dataset_variant == "drugprot_source":
        return TextDocumentWithLabeledSpansAndBinaryRelations
    elif dataset_variant == "drugprot_bigbio_kb":
        return TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")


@pytest.fixture(scope="module")
def converted_pie_dataset(pie_dataset, converted_document_type) -> DatasetDict:
    pie_dataset_converted = pie_dataset.to_document_type(document_type=converted_document_type)
    return pie_dataset_converted


def test_converted_pie_dataset(converted_pie_dataset, converted_document_type):
    assert set(converted_pie_dataset) == SPLIT_NAMES
    split_sizes = {split_name: len(ds) for split_name, ds in converted_pie_dataset.items()}
    assert split_sizes == SPLIT_SIZES
    for ds in converted_pie_dataset.values():
        for document in ds:
            assert isinstance(document, converted_document_type)


@pytest.fixture(scope="module")
def converted_document(converted_pie_dataset) -> Type[TextBasedDocument]:
    return converted_pie_dataset["train"][0]


def test_converted_document(converted_document, converted_document_type):
    assert isinstance(converted_document, converted_document_type)
    if converted_document_type == TextDocumentWithLabeledSpansAndBinaryRelations:
        pass  # no specific tests for this type
    elif (
        converted_document_type == TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
    ):
        labeled_partitions = list(converted_document.labeled_partitions)
        assert len(labeled_partitions)
        assert (
            str(labeled_partitions[0])
            == "RDH12, a retinol dehydrogenase causing Leber's congenital amaurosis, is also involved in steroid metabolism."
        )
        assert (
            str(labeled_partitions[1])
            == "Three retinol dehydrogenases (RDHs) were tested for steroid converting abilities: human and murine RDH 12 and human RDH13. RDH12 is involved in retinal degeneration in Leber's congenital amaurosis (LCA). We show that murine Rdh12 and human RDH13 do not reveal activity towards the checked steroids, but that human type 12 RDH reduces dihydrotestosterone to androstanediol, and is thus also involved in steroid metabolism. Furthermore, we analyzed both expression and subcellular localization of these enzymes."
        )
    else:
        raise ValueError(f"Unknown document type: {converted_document_type}")

    assert (
        converted_document.text
        == "RDH12, a retinol dehydrogenase causing Leber's congenital amaurosis, is also involved in steroid metabolism. Three retinol dehydrogenases (RDHs) were tested for steroid converting abilities: human and murine RDH 12 and human RDH13. RDH12 is involved in retinal degeneration in Leber's congenital amaurosis (LCA). We show that murine Rdh12 and human RDH13 do not reveal activity towards the checked steroids, but that human type 12 RDH reduces dihydrotestosterone to androstanediol, and is thus also involved in steroid metabolism. Furthermore, we analyzed both expression and subcellular localization of these enzymes."
    )
    labeled_spans = list(converted_document.labeled_spans)
    assert len(labeled_spans) == 13
    assert str(labeled_spans[0]) == "RDH12"
    assert str(labeled_spans[1]) == "retinol"
    assert str(labeled_spans[-1]) == "androstanediol"
    assert converted_document.metadata["entity_ids"][0] == "17512723_T12"
    assert converted_document.metadata["entity_ids"][1] == "17512723_T3"
    assert converted_document.metadata["entity_ids"][-1] == "17512723_T1"

    binary_relations = list(converted_document.binary_relations)
    assert len(binary_relations) == 1
    assert binary_relations[0].label == "PRODUCT-OF"
    assert str(binary_relations[0].head) == "androstanediol"


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize(
    document: TextBasedDocument, tokenizer: PreTrainedTokenizer
) -> List[TokenBasedDocument]:
    if isinstance(document, TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions):
        result_document_type = TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
        partition_layer = "labeled_partitions"
    elif isinstance(document, TextDocumentWithLabeledSpansAndBinaryRelations):
        result_document_type = TokenDocumentWithLabeledSpansAndBinaryRelations
        partition_layer = None
    else:
        raise ValueError(f"Unsupported document type: {type(document)}")
    tokenized_documents = tokenize_document(
        document,
        tokenizer=tokenizer,
        return_overflowing_tokens=True,
        result_document_type=result_document_type,
        partition_layer=partition_layer,
        strict_span_conversion=True,
        verbose=True,
    )
    return tokenized_documents


def test_tokenize_document(converted_document, tokenizer):
    tokenized_docs = tokenize(converted_document, tokenizer=tokenizer)
    # we just ensure that we get at least one tokenized document
    assert tokenized_docs is not None
    assert len(tokenized_docs) > 0
    if isinstance(
        converted_document,
        TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
    ):
        assert len(tokenized_docs) == 2
        doc = tokenized_docs
        assert len(doc[0].tokens) == 32
        assert len(doc[1].tokens) == 132

        assert len(doc[0].labeled_spans) == 3
        assert len(doc[1].labeled_spans) == 10
        ent = doc[0].labeled_spans[0]
        assert ent.target[ent.start : ent.end] == ("rd", "##h", "##12")
        ent = doc[1].labeled_spans[0]
        assert ent.target[ent.start : ent.end] == ("re", "##tino", "##l")
        ent = doc[1].labeled_spans[-1]
        assert ent.target[ent.start : ent.end] == ("and", "##ros", "##tan", "##ed", "##iol")

        assert len(doc[0].binary_relations) == 0
        assert len(doc[1].binary_relations) == 1
        rel = doc[1].binary_relations[0]
        assert rel.label == "PRODUCT-OF"
        assert rel.head.target[rel.head.start : rel.head.end] == (
            "and",
            "##ros",
            "##tan",
            "##ed",
            "##iol",
        )

    elif isinstance(
        converted_document,
        TextDocumentWithLabeledSpansAndBinaryRelations,
    ):
        assert len(tokenized_docs) == 1
        doc = tokenized_docs[0]
        assert len(doc.tokens) == 162

        assert len(doc.labeled_spans) == 13
        ent = doc.labeled_spans[0]
        assert ent.target[ent.start : ent.end] == ("rd", "##h", "##12")
        ent = doc.labeled_spans[1]
        assert ent.target[ent.start : ent.end] == ("re", "##tino", "##l")
        ent = doc.labeled_spans[-1]
        assert ent.target[ent.start : ent.end] == ("and", "##ros", "##tan", "##ed", "##iol")

        assert len(doc.binary_relations) == 1
        rel = doc.binary_relations[0]
        assert rel.label == "PRODUCT-OF"
        assert rel.head.target[rel.head.start : rel.head.end] == (
            "and",
            "##ros",
            "##tan",
            "##ed",
            "##iol",
        )
    else:
        raise ValueError(f"Converted document has an unsupported type: {type(converted_document)}")


@pytest.mark.slow
def test_tokenize_documents_all(converted_pie_dataset, tokenizer):
    for split, docs in converted_pie_dataset.items():
        for doc in docs:
            # Note, that this is a list of documents, because the document may be split into chunks
            # if the input text is too long.
            tokenized_docs = tokenize(doc, tokenizer=tokenizer)
            # we just ensure that we get at least one tokenized document
            assert tokenized_docs is not None
            assert len(tokenized_docs) > 0
