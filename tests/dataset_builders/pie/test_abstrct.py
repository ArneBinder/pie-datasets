from typing import List, Optional, Union

import pytest
from datasets import disable_caching
from pie_models.document.processing import tokenize_document
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndBinaryRelations
from transformers import AutoTokenizer, PreTrainedTokenizer

from dataset_builders.pie.abstrct.abstrct import AbstRCT
from pie_datasets import DatasetDict
from pie_datasets.builders.brat import BratDocument, BratDocumentWithMergedSpans
from pie_datasets.document.types import TokenDocumentWithLabeledSpansAndBinaryRelations
from tests.dataset_builders.common import PIE_BASE_PATH

disable_caching()

DATASET_NAME = "abstrct"
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
SPLIT_SIZES = {
    "glaucoma_test": 100,
    "mixed_test": 100,
    "neoplasm_dev": 50,
    "neoplasm_test": 100,
    "neoplasm_train": 350,
}
SPLIT = "neoplasm_train"


@pytest.fixture(scope="module", params=[config.name for config in AbstRCT.BUILDER_CONFIGS])
def dataset_variant(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def dataset(dataset_variant) -> DatasetDict:
    return DatasetDict.load_dataset(str(PIE_DATASET_PATH), name=dataset_variant)


def test_dataset(dataset):
    assert dataset is not None
    assert {name: len(ds) for name, ds in dataset.items()} == SPLIT_SIZES


@pytest.fixture(scope="module")
def document(dataset, dataset_variant) -> Union[BratDocument, BratDocumentWithMergedSpans]:
    result = dataset[SPLIT][0]
    if dataset_variant == "default":
        assert isinstance(result, BratDocument)
    elif dataset_variant == "merge_fragmented_spans":
        assert isinstance(result, BratDocumentWithMergedSpans)
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")
    return result


def test_document(document, dataset_variant):
    assert document.text.startswith("Should students be taught to compete or to cooperate?")
    if dataset_variant == "default":
        # TODO
        raise NotImplementedError()
    elif dataset_variant == "merge_fragmented_spans":
        # TODO
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")


@pytest.fixture(scope="module")
def dataset_of_text_documents_with_labeled_spans_and_binary_relations(
    dataset, dataset_variant
) -> Optional[DatasetDict]:
    if dataset_variant == "default":
        with pytest.raises(ValueError) as excinfo:
            dataset.to_document_type(TextDocumentWithLabeledSpansAndBinaryRelations)
        assert (
            str(excinfo.value)
            == "No valid key (either subclass or superclass) was found for the document type "
            "'<class 'pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations'>' in the "
            "document_converters of the dataset. Available keys: set(). Consider adding a respective "
            "converter to the dataset with dataset.register_document_converter(my_converter_method) "
            "where my_converter_method should accept <class 'pie_datasets.builders.brat.BratDocument'> "
            "as input and return '<class 'pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations'>'."
        )
        converted_dataset = None
    elif dataset_variant == "merge_fragmented_spans":
        converted_dataset = dataset.to_document_type(
            TextDocumentWithLabeledSpansAndBinaryRelations
        )
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")
    return converted_dataset


def test_dataset_of_text_documents_with_labeled_spans_and_binary_relations(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations,
):
    if dataset_of_text_documents_with_labeled_spans_and_binary_relations is not None:
        # Check that the conversion is correct and the data makes sense
        # get a document to check
        doc = dataset_of_text_documents_with_labeled_spans_and_binary_relations[SPLIT][0]
        assert isinstance(doc, TextDocumentWithLabeledSpansAndBinaryRelations)
        # check the entities
        assert len(doc.labeled_spans) == 183
        # sort the entities by their start position and convert them to tuples
        # check the first ten entities after sorted
        sorted_entity_tuples = [
            (str(ent), ent.label)
            for ent in sorted(doc.labeled_spans, key=lambda ent: ent.start)[:10]
        ]
        # Checking the first ten entities
        assert sorted_entity_tuples[0] == (
            "complicated 3D character models are widely used in fields of entertainment, virtual reality, medicine etc",
            "background_claim",
        )
        assert sorted_entity_tuples[1] == (
            "The range of breathtaking realistic 3D models is only limited by the creativity of artists and resolution "
            "of devices",
            "background_claim",
        )
        assert sorted_entity_tuples[2] == (
            "Driving 3D models in a natural and believable manner is not trivial",
            "background_claim",
        )
        assert sorted_entity_tuples[3] == ("the model is very detailed", "data")
        assert sorted_entity_tuples[4] == (
            "playback of animation becomes quite heavy and time consuming",
            "data",
        )
        assert sorted_entity_tuples[5] == ("a frame goes wrong", "data")
        assert sorted_entity_tuples[6] == (
            "a production cannot afford major revisions",
            "background_claim",
        )
        assert sorted_entity_tuples[7] == ("resculpting models", "data")
        assert sorted_entity_tuples[8] == ("re-rigging skeletons", "data")
        assert sorted_entity_tuples[9] == (
            "providing a flexible and efficient solution to animation remains an open problem",
            "own_claim",
        )

        # check the relations
        assert len(doc.binary_relations) == 116
        # check the first ten relations
        relation_tuples = [
            (str(rel.head), rel.label, str(rel.tail)) for rel in doc.binary_relations[:10]
        ]
        assert relation_tuples[0] == (
            "a production cannot afford major revisions",
            "supports",
            "providing a flexible and efficient solution to animation remains an open problem",
        )
        assert relation_tuples[1] == (
            "its ease of implementation",
            "supports",
            "SSD is widely used in games, virtual reality and other realtime applications",
        )
        assert relation_tuples[2] == (
            "low cost of computing",
            "supports",
            "SSD is widely used in games, virtual reality and other realtime applications",
        )
        assert relation_tuples[3] == (
            "editing in the rest pose will influence most other poses",
            "supports",
            "This approach is not commonly applied",
        )
        assert relation_tuples[4] == (
            "This approach is not commonly applied",
            "contradicts",
            "artists will edit the geometry of characters in the rest pose to fine-tune animations",
        )
        assert relation_tuples[5] == (
            "the animator specifies the PSD examples after the SSD has been performed",
            "contradicts",
            "the examples are best interpolated in the rest pose, before the SSD has been applied",
        )
        assert relation_tuples[6] == (
            "PSD may be used as a compensation to the underlying SSD",
            "contradicts",
            "the examples are best interpolated in the rest pose, before the SSD has been applied",
        )
        assert relation_tuples[7] == (
            "the examples are best interpolated in the rest pose, before the SSD has been applied",
            "supports",
            "the action of the SSD and any other deformations must be “inverted” in order to push the example "
            "compensation before these operations",
        )
        assert relation_tuples[8] == (
            "this inverse strategy has a better performance than the same framework without it",
            "semantically_same",
            "this approach will improve the quality of deformation",
        )
        assert relation_tuples[9] == (
            "the high cost of computing",
            "supports",
            "they are seldom applied to interactive applications",
        )


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def tokenized_documents_with_labeled_spans_and_binary_relations(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations, tokenizer
) -> Optional[List[TokenDocumentWithLabeledSpansAndBinaryRelations]]:
    if dataset_of_text_documents_with_labeled_spans_and_binary_relations is None:
        return None

    # get a document to check
    doc = dataset_of_text_documents_with_labeled_spans_and_binary_relations[SPLIT][0]
    # Note, that this is a list of documents, because the document may be split into chunks
    # if the input text is too long.
    tokenized_docs = tokenize_document(
        doc,
        tokenizer=tokenizer,
        return_overflowing_tokens=True,
        result_document_type=TokenDocumentWithLabeledSpansAndBinaryRelations,
        strict_span_conversion=False,
        verbose=True,
    )
    return tokenized_docs


def test_tokenized_documents_with_labeled_spans_and_binary_relations(
    tokenized_documents_with_labeled_spans_and_binary_relations,
):
    if tokenized_documents_with_labeled_spans_and_binary_relations is not None:
        docs = tokenized_documents_with_labeled_spans_and_binary_relations
        # check that the tokenization was fine
        assert len(docs) == 1
        doc = docs[0]
        assert len(doc.labeled_spans) == 183
        assert len(doc.tokens) == 7689
        # Check the first ten tokens
        assert doc.tokens[:10] == ("[CLS]", "<", "?", "xml", "version", "=", '"', "1", ".", "0")
        # Check the first ten tokenized entities after sorted by their start position
        sorted_entities = sorted(doc.labeled_spans, key=lambda ent: ent.start)
        assert (
            str(sorted_entities[0])
            == "('complicated', '3d', 'character', 'models', 'are', 'widely', 'used', 'in', 'fields', 'of', "
            "'entertainment', ',', 'virtual', 'reality', ',', 'medicine', 'etc')"
        )
        assert (
            str(sorted_entities[1])
            == "('the', 'range', 'of', 'breath', '##taking', 'realistic', '3d', 'models', 'is', 'only', 'limited', "
            "'by', 'the', 'creativity', 'of', 'artists', 'and', 'resolution', 'of', 'devices')"
        )
        assert (
            str(sorted_entities[2])
            == "('driving', '3d', 'models', 'in', 'a', 'natural', 'and', 'bel', '##ie', '##vable', 'manner', 'is', "
            "'not', 'trivial')"
        )
        assert str(sorted_entities[3]) == "('the', 'model', 'is', 'very', 'detailed')"
        assert (
            str(sorted_entities[4])
            == "('playback', 'of', 'animation', 'becomes', 'quite', 'heavy', 'and', 'time', 'consuming')"
        )
        assert str(sorted_entities[5]) == "('a', 'frame', 'goes', 'wrong')"
        assert (
            str(sorted_entities[6])
            == "('a', 'production', 'cannot', 'afford', 'major', 'revisions')"
        )
        assert str(sorted_entities[7]) == "('res', '##cu', '##lp', '##ting', 'models')"
        assert str(sorted_entities[8]) == "('re', '-', 'rig', '##ging', 'skeletons')"
        assert (
            str(sorted_entities[9])
            == "('providing', 'a', 'flexible', 'and', 'efficient', 'solution', 'to', 'animation', 'remains', 'an', "
            "'open', 'problem')"
        )


def test_tokenized_documents_with_entities_and_relations_all(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations, tokenizer, dataset_variant
):
    if dataset_of_text_documents_with_labeled_spans_and_binary_relations is not None:
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
                    result_document_type=TokenDocumentWithLabeledSpansAndBinaryRelations,
                    strict_span_conversion=False,
                    verbose=True,
                )
                # we just ensure that we get at least one tokenized document
                assert tokenized_docs is not None
                assert len(tokenized_docs) > 0


def test_document_converters(dataset_variant):
    builder = AbstRCT(config_name=dataset_variant)
    document_converters = builder.document_converters

    if dataset_variant == "default":
        assert document_converters == {}
    elif dataset_variant == "merge_fragmented_spans":
        assert len(document_converters) == 1
        assert set(document_converters) == {
            TextDocumentWithLabeledSpansAndBinaryRelations,
        }
        assert all(callable(v) for k, v in document_converters.items())
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")
