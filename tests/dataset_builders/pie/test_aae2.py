from typing import List, Optional, Union

import pytest
from datasets import disable_caching
from pie_modules.document.processing import tokenize_document
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)
from transformers import AutoTokenizer, PreTrainedTokenizer

from dataset_builders.pie.aae2.aae2 import ArgumentAnnotatedEssaysV2
from pie_datasets import DatasetDict
from pie_datasets.builders.brat import BratDocument, BratDocumentWithMergedSpans
from tests.dataset_builders.common import (
    PIE_BASE_PATH,
    TestTokenDocumentWithLabeledSpansAndBinaryRelations,
    TestTokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

disable_caching()

DATASET_NAME = "aae2"
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
SPLIT_SIZES = {"test": 80, "train": 322}


@pytest.fixture(
    scope="module", params=[config.name for config in ArgumentAnnotatedEssaysV2.BUILDER_CONFIGS]
)
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
    result = dataset["train"][0]
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
        doc = dataset_of_text_documents_with_labeled_spans_and_binary_relations["train"][0]
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
def dataset_of_text_documents_with_labeled_spans_binary_relations_and_labeled_partitions(
    dataset, dataset_variant
) -> Optional[DatasetDict]:
    if dataset_variant == "default":
        with pytest.raises(ValueError) as excinfo:
            dataset.to_document_type(
                TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
            )
        assert (
            str(excinfo.value)
            == "No valid key (either subclass or superclass) was found for the document type "
            "'<class 'pytorch_ie.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions'>' "
            "in the document_converters of the dataset. Available keys: set(). Consider adding a respective "
            "converter to the dataset with dataset.register_document_converter(my_converter_method) where "
            "my_converter_method should accept <class 'pie_datasets.builders.brat.BratDocument'> as input and "
            "return '<class 'pytorch_ie.documents."
            "TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions'>'."
        )
        converted_dataset = None
    elif dataset_variant == "merge_fragmented_spans":
        converted_dataset = dataset.to_document_type(
            TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
        )
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")
    return converted_dataset


def test_dataset_of_text_documents_with_labeled_spans_binary_relations_and_labeled_partitions(
    dataset_of_text_documents_with_labeled_spans_binary_relations_and_labeled_partitions,
    dataset_of_text_documents_with_labeled_spans_and_binary_relations,
):
    if (
        dataset_of_text_documents_with_labeled_spans_binary_relations_and_labeled_partitions
        is not None
    ):
        # Check that the conversion is correct and the data makes sense
        # get a document to check
        doc_without_partitions = dataset_of_text_documents_with_labeled_spans_and_binary_relations[
            "train"
        ][0]
        doc_with_partitions = (
            dataset_of_text_documents_with_labeled_spans_binary_relations_and_labeled_partitions[
                "train"
            ][0]
        )
        assert isinstance(doc_without_partitions, TextDocumentWithLabeledSpansAndBinaryRelations)
        assert isinstance(
            doc_with_partitions, TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
        )

        # check the partitions with startswiths(), endswith(), and label
        partitions = doc_with_partitions.labeled_partitions
        assert len(partitions) == 10
        # Only check the first sentence in each partition, as it could be really long
        assert str(partitions[0]).startswith(
            "<Title>A Powell Optimization Approach for Example-Based Skinning in a Production Animation "
            "Environment</Title>\n"
        )
        assert str(partitions[0]).endswith(" Tian Feng Nanyang Technological University")
        assert partitions[0].label == "Title"
        assert str(partitions[1]).startswith(
            "<Abstract>We propose a layered framework for incorporating example-based skinning algorithms such as "
            "Pose Space Deformation or Shape-by-Example into an existing character animation system."
        )
        assert str(partitions[1]).endswith(
            "\n      <H2>Keywords: </H2>skinning, Powell optimization, computer animation"
        )
        assert partitions[1].label == "Abstract"
        assert str(partitions[2]).startswith(
            "<H1>1 Introduction</H1>\n      \n      With the help of modelling tools or capture devices, complicated "
            "3D character models are widely used in fields of entertainment, virtual reality, medicine etc."
        )
        assert str(partitions[2]).endswith(
            " Conclusion and some discussions of future work are presented in the last section.\n      1"
        )
        assert partitions[2].label == "H1"
        assert str(partitions[3]).startswith(
            "<H1>2 Related Work</H1>\n        Besides the geometric solutions mentioned in the previous section, "
            "physical modelling and animation is another field providing realistic character simulations."
        )
        assert str(partitions[3]).endswith(
            " while allowing any proprietary skinning operators to be incorporated."
        )
        assert partitions[3].label == "H1"
        assert str(partitions[4]).startswith(
            "<H1>3 Skeleton Sub-Space Deformation</H1>\n        6 5 4 3 2 1 0\n        \n          \n        \n        "
            "0 1 2 3 4 5 6 7 8 (a). (b)."
        )
        assert str(partitions[4]).endswith(
            " in order to interpolate the examples in the rest pose is a the right choice."
        )
        assert partitions[4].label == "H1"
        assert str(partitions[5]).startswith(
            "<H1>4 Inverse Operation</H1>\n        This section will describe the implementation of our inverse "
            "algorithm and why it is an improvement."
        )
        assert str(partitions[5]).endswith(
            "\n          Figure 4: A simple test case: two example poses with one vertex sculpted"
        )
        assert partitions[5].label == "H1"
        assert str(partitions[6]).startswith(
            "<H1>5 A Unified Framework for Inverse Skinning Model</H1>\n        The above discussions assume that "
            "the basic skinning algorithm is SSD, but in many circumstances, other deformation schemes will be "
            "adopted \n[ 9 ], [ 10 ], most of which have been implemented in most animation packages."
        )
        assert str(partitions[6]).endswith(
            "\n          Figure 8: toad: closeup of circled part from figure 9 . left: PSD; right: inverse PSD."
        )
        assert partitions[6].label == "H1"
        assert str(partitions[7]).startswith(
            "<H1>6 Conclusions and Discussions</H1>\n        Inverse skinning integrates SSD and shape interpolation "
            "more firmly than its forward rival."
        )
        assert str(partitions[7]).endswith(
            " It interoperates with their closed-source “Smooth Skinning” deformation.\n        8"
        )
        assert partitions[7].label == "H1"
        assert str(partitions[8]).startswith(
            "<H1>Acknowledgements</H1>\n        Authors would like to thank artists from EggStory Creative Production."
        )
        assert str(partitions[8]).endswith(
            " We also thank several of the reviewers for comments which improved this work"
        )
        assert partitions[8].label == "H1"
        assert str(partitions[9]).startswith(
            "<H1>References</H1>\n        \n          [1] J. P. Lewis, Matt Cordner, and Nickson Fong. "
            "Pose space deformation: a unified approach to shape interpolation and skeleton-driven deformation."
        )
        assert str(partitions[9]).endswith(
            "\n          Figure 10: human arm\n        \n        10\n      \n    \n  \n\n</Document>"
        )
        assert partitions[9].label == "H1"

        # check the entities
        assert doc_with_partitions.labeled_spans == doc_without_partitions.labeled_spans

        # check the relations
        assert doc_with_partitions.binary_relations == doc_without_partitions.binary_relations


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def tokenized_documents_with_labeled_spans_and_binary_relations(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations, tokenizer
) -> Optional[List[TestTokenDocumentWithLabeledSpansAndBinaryRelations]]:
    if dataset_of_text_documents_with_labeled_spans_and_binary_relations is None:
        return None

    # get a document to check
    doc = dataset_of_text_documents_with_labeled_spans_and_binary_relations["train"][0]
    # Note, that this is a list of documents, because the document may be split into chunks
    # if the input text is too long.
    tokenized_docs = tokenize_document(
        doc,
        tokenizer=tokenizer,
        return_overflowing_tokens=True,
        result_document_type=TestTokenDocumentWithLabeledSpansAndBinaryRelations,
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
                    result_document_type=TestTokenDocumentWithLabeledSpansAndBinaryRelations,
                    strict_span_conversion=False,
                    verbose=True,
                )
                # we just ensure that we get at least one tokenized document
                assert tokenized_docs is not None
                assert len(tokenized_docs) > 0


@pytest.fixture(scope="module")
def tokenized_documents_with_labeled_spans_binary_relations_and_labeled_partitions(
    dataset_of_text_documents_with_labeled_spans_binary_relations_and_labeled_partitions, tokenizer
) -> List[TestTokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions]:
    if (
        dataset_of_text_documents_with_labeled_spans_binary_relations_and_labeled_partitions
        is not None
    ):
        # get a document to check
        doc = dataset_of_text_documents_with_labeled_spans_binary_relations_and_labeled_partitions[
            "train"
        ][0]
        # Note, that this is a list of documents, because the document may be split into chunks
        # if the input text is too long.
        tokenized_docs = tokenize_document(
            doc,
            tokenizer=tokenizer,
            return_overflowing_tokens=True,
            result_document_type=TestTokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
            strict_span_conversion=False,
            verbose=True,
        )
        return tokenized_docs


def test_tokenized_documents_with_labeled_spans_binary_relations_and_labeled_partitions(
    tokenized_documents_with_labeled_spans_binary_relations_and_labeled_partitions,
    tokenized_documents_with_labeled_spans_and_binary_relations,
):
    if tokenized_documents_with_labeled_spans_binary_relations_and_labeled_partitions is not None:
        docs_with_partitions = (
            tokenized_documents_with_labeled_spans_binary_relations_and_labeled_partitions
        )
        docs_without_partitions = tokenized_documents_with_labeled_spans_and_binary_relations

        # check that the tokenization was fine
        assert len(docs_with_partitions) == 1
        doc_with_partitions = docs_with_partitions[0]
        doc_without_partitions = docs_without_partitions[0]
        assert len(doc_with_partitions.labeled_partitions) == 10
        assert doc_with_partitions.labeled_spans == doc_without_partitions.labeled_spans
        assert doc_with_partitions.binary_relations == doc_without_partitions.binary_relations
        assert doc_with_partitions.tokens == doc_without_partitions.tokens


def test_tokenized_documents_with_entities_relations_and_partitions_all(
    dataset_of_text_documents_with_labeled_spans_binary_relations_and_labeled_partitions, tokenizer
):
    if (
        dataset_of_text_documents_with_labeled_spans_binary_relations_and_labeled_partitions
        is not None
    ):
        for (
            split,
            docs,
        ) in (
            dataset_of_text_documents_with_labeled_spans_binary_relations_and_labeled_partitions.items()
        ):
            for doc in docs:
                # Note, that this is a list of documents, because the document may be split into chunks
                # if the input text is too long.
                tokenized_docs = tokenize_document(
                    doc,
                    tokenizer=tokenizer,
                    return_overflowing_tokens=True,
                    result_document_type=TestTokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
                    strict_span_conversion=False,
                    verbose=True,
                )
                # we just ensure that we get at least one tokenized document
                assert tokenized_docs is not None
                assert len(tokenized_docs) > 0


def test_document_converters(dataset_variant):
    builder = ArgumentAnnotatedEssaysV2(config_name=dataset_variant)
    document_converters = builder.document_converters

    if dataset_variant == "default":
        assert document_converters == {}
    elif dataset_variant == "merge_fragmented_spans":
        assert len(document_converters) == 2
        assert set(document_converters) == {
            TextDocumentWithLabeledSpansAndBinaryRelations,
            TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
        }
        assert all(callable(v) for k, v in document_converters.items())
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")
