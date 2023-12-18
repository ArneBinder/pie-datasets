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


def test_document(document):
    assert document.text.startswith("Should students be taught to compete or to cooperate?")
    assert document.text.endswith(
        "Consequently, no matter from the view of individual development or the relationship "
        "between competition and cooperation we can receive the same conclusion that a more "
        "cooperative attitudes towards life is more profitable in one's success."
    )


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

        entities = doc.labeled_spans
        assert len(entities) == 11
        # sort the entities by their start position and convert them to tuples
        sorted_entity_tuples = [
            (str(ent), ent.label) for ent in sorted(doc.labeled_spans, key=lambda ent: ent.start)
        ]
        assert sorted_entity_tuples[0] == (
            "we should attach more importance to cooperation during primary education",
            "MajorClaim",
        )
        assert sorted_entity_tuples[1] == (
            "through cooperation, children can learn about interpersonal skills which are significant in the future life "
            "of all students",
            "Claim",
        )
        assert sorted_entity_tuples[2] == (
            "What we acquired from team work is not only how to achieve the same goal with others but more importantly, "
            "how to get along with others",
            "Premise",
        )
        assert sorted_entity_tuples[3] == (
            "During the process of cooperation, children can learn about how to listen to opinions of others, how to "
            "communicate with others, how to think comprehensively, and even how to compromise with other team members "
            "when conflicts occurred",
            "Premise",
        )
        assert sorted_entity_tuples[4] == (
            "All of these skills help them to get on well with other people and will benefit them for the whole life",
            "Premise",
        )
        assert sorted_entity_tuples[5] == (
            "the significance of competition is that how to become more excellence to gain the victory",
            "Premise",
        )
        assert sorted_entity_tuples[6] == ("competition makes the society more effective", "Claim")
        assert sorted_entity_tuples[7] == (
            "when we consider about the question that how to win the game, we always find that we need the cooperation",
            "Premise",
        )
        assert sorted_entity_tuples[8] == (
            "Take Olympic games which is a form of competition for instance, it is hard to imagine how an athlete could "
            "win the game without the training of his or her coach, and the help of other professional staffs such as "
            "the people who take care of his diet, and those who are in charge of the medical care",
            "Premise",
        )
        assert sorted_entity_tuples[9] == (
            "without the cooperation, there would be no victory of competition",
            "Claim",
        )
        assert sorted_entity_tuples[10] == (
            "a more cooperative attitudes towards life is more profitable in one's success",
            "MajorClaim",
        )

        # check the relations
        # for conversion_method="connect_first"
        assert len(doc.binary_relations) == 10
        relation_tuples = [
            (str(rel.head), rel.label, str(rel.tail)) for rel in doc.binary_relations
        ]
        assert relation_tuples[0] == (
            "What we acquired from team work is not only how to achieve the same goal with others but more importantly, "
            "how to get along with others",
            "supports",
            "through cooperation, children can learn about interpersonal skills which are significant in the future life "
            "of all students",
        )
        assert relation_tuples[1] == (
            "During the process of cooperation, children can learn about how to listen to opinions of others, how to "
            "communicate with others, how to think comprehensively, and even how to compromise with other team members "
            "when conflicts occurred",
            "supports",
            "through cooperation, children can learn about interpersonal skills which are significant in the future life "
            "of all students",
        )
        assert relation_tuples[2] == (
            "All of these skills help them to get on well with other people and will benefit them for the whole life",
            "supports",
            "through cooperation, children can learn about interpersonal skills which are significant in the future life "
            "of all students",
        )
        assert relation_tuples[3] == (
            "Take Olympic games which is a form of competition for instance, it is hard to imagine how an athlete could "
            "win the game without the training of his or her coach, and the help of other professional staffs such as "
            "the people who take care of his diet, and those who are in charge of the medical care",
            "supports",
            "without the cooperation, there would be no victory of competition",
        )
        assert relation_tuples[4] == (
            "when we consider about the question that how to win the game, we always find that we need the cooperation",
            "supports",
            "without the cooperation, there would be no victory of competition",
        )
        assert relation_tuples[5] == (
            "the significance of competition is that how to become more excellence to gain the victory",
            "supports",
            "competition makes the society more effective",
        )
        assert relation_tuples[6] == (
            "through cooperation, children can learn about interpersonal skills which are significant in the future "
            "life of all students",
            "supports",
            "we should attach more importance to cooperation during primary education",
        )
        assert relation_tuples[7] == (
            "competition makes the society more effective",
            "attacks",
            "we should attach more importance to cooperation during primary education",
        )
        assert relation_tuples[8] == (
            "without the cooperation, there would be no victory of competition",
            "supports",
            "we should attach more importance to cooperation during primary education",
        )
        assert relation_tuples[9] == (
            "a more cooperative attitudes towards life is more profitable in one's success",
            "semantically_same",
            "we should attach more importance to cooperation during primary education",
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

        partitions = doc_with_partitions.labeled_partitions
        assert len(partitions) == 5
        assert [partition.label == "partition" for partition in partitions]
        assert str(partitions[0]) == "Should students be taught to compete or to cooperate?"
        assert (
            str(partitions[1])
            == "It is always said that competition can effectively promote the development of economy. In order to "
            "survive in the competition, companies continue to improve their products and service, and as a result, "
            "the whole society prospers. However, when we discuss the issue of competition or cooperation, what "
            "we are concerned about is not the whole society, but the development of an individual's whole life. "
            "From this point of view, I firmly believe that we should attach more importance to cooperation during "
            "primary education."
        )
        assert (
            str(partitions[2])
            == "First of all, through cooperation, children can learn about interpersonal skills which are "
            "significant in the future life of all students. What we acquired from team work is not only how to "
            "achieve the same goal with others but more importantly, how to get along with others. During the "
            "process of cooperation, children can learn about how to listen to opinions of others, how to "
            "communicate with others, how to think comprehensively, and even how to compromise with other team "
            "members when conflicts occurred. All of these skills help them to get on well with other people and "
            "will benefit them for the whole life."
        )
        assert (
            str(partitions[3])
            == "On the other hand, the significance of competition is that how to become more excellence to gain the "
            "victory. Hence it is always said that competition makes the society more effective. However, when we "
            "consider about the question that how to win the game, we always find that we need the cooperation. "
            "The greater our goal is, the more competition we need. Take Olympic games which is a form of "
            "competition for instance, it is hard to imagine how an athlete could win the game without the "
            "training of his or her coach, and the help of other professional staffs such as the people who take "
            "care of his diet, and those who are in charge of the medical care. The winner is the athlete but the "
            "success belongs to the whole team. Therefore without the cooperation, there would be no victory of "
            "competition."
        )
        assert (
            str(partitions[4])
            == "Consequently, no matter from the view of individual development or the relationship between "
            "competition and cooperation we can receive the same conclusion that a more cooperative attitudes "
            "towards life is more profitable in one's success."
        )

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
        strict_span_conversion=True,
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
        assert len(doc.labeled_spans) == 11
        assert len(doc.binary_relations) == 10
        assert len(doc.tokens) == 427
        # Check the first ten tokens
        assert doc.tokens[:10] == (
            "[CLS]",
            "should",
            "students",
            "be",
            "taught",
            "to",
            "compete",
            "or",
            "to",
            "cooperate",
        )
        # sort the entities by their start position
        sorted_entities = sorted(doc.labeled_spans, key=lambda ent: ent.start)
        assert (
            str(sorted_entities[0])
            == "('we', 'should', 'attach', 'more', 'importance', 'to', 'cooperation', 'during', 'primary', 'education')"
        )
        assert (
            str(sorted_entities[1])
            == "('through', 'cooperation', ',', 'children', 'can', 'learn', 'about', 'inter', '##personal', 'skills', "
            "'which', 'are', 'significant', 'in', 'the', 'future', 'life', 'of', 'all', 'students')"
        )
        assert (
            str(sorted_entities[2])
            == "('what', 'we', 'acquired', 'from', 'team', 'work', 'is', 'not', 'only', 'how', 'to', 'achieve', 'the', "
            "'same', 'goal', 'with', 'others', 'but', 'more', 'importantly', ',', 'how', 'to', 'get', 'along', "
            "'with', 'others')"
        )
        assert (
            str(sorted_entities[3])
            == "('during', 'the', 'process', 'of', 'cooperation', ',', 'children', 'can', 'learn', 'about', 'how', 'to', "
            "'listen', 'to', 'opinions', 'of', 'others', ',', 'how', 'to', 'communicate', 'with', 'others', ',', "
            "'how', 'to', 'think', 'comprehensive', '##ly', ',', 'and', 'even', 'how', 'to', 'compromise', 'with', "
            "'other', 'team', 'members', 'when', 'conflicts', 'occurred')"
        )
        assert (
            str(sorted_entities[4])
            == "('all', 'of', 'these', 'skills', 'help', 'them', 'to', 'get', 'on', 'well', 'with', 'other', 'people', "
            "'and', 'will', 'benefit', 'them', 'for', 'the', 'whole', 'life')"
        )
        assert (
            str(sorted_entities[5])
            == "('the', 'significance', 'of', 'competition', 'is', 'that', 'how', 'to', 'become', 'more', 'excellence', "
            "'to', 'gain', 'the', 'victory')"
        )
        assert (
            str(sorted_entities[6])
            == "('competition', 'makes', 'the', 'society', 'more', 'effective')"
        )
        assert (
            str(sorted_entities[7])
            == "('when', 'we', 'consider', 'about', 'the', 'question', 'that', 'how', 'to', 'win', 'the', 'game', ',', "
            "'we', 'always', 'find', 'that', 'we', 'need', 'the', 'cooperation')"
        )
        assert (
            str(sorted_entities[8])
            == "('take', 'olympic', 'games', 'which', 'is', 'a', 'form', 'of', 'competition', 'for', 'instance', ',', "
            "'it', 'is', 'hard', 'to', 'imagine', 'how', 'an', 'athlete', 'could', 'win', 'the', 'game', 'without', "
            "'the', 'training', 'of', 'his', 'or', 'her', 'coach', ',', 'and', 'the', 'help', 'of', 'other', "
            "'professional', 'staff', '##s', 'such', 'as', 'the', 'people', 'who', 'take', 'care', 'of', 'his', "
            "'diet', ',', 'and', 'those', 'who', 'are', 'in', 'charge', 'of', 'the', 'medical', 'care')"
        )
        assert (
            str(sorted_entities[9])
            == "('without', 'the', 'cooperation', ',', 'there', 'would', 'be', 'no', 'victory', 'of', 'competition')"
        )
        assert (
            str(sorted_entities[10])
            == "('a', 'more', 'cooperative', 'attitudes', 'towards', 'life', 'is', 'more', 'profitable', "
            "'in', 'one', \"'\", 's', 'success')"
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
                    return_overflowing_tokens=True,
                    result_document_type=TestTokenDocumentWithLabeledSpansAndBinaryRelations,
                    strict_span_conversion=True,
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
            partition_layer="labeled_partitions",
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

        # check that the tokenization was fine
        assert len(docs_with_partitions) == 5
        doc_with_partitions = docs_with_partitions[0]
        assert len(doc_with_partitions.labeled_partitions) == 1
        assert len(doc_with_partitions.labeled_spans) == 0
        assert len(doc_with_partitions.binary_relations) == 0
        assert doc_with_partitions.tokens == (
            "[CLS]",
            "should",
            "students",
            "be",
            "taught",
            "to",
            "compete",
            "or",
            "to",
            "cooperate",
            "?",
            "[SEP]",
        )


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
                    partition_layer="labeled_partitions",
                    return_overflowing_tokens=True,
                    result_document_type=TestTokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
                    strict_span_conversion=False,
                    verbose=True,
                )
                # we just ensure that we get at least one tokenized document
                assert tokenized_docs is not None
                assert len(tokenized_docs) > 0
                for tokenized_doc in tokenized_docs:
                    assert tokenized_doc.labeled_partitions is not None
                    # We use the partitions to partition the input, so each tokenized
                    # document should have exactly one partition annotation.
                    assert len(tokenized_doc.labeled_partitions) == 1


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
