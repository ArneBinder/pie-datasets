from typing import List

import pytest
from datasets import disable_caching
from pie_core import Document
from pie_modules.annotations import BinaryRelation, LabeledSpan
from pie_modules.document.processing import tokenize_document
from pie_modules.documents import (
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)
from transformers import AutoTokenizer, PreTrainedTokenizer

from dataset_builders.pie.aae2.aae2 import (
    ArgumentAnnotatedEssaysV2,
    convert_aae2_claim_attributions_to_relations,
    remove_cross_partition_relations,
)
from pie_datasets import DatasetDict
from pie_datasets.builders.brat import BratAttribute, BratDocumentWithMergedSpans
from tests.dataset_builders.common import (
    PIE_BASE_PATH,
    TestTokenDocumentWithLabeledSpansAndBinaryRelations,
    TestTokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

disable_caching()

DATASET_NAME = "aae2"
BUILDER_CLASS = ArgumentAnnotatedEssaysV2
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
SPLIT_SIZES = {"test": 80, "train": 322}


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
    result = dataset["train"][0]
    # we can not assert the real document type because it may come from a dataset loading script
    # downloaded to a temporary directory and thus have a different type object, although it is
    # semantically the same
    assert isinstance(result, Document)
    return result


def test_document(document, dataset_variant):
    assert document is not None
    assert document.id == "essay001"

    # check spans
    assert len(document.spans) == 11
    span_texts_labels_tuples = [(str(span), span.label) for span in document.spans]
    assert span_texts_labels_tuples[0] == (
        "we should attach more importance to cooperation during primary education",
        "MajorClaim",
    )
    assert span_texts_labels_tuples[1] == (
        "a more cooperative attitudes towards life is more profitable in one's success",
        "MajorClaim",
    )
    assert span_texts_labels_tuples[2] == (
        "through cooperation, children can learn about interpersonal skills which are significant in the future life "
        "of all students",
        "Claim",
    )
    assert span_texts_labels_tuples[3] == (
        "What we acquired from team work is not only how to achieve the same goal with others but more importantly, "
        "how to get along with others",
        "Premise",
    )
    assert span_texts_labels_tuples[4] == (
        "During the process of cooperation, children can learn about how to listen to opinions of others, how to "
        "communicate with others, how to think comprehensively, and even how to compromise with other team members "
        "when conflicts occurred",
        "Premise",
    )
    assert span_texts_labels_tuples[5] == (
        "All of these skills help them to get on well with other people and will benefit them for the whole life",
        "Premise",
    )
    assert span_texts_labels_tuples[6] == ("competition makes the society more effective", "Claim")
    assert span_texts_labels_tuples[7] == (
        "the significance of competition is that how to become more excellence to gain the victory",
        "Premise",
    )
    assert span_texts_labels_tuples[8] == (
        "when we consider about the question that how to win the game, we always find that we need the cooperation",
        "Premise",
    )
    assert span_texts_labels_tuples[9] == (
        "Take Olympic games which is a form of competition for instance, it is hard to imagine how an athlete could "
        "win the game without the training of his or her coach, and the help of other professional staffs such as "
        "the people who take care of his diet, and those who are in charge of the medical care",
        "Premise",
    )
    assert span_texts_labels_tuples[10] == (
        "without the cooperation, there would be no victory of competition",
        "Claim",
    )

    # check relations
    assert len(document.relations) == 6
    assert document.relations[0].label == "supports"
    assert document.relations[0].head == document.spans[3]
    assert document.relations[0].tail == document.spans[2]
    assert document.relations[1].label == "supports"
    assert document.relations[1].head == document.spans[4]
    assert document.relations[1].tail == document.spans[2]
    assert document.relations[2].label == "supports"
    assert document.relations[2].head == document.spans[5]
    assert document.relations[2].tail == document.spans[2]
    assert document.relations[3].label == "supports"
    assert document.relations[3].head == document.spans[9]
    assert document.relations[3].tail == document.spans[10]
    assert document.relations[4].label == "supports"
    assert document.relations[4].head == document.spans[8]
    assert document.relations[4].tail == document.spans[10]
    assert document.relations[5].label == "supports"
    assert document.relations[5].head == document.spans[7]
    assert document.relations[5].tail == document.spans[6]


@pytest.fixture(scope="module")
def dataset_of_text_documents_with_labeled_spans_and_binary_relations(
    dataset, dataset_variant
) -> DatasetDict:
    if dataset_variant == "default":
        converted_dataset = dataset.to_document_type(
            TextDocumentWithLabeledSpansAndBinaryRelations
        )
    elif dataset_variant == "paragraphs":
        converted_dataset = dataset.to_document_type(
            TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
        ).cast_document_type(TextDocumentWithLabeledSpansAndBinaryRelations)
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")
    return converted_dataset


@pytest.mark.parametrize("method", ["connect_first", "connect_all"])
def test_convert_aae2_claim_attributions_to_relations(method):
    # create sample document for testing
    sample_doc = BratDocumentWithMergedSpans(
        text="This is an example claim. This is the first major claim. "
        "This is the second major claim."
    )
    claim = LabeledSpan(start=0, end=25, label="Claim")
    first_majorclaim = LabeledSpan(start=26, end=56, label="MajorClaim")
    second_majorclaim = LabeledSpan(start=57, end=88, label="MajorClaim")
    sample_doc.spans.extend([claim, first_majorclaim, second_majorclaim])
    # sanity check (works only after labeled spans were added to the document)
    assert str(claim) == "This is an example claim."
    assert str(first_majorclaim) == "This is the first major claim."
    assert str(second_majorclaim) == "This is the second major claim."
    # create claim attribute
    claim_attribute = BratAttribute(annotation=claim, label="Stance", value="For")
    sample_doc.attributes.append(claim_attribute)

    # check results
    converted_doc = convert_aae2_claim_attributions_to_relations(sample_doc, method)
    converted_binary_relations_tuples = [
        (str(relation.head), relation.label, str(relation.tail))
        for relation in converted_doc.binary_relations
    ]
    assert len(converted_binary_relations_tuples) == 2
    assert converted_binary_relations_tuples[0] == (
        "This is an example claim.",
        "supports",
        "This is the first major claim.",
    )
    if method == "connect_first":
        assert converted_binary_relations_tuples[1] == (
            "This is the second major claim.",
            "semantically_same",
            "This is the first major claim.",
        )
    elif method == "connect_all":
        assert converted_binary_relations_tuples[1] == (
            "This is an example claim.",
            "supports",
            "This is the second major claim.",
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def test_remove_cross_partition_relations():
    adu_texts = [
        "This is a test document.",
        "It has two partitions.",
        "This is another paragraph.",
        "Believe me.",
        "That's why this is a test document.",
    ]
    adu_labels = ["MajorClaim", "Claim", "Claim", "Premise", "MajorClaim"]
    paragraphs = [" ".join(adu_texts[:2]), " ".join(adu_texts[2:])]
    document = TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(
        id="test",
        text=" ".join(adu_texts),
    )
    document.labeled_partitions.extend(
        [
            LabeledSpan(start=0, end=len(paragraphs[0]), label="paragraph"),
            LabeledSpan(
                start=len(paragraphs[0]) + 1,
                end=len(paragraphs[0]) + 1 + len(paragraphs[1]),
                label="paragraph",
            ),
        ]
    )
    assert document.labeled_partitions.resolve() == [
        ("paragraph", paragraphs[0]),
        ("paragraph", paragraphs[1]),
    ]
    start = 0
    for sentence, adu_label in zip(adu_texts, adu_labels):
        document.labeled_spans.append(
            LabeledSpan(start=start, end=start + len(sentence), label=adu_label)
        )
        start += len(sentence) + 1
    assert document.labeled_spans.resolve() == [
        ("MajorClaim", "This is a test document."),
        ("Claim", "It has two partitions."),
        ("Claim", "This is another paragraph."),
        ("Premise", "Believe me."),
        ("MajorClaim", "That's why this is a test document."),
    ]
    document.binary_relations.extend(
        [
            # in-paragraph support
            BinaryRelation(
                head=document.labeled_spans[1], tail=document.labeled_spans[0], label="supports"
            ),
            # cross-paragraph support
            BinaryRelation(
                head=document.labeled_spans[1], tail=document.labeled_spans[4], label="supports"
            ),
            # in paragraph support
            BinaryRelation(
                head=document.labeled_spans[2], tail=document.labeled_spans[4], label="supports"
            ),
            # in-paragraph support
            BinaryRelation(
                head=document.labeled_spans[3], tail=document.labeled_spans[2], label="supports"
            ),
            # cross-paragraph support
            BinaryRelation(
                head=document.labeled_spans[2], tail=document.labeled_spans[0], label="supports"
            ),
        ]
    )
    assert document.binary_relations.resolve() == [
        # in-paragraph support
        (
            "supports",
            (("Claim", "It has two partitions."), ("MajorClaim", "This is a test document.")),
        ),
        # cross-paragraph support
        (
            "supports",
            (
                ("Claim", "It has two partitions."),
                ("MajorClaim", "That's why this is a test document."),
            ),
        ),
        # in-paragraph support
        (
            "supports",
            (
                ("Claim", "This is another paragraph."),
                ("MajorClaim", "That's why this is a test document."),
            ),
        ),
        # in-paragraph support
        ("supports", (("Premise", "Believe me."), ("Claim", "This is another paragraph."))),
        # cross-paragraph support
        (
            "supports",
            (("Claim", "This is another paragraph."), ("MajorClaim", "This is a test document.")),
        ),
    ]

    result = remove_cross_partition_relations(document)
    assert result != document
    assert result.labeled_partitions.resolve() == document.labeled_partitions.resolve()
    assert result.labeled_spans.resolve() == document.labeled_spans.resolve()
    assert result.binary_relations.resolve() == [
        resolved_rel
        for idx, resolved_rel in enumerate(document.binary_relations.resolve())
        if idx not in [1, 4]
    ]


def test_dataset_of_text_documents_with_labeled_spans_and_binary_relations(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations, dataset_variant
):
    # Check that the conversion is correct and the data makes sense
    # get a document to check
    doc = dataset_of_text_documents_with_labeled_spans_and_binary_relations["train"][0]
    assert isinstance(doc, TextDocumentWithLabeledSpansAndBinaryRelations)

    # check the entities
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
    if dataset_variant == "paragraphs":
        assert doc.binary_relations.resolve() == [
            (
                "supports",
                (
                    (
                        "Premise",
                        "What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others",
                    ),
                    (
                        "Claim",
                        "through cooperation, children can learn about interpersonal skills which are significant in the future life of all students",
                    ),
                ),
            ),
            (
                "supports",
                (
                    (
                        "Premise",
                        "During the process of cooperation, children can learn about how to listen to opinions of others, how to communicate with others, how to think comprehensively, and even how to compromise with other team members when conflicts occurred",
                    ),
                    (
                        "Claim",
                        "through cooperation, children can learn about interpersonal skills which are significant in the future life of all students",
                    ),
                ),
            ),
            (
                "supports",
                (
                    (
                        "Premise",
                        "All of these skills help them to get on well with other people and will benefit them for the whole life",
                    ),
                    (
                        "Claim",
                        "through cooperation, children can learn about interpersonal skills which are significant in the future life of all students",
                    ),
                ),
            ),
            (
                "supports",
                (
                    (
                        "Premise",
                        "Take Olympic games which is a form of competition for instance, it is hard to imagine how an athlete could win the game without the training of his or her coach, and the help of other professional staffs such as the people who take care of his diet, and those who are in charge of the medical care",
                    ),
                    ("Claim", "without the cooperation, there would be no victory of competition"),
                ),
            ),
            (
                "supports",
                (
                    (
                        "Premise",
                        "when we consider about the question that how to win the game, we always find that we need the cooperation",
                    ),
                    ("Claim", "without the cooperation, there would be no victory of competition"),
                ),
            ),
            (
                "supports",
                (
                    (
                        "Premise",
                        "the significance of competition is that how to become more excellence to gain the victory",
                    ),
                    ("Claim", "competition makes the society more effective"),
                ),
            ),
        ]
    else:
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
) -> DatasetDict:
    if dataset_variant == "default":
        converted_dataset = dataset.to_document_type(
            TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
        )
    elif dataset_variant == "paragraphs":
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
    assert all([partition.label == "paragraph" for partition in partitions])
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
) -> List[TestTokenDocumentWithLabeledSpansAndBinaryRelations]:
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
    tokenized_documents_with_labeled_spans_and_binary_relations, dataset_variant
):
    docs = tokenized_documents_with_labeled_spans_and_binary_relations
    # check that the tokenization was fine
    assert len(docs) == 1
    doc = docs[0]
    assert len(doc.labeled_spans) == 11
    if dataset_variant == "paragraphs":
        assert len(doc.binary_relations) == 6
    else:
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


@pytest.fixture(scope="module")
def tokenized_documents_with_labeled_spans_binary_relations_and_labeled_partitions(
    dataset_of_text_documents_with_labeled_spans_binary_relations_and_labeled_partitions, tokenizer
) -> List[TestTokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions]:
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
        # We set strict_span_conversion to False
        # because we added relations between Claims and MajorClaims from different paragraphs
        # and those relations are lost in annotations
        strict_span_conversion=False,
        verbose=True,
    )
    return tokenized_docs


def test_tokenized_documents_with_labeled_spans_binary_relations_and_labeled_partitions(
    tokenized_documents_with_labeled_spans_binary_relations_and_labeled_partitions,
    tokenized_documents_with_labeled_spans_and_binary_relations,
):
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
                # We set strict_span_conversion to False
                # because we added relations between Claims and MajorClaims from different paragraphs
                # and those relations are lost in annotations
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
    builder = BUILDER_CLASS(config_name=dataset_variant)
    document_converters = builder.document_converters

    if dataset_variant == "default":
        assert len(document_converters) == 2
        assert set(document_converters) == {
            TextDocumentWithLabeledSpansAndBinaryRelations,
            TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
        }
        assert all(callable(v) for k, v in document_converters.items())
    elif dataset_variant == "paragraphs":
        assert len(document_converters) == 1
        assert set(document_converters) == {
            TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
        }
        assert all(callable(v) for k, v in document_converters.items())
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")
