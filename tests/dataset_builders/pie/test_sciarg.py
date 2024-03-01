from collections import Counter
from typing import Optional, Type

import datasets
import pytest
from pie_modules.document.processing import tokenize_document
from pie_modules.documents import (
    TextDocumentWithLabeledMultiSpansAndBinaryRelations,
    TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)
from pytorch_ie.core import Document
from pytorch_ie.documents import TextDocumentWithLabeledPartitions
from transformers import AutoTokenizer, PreTrainedTokenizer

from dataset_builders.pie.sciarg.sciarg import SciArg
from pie_datasets import DatasetDict
from pie_datasets.builders.brat import BratDocument, BratDocumentWithMergedSpans
from tests.dataset_builders.common import (
    PIE_BASE_PATH,
    PIE_DS_FIXTURE_DATA_PATH,
    TestTokenDocumentWithLabeledMultiSpansAndBinaryRelations,
    TestTokenDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions,
    TestTokenDocumentWithLabeledPartitions,
    TestTokenDocumentWithLabeledSpansAndBinaryRelations,
    TestTokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
    resolve_annotations,
)

datasets.disable_caching()

TEST_FULL_DATASET = True

DATASET_NAME = "sciarg"
BUILDER_CLASS = SciArg
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
DATA_DIR = PIE_DS_FIXTURE_DATA_PATH / DATASET_NAME
SPLIT_SIZES = {"train": 40 if TEST_FULL_DATASET else 3}


@pytest.fixture(scope="module", params=[config.name for config in BUILDER_CLASS.BUILDER_CONFIGS])
def dataset_variant(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def hf_dataset(dataset_variant) -> datasets.DatasetDict:
    kwargs = dict(BUILDER_CLASS.BASE_BUILDER_KWARGS_DICT[dataset_variant])
    if not TEST_FULL_DATASET:
        kwargs["data_dir"] = str(DATA_DIR)
    result = datasets.load_dataset(BUILDER_CLASS.BASE_DATASET_PATH, name=dataset_variant, **kwargs)
    return result


def test_hf_dataset(hf_dataset):
    assert hf_dataset is not None
    split_sizes = {name: len(ds) for name, ds in hf_dataset.items()}
    assert split_sizes == SPLIT_SIZES


@pytest.fixture(scope="module")
def builder(dataset_variant) -> SciArg:
    return SciArg(config_name=dataset_variant)


def test_builder(builder, dataset_variant):
    assert builder is not None
    assert builder.config_id == dataset_variant
    assert builder.dataset_name == DATASET_NAME


@pytest.fixture(scope="module")
def dataset(dataset_variant) -> DatasetDict:
    if TEST_FULL_DATASET:
        base_dataset_kwargs = None
    else:
        base_dataset_kwargs = {"data_dir": str(PIE_DS_FIXTURE_DATA_PATH / DATASET_NAME)}
    return DatasetDict.load_dataset(
        str(PIE_DATASET_PATH), name=dataset_variant, base_dataset_kwargs=base_dataset_kwargs
    )


def test_dataset(dataset):
    assert dataset is not None
    assert {name: len(ds) for name, ds in dataset.items()} == SPLIT_SIZES


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
    assert document.text.startswith(
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<Document xmlns:gate="http://www.gate.ac.uk" '
        'name="A01_S01_A_Powell_Optimization_Approach__for_Example-Based_Skinning_CITATION_PURPOSE_M_v1.xml">'
    )
    if dataset_variant == "default":
        assert isinstance(document, BratDocumentWithMergedSpans)
        assert len(document.spans) == 183
        counter = Counter([i.label for i in document.relations])
        assert dict(counter) == {
            "supports": 93,
            "semantically_same": 9,
            "contradicts": 8,
            "parts_of_same": 6,
        }
    elif dataset_variant == "resolve_parts_of_same":
        assert isinstance(document, BratDocument)
        assert len(document.spans) == 177
        counter = Counter([i.label for i in document.relations])
        assert dict(counter) == {"supports": 93, "semantically_same": 9, "contradicts": 8}
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")


@pytest.fixture(
    scope="module",
    params=[
        TextDocumentWithLabeledSpansAndBinaryRelations,
        TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
        TextDocumentWithLabeledMultiSpansAndBinaryRelations,
        TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions,
    ],
)
def target_document_type(builder, request) -> Optional[Type[Document]]:
    if request.param in set(builder.document_converters):
        return request.param
    return None


@pytest.fixture(scope="module")
def converted_dataset(dataset, target_document_type) -> Optional[DatasetDict]:
    if target_document_type is None:
        return None
    return dataset.to_document_type(target_document_type)


@pytest.fixture(scope="module")
def converted_document(converted_dataset) -> Optional[Document]:
    if converted_dataset is None:
        return None
    return converted_dataset["train"][0]


def test_converted_document(converted_document, dataset_variant):
    # Check that the conversion is correct and the data makes sense
    # get a document to check
    doc = converted_document
    if dataset_variant == "default":
        if isinstance(doc, TextDocumentWithLabeledSpansAndBinaryRelations):
            # check the entities
            assert len(doc.labeled_spans) == 183
            # sort the entities by their start position and convert them to tuples
            # check the first ten entities after sorted
            sorted_entity_tuples = resolve_annotations(doc.labeled_spans)
            # Checking the first ten entities
            assert sorted_entity_tuples[:10] == [
                (
                    "complicated 3D character models are widely used in fields of entertainment, virtual reality, medicine etc",
                    "background_claim",
                ),
                (
                    "The range of breathtaking realistic 3D models is only limited by the creativity of artists and resolution of devices",
                    "background_claim",
                ),
                (
                    "Driving 3D models in a natural and believable manner is not trivial",
                    "background_claim",
                ),
                ("the model is very detailed", "data"),
                ("playback of animation becomes quite heavy and time consuming", "data"),
                ("a frame goes wrong", "data"),
                ("a production cannot afford major revisions", "background_claim"),
                ("resculpting models", "data"),
                ("re-rigging skeletons", "data"),
                (
                    "providing a flexible and efficient solution to animation remains an open problem",
                    "own_claim",
                ),
            ]

            # check the relations
            assert len(doc.binary_relations) == 116
            relation_tuples = resolve_annotations(doc.binary_relations)
            # check the first ten relations
            assert relation_tuples[:13] == [
                (
                    ("the model is very detailed", "data"),
                    "supports",
                    (
                        "Driving 3D models in a natural and believable manner is not trivial",
                        "background_claim",
                    ),
                ),
                (
                    ("playback of animation becomes quite heavy and time consuming", "data"),
                    "supports",
                    (
                        "Driving 3D models in a natural and believable manner is not trivial",
                        "background_claim",
                    ),
                ),
                (
                    ("a frame goes wrong", "data"),
                    "supports",
                    ("a production cannot afford major revisions", "background_claim"),
                ),
                (
                    ("a production cannot afford major revisions", "background_claim"),
                    "supports",
                    (
                        "providing a flexible and efficient solution to animation remains an open problem",
                        "own_claim",
                    ),
                ),
                (
                    ("resculpting models", "data"),
                    "supports",
                    ("a production cannot afford major revisions", "background_claim"),
                ),
                (
                    ("re-rigging skeletons", "data"),
                    "supports",
                    ("a production cannot afford major revisions", "background_claim"),
                ),
                (("1", "data"), "supports", ("A nice review of SSD is given", "background_claim")),
                (
                    (
                        "SSD is widely used in games, virtual reality and other realtime applications",
                        "background_claim",
                    ),
                    "supports",
                    (
                        "Skeleton Subspace Deformation (SSD) is the predominant approach to character skinning at present",
                        "background_claim",
                    ),
                ),
                (
                    ("its ease of implementation", "data"),
                    "supports",
                    (
                        "SSD is widely used in games, virtual reality and other realtime applications",
                        "background_claim",
                    ),
                ),
                (
                    ("low cost of computing", "data"),
                    "supports",
                    (
                        "SSD is widely used in games, virtual reality and other realtime applications",
                        "background_claim",
                    ),
                ),
                (
                    ("This approach is not commonly applied", "background_claim"),
                    "contradicts",
                    (
                        "artists will edit the geometry of characters in the rest pose to fine-tune animations",
                        "background_claim",
                    ),
                ),
                (
                    (
                        "editing in the rest pose will influence most other poses",
                        "background_claim",
                    ),
                    "supports",
                    ("This approach is not commonly applied", "background_claim"),
                ),
                (
                    ("For those applications that require visual fidelity", "background_claim"),
                    "parts_of_same",
                    ("SSD serves only as a basic framework", "background_claim"),
                ),
            ]
            counter = Counter([rt[1] for rt in relation_tuples])
            assert dict(counter) == {
                "supports": 93,
                "contradicts": 8,
                "parts_of_same": 6,
                "semantically_same": 9,
            }
        elif doc is None:
            pass
        else:
            raise ValueError(f"Unknown document type: {type(doc)}")
    elif dataset_variant == "resolve_parts_of_same":
        # if isinstance(doc, TextDocumentWithLabeledSpansAndBinaryRelations):
        #     # check the entities
        #     assert len(doc.labeled_spans) == 177
        #     # sort the labeled spans by their start position and convert them to tuples
        #     sorted_entity_tuples = resolve_annotations(doc.labeled_spans)
        #     # check the first ten entities
        #     assert sorted_entity_tuples[:10] == [
        #         (
        #             "complicated 3D character models are widely used in fields of entertainment, virtual reality, medicine etc",
        #             "background_claim",
        #         ),
        #         (
        #             "The range of breathtaking realistic 3D models is only limited by the creativity of artists and resolution of devices",
        #             "background_claim",
        #         ),
        #         (
        #             "Driving 3D models in a natural and believable manner is not trivial",
        #             "background_claim",
        #         ),
        #         ("the model is very detailed", "data"),
        #         ("playback of animation becomes quite heavy and time consuming", "data"),
        #         ("a frame goes wrong", "data"),
        #         ("a production cannot afford major revisions", "background_claim"),
        #         ("resculpting models", "data"),
        #         ("re-rigging skeletons", "data"),
        #         (
        #             "providing a flexible and efficient solution to animation remains an open problem",
        #             "own_claim",
        #         ),
        #     ]
        #
        #     # this comes out of the 13th relation which is a parts_of_same relation (see above)
        #     assert sorted_entity_tuples[20] == (
        #         "For those applications that require visual fidelity, such as movies, SSD serves only as a basic framework",
        #         "background_claim",
        #     )
        #
        #     # check the relations
        #     assert len(doc.binary_relations) == 110
        #     relation_tuples = resolve_annotations(doc.binary_relations)
        #     # check the first ten relations
        #     assert relation_tuples[:10] == [
        #         (
        #             ("the model is very detailed", "data"),
        #             "supports",
        #             (
        #                 "Driving 3D models in a natural and believable manner is not trivial",
        #                 "background_claim",
        #             ),
        #         ),
        #         (
        #             ("playback of animation becomes quite heavy and time consuming", "data"),
        #             "supports",
        #             (
        #                 "Driving 3D models in a natural and believable manner is not trivial",
        #                 "background_claim",
        #             ),
        #         ),
        #         (
        #             ("a frame goes wrong", "data"),
        #             "supports",
        #             ("a production cannot afford major revisions", "background_claim"),
        #         ),
        #         (
        #             ("a production cannot afford major revisions", "background_claim"),
        #             "supports",
        #             (
        #                 "providing a flexible and efficient solution to animation remains an open problem",
        #                 "own_claim",
        #             ),
        #         ),
        #         (
        #             ("resculpting models", "data"),
        #             "supports",
        #             ("a production cannot afford major revisions", "background_claim"),
        #         ),
        #         (
        #             ("re-rigging skeletons", "data"),
        #             "supports",
        #             ("a production cannot afford major revisions", "background_claim"),
        #         ),
        #         (("1", "data"), "supports", ("A nice review of SSD is given", "background_claim")),
        #         (
        #             (
        #                 "SSD is widely used in games, virtual reality and other realtime applications",
        #                 "background_claim",
        #             ),
        #             "supports",
        #             (
        #                 "Skeleton Subspace Deformation (SSD) is the predominant approach to character skinning at present",
        #                 "background_claim",
        #             ),
        #         ),
        #         (
        #             ("its ease of implementation", "data"),
        #             "supports",
        #             (
        #                 "SSD is widely used in games, virtual reality and other realtime applications",
        #                 "background_claim",
        #             ),
        #         ),
        #         (
        #             ("low cost of computing", "data"),
        #             "supports",
        #             (
        #                 "SSD is widely used in games, virtual reality and other realtime applications",
        #                 "background_claim",
        #             ),
        #         ),
        #     ]
        #     counter = Counter([rt[1] for rt in relation_tuples])
        #     assert dict(counter) == {"supports": 93, "contradicts": 8, "semantically_same": 9}

        if isinstance(doc, TextDocumentWithLabeledMultiSpansAndBinaryRelations):
            # check the entities
            assert len(doc.labeled_multi_spans) == 177
            # sort the labeled spans by their start position and convert them to tuples
            sorted_span_tuples = resolve_annotations(doc.labeled_multi_spans)
            # check the first ten entities
            assert sorted_span_tuples[:10] == [
                (
                    (
                        "complicated 3D character models are widely used in fields of entertainment, virtual reality, medicine etc",
                    ),
                    "background_claim",
                ),
                (
                    (
                        "The range of breathtaking realistic 3D models is only limited by the creativity of artists and resolution of devices",
                    ),
                    "background_claim",
                ),
                (
                    ("Driving 3D models in a natural and believable manner is not trivial",),
                    "background_claim",
                ),
                (("the model is very detailed",), "data"),
                (("playback of animation becomes quite heavy and time consuming",), "data"),
                (("a frame goes wrong",), "data"),
                (("a production cannot afford major revisions",), "background_claim"),
                (("resculpting models",), "data"),
                (("re-rigging skeletons",), "data"),
                (
                    (
                        "providing a flexible and efficient solution to animation remains an open problem",
                    ),
                    "own_claim",
                ),
            ]

            # this comes out of the 13th relation which is a parts_of_same relation (see above)
            assert sorted_span_tuples[20] == (
                (
                    "For those applications that require visual fidelity",
                    "SSD serves only as a basic framework",
                ),
                "background_claim",
            )

            # check the relations
            assert len(doc.binary_relations) == 110
            relation_tuples = resolve_annotations(doc.binary_relations)
            # check the first ten relations
            assert relation_tuples[:10] == [
                (
                    (("the model is very detailed",), "data"),
                    "supports",
                    (
                        ("Driving 3D models in a natural and believable manner is not trivial",),
                        "background_claim",
                    ),
                ),
                (
                    (("playback of animation becomes quite heavy and time consuming",), "data"),
                    "supports",
                    (
                        ("Driving 3D models in a natural and believable manner is not trivial",),
                        "background_claim",
                    ),
                ),
                (
                    (("a frame goes wrong",), "data"),
                    "supports",
                    (("a production cannot afford major revisions",), "background_claim"),
                ),
                (
                    (("a production cannot afford major revisions",), "background_claim"),
                    "supports",
                    (
                        (
                            "providing a flexible and efficient solution to animation remains an open problem",
                        ),
                        "own_claim",
                    ),
                ),
                (
                    (("resculpting models",), "data"),
                    "supports",
                    (("a production cannot afford major revisions",), "background_claim"),
                ),
                (
                    (("re-rigging skeletons",), "data"),
                    "supports",
                    (("a production cannot afford major revisions",), "background_claim"),
                ),
                (
                    (("1",), "data"),
                    "supports",
                    (("A nice review of SSD is given",), "background_claim"),
                ),
                (
                    (
                        (
                            "SSD is widely used in games, virtual reality and other realtime applications",
                        ),
                        "background_claim",
                    ),
                    "supports",
                    (
                        (
                            "Skeleton Subspace Deformation (SSD) is the predominant approach to character skinning at present",
                        ),
                        "background_claim",
                    ),
                ),
                (
                    (("its ease of implementation",), "data"),
                    "supports",
                    (
                        (
                            "SSD is widely used in games, virtual reality and other realtime applications",
                        ),
                        "background_claim",
                    ),
                ),
                (
                    (("low cost of computing",), "data"),
                    "supports",
                    (
                        (
                            "SSD is widely used in games, virtual reality and other realtime applications",
                        ),
                        "background_claim",
                    ),
                ),
            ]
            counter = Counter([rt[1] for rt in relation_tuples])
            assert dict(counter) == {"supports": 93, "contradicts": 8, "semantically_same": 9}
        elif doc is None:
            pass
        else:
            raise ValueError(f"Unknown document type: {type(doc)}")
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")

    if isinstance(doc, TextDocumentWithLabeledPartitions):
        # check the partitions with startswiths(), endswith(), and label
        partitions = doc.labeled_partitions
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


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def dataset_of_text_documents_with_labeled_spans_and_binary_relations(
    dataset, dataset_variant
) -> Optional[DatasetDict]:
    return dataset.to_document_type(TextDocumentWithLabeledSpansAndBinaryRelations)


TOKENIZED_DOCUMENT_TYPE_MAPPING = {
    TextDocumentWithLabeledSpansAndBinaryRelations: TestTokenDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions: TestTokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
    TextDocumentWithLabeledMultiSpansAndBinaryRelations: TestTokenDocumentWithLabeledMultiSpansAndBinaryRelations,
    TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions: TestTokenDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions,
}


def test_tokenize_documents_all(converted_dataset, tokenizer, dataset_variant):
    if converted_dataset is None:
        return
    for split, docs in converted_dataset.items():
        for doc in docs:
            # Note, that this is a list of documents, because the document may be split into chunks
            # if the input text is too long.
            tokenized_docs = tokenize_document(
                doc,
                tokenizer=tokenizer,
                return_overflowing_tokens=True,
                result_document_type=TOKENIZED_DOCUMENT_TYPE_MAPPING[type(doc)],
                partition_layer="labeled_partitions"
                if isinstance(doc, TextDocumentWithLabeledPartitions)
                else None,
                strict_span_conversion=False
                if isinstance(doc, TextDocumentWithLabeledPartitions)
                else True,
                verbose=True,
            )
            # we just ensure that we get at least one tokenized document
            assert tokenized_docs is not None
            assert len(tokenized_docs) > 0


def test_document_converters(dataset_variant):
    builder = BUILDER_CLASS(config_name=dataset_variant)
    document_converters = builder.document_converters

    if dataset_variant == "default":
        assert set(document_converters) == {
            TextDocumentWithLabeledSpansAndBinaryRelations,
            TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
        }
    elif dataset_variant == "resolve_parts_of_same":
        assert set(document_converters) == {
            # TextDocumentWithLabeledSpansAndBinaryRelations,
            # TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
            TextDocumentWithLabeledMultiSpansAndBinaryRelations,
            TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions,
        }
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")

    assert all(callable(v) for k, v in document_converters.items())
