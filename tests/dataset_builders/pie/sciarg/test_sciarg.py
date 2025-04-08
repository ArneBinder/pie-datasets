import dataclasses
from collections import Counter
from typing import Any, List, Optional, Sequence, Type

import datasets
import pytest
from pie_core import Annotation, AnnotationLayer, Document, annotation_field
from pie_modules.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan, Span
from pie_modules.document.processing import tokenize_document
from pie_modules.documents import (
    TextDocumentWithLabeledMultiSpansAndBinaryRelations,
    TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions,
    TextDocumentWithLabeledPartitions,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
    TokenBasedDocument,
)
from transformers import AutoTokenizer, PreTrainedTokenizer

from dataset_builders.pie.sciarg.sciarg import SciArg, remove_duplicate_relations
from pie_datasets import DatasetDict
from pie_datasets.builders.brat import (
    BratAttribute,
    BratDocument,
    BratDocumentWithMergedSpans,
    BratMultiSpan,
    BratRelation,
    BratSpan,
)
from tests.dataset_builders.common import (
    PIE_BASE_PATH,
    PIE_DS_FIXTURE_DATA_PATH,
    TestTokenDocumentWithLabeledPartitions,
    TestTokenDocumentWithLabeledSpansAndBinaryRelations,
    TestTokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

datasets.disable_caching()

TEST_FULL_DATASET = True

DATASET_NAME = "sciarg"
BUILDER_CLASS = SciArg
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
DATA_DIR = PIE_DS_FIXTURE_DATA_PATH / DATASET_NAME
SPLIT_SIZES = {"train": 40 if TEST_FULL_DATASET else 3}
FULL_LABEL_COUNTS = {
    "default": {
        "relations": {
            "contradicts": 696,
            "parts_of_same": 1298,
            "semantically_same": 44,
            "supports": 5789,
        },
        "spans": {"background_claim": 3291, "data": 4297, "own_claim": 6004},
    },
    "resolve_parts_of_same": {
        "relations": {"contradicts": 696, "semantically_same": 44, "supports": 5788},
        "spans": {"background_claim": 2752, "data": 4093, "own_claim": 5450},
    },
}
CONVERTED_LAYER_MAPPING = {
    "default": {
        "spans": "labeled_spans",
        "relations": "binary_relations",
    },
    "resolve_parts_of_same": {
        "spans": "labeled_multi_spans",
        "relations": "binary_relations",
    },
}
FULL_LABEL_COUNTS_CONVERTED = {
    variant: {CONVERTED_LAYER_MAPPING[variant][ln]: value for ln, value in counts.items()}
    for variant, counts in FULL_LABEL_COUNTS.items()
}
LABELED_PARTITION_COUNTS = {"Abstract": 40, "H1": 340, "Title": 40}


def resolve_annotation(annotation: Annotation) -> Any:
    if annotation.target is None:
        return None
    if isinstance(annotation, (LabeledSpan, BratSpan)):
        return annotation.target[annotation.start : annotation.end], annotation.label
    elif isinstance(annotation, (LabeledMultiSpan, BratMultiSpan)):
        return (
            tuple(annotation.target[start:end] for start, end in annotation.slices),
            annotation.label,
        )
    elif isinstance(annotation, (BinaryRelation, BratRelation)):
        return (
            resolve_annotation(annotation.head),
            annotation.label,
            resolve_annotation(annotation.tail),
        )
    elif isinstance(annotation, BratAttribute):
        result = (resolve_annotation(annotation.annotation), annotation.label)
        if annotation.value is not None:
            return result + (annotation.value,)
        else:
            return result
    elif isinstance(annotation, BinaryRelation):
        return (
            resolve_annotation(annotation.head),
            annotation.label,
            resolve_annotation(annotation.tail),
        )
    else:
        raise TypeError(f"Unknown annotation type: {type(annotation)}")


def sort_annotations(annotations: Sequence[Annotation]) -> List[Annotation]:
    if len(annotations) == 0:
        return []
    annotation = annotations[0]
    if isinstance(annotation, (LabeledSpan, BratSpan)):
        return sorted(annotations, key=lambda a: (a.start, a.end, a.label))
    elif isinstance(annotation, Span):
        return sorted(annotations, key=lambda a: (a.start, a.end))
    elif isinstance(annotation, (LabeledMultiSpan, BratMultiSpan)):
        return sorted(annotations, key=lambda a: (a.slices, a.label))
    elif isinstance(annotation, (BinaryRelation, BratRelation)):
        if isinstance(annotation.head, LabeledSpan) and isinstance(annotation.tail, LabeledSpan):
            return sorted(
                annotations,
                key=lambda a: (a.head.start, a.head.end, a.label, a.tail.start, a.tail.end),
            )
        elif isinstance(annotation.head, LabeledMultiSpan) and isinstance(
            annotation.tail, LabeledMultiSpan
        ):
            return sorted(
                annotations,
                key=lambda a: (a.head.slices, a.label, a.tail.slices),
            )
        else:
            raise ValueError(
                f"Unsupported relation type for BinaryRelation arguments: "
                f"{type(annotation.head)}, {type(annotation.tail)}"
            )
    else:
        raise ValueError(f"Unsupported annotation type: {type(annotation)}")


def resolve_annotations(annotations: Sequence[Annotation]) -> List[Any]:
    sorted_annotations = sort_annotations(annotations)
    return [resolve_annotation(a) for a in sorted_annotations]


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


def test_generate_document(builder, hf_dataset, dataset_variant):
    hf_example = hf_dataset["train"][0]
    document = builder._generate_document(hf_example)
    if dataset_variant == "default":
        assert isinstance(document, BratDocumentWithMergedSpans)
        assert len(document.spans) == 183
    elif dataset_variant == "resolve_parts_of_same":
        assert isinstance(document, BratDocument)
        assert len(document.spans) == 177
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")


@pytest.fixture(scope="module")
def dataset(dataset_variant) -> DatasetDict:
    if TEST_FULL_DATASET:
        base_dataset_kwargs = None
    else:
        base_dataset_kwargs = {"data_dir": str(PIE_DS_FIXTURE_DATA_PATH / DATASET_NAME)}
    return DatasetDict.load_dataset(
        str(PIE_DATASET_PATH), name=dataset_variant, base_dataset_kwargs=base_dataset_kwargs
    )


def assert_dataset_label_counts(dataset, expected_label_counts):
    label_counts = {
        ln: dict(Counter(ann.label for doc in dataset["train"] for ann in doc[ln]))
        for ln in expected_label_counts.keys()
    }
    assert label_counts == expected_label_counts


def test_dataset(dataset, dataset_variant):
    assert dataset is not None
    assert {name: len(ds) for name, ds in dataset.items()} == SPLIT_SIZES

    if TEST_FULL_DATASET:
        assert_dataset_label_counts(
            dataset, expected_label_counts=FULL_LABEL_COUNTS[dataset_variant]
        )


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


def test_converted_datasets(converted_dataset, dataset_variant, target_document_type):
    if converted_dataset is not None:
        split_sizes = {name: len(ds) for name, ds in converted_dataset.items()}
        assert split_sizes == SPLIT_SIZES
        if dataset_variant == "default":
            expected_document_type = TextDocumentWithLabeledSpansAndBinaryRelations
        elif dataset_variant == "resolve_parts_of_same":
            expected_document_type = TextDocumentWithLabeledMultiSpansAndBinaryRelations
        else:
            raise ValueError(f"Unknown dataset variant: {dataset_variant}")

        assert issubclass(converted_dataset.document_type, expected_document_type)
        assert isinstance(converted_dataset["train"][0], expected_document_type)

        if TEST_FULL_DATASET:
            # copy to avoid modifying the original dict
            expected_label_counts = {**FULL_LABEL_COUNTS_CONVERTED[dataset_variant]}
            if issubclass(target_document_type, TextDocumentWithLabeledPartitions):
                expected_label_counts["labeled_partitions"] = LABELED_PARTITION_COUNTS
            assert_dataset_label_counts(converted_dataset, expected_label_counts)


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
                    (
                        "artists will edit the geometry of characters in the rest pose to fine-tune animations",
                        "background_claim",
                    ),
                    "contradicts",
                    ("This approach is not commonly applied", "background_claim"),
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


@dataclasses.dataclass
class TestTokenDocumentWithLabeledMultiSpansAndBinaryRelations(TokenBasedDocument):
    labeled_multi_spans: AnnotationLayer[LabeledMultiSpan] = annotation_field(target="tokens")
    binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(
        target="labeled_multi_spans"
    )


@dataclasses.dataclass
class TestTokenDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions(
    TestTokenDocumentWithLabeledMultiSpansAndBinaryRelations,
    TestTokenDocumentWithLabeledPartitions,
):
    pass


TOKENIZED_DOCUMENT_TYPE_MAPPING = {
    TextDocumentWithLabeledSpansAndBinaryRelations: TestTokenDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions: TestTokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
    TextDocumentWithLabeledMultiSpansAndBinaryRelations: TestTokenDocumentWithLabeledMultiSpansAndBinaryRelations,
    TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions: TestTokenDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions,
}


def test_tokenize_documents_all(converted_dataset, tokenizer, dataset_variant):
    if converted_dataset is None:
        return
    # docs that cause errors when using strict_span_conversion (and the spans that can not be converted):
    # - A11: "́ Ø 1⁄2 μ 3⁄4È and  ́ Ø 3⁄4 μ 3⁄4Ç"
    # - A19: "̃ l CE is the normalized muscle fiber length"
    # - A20: "̇ to generalized forces (u)" / "('a force field function u = g(q, q)  ̇ maps kinematic states', '̇ to generalized forces (u)')"
    # - A24: "φ n = φ  ̄"
    # - A25: " M 0 −1 I −1 0 A 1 b T 1 ··· A k b T k b k" and " b 1  R = " / "('\uf8ee b 1 \uf8f9 R = \uf8f0',)" and "('\uf8fb M 0 −1 I −1 0 A 1 b T 1 ··· A k b T k b k',)"
    docs_with_span_errors = {"A11", "A19", "A20", "A24", "A25"}
    for split, docs in converted_dataset.items():
        for doc in docs:
            strict_span_conversion = doc.id not in docs_with_span_errors and not isinstance(
                doc, TextDocumentWithLabeledPartitions
            )
            # Note, that this is a list of documents, because the document may be split into chunks
            # if the input text is too long.
            tokenized_docs = tokenize_document(
                doc,
                tokenizer=tokenizer,
                return_overflowing_tokens=True,
                result_document_type=TOKENIZED_DOCUMENT_TYPE_MAPPING[type(doc)],
                partition_layer=(
                    "labeled_partitions"
                    if isinstance(doc, TextDocumentWithLabeledPartitions)
                    else None
                ),
                strict_span_conversion=strict_span_conversion,
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


def test_remove_duplicate_relations():
    doc = BratDocumentWithMergedSpans(id="test", text="This is a test sentence.")
    doc.spans.append(LabeledSpan(start=0, end=4, label="test"))
    assert str(doc.spans[0]) == "This"
    doc.spans.append(LabeledSpan(start=10, end=23, label="sentence"))
    assert str(doc.spans[1]) == "test sentence"
    # reference relation
    doc.relations.append(BinaryRelation(head=doc.spans[0], tail=doc.spans[1], label="a"))
    # swapped arguments
    doc.relations.append(BinaryRelation(head=doc.spans[1], tail=doc.spans[0], label="a"))
    # this is the only duplicate relation, it should be removed
    doc.relations.append(BinaryRelation(head=doc.spans[0], tail=doc.spans[1], label="a"))
    # different label
    doc.relations.append(BinaryRelation(head=doc.spans[0], tail=doc.spans[1], label="b"))

    assert len(doc.relations) == 4
    remove_duplicate_relations(doc)
    assert len(doc.relations) == 3
    assert doc.relations[0].head == doc.spans[0]
    assert doc.relations[0].tail == doc.spans[1]
    assert doc.relations[0].label == "a"
    assert doc.relations[1].label == "a"
    assert doc.relations[1].head == doc.spans[1]
    assert doc.relations[1].tail == doc.spans[0]
    assert doc.relations[2].label == "b"
    assert doc.relations[2].head == doc.spans[0]
    assert doc.relations[2].tail == doc.spans[1]
