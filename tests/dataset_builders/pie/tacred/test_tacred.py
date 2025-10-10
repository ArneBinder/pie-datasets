import logging
import os

import pytest
from datasets import ClassLabel, load_dataset
from pie_core import Document
from pie_documents.documents import TextDocumentWithLabeledSpansAndBinaryRelations

from dataset_builders.pie.tacred.tacred import (
    Tacred,
    convert_to_text_document_with_labeled_spans_and_binary_relations,
    document_to_example,
    example_to_document,
)
from tests import FIXTURES_ROOT
from tests.conftest import CREATE_FIXTURE_DATA
from tests.dataset_builders.common import (
    PIE_BASE_PATH,
    _deep_compare,
    _dump_json,
    _load_json,
)

logger = logging.getLogger(__name__)

PIE_DATASET_PATH = f"{PIE_BASE_PATH}/tacred"
BUILDER_CLASS = Tacred
HF_DATASET_PATH = BUILDER_CLASS.BASE_DATASET_PATH
SPLIT_NAMES = {"train", "validation", "test"}
EXAMPLE_IDX = 0
NUM_SAMPLES = 3

TACRED_DATA_DIR = os.getenv("TACRED_DATA_DIR", "") or None  # ~/datasets/tacred/data/json


@pytest.fixture(params=[config.name for config in BUILDER_CLASS.BUILDER_CONFIGS], scope="module")
def dataset_variant(request):
    return request.param


@pytest.fixture(params=SPLIT_NAMES, scope="module")
def split_name(request):
    return request.param


@pytest.fixture(scope="module")
def hf_example_path(dataset_variant):
    return f"{FIXTURES_ROOT}/dataset_builders/hf/{HF_DATASET_PATH}/{dataset_variant}"


@pytest.fixture(scope="module")
def hf_samples_fn(hf_example_path):
    return f"{hf_example_path}/{{split}}.samples-{NUM_SAMPLES}.json"


@pytest.fixture(scope="module")
def hf_metadata_fn(hf_example_path):
    return f"{hf_example_path}/{{split}}.{{idx_or_feature}}.json"


@pytest.fixture(scope="module")
def hf_dataset(dataset_variant):
    if TACRED_DATA_DIR is None:
        raise ValueError("TACRED_DATA_DIR is required to load the Huggingface TacRED dataset")
    else:
        return load_dataset(
            HF_DATASET_PATH,
            name=dataset_variant,
            data_dir=TACRED_DATA_DIR,
            **BUILDER_CLASS.BASE_BUILDER_KWARGS_DICT[dataset_variant],
        )


@pytest.fixture(scope="module")
def hf_dataset_samples(hf_samples_fn):
    data_files = {split: hf_samples_fn.format(split=split) for split in SPLIT_NAMES}
    return load_dataset("json", data_files=data_files)


def test_hf_dataset_samples(hf_dataset_samples):
    assert set(hf_dataset_samples) == SPLIT_NAMES
    for ds in hf_dataset_samples.values():
        assert len(ds) == NUM_SAMPLES


@pytest.mark.skipif(condition=not CREATE_FIXTURE_DATA, reason="don't dump fixture data")
def test_dump_hf(hf_dataset, hf_samples_fn, hf_metadata_fn):
    for split, ds in hf_dataset.items():
        # save the dataset split
        samples = [ds[i] for i in range(NUM_SAMPLES)]
        _dump_json(samples, hf_samples_fn.format(split=split))
        # save the metadata
        _dump_json(
            obj=ds.features["subj_type"].names,
            fn=hf_metadata_fn.format(split=split, idx_or_feature="ner_names"),
        )
        _dump_json(
            obj=ds.features["relation"].names,
            fn=hf_metadata_fn.format(split=split, idx_or_feature="relation_names"),
        )


@pytest.fixture(params=range(NUM_SAMPLES), scope="module")
def hf_example(hf_dataset_samples, split_name, request):
    return hf_dataset_samples[split_name][request.param]


@pytest.fixture(scope="module")
def ner_labels(hf_metadata_fn, split_name):
    return ClassLabel(
        names=_load_json(hf_metadata_fn.format(split=split_name, idx_or_feature="ner_names"))
    )


@pytest.fixture(scope="module")
def relation_labels(hf_metadata_fn, split_name):
    return ClassLabel(
        names=_load_json(hf_metadata_fn.format(split=split_name, idx_or_feature="relation_names"))
    )


@pytest.fixture(scope="module")
def document(hf_example, ner_labels, relation_labels):
    return example_to_document(
        hf_example,
        ner_labels=ner_labels,
        relation_labels=relation_labels,
    )


def test_document(document):
    assert document is not None
    assert isinstance(document, BUILDER_CLASS.DOCUMENT_TYPE)


def test_example_to_document_and_back(hf_example, ner_labels, relation_labels):
    doc = example_to_document(
        hf_example,
        ner_labels=ner_labels,
        relation_labels=relation_labels,
    )
    example_back = document_to_example(doc, ner_labels=ner_labels, relation_labels=relation_labels)

    _deep_compare(obj=example_back, obj_expected=hf_example)


@pytest.mark.skipif(
    condition=TACRED_DATA_DIR is None,
    reason="environment variable TACRED_DATA_DIR is not set",
)
@pytest.mark.slow
def test_example_to_document_and_back_all(hf_dataset):
    for hf_ds in hf_dataset.values():
        ner_labels = hf_ds.features["subj_type"]
        relation_labels = hf_ds.features["relation"]
        for hf_ex in hf_ds:
            doc = example_to_document(
                hf_ex,
                ner_labels=ner_labels,
                relation_labels=relation_labels,
            )
            assert isinstance(doc, BUILDER_CLASS.DOCUMENT_TYPE)
            example_back = document_to_example(
                doc, ner_labels=ner_labels, relation_labels=relation_labels
            )

            _deep_compare(obj=example_back, obj_expected=hf_ex)


@pytest.mark.skipif(
    condition=TACRED_DATA_DIR is None,
    reason="environment variable TACRED_DATA_DIR is not set",
)
@pytest.mark.slow
def test_pie_document_all(dataset_variant):
    pie_dataset = load_dataset(
        PIE_DATASET_PATH,
        name=dataset_variant,
        data_dir=TACRED_DATA_DIR,
    )
    for split, ds in pie_dataset.items():
        for doc in ds:
            assert doc is not None
            # Note: we don't check the actual type of the document here, because the real type
            # comes from the dataset builder script which Huggingface load_dataset() copies
            # to a temporary directory and then imports. This means that the type of the document
            # is not the same as the type of the document in the original dataset builder script.
            assert isinstance(doc, Document)


def test_convert_to_text_document_with_labeled_spans_and_binary_relations(document):
    converted_doc = convert_to_text_document_with_labeled_spans_and_binary_relations(document)
    assert converted_doc is not None
    assert isinstance(converted_doc, TextDocumentWithLabeledSpansAndBinaryRelations)
