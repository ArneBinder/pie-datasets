import logging
import os

import pytest
from datasets import load_dataset
from pytorch_ie.core import Document
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndBinaryRelations

from dataset_builders.pie.tacred.tacred import (
    Tacred,
    convert_to_text_document_with_labeled_spans_and_binary_relations,
    document_to_example,
    example_to_document,
)
from tests import FIXTURES_ROOT
from tests.dataset_builders.common import (
    PIE_BASE_PATH,
    _deep_compare,
    _dump_json,
    _load_json,
)

logger = logging.getLogger(__name__)

HF_DATASET_PATH = "DFKI-SLT/tacred"
PIE_DATASET_PATH = f"{PIE_BASE_PATH}/tacred"
SPLITS = ["train", "validation", "test"]
EXAMPLE_IDX = 0
NUM_SAMPLES = 3

DUMP_FIXTURE_DATA = False

TACRED_DATA_DIR = os.getenv("TACRED_DATA_DIR", "") or None  # ~/datasets/tacred/data/json


@pytest.fixture(params=[config.name for config in Tacred.BUILDER_CONFIGS], scope="module")
def dataset_variant(request):
    return request.param


@pytest.fixture(params=SPLITS, scope="module")
def split(request):
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
        return load_dataset(HF_DATASET_PATH, name=dataset_variant, data_dir=TACRED_DATA_DIR)


@pytest.fixture(scope="module")
def hf_dataset_samples(hf_samples_fn):
    data_files = {split: hf_samples_fn.format(split=split) for split in SPLITS}
    return load_dataset("json", data_files=data_files)


def test_hf_dataset_samples(hf_dataset_samples):
    assert set(hf_dataset_samples) == {"train", "validation", "test"}
    for ds in hf_dataset_samples.values():
        assert len(ds) == NUM_SAMPLES


@pytest.mark.skipif(condition=not DUMP_FIXTURE_DATA, reason="don't dump fixture data")
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
def hf_example(hf_dataset_samples, split, request):
    return hf_dataset_samples[split][request.param]


@pytest.fixture(scope="module")
def ner_names(hf_metadata_fn, split):
    return _load_json(hf_metadata_fn.format(split=split, idx_or_feature="ner_names"))


@pytest.fixture(scope="module")
def relation_names(hf_metadata_fn, split):
    return _load_json(hf_metadata_fn.format(split=split, idx_or_feature="relation_names"))


@pytest.fixture(scope="module")
def document(hf_example, ner_names, relation_names):
    return example_to_document(
        hf_example,
        ner_int2str=lambda idx: ner_names[idx],
        relation_int2str=lambda idx: relation_names[idx],
    )


def test_document(document):
    assert document is not None
    assert isinstance(document, Document)


def test_example_to_document_and_back(hf_example, ner_names, relation_names):
    doc = example_to_document(
        hf_example,
        ner_int2str=lambda idx: ner_names[idx],
        relation_int2str=lambda idx: relation_names[idx],
    )
    example_back = document_to_example(doc, ner_names=ner_names, relation_names=relation_names)

    _deep_compare(obj=example_back, obj_expected=hf_example)


@pytest.mark.skipif(
    condition=TACRED_DATA_DIR is None,
    reason="environment variable TACRED_DATA_DIR is not set",
)
@pytest.mark.slow
def test_example_to_document_and_back_all(hf_dataset):
    for hf_ds in hf_dataset.values():
        ner_names = hf_ds.features["subj_type"].names
        relation_names = hf_ds.features["relation"].names
        for hf_ex in hf_ds:
            doc = example_to_document(
                hf_ex,
                ner_int2str=lambda idx: ner_names[idx],
                relation_int2str=lambda idx: relation_names[idx],
            )
            example_back = document_to_example(
                doc, ner_names=ner_names, relation_names=relation_names
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
            assert isinstance(doc, Document)


def test_convert_to_text_document_with_labeled_spans_and_binary_relations(document):
    converted_doc = convert_to_text_document_with_labeled_spans_and_binary_relations(document)
    assert converted_doc is not None
    assert isinstance(converted_doc, TextDocumentWithLabeledSpansAndBinaryRelations)
