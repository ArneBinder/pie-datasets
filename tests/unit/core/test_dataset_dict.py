import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import datasets
import pytest
from pie_core import (
    AnnotationLayer,
    Document,
    EnterDatasetDictMixin,
    EnterDatasetMixin,
    ExitDatasetDictMixin,
    ExitDatasetMixin,
    WithDocumentTypeMixin,
    annotation_field,
)
from pie_modules.annotations import Label, LabeledSpan
from pie_modules.documents import TextBasedDocument

from pie_datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    concatenate_dataset_dicts,
    load_dataset,
)
from tests import DATASET_BUILDERS_ROOT, FIXTURES_ROOT
from tests.conftest import CREATE_FIXTURE_DATA, TestDocument
from tests.unit.core import TEST_PACKAGE

TEST_MODULE = f"{TEST_PACKAGE}.{Path(__file__).stem}"

logger = logging.getLogger(__name__)

DATASET_NAME = "conll2003"
# restrict all splits to 3 examples
N_FIXTURE_SAMPLES = 3
PIE_DATASET_PATH = DATASET_BUILDERS_ROOT / "pie" / DATASET_NAME
FIXTURE_DATA_PATH = FIXTURES_ROOT / "dataset_dict" / f"{DATASET_NAME}_extract"


@pytest.mark.skipif(condition=not CREATE_FIXTURE_DATA, reason="don't create fixture data again")
def test_create_fixture_data():
    conll2003 = DatasetDict(datasets.load_dataset(str(PIE_DATASET_PATH)))
    for split in list(conll2003):
        conll2003 = conll2003.select(split=split, stop=N_FIXTURE_SAMPLES)
    conll2003.to_json(FIXTURE_DATA_PATH)


@dataclass
class DocumentWithEntitiesAndRelations(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


@pytest.fixture(scope="module")
def dataset_dict():
    return DatasetDict.from_json(
        data_dir=FIXTURE_DATA_PATH, document_type=DocumentWithEntitiesAndRelations
    )


def test_from_json(dataset_dict):
    assert set(dataset_dict) == {"train", "test", "validation"}
    assert len(dataset_dict["train"]) == N_FIXTURE_SAMPLES
    assert len(dataset_dict["test"]) == N_FIXTURE_SAMPLES
    assert len(dataset_dict["validation"]) == N_FIXTURE_SAMPLES


def test_from_json_no_serialized_document_type(dataset_dict):
    with pytest.raises(ValueError) as excinfo:
        DatasetDict.from_json(data_dir=FIXTURE_DATA_PATH)
    assert (
        str(excinfo.value)
        == "document_type must be provided if it cannot be loaded from the metadata file"
    )


@pytest.fixture(scope="module")
def iterable_dataset_dict():
    return DatasetDict.from_json(
        data_dir=FIXTURE_DATA_PATH,
        document_type=DocumentWithEntitiesAndRelations,
        streaming=True,
    )


def test_iterable_dataset_dict(iterable_dataset_dict):
    assert set(iterable_dataset_dict) == {"train", "test", "validation"}


def test_to_json_and_back(dataset_dict, tmp_path):
    path = Path(tmp_path) / "dataset_dict"
    dataset_dict.to_json(path)
    dataset_dict_from_json = DatasetDict.from_json(
        data_dir=str(path),
        document_type=dataset_dict.document_type,
    )
    assert set(dataset_dict_from_json) == set(dataset_dict)
    for split in dataset_dict:
        assert len(dataset_dict_from_json[split]) == len(dataset_dict[split])
        for doc1, doc2 in zip(dataset_dict_from_json[split], dataset_dict[split]):
            assert doc1 == doc2


def test_to_json_and_back_serialize_document_type(dataset_dict, tmp_path):
    path = Path(tmp_path) / "dataset_dict"
    dataset_dict.to_json(path)
    dataset_dict_from_json = DatasetDict.from_json(
        data_dir=str(path),
    )
    assert set(dataset_dict_from_json) == set(dataset_dict)
    for split in dataset_dict:
        assert len(dataset_dict_from_json[split]) == len(dataset_dict[split])
        for doc1, doc2 in zip(dataset_dict_from_json[split], dataset_dict[split]):
            assert doc1 == doc2


def test_to_json_and_back_append(dataset_dict, tmp_path, caplog):
    path = Path(tmp_path) / "dataset_dict"

    dataset_dict1 = DatasetDict(
        {split_name: Dataset.from_documents(docs[:2]) for split_name, docs in dataset_dict.items()}
    )
    dataset_dict2 = DatasetDict(
        {split_name: Dataset.from_documents(docs[2:]) for split_name, docs in dataset_dict.items()}
    )
    caplog.clear()
    dataset_dict1.to_json(path)
    dataset_dict2.to_json(path)
    assert len(caplog.messages) == 0
    dataset_dict_from_json = DatasetDict.from_json(
        data_dir=str(path),
    )
    assert set(dataset_dict_from_json) == set(dataset_dict)
    for split in dataset_dict:
        assert len(dataset_dict_from_json[split]) == len(dataset_dict[split])
        for doc1, doc2 in zip(dataset_dict_from_json[split], dataset_dict[split]):
            assert doc1 == doc2


def test_to_json_and_back_append_overwrite(dataset_dict, tmp_path, caplog):
    path = Path(tmp_path) / "dataset_dict"

    dataset_dict1 = DatasetDict(
        {split_name: Dataset.from_documents(docs[:2]) for split_name, docs in dataset_dict.items()}
    )
    dataset_dict2 = DatasetDict(
        {split_name: Dataset.from_documents(docs[2:]) for split_name, docs in dataset_dict.items()}
    )
    caplog.clear()
    dataset_dict1.to_json(path)
    dataset_dict2.to_json(path, mode="w")
    assert len(caplog.messages) == 1
    assert caplog.messages[0].startswith("Dataset serialization directory")
    assert caplog.messages[0].endswith("already exists, removing it to overwrite existing files.")

    dataset_dict_from_json = DatasetDict.from_json(
        data_dir=str(path),
    )
    assert set(dataset_dict_from_json) == set(dataset_dict2)
    for split in dataset_dict:
        assert len(dataset_dict_from_json[split]) == len(dataset_dict2[split])
        for doc1, doc2 in zip(dataset_dict_from_json[split], dataset_dict2[split]):
            assert doc1 == doc2


def test_to_json_and_back_append_metadata_mismatch(dataset_dict, tmp_path):
    path = Path(tmp_path) / "dataset_dict"

    dataset_dict1 = DatasetDict(
        {split_name: Dataset.from_documents(docs[:2]) for split_name, docs in dataset_dict.items()}
    )
    dataset_dict2 = DatasetDict(
        {split_name: Dataset.from_documents(docs[2:]) for split_name, docs in dataset_dict.items()}
    )
    dataset_dict2_converted = dataset_dict2.cast_document_type(TextBasedDocument)
    dataset_dict1.to_json(path)
    with pytest.raises(ValueError) as excinfo:
        dataset_dict2_converted.to_json(path)
    assert str(excinfo.value).endswith(
        "metadata.json already exists, but the content does not match the current metadata. "
        "Can not append the current dataset to already serialized data."
        "\nprevious metadata: {'document_type': 'tests.unit.core.test_dataset_dict.DocumentWithEntitiesAndRelations'}"
        "\ncurrent metadata: {'document_type': 'pytorch_ie.documents.TextBasedDocument'}"
    )


def test_to_json_unknown_mode(dataset_dict, tmp_path):
    path = Path(tmp_path) / "dataset_dict"

    with pytest.raises(ValueError) as excinfo:
        dataset_dict.to_json(path, mode="unknown_mode")
    assert (
        str(excinfo.value)
        == 'mode must be "a" (append) or "w" (overwrite), but is "unknown_mode".'
    )


def test_document_type_empty_no_splits():
    with pytest.raises(ValueError) as excinfo:
        DatasetDict().document_type
    assert (
        str(excinfo.value) == "dataset does not contain any splits, cannot determine document type"
    )


def test_document_type_different_types(dataset_dict):
    # load the example dataset as a different document type
    dataset_dict_different_type = DatasetDict.from_json(
        data_dir=FIXTURE_DATA_PATH,
        document_type=TextBasedDocument,
    )
    assert dataset_dict_different_type.document_type is TextBasedDocument
    # create a dataset dict with different document types for train and test splits
    dataset_dict_different_types = DatasetDict(
        {
            "train": dataset_dict["train"],
            "test": dataset_dict_different_type["test"],
        }
    )
    # accessing the document type should raise an error with the message that starts with
    # "dataset contains splits with different document types:"
    with pytest.raises(ValueError) as excinfo:
        dataset_dict_different_types.document_type
    assert str(excinfo.value).startswith("dataset contains splits with different document types:")


def test_dataset_type(dataset_dict):
    assert dataset_dict.dataset_type is Dataset


def test_dataset_type_no_splits():
    with pytest.raises(ValueError) as excinfo:
        DatasetDict().dataset_type
    assert (
        str(excinfo.value)
        == "dataset does not contain any splits, cannot determine the dataset type"
    )


def test_dataset_type_different_type(dataset_dict, iterable_dataset_dict):
    dataset_dict_different_type = DatasetDict(
        {
            "train": dataset_dict["train"],
            "test": iterable_dataset_dict["test"],
        }
    )
    with pytest.raises(ValueError) as excinfo:
        dataset_dict_different_type.dataset_type
    assert str(excinfo.value).startswith("dataset contains splits with different dataset types:")


def map_fn(doc):
    doc.text = doc.text.upper()
    return doc


@pytest.mark.parametrize(
    "function",
    [map_fn, f"{TEST_MODULE}.map_fn"],
)
def test_map(dataset_dict, function):
    dataset_dict_mapped = dataset_dict.map(function)
    for split in dataset_dict:
        assert len(dataset_dict_mapped[split]) == len(dataset_dict[split])
        for doc1, doc2 in zip(dataset_dict_mapped[split], dataset_dict[split]):
            assert doc1.text == doc2.text.upper()


def test_map_noop(dataset_dict):
    dataset_dict_mapped = dataset_dict.map()
    for split in dataset_dict:
        assert len(dataset_dict_mapped[split]) == len(dataset_dict[split])
        for doc1, doc2 in zip(dataset_dict_mapped[split], dataset_dict[split]):
            assert doc1 == doc2


def test_map_with_result_document_type(dataset_dict):
    dataset_dict_mapped = dataset_dict.map(result_document_type=TextBasedDocument)
    for split in dataset_dict:
        assert len(dataset_dict_mapped[split]) == len(dataset_dict[split])
        for doc1, doc2 in zip(dataset_dict_mapped[split], dataset_dict[split]):
            assert isinstance(doc1, TextBasedDocument)
            assert isinstance(doc2, DocumentWithEntitiesAndRelations)
            assert doc1.text == doc2.text


def test_map_with_context_manager(dataset_dict):
    class DocumentCounter(
        EnterDatasetMixin, ExitDatasetMixin, EnterDatasetDictMixin, ExitDatasetDictMixin
    ):
        def reset_statistics(self):
            self.number = 0

        def __call__(self, doc):
            self.number += 1
            return doc

        def enter_dataset(
            self, dataset: Union[Dataset, IterableDataset], name: Optional[str] = None
        ) -> None:
            self.reset_statistics()
            self.split = name

        def exit_dataset(
            self, dataset: Union[Dataset, IterableDataset], name: Optional[str] = None
        ) -> None:
            self.all_docs[self.split] = self.number

        def enter_dataset_dict(self, dataset_dict: DatasetDict) -> None:
            self.all_docs: Dict[Optional[str], int] = {}
            self.split = None

        def exit_dataset_dict(self, dataset_dict: DatasetDict) -> None:
            logger.info(f"Number of documents per split: {self.all_docs}")

    document_counter = DocumentCounter()
    # note that we need to disable caching here, otherwise the __call__ method may not be called for any dataset split
    dataset_dict_mapped = dataset_dict.map(function=document_counter, load_from_cache_file=False)
    assert document_counter.all_docs == {"train": 3, "test": 3, "validation": 3}

    # the document_counter should not have been modified the dataset
    assert set(dataset_dict_mapped) == set(dataset_dict)
    for split in dataset_dict:
        assert len(dataset_dict_mapped[split]) == len(dataset_dict[split])
        for doc1, doc2 in zip(dataset_dict_mapped[split], dataset_dict[split]):
            assert doc1 == doc2


def test_map_set_max_batch_size(dataset_dict):
    def join_docs(docs):
        return [TextBasedDocument(text=" ".join([doc.text for doc in docs]))]

    dataset_dict_mapped = dataset_dict.map(
        join_docs,
        batched=True,
        set_batch_size_to_split_size=True,
        result_document_type=TextBasedDocument,
    )
    assert dataset_dict_mapped.document_type is TextBasedDocument
    for split in dataset_dict:
        assert len(dataset_dict_mapped[split]) == 1
        new_doc = dataset_dict_mapped[split][0]
        assert isinstance(new_doc, TextBasedDocument)
        original_texts = [doc.text for doc in dataset_dict[split]]
        assert new_doc.text == " ".join(original_texts)


def test_select(dataset_dict):
    # select documents by index
    dataset_dict_selected = dataset_dict.select(
        split="train",
        indices=[0, 2],
    )
    assert len(dataset_dict_selected["train"]) == 2
    assert dataset_dict_selected["train"][0] == dataset_dict["train"][0]
    assert dataset_dict_selected["train"][1] == dataset_dict["train"][2]

    # select documents by range
    dataset_dict_selected = dataset_dict.select(
        split="train",
        stop=2,
        start=1,
        step=1,
    )
    assert len(dataset_dict_selected["train"]) == 1
    assert dataset_dict_selected["train"][0] == dataset_dict["train"][1]

    # calling with no arguments that do result in the creation of indices should return the same dataset,
    # but will log a warning if other arguments (here "any_arg") are passed
    dataset_dict_selected = dataset_dict.select(split="train", any_arg="ignored")
    assert len(dataset_dict_selected["train"]) == len(dataset_dict["train"])
    assert dataset_dict_selected["train"][0] == dataset_dict["train"][0]
    assert dataset_dict_selected["train"][1] == dataset_dict["train"][1]
    assert dataset_dict_selected["train"][2] == dataset_dict["train"][2]


def test_rename_splits(dataset_dict):
    mapping = {
        "train": "train_renamed",
        "test": "test_renamed",
        "validation": "validation_renamed",
    }
    dataset_dict_renamed = dataset_dict.rename_splits(mapping)
    assert set(dataset_dict_renamed) == set(mapping.values())
    for split in dataset_dict:
        split_renamed = mapping[split]
        assert len(dataset_dict_renamed[split_renamed]) == len(dataset_dict[split])
        for doc1, doc2 in zip(dataset_dict_renamed[split_renamed], dataset_dict[split]):
            assert doc1 == doc2


def test_rename_split_noop(dataset_dict):
    dataset_dict_renamed = dataset_dict.rename_splits()
    assert set(dataset_dict_renamed) == set(dataset_dict)
    for split in dataset_dict:
        assert len(dataset_dict_renamed[split]) == len(dataset_dict[split])
        for doc1, doc2 in zip(dataset_dict_renamed[split], dataset_dict[split]):
            assert doc1 == doc2


def assert_doc_lists_equal(docs: Iterable[Document], other_docs: Iterable[Document]):
    assert all(doc1 == doc2 for doc1, doc2 in zip(docs, other_docs))


def test_add_test_split(dataset_dict):
    dataset_dict_with_test = dataset_dict.add_test_split(
        source_split="test", target_split="new_test", test_size=1, shuffle=False
    )
    assert "new_test" in dataset_dict_with_test
    assert len(dataset_dict_with_test["new_test"]) + len(dataset_dict_with_test["test"]) == len(
        dataset_dict["test"]
    )
    assert len(dataset_dict_with_test["new_test"]) == 1
    assert len(dataset_dict_with_test["test"]) == 2
    assert_doc_lists_equal(dataset_dict_with_test["new_test"], dataset_dict["test"][2:])
    assert_doc_lists_equal(dataset_dict_with_test["test"], dataset_dict["test"][:2])
    test_ids = [doc.id for doc in dataset_dict_with_test["test"]]
    new_test_ids = [doc.id for doc in dataset_dict_with_test["new_test"]]
    assert set(test_ids).intersection(set(new_test_ids)) == set()

    # remaining splits should be unchanged
    assert len(dataset_dict_with_test["train"]) == len(dataset_dict["train"])
    assert len(dataset_dict_with_test["validation"]) == len(dataset_dict["validation"])
    assert_doc_lists_equal(dataset_dict_with_test["train"], dataset_dict["train"])
    assert_doc_lists_equal(dataset_dict_with_test["validation"], dataset_dict["validation"])


def test_drop_splits(dataset_dict):
    dataset_dict_dropped = dataset_dict.drop_splits(["train", "validation"])
    assert set(dataset_dict_dropped) == {"test"}
    assert len(dataset_dict_dropped["test"]) == len(dataset_dict["test"])
    assert_doc_lists_equal(dataset_dict_dropped["test"], dataset_dict["test"])


def test_concat_splits(dataset_dict):
    dataset_dict_concatenated = dataset_dict.concat_splits(["train", "validation"], target="train")
    assert set(dataset_dict_concatenated) == {"test", "train"}
    assert len(dataset_dict_concatenated["train"]) == len(dataset_dict["train"]) + len(
        dataset_dict["validation"]
    )
    assert_doc_lists_equal(
        dataset_dict_concatenated["train"],
        list(dataset_dict["train"]) + list(dataset_dict["validation"]),
    )


def test_concat_splits_no_splits(dataset_dict):
    with pytest.raises(ValueError) as excinfo:
        dataset_dict.concat_splits(splits=[], target="train")
    assert str(excinfo.value) == "please provide at least one split to concatenate"


def test_concat_splits_different_dataset_types(dataset_dict, iterable_dataset_dict):
    dataset_dict_to_concat = DatasetDict(
        {
            "train": dataset_dict["train"],
            "validation": iterable_dataset_dict["validation"],
        }
    )
    with pytest.raises(ValueError) as excinfo:
        dataset_dict_to_concat.concat_splits(splits=["train", "validation"], target="train")
    assert str(excinfo.value).startswith("dataset contains splits with different dataset types:")


def test_filter(dataset_dict):
    dataset_dict_filtered = dataset_dict.filter(
        function=lambda doc: len(doc["text"]) > 15,
        split="train",
    )
    assert all(len(doc.text) > 15 for doc in dataset_dict_filtered["train"])
    assert len(dataset_dict["train"]) == N_FIXTURE_SAMPLES
    assert len(dataset_dict_filtered["train"]) == 2
    assert dataset_dict_filtered["train"][0] == dataset_dict["train"][0]
    assert dataset_dict_filtered["train"][1] == dataset_dict["train"][2]

    # remaining splits should be unchanged
    assert (
        len(dataset_dict_filtered["validation"])
        == len(dataset_dict["validation"])
        == N_FIXTURE_SAMPLES
    )
    assert len(dataset_dict_filtered["test"]) == len(dataset_dict["test"]) == N_FIXTURE_SAMPLES
    assert_doc_lists_equal(dataset_dict_filtered["validation"], dataset_dict["validation"])
    assert_doc_lists_equal(dataset_dict_filtered["test"], dataset_dict["test"])


def test_filter_iterable(iterable_dataset_dict):
    dataset_dict_filtered = iterable_dataset_dict.filter(
        function=lambda doc: len(doc["text"]) > 15,
        split="train",
    )
    docs_train = list(dataset_dict_filtered["train"])
    assert len(docs_train) == 2
    assert all(len(doc.text) > 15 for doc in docs_train)


def test_filter_unknown_dataset_type():
    dataset_dict = DatasetDict({"train": "foo"})
    with pytest.raises(TypeError) as excinfo:
        dataset_dict.filter(function=lambda doc: True, split="train")
    assert str(excinfo.value) == "dataset must be of type Dataset, but is <class 'str'>"


def test_filter_noop(dataset_dict):
    # passing no filter function should be a noop
    dataset_dict_filtered = dataset_dict.filter(split="train")
    assert len(dataset_dict_filtered["train"]) == len(dataset_dict["train"]) == N_FIXTURE_SAMPLES
    assert (
        len(dataset_dict_filtered["validation"])
        == len(dataset_dict["validation"])
        == N_FIXTURE_SAMPLES
    )
    assert len(dataset_dict_filtered["test"]) == len(dataset_dict["test"]) == N_FIXTURE_SAMPLES
    assert_doc_lists_equal(dataset_dict_filtered["train"], dataset_dict["train"])
    assert_doc_lists_equal(dataset_dict_filtered["validation"], dataset_dict["validation"])
    assert_doc_lists_equal(dataset_dict_filtered["test"], dataset_dict["test"])


@pytest.mark.parametrize(
    # we can either provide ids or a filter function
    "ids,filter_function",
    [
        (["1", "2"], None),
        (None, lambda doc: doc["id"] in ["1", "2"]),
    ],
)
def test_move_to_new_split(dataset_dict, ids, filter_function):
    # move the second and third document from train to new_validation
    dataset_dict_moved = dataset_dict.move_to_new_split(
        ids=ids,
        filter_function=filter_function,
        source_split="train",
        target_split="new_validation",
    )
    assert len(dataset_dict_moved["train"]) == 1
    assert len(dataset_dict_moved["new_validation"]) == 2
    assert_doc_lists_equal(dataset_dict_moved["train"], dataset_dict["train"][:1])

    # the remaining splits should be unchanged
    assert (
        len(dataset_dict_moved["validation"])
        == len(dataset_dict["validation"])
        == N_FIXTURE_SAMPLES
    )
    assert len(dataset_dict_moved["test"]) == len(dataset_dict["test"]) == N_FIXTURE_SAMPLES
    assert_doc_lists_equal(dataset_dict_moved["validation"], dataset_dict["validation"])
    assert_doc_lists_equal(dataset_dict_moved["test"], dataset_dict["test"])


def test_move_to_new_split_missing_arguments(dataset_dict):
    with pytest.raises(ValueError) as excinfo:
        dataset_dict.move_to_new_split(
            ids=None,
            filter_function=None,
            source_split="train",
            target_split="new_validation",
        )
    assert str(excinfo.value) == "please provide either a list of ids or a filter function"


def test_move_shard_to_new_split(dataset_dict):
    ids_per_split_original = {}
    for split in dataset_dict:
        ids_per_split_original[split] = [doc.id for doc in dataset_dict[split]]
    assert ids_per_split_original == {
        "test": ["0", "1", "2"],
        "train": ["0", "1", "2"],
        "validation": ["0", "1", "2"],
    }
    # move the second and third document from train to new_validation
    dataset_dict_moved = dataset_dict.move_shard_to_new_split(
        source_split="train",
        target_split="new_validation",
        num_shards=3,
        index=2,
        contiguous=True,
    )
    ids_per_split = {}
    for split in dataset_dict_moved:
        ids_per_split[split] = [doc.id for doc in dataset_dict_moved[split]]
    assert ids_per_split == {
        "train": ["0", "1"],
        "validation": ["0", "1", "2"],
        "test": ["0", "1", "2"],
        "new_validation": ["2"],
    }

    assert len(dataset_dict_moved["train"]) == 2
    assert len(dataset_dict_moved["new_validation"]) == 1
    assert_doc_lists_equal(
        list(dataset_dict_moved["train"]) + list(dataset_dict_moved["new_validation"]),
        dataset_dict["train"],
    )

    # the remaining splits should be unchanged
    assert (
        len(dataset_dict_moved["validation"])
        == len(dataset_dict["validation"])
        == N_FIXTURE_SAMPLES
    )
    assert len(dataset_dict_moved["test"]) == len(dataset_dict["test"]) == N_FIXTURE_SAMPLES
    assert_doc_lists_equal(dataset_dict_moved["validation"], dataset_dict["validation"])
    assert_doc_lists_equal(dataset_dict_moved["test"], dataset_dict["test"])


def test_cast_document_type(dataset_dict):
    dataset_dict_cast = dataset_dict.cast_document_type(TextBasedDocument)
    assert dataset_dict_cast.document_type == TextBasedDocument
    for split in dataset_dict_cast:
        assert all(isinstance(doc, TextBasedDocument) for doc in dataset_dict_cast[split])


@dataclass
class TestDocumentWithLabel(TextBasedDocument):
    label: AnnotationLayer[Label] = annotation_field()


def convert_to_document_with_label(document: TestDocument) -> TestDocumentWithLabel:
    result = TestDocumentWithLabel(text=document.text)
    result.label.append(Label(label="label"))
    return result


def test_register_document_converter(dataset_dict):
    dataset_dict.register_document_converter(
        convert_to_document_with_label, document_type=TestDocumentWithLabel
    )

    for name, split in dataset_dict.items():
        assert split.document_converters[TestDocumentWithLabel] == convert_to_document_with_label


def test_register_document_converter_resolve(dataset_dict):
    dataset_dict.register_document_converter(
        f"{TEST_MODULE}.convert_to_document_with_label",
        document_type=f"{TEST_MODULE}.TestDocumentWithLabel",
    )

    for name, split in dataset_dict.items():
        assert split.document_converters[TestDocumentWithLabel] == convert_to_document_with_label


class NoDocument:
    pass


def test_register_document_converter_resolve_wrong_document_type(dataset_dict):
    with pytest.raises(TypeError) as excinfo:
        dataset_dict.register_document_converter(
            convert_to_document_with_label, document_type=f"{TEST_MODULE}.NoDocument"
        )
    assert (
        str(excinfo.value)
        == f"document_type must be or resolve to a subclass of Document, but is '{TEST_MODULE}.NoDocument'"
    )


def test_register_document_converter_resolve_wrong_converter(dataset_dict):
    with pytest.raises(TypeError) as excinfo:
        dataset_dict.register_document_converter([1, 2, 3], document_type=TestDocumentWithLabel)
    assert str(excinfo.value) == "converter must be a callable or a dict, but is <class 'list'>"


@pytest.mark.parametrize("downcast", [True, False, None])
def test_to_document_type(dataset_dict, downcast):
    dataset_dict.register_document_converter(convert_to_document_with_label)
    if downcast is None:
        dataset_dict_converted = dataset_dict.to_document_type(TestDocumentWithLabel)
    else:
        dataset_dict_converted = dataset_dict.to_document_type(
            TestDocumentWithLabel, downcast=downcast
        )
    assert dataset_dict_converted.document_type == TestDocumentWithLabel
    for split in dataset_dict_converted.values():
        assert all(isinstance(doc, TestDocumentWithLabel) for doc in split)


def test_to_document_type_from_with_document_type(dataset_dict, caplog):
    class MyWithDocumentType(WithDocumentTypeMixin):
        DOCUMENT_TYPE = TestDocumentWithLabel

    with_document_type = MyWithDocumentType()
    dataset_dict.register_document_converter(convert_to_document_with_label)
    with caplog.at_level(logging.INFO):
        dataset_dict_converted = dataset_dict.to_document_type(with_document_type)
    assert (
        caplog.messages[-1]
        == f"convert the dataset to the document type that is specified by MyWithDocumentType: {TestDocumentWithLabel}"
    )
    assert dataset_dict_converted.document_type == TestDocumentWithLabel
    for split in dataset_dict_converted.values():
        assert all(isinstance(doc, TestDocumentWithLabel) for doc in split)


def test_to_document_type_from_with_document_type_no_document_type(dataset_dict, caplog):
    class MyWithDocumentType(WithDocumentTypeMixin):
        pass

    without_document_type = MyWithDocumentType()

    with caplog.at_level(logging.WARNING):
        dataset_dict_converted = dataset_dict.to_document_type(without_document_type)
    assert caplog.messages == [
        "MyWithDocumentType does not specify a document type. The dataset can not be automatically converted to a document type."
    ]
    assert dataset_dict_converted.document_type == dataset_dict.document_type


def test_to_document_type_resolve(dataset_dict):
    dataset_dict.register_document_converter(convert_to_document_with_label)
    dataset_dict_converted = dataset_dict.to_document_type(f"{TEST_MODULE}.TestDocumentWithLabel")
    assert dataset_dict_converted.document_type == TestDocumentWithLabel
    for split in dataset_dict_converted.values():
        assert all(isinstance(doc, TestDocumentWithLabel) for doc in split)


def test_to_document_type_resolve_wrong_document_type(dataset_dict):
    dataset_dict.register_document_converter(convert_to_document_with_label)
    with pytest.raises(TypeError) as excinfo:
        dataset_dict.to_document_type(f"{TEST_MODULE}.NoDocument")
    assert (
        str(excinfo.value)
        == f"document_type must be a document type or a string that can be resolved to such a type, but got "
        f"{TEST_MODULE}.NoDocument."
    )


def test_to_document_type_noop(dataset_dict):
    assert dataset_dict.document_type == DocumentWithEntitiesAndRelations
    dataset_dict_converted = dataset_dict.to_document_type(DocumentWithEntitiesAndRelations)
    assert dataset_dict_converted.document_type == DocumentWithEntitiesAndRelations
    assert dataset_dict_converted == dataset_dict


def test_to_document_type_dont_downcast_noop(dataset_dict, caplog):
    @dataclass
    class DocumentWithEntitiesRelationsAndPartitions(DocumentWithEntitiesAndRelations):
        partitions: AnnotationLayer[LabeledSpan] = annotation_field(target="text")

    # nothing should happen, since the document is a superclass of the requested type
    with caplog.at_level(logging.INFO):
        dataset_dict_converted = dataset_dict.to_document_type(
            DocumentWithEntitiesRelationsAndPartitions, downcast=False
        )
    assert caplog.messages == [
        f"The dataset has already the requested document type {DocumentWithEntitiesRelationsAndPartitions}."
    ]
    assert issubclass(
        DocumentWithEntitiesRelationsAndPartitions, dataset_dict_converted.document_type
    )


def test_load_dataset_conll2003():
    dataset_dict = load_dataset("pie/conll2003")
    assert isinstance(dataset_dict, DatasetDict)
    assert set(dataset_dict) == {"train", "test", "validation"}
    split_sizes = {split: len(dataset_dict[split]) for split in dataset_dict}
    assert split_sizes == {"train": 14041, "test": 3453, "validation": 3250}
    doc = dataset_dict["train"][0]
    assert isinstance(doc, TextBasedDocument)
    assert doc.text == "EU rejects German call to boycott British lamb ."
    resolved_entities = [(str(ent), ent.label) for ent in doc.entities]
    assert resolved_entities == [("EU", "ORG"), ("German", "MISC"), ("British", "MISC")]


def test_load_dataset_conll2003_single_split():
    dataset = load_dataset("pie/conll2003", split="train")
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 14041
    doc = dataset[0]
    assert isinstance(doc, TextBasedDocument)
    assert doc.text == "EU rejects German call to boycott British lamb ."
    resolved_entities = [(str(ent), ent.label) for ent in doc.entities]
    assert resolved_entities == [("EU", "ORG"), ("German", "MISC"), ("British", "MISC")]


def test_load_dataset_conll2003_wrong_type():
    with pytest.raises(TypeError) as excinfo:
        load_dataset("conll2003")
    assert (
        str(excinfo.value)
        == "expected all splits to be <class 'pie_datasets.core.dataset.Dataset'> or "
        "<class 'pie_datasets.core.dataset.IterableDataset'>, but split \"train\" is of type "
        "<class 'datasets.arrow_dataset.Dataset'>"
    )


def test_load_dataset_conll2003_wrong_type_single_split():
    with pytest.raises(TypeError) as excinfo:
        load_dataset("conll2003", split="train")
    assert (
        str(excinfo.value)
        == "expected datasets.load_dataset to return <class 'datasets.dataset_dict.DatasetDict'>, "
        "<class 'datasets.dataset_dict.IterableDatasetDict'>, <class 'pie_datasets.core.dataset.Dataset'>, "
        "or <class 'pie_datasets.core.dataset.IterableDataset'>, but got <class 'datasets.arrow_dataset.Dataset'>"
    )


@pytest.fixture
def tbga_extract():
    return DatasetDict.from_json(data_dir=FIXTURES_ROOT / "dataset_dict" / "tbga_extract")


@pytest.fixture
def comagc_extract():
    return DatasetDict.from_json(data_dir=FIXTURES_ROOT / "dataset_dict" / "comagc_extract")


def test_concatenate_dataset_dicts(tbga_extract, comagc_extract):
    concatenated_dataset = concatenate_dataset_dicts(
        inputs={"tbga": tbga_extract, "comagc": comagc_extract},
        split_mappings={"train": {"tbga": "train", "comagc": "train"}},
        clear_metadata=True,
    )

    assert len(concatenated_dataset["train"]) == len(tbga_extract["train"]) + len(
        comagc_extract["train"]
    )
    assert all(
        [ds.metadata["dataset_name"] in ["tbga", "comagc"] for ds in concatenated_dataset["train"]]
    )

    concatenated_dataset_with_list_in_mapping = concatenate_dataset_dicts(
        inputs={"tbga": tbga_extract, "comagc": comagc_extract},
        split_mappings={"train": {"tbga": ["train", "test"], "comagc": "train"}},
        clear_metadata=True,
    )

    assert len(concatenated_dataset_with_list_in_mapping["train"]) == len(
        tbga_extract["train"]
    ) + len(tbga_extract["test"]) + len(comagc_extract["train"])
    assert all(
        [
            ds.metadata["dataset_name"] in ["tbga", "comagc"]
            for ds in concatenated_dataset_with_list_in_mapping["train"]
        ]
    )
