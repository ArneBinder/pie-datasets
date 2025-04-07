import datasets
import pytest
from pie_core import Document
from pie_modules.documents import ExtractiveQADocument

from dataset_builders.pie.squad_v2.squad_v2 import SquadV2
from pie_datasets import Dataset, IterableDataset, load_dataset
from tests.dataset_builders.common import PIE_BASE_PATH

datasets.disable_caching()

DATASET_NAME = "squad_v2"
BUILDER_CLASS = SquadV2
DOCUMENT_TYPE = SquadV2.DOCUMENT_TYPE
SPLIT_SIZES = {"train": 130319, "validation": 11873}
HF_DATASET_PATH = BUILDER_CLASS.BASE_DATASET_PATH
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME

# fast testing parameters
SPLIT = "train"
STREAM_SIZE = 3


@pytest.fixture(scope="module", params=list(SPLIT_SIZES))
def split(request):
    return request.param


@pytest.fixture(scope="module")
def hf_dataset(split):
    return datasets.load_dataset(str(HF_DATASET_PATH), split=split)


@pytest.mark.slow
def test_hf_dataset(hf_dataset, split):
    assert hf_dataset is not None
    assert len(hf_dataset) == SPLIT_SIZES[split]


@pytest.fixture(scope="module")
def hf_dataset_fast() -> datasets.IterableDataset:
    return datasets.load_dataset(str(HF_DATASET_PATH), split=SPLIT, streaming=True).take(
        STREAM_SIZE
    )


def test_hf_dataset_fast(hf_dataset_fast):
    assert hf_dataset_fast is not None


@pytest.fixture(scope="module")
def hf_example_fast(hf_dataset_fast):
    return list(hf_dataset_fast)[0]


def test_hf_example(hf_example_fast, split):
    assert hf_example_fast is not None
    assert hf_example_fast == {
        "id": "56be85543aeaaa14008c9063",
        "title": "Beyoncé",
        "context": "Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an "
        "American singer, songwriter, record producer and actress. Born and raised in Houston, "
        "Texas, she performed in various singing and dancing competitions as a child, and rose to "
        "fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her "
        "father, Mathew Knowles, the group became one of the world's best-selling girl groups of "
        "all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love "
        "(2003), which established her as a solo artist worldwide, earned five Grammy Awards and "
        'featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".',
        "question": "When did Beyonce start becoming popular?",
        "answers": {"text": ["in the late 1990s"], "answer_start": [269]},
    }


@pytest.fixture(scope="module")
def generated_document_fast(hf_example_fast) -> DOCUMENT_TYPE:
    return BUILDER_CLASS()._generate_document(hf_example_fast)


def test_generated_document_fast(generated_document_fast):
    assert isinstance(generated_document_fast, DOCUMENT_TYPE)

    assert generated_document_fast.id == "56be85543aeaaa14008c9063"
    assert generated_document_fast.title == "Beyoncé"
    assert generated_document_fast.text.startswith(
        "Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981)"
    )
    assert len(generated_document_fast.questions) == 1
    assert str(generated_document_fast.questions[0]) == "When did Beyonce start becoming popular?"

    assert len(generated_document_fast.answers) == 1
    assert str(generated_document_fast.answers[0]) == "in the late 1990s"


def test_example_to_document_and_back_fast(hf_example_fast, generated_document_fast):
    hf_example_back = BUILDER_CLASS()._generate_example(generated_document_fast)
    assert hf_example_back == hf_example_fast


@pytest.mark.slow
def test_example_to_document_and_back_all(hf_dataset):
    builder = BUILDER_CLASS()
    for hf_ex in hf_dataset:
        doc = builder._generate_document(hf_ex)
        hf_ex_back = builder._generate_example(doc)
        assert hf_ex_back == hf_ex


@pytest.fixture(scope="module")
def pie_dataset(split) -> Dataset:
    return load_dataset(str(PIE_DATASET_PATH), split=split)


@pytest.mark.slow
def test_pie_dataset(pie_dataset, split):
    assert pie_dataset is not None
    assert len(pie_dataset) == SPLIT_SIZES[split]


@pytest.fixture(scope="module")
def pie_dataset_fast() -> IterableDataset:
    return load_dataset(str(PIE_DATASET_PATH), split=SPLIT, streaming=True).take(STREAM_SIZE)


def test_pie_dataset_fast(pie_dataset_fast):
    assert pie_dataset is not None


@pytest.fixture(scope="module")
def document_fast(pie_dataset_fast) -> DOCUMENT_TYPE:
    doc = list(pie_dataset_fast)[0]
    # we can not assert the real document type because it may come from a dataset loading script
    # downloaded to a temporary directory and thus have a different type object, although it is
    # semantically the same
    assert isinstance(doc, Document)
    casted = doc.as_type(DOCUMENT_TYPE)
    return casted


def test_compare_document_and_generated_document(document_fast, generated_document_fast):
    assert document_fast == generated_document_fast


def test_dataset_with_extractive_qa_documents_fast(pie_dataset_fast, document_fast):
    dataset_with_extractive_qa_documents_fast = pie_dataset_fast.to_document_type(
        ExtractiveQADocument
    )
    assert dataset_with_extractive_qa_documents_fast is not None
    doc = list(dataset_with_extractive_qa_documents_fast)[0]
    assert isinstance(doc, ExtractiveQADocument)
    doc_casted = document_fast.as_type(ExtractiveQADocument)
    assert doc == doc_casted


@pytest.mark.slow
def test_dataset_with_extractive_qa_documents_all(pie_dataset):
    dataset_with_extractive_qa_documents = pie_dataset.to_document_type(ExtractiveQADocument)
    assert dataset_with_extractive_qa_documents is not None
    for doc in dataset_with_extractive_qa_documents:
        assert isinstance(doc, ExtractiveQADocument)
