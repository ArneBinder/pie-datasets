import pytest
from datasets import disable_caching, load_dataset
from pie_modules.documents import ExtractiveQADocument
from pytorch_ie.core import Document

from dataset_builders.pie.squad_v2.squad_v2 import SquadV2
from pie_datasets import DatasetDict
from tests.dataset_builders.common import PIE_BASE_PATH

disable_caching()

DATASET_NAME = "squad_v2"
BUILDER_CLASS = SquadV2
DOCUMENT_TYPE = SquadV2.DOCUMENT_TYPE
SPLIT_SIZES = {"train": 130319, "validation": 11873}
HF_DATASET_PATH = BUILDER_CLASS.BASE_DATASET_PATH
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME


@pytest.fixture(scope="module", params=list(SPLIT_SIZES))
def split(request):
    return request.param


@pytest.fixture(scope="module")
def hf_dataset():
    return load_dataset(str(HF_DATASET_PATH))


def test_hf_dataset(hf_dataset):
    assert hf_dataset is not None
    assert {name: len(ds) for name, ds in hf_dataset.items()} == SPLIT_SIZES


@pytest.fixture(scope="module")
def hf_example(hf_dataset, split):
    return hf_dataset[split][0]


def test_hf_example(hf_example, split):
    assert hf_example is not None
    if split == "train":
        assert hf_example == {
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
    elif split == "validation":
        assert hf_example == {
            "id": "56ddde6b9a695914005b9628",
            "title": "Normans",
            "context": "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in "
            "the 10th and 11th centuries gave their name to Normandy, a region in France. They were "
            'descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, '
            "Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles "
            "III of West Francia. Through generations of assimilation and mixing with the native "
            "Frankish and Roman-Gaulish populations, their descendants would gradually merge with the "
            "Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of "
            "the Normans emerged initially in the first half of the 10th century, and it continued to "
            "evolve over the succeeding centuries.",
            "question": "In what country is Normandy located?",
            "answers": {
                "text": ["France", "France", "France", "France"],
                "answer_start": [159, 159, 159, 159],
            },
        }
    else:
        raise ValueError(f"Unknown split: {split}")


@pytest.fixture(scope="module")
def generate_document_kwargs(hf_dataset, split) -> dict:
    return BUILDER_CLASS()._generate_document_kwargs(hf_dataset[split]) or {}


@pytest.fixture(scope="module")
def generate_example_kwargs(hf_dataset, split) -> dict:
    return BUILDER_CLASS()._generate_example_kwargs(hf_dataset[split]) or {}


@pytest.fixture(scope="module")
def generated_document(hf_example, generate_document_kwargs) -> DOCUMENT_TYPE:
    return BUILDER_CLASS()._generate_document(hf_example, **generate_document_kwargs)


def test_generated_document(generated_document, split):
    assert isinstance(generated_document, DOCUMENT_TYPE)
    if split == "train":
        assert generated_document.id == "56be85543aeaaa14008c9063"
        assert generated_document.title == "Beyoncé"
        assert generated_document.text.startswith(
            "Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981)"
        )
        assert len(generated_document.questions) == 1
        assert str(generated_document.questions[0]) == "When did Beyonce start becoming popular?"

        assert len(generated_document.answers) == 1
        assert str(generated_document.answers[0]) == "in the late 1990s"

    elif split == "validation":
        assert generated_document.id == "56ddde6b9a695914005b9628"
        assert generated_document.title == "Normans"
        assert generated_document.text.startswith(
            "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the "
            "10th and 11th centuries gave their name to Normandy, a region in France."
        )
        assert len(generated_document.questions) == 1
        assert str(generated_document.questions[0]) == "In what country is Normandy located?"

        assert len(generated_document.answers) == 4
        assert str(generated_document.answers[0]) == "France"
        assert str(generated_document.answers[1]) == "France"
        assert str(generated_document.answers[2]) == "France"
        assert str(generated_document.answers[3]) == "France"

    else:
        raise ValueError(f"Unknown split: {split}")


@pytest.fixture(scope="module")
def hf_example_back(generated_document, generate_document_kwargs):
    return BUILDER_CLASS()._generate_example(generated_document, **generate_document_kwargs)


def test_example_to_document_and_back(hf_example, hf_example_back):
    assert hf_example_back == hf_example


@pytest.mark.slow
def test_example_to_document_and_back_all(
    hf_dataset, generate_document_kwargs, generate_example_kwargs, split
):
    builder = BUILDER_CLASS()
    for hf_ex in hf_dataset[split]:
        doc = builder._generate_example(hf_ex, **generate_document_kwargs)
        hf_ex_back = builder._generate_example(doc, **generate_example_kwargs)
        assert hf_ex_back == hf_ex


@pytest.fixture(scope="module")
def dataset() -> DatasetDict:
    return DatasetDict.load_dataset(str(PIE_DATASET_PATH))


def test_pie_dataset(dataset):
    assert dataset is not None
    assert {name: len(ds) for name, ds in dataset.items()} == SPLIT_SIZES


@pytest.fixture(scope="module")
def document(dataset, split) -> DOCUMENT_TYPE:
    doc = dataset[split][0]
    # we can not assert the real document type because it may come from a dataset loading script
    # downloaded to a temporary directory and thus have a different type object, although it is
    # semantically the same
    assert isinstance(doc, Document)
    casted = doc.as_type(DOCUMENT_TYPE)
    return casted


def test_compare_document_and_generated_document(document, generated_document):
    assert document == generated_document


@pytest.fixture(scope="module")
def dataset_with_extractive_qa_documents(dataset) -> DatasetDict:
    return dataset.to_document_type(ExtractiveQADocument)


def test_dataset_with_extractive_qa_documents(
    dataset_with_extractive_qa_documents, document, split
):
    assert dataset_with_extractive_qa_documents is not None
    doc = dataset_with_extractive_qa_documents[split][0]
    assert isinstance(doc, ExtractiveQADocument)
    doc_casted = document.as_type(ExtractiveQADocument)
    assert doc == doc_casted
