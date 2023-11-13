# pie-datasets

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://github.com/ChristophAlt/pytorch-ie"><img alt="PyTorch-IE" src="https://img.shields.io/badge/-PyTorch--IE-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

[![PyPI](https://img.shields.io/pypi/v/pie-datasets.svg)][pypi status]
[![Tests](https://github.com/arnebinder/pie-datasets/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/arnebinder/pie-datasets/branch/main/graph/badge.svg)][codecov]
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

Dataset building scripts and utilities for [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie). We parse all datasets into a common format that can be
loaded directly from the Huggingface Hub. Taking advantage of
[Huggingface datasets](https://huggingface.co/docs/datasets), the documents are cached in an arrow table and
serialized / deserialized on the fly. Any changes or preprocessing applied to the documents will be cached as well.

## Setup

```bash
pip install pie-datasets
```

To install the latest version from GitHub:

```bash
pip install git+https://git@github.com/ArneBinder/pie-datasets.git
```

## Available datasets

See [here](https://huggingface.co/pie) for a list of available datasets. Note, that you can easily add your own
datasets by following the [instructions below](#how-to-create-your-own-pie-dataset).

## Usage

### General

```python
from pie_datasets import load_dataset

# load the dataset from https://huggingface.co/datasets/pie/conll2003
dataset = load_dataset("pie/conll2003")

print(dataset["train"][0])
# >>> CoNLL2003Document(text='EU rejects German call to boycott British lamb .', id='0', metadata={})

dataset["train"][0].entities
# >>> AnnotationLayer([LabeledSpan(start=0, end=2, label='ORG', score=1.0), LabeledSpan(start=11, end=17, label='MISC', score=1.0), LabeledSpan(start=34, end=41, label='MISC', score=1.0)])

entity = dataset["train"][0].entities[1]

print(f"[{entity.start}, {entity.end}] {entity}")
# >>> [11, 17] German

{name: len(split) for name, split in dataset.items()}
# >>> {'train': 14041, 'validation': 3250, 'test': 3453}
```

### Adjusting splits

Similar to [Huggingface datasets](https://huggingface.co/docs/datasets), you can adjust the splits of a dataset in
various ways. Here are some examples:

```python
from pie_datasets import load_dataset

dataset = load_dataset("pie/conll2003")

# re-create a validation split from train split concatenated with the original validation split
dataset_with_new_val = dataset.concat_splits(
   ["train", "validation"], target="train"
).add_test_split(
   source_split="train", target_split="my_validation", test_size=0.2, seed=42
)
{name: len(split) for name, split in dataset_with_new_val.items()}
# >>> {'test': 3453, 'train': 13832, 'my_validation': 3459}

# drop the test split
dataset_without_test = dataset_with_new_val.drop_splits(["test"])
{name: len(split) for name, split in dataset_without_test.items()}
# >>> {'train': 13832, 'my_validation': 3459}
```

### Adjusting dataset entries

Calling `map` on the dataset will apply the given function to all its documents. Internally, that relies
on [datasets.Dataset.map](https://huggingface.co/docs/datasets/v2.4.0/package_reference/main_classes.html#datasets.Dataset.map).
Thus, the function can be any function that takes a document as input and returns a document as output. If the
function returns a different document type, you need to provide it as `result_document_type` argument to
`map`. Note, that **the result is cached for each split, so that re-running the same function on the
same dataset will be a no-op**.

Example where the function returns the same document type:

```python
from pie_datasets import load_dataset

def duplicate_entities(document):
    new_document = document.copy()
    for entity in document.entities:
        # we need to copy the entity because each annotation can only be part of one document
        new_document.entities.append(entity.copy())
    return new_document

dataset = load_dataset("pie/conll2003")
len(dataset["train"][0].entities)
# >>> 3

converted_dataset = dataset.map(duplicate_entities)
# Map: 100%|██████████| 14041/14041 [00:02<00:00, 4697.18 examples/s]
# Map: 100%|██████████| 3250/3250 [00:00<00:00, 4583.95 examples/s]
# Map: 100%|██████████| 3453/3453 [00:00<00:00, 4614.67 examples/s]
len(converted_dataset["train"][0].entities)
# >>> 6
```

Example where the function returns a different document type:

```python
from dataclasses import dataclass

from pytorch_ie.core import AnnotationLayer, annotation_field
from pytorch_ie.documents import TextBasedDocument
from pytorch_ie.annotations import LabeledSpan, Span
from pie_datasets import load_dataset

@dataclass
class CoNLL2003DocumentWithWords(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    words: AnnotationLayer[Span] = annotation_field(target="text")

def add_words(document) -> CoNLL2003DocumentWithWords:
    new_document = CoNLL2003DocumentWithWords(text=document.text, id=document.id)
    for entity in document.entities:
        new_document.entities.append(entity.copy())
    start = 0
    for word in document.text.split():
        word_start = document.text.index(word, start)
        word_annotation = Span(start=word_start, end=word_start + len(word))
        new_document.words.append(word_annotation)
    return new_document

dataset = load_dataset("pie/conll2003")
dataset.document_type
# >>> <class 'datasets_modules.datasets.pie--conll2003.821bfce48d2ebc3533db067c4d8e89396155c65cd311d2341a82acf81f561885.conll2003.CoNLL2003Document'>

converted_dataset = dataset.map(add_words, result_document_type=CoNLL2003DocumentWithWords)
# Map: 100%|██████████| 14041/14041 [00:03<00:00, 3902.00 examples/s]
# Map: 100%|██████████| 3250/3250 [00:00<00:00, 3929.52 examples/s]
# Map: 100%|██████████| 3453/3453 [00:00<00:00, 3947.49 examples/s]

converted_dataset.document_type
# >>> <class '__main__.CoNLL2003DocumentWithWords'>

converted_dataset["train"][0].words
# >>> AnnotationLayer([Span(start=0, end=2), Span(start=3, end=10), Span(start=11, end=17), Span(start=18, end=22), Span(start=23, end=25), Span(start=26, end=33), Span(start=34, end=41), Span(start=42, end=46), Span(start=47, end=48)])

[str(word) for word in converted_dataset["train"][0].words]
# >>> ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
```

We can also **register a document converter** for a specific document type. This will be used when calling
`to_document_type` with the respective document type. The following code will produce the same result
as the previous one:

```python
dataset = load_dataset("pie/conll2003")

# Register add_words as a converter function for the target document type CoNLL2003DocumentWithWords.
# Since add_words specifies the return type, we can omit the document type here.
dataset.register_document_converter(add_words)

# Determine the matching converter entry for the target document type and apply it with dataset.map.
converted_dataset = dataset.to_document_type(CoNLL2003DocumentWithWords)
```

Note, that some of the PIE datasets come with default document converters. For instance, the
[PIE conll2003 dataset](https://huggingface.co/datasets/pie/conll2003) comes with one that converts
the dataset to `pytorch_ie.documents.TextDocumentWithLabeledSpans`. These documents work with the
PIE taskmodules for
[token classification](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/taskmodules/transformer_token_classification.py)
and [span classification](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/taskmodules/transformer_span_classification.py)
out-of-the-box. The following code will load the dataset and convert it to the required document type:

```python
from pie_datasets import load_dataset
from pytorch_ie.taskmodules import TransformerTokenClassificationTaskModule

taskmodule = TransformerTokenClassificationTaskModule(tokenizer_name_or_path="bert-base-cased")
# the taskmodule expects TextDocumentWithLabeledSpans as input and the conll2003 dataset comes with a
# default converter for that document type. Thus, we can directly load the dataset and convert it.
dataset = load_dataset("pie/conll2003").to_document_type(taskmodule.document_type)
...
```

### How to create your own PIE dataset

PIE datasets are built on top of Huggingface datasets. For instance, consider
[conll2003 at the Huggingface Hub](https://huggingface.co/datasets/conll2003) and especially its respective
[dataset loading script](https://huggingface.co/datasets/conll2003/blob/main/conll2003.py). To create a PIE
dataset from that, you have to implement:

1. A Document class. This will be the type of individual dataset examples.

```python
from dataclasses import dataclass

from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationLayer, annotation_field
from pytorch_ie.documents import TextBasedDocument

@dataclass
class CoNLL2003Document(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
```

Here we derive from `TextBasedDocument` that has a simple `text` string as base annotation target. The
`CoNLL2003Document` adds one single annotation layer called `entities` that consists of `LabeledSpan`s which
reference the `text` field of the document. You can add further annotation types by adding `AnnotationLayer`
fields that may also reference (i.e. `target`) other annotations as you like. The package
[pytorch_ie.annotations](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py)
contains some predefined annotation types and the package
[pytorch_ie.documents](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) defines
some document types that you can use as base classes.

2. A dataset config. This is similar to
   [creating a Huggingface dataset config](https://huggingface.co/docs/datasets/dataset_script#multiple-configurations).

```python
import datasets

class CoNLL2003Config(datasets.BuilderConfig):
    """BuilderConfig for CoNLL2003"""

    def __init__(self, **kwargs):
        """BuilderConfig for CoNLL2003.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)
```

3. A dataset builder class. This should inherit from
   [`pie_datasets.GeneratorBasedBuilder`](src/pie_datasets/core/builder.py) which is a wrapper around the
   [Huggingface dataset builder class](https://huggingface.co/docs/datasets/v2.4.0/en/package_reference/builder_classes#datasets.GeneratorBasedBuilder)
   with some utility functionality to work with PyTorch-IE `Documents`. The key elements to implement are: `DOCUMENT_TYPE`,
   `BASE_DATASET_PATH`, and `_generate_document`.

```python

from pytorch_ie.documents import TextDocumentWithLabeledSpans
from pytorch_ie.utils.span import tokens_and_tags_to_text_and_labeled_spans
from pie_datasets import GeneratorBasedBuilder

class Conll2003(GeneratorBasedBuilder):
    # Specify the document type. This will be the class of individual dataset examples.
    DOCUMENT_TYPE = CoNLL2003Document

    # The Huggingface identifier that points to the base dataset. This may be any string that works
    # as path with Huggingface `datasets.load_dataset`.
    BASE_DATASET_PATH = "conll2003"
    # It is strongly recommended to also specify the revision (tag name, or branch name, or commit hash)
    # of the base dataset. This ensures that the dataset will not change unexpectedly when the base dataset
    # is updated.
    BASE_DATASET_REVISION = "01ad4ad271976c5258b9ed9b910469a806ff3288"

    # The builder configs, see https://huggingface.co/docs/datasets/dataset_script for further information.
    BUILDER_CONFIGS = [
        CoNLL2003Config(
            name="conll2003", version=datasets.Version("1.0.0"), description="CoNLL2003 dataset"
        ),
    ]

    # [Optional] Define additional keyword arguments which will be passed to `_generate_document` below.
    def _generate_document_kwargs(self, dataset):
        return {"int_to_str": dataset.features["ner_tags"].feature.int2str}

    # Define how a Pytorch-IE Document will be created from a Huggingface dataset example.
    def _generate_document(self, example, int_to_str):
        doc_id = example["id"]
        tokens = example["tokens"]
        ner_tags = [int_to_str(tag) for tag in example["ner_tags"]]

        text, ner_spans = tokens_and_tags_to_text_and_labeled_spans(tokens=tokens, tags=ner_tags)

        document = CoNLL2003Document(text=text, id=doc_id)

        for span in sorted(ner_spans, key=lambda span: span.start):
            document.entities.append(span)

        return document

    # [OPTIONAL] Define how the dataset will be converted to a different document type. Here, we add a
    # converter for the generic document type `TextDocumentWithLabeledSpans` that is used by the PIE
    # taskmodules for token and span classification. This allows to directly call
    # `pie_datasets.load_dataset("pie/conll2003").to_document_type(TextDocumentWithLabeledSpans)`.
    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpans: {
            # if the converter is a simple dictionary, just rename the layer according that
            "entities": "labeled_spans",
        }
    }
```

The full script can be found here: [dataset_builders/pie/conll2003/conll2003.py](dataset_builders/pie/conll2003/conll2003.py). Note, that to
load the dataset with `pie_datasets.load_dataset`, the script has to be located in a directory with the same name
(as it is the case for standard Huggingface dataset loading scripts).

## Development

### Setup

1. This project is build with [Poetry](https://python-poetry.org/). See here for [installation instructions](https://python-poetry.org/docs/#installation).
2. Get the code and switch into the project directory:
   ```bash
   git clone https://github.com/ArneBinder/pie-datasets
   cd pie-datasets
   ```
3. Create a virtual environment and install the dependencies:
   ```bash
   poetry install
   ```

Finally, to run any of the below commands, you need to activate the virtual environment:

```bash
poetry shell
```

Note: You can also run commands in the virtual environment without activating it first: `poetry run <command>`.

### Code Formatting, Linting and Static Type Checking

```bash
pre-commit run -a
```

### Testing

run all tests with coverage:

```bash
pytest --cov --cov-report term-missing
```

### Releasing

1. Create the release branch:
   `git switch --create release main`
2. Increase the version:
   `poetry version <PATCH|MINOR|MAJOR>`,
   e.g. `poetry version patch` for a patch release. If the release contains new features, or breaking changes,
   bump the minor version (this project has no main release yet). If the release contains only bugfixes, bump
   the patch version. See [Semantic Versioning](https://semver.org/) for more information.
3. Commit the changes:
   `git commit --message="release <NEW VERSION>" pyproject.toml`,
   e.g. `git commit --message="release 0.13.0" pyproject.toml`
4. Push the changes to GitHub:
   `git push origin release`
5. Create a PR for that `release` branch on GitHub.
6. Wait until checks passed successfully.
7. Merge the PR into the main branch. This triggers the GitHub Action that creates all relevant release
   artefacts and also uploads them to PyPI.
8. Cleanup: Delete the `release` branch. This is important, because otherwise the next release will fail.

[black]: https://github.com/psf/black
[codecov]: https://app.codecov.io/gh/arnebinder/pie-datasets
[pre-commit]: https://github.com/pre-commit/pre-commit
[pypi status]: https://pypi.org/project/pie-datasets/
[tests]: https://github.com/arnebinder/pie-datasets/actions?workflow=Tests
