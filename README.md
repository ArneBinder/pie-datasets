# pie-datasets

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://github.com/ChristophAlt/pytorch-ie"><img alt="PyTorch-IE" src="https://img.shields.io/badge/-PyTorch--IE-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

[![PyPI](https://img.shields.io/pypi/v/pie-datasets.svg)][pypi status]
[![Tests](https://github.com/arnebinder/pie-datasets/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/arnebinder/pie-datasets/branch/main/graph/badge.svg)][codecov]
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

Building Scripts for [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) Datasets, also see [here](https://huggingface.co/pie).

## Setup

```bash
pip install pie-datasets
```

To install the latest version from GitHub:

```bash
pip install git+https://git@github.com/ArneBinder/pie-datasets.git
```

## Usage

### Available PIE Datasets

```python
import datasets

dataset = datasets.load_dataset("pie/conll2003")

print(dataset["train"][0])
# >>> CoNLL2003Document(text='EU rejects German call to boycott British lamb .', id='0', metadata={})

dataset["train"][0].entities
# >>> AnnotationLayer([LabeledSpan(start=0, end=2, label='ORG', score=1.0), LabeledSpan(start=11, end=17, label='MISC', score=1.0), LabeledSpan(start=34, end=41, label='MISC', score=1.0)])

entity = dataset["train"][0].entities[1]

print(f"[{entity.start}, {entity.end}] {entity}")
# >>> [11, 17] German
```

### How to create your own PIE dataset

PIE datasets are built on top of Huggingface datasets. For instance, consider the
[conll2003 from the Huggingface Hub](https://huggingface.co/datasets/conll2003) and especially their respective
[dataset loading script](https://huggingface.co/datasets/conll2003/blob/main/conll2003.py). To create a PIE
dataset from that, you have to implement:

1. A Document class. This will be the type of individual dataset examples.

```python
from dataclasses import dataclass

from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationLayer, annotation_field
from pytorch_ie.documents import TextDocument

@dataclass
class CoNLL2003Document(TextDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
```

Here we derive from `TextDocument` that has a simple `text` string as base annotation target. The `CoNLL2003Document`
adds one single annotation list called `entities` that consists of `LabeledSpan`s which reference the `text` field of
the document. You can add further annotation types by adding `AnnotationLayer` fields that may also reference (i.e.
`target`) other annotations as you like. See ['pytorch_ie.annotations\`](src/pytorch_ie/annotations.py) for predefined
annotation types.

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
   [`pytorch_ie.data.builder.GeneratorBasedBuilder`](src/pytorch_ie/data/builder.py) which is a wrapper around the
   [Huggingface dataset builder class](https://huggingface.co/docs/datasets/v2.4.0/en/package_reference/builder_classes#datasets.GeneratorBasedBuilder)
   with some utility functionality to work with PyTorch-IE `Documents`. The key elements to implement are: `DOCUMENT_TYPE`,
   `BASE_DATASET_PATH`, and `_generate_document`.

```python

from pytorch_ie.utils.span import tokens_and_tags_to_text_and_labeled_spans
from pie_datasets import GeneratorBasedBuilder

class Conll2003(GeneratorBasedBuilder):
    # Specify the document type. This will be the class of individual dataset examples.
    DOCUMENT_TYPE = CoNLL2003Document

    # The Huggingface identifier that points to the base dataset. This may be any string that works
    # as path with Huggingface `datasets.load_dataset`.
    BASE_DATASET_PATH = "conll2003"

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
```

The full script can be found here: [dataset_builders/conll2003/conll2003.py](dataset_builders/conll2003/conll2003.py). Note, that to
load the dataset with `datasets.load_dataset`, the script has to be located in a directory with the same name (as it
is the case for standard Huggingface dataset loading scripts).

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
