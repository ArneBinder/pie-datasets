---
annotations_creators:
  - expert-generated
language_creators:
  - found
license: []
task_categories:
  - token-classification
task_ids:
  - parsing
---

# Information Card for Brat

## Table of Contents

- [Description](#description)
  - [Summary](#summary)
- [Dataset Structure](#dataset-structure)
- [Data Instances](#data-instances)
- [Data Fields](#data-instances)
- [Usage](#usage)
- [Additional Information](#additional-information)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)

## Description

- **Homepage:** https://brat.nlplab.org
- **Paper:** https://aclanthology.org/E12-2021/
- **Leaderboard:** \[Needs More Information\]
- **Point of Contact:** \[Needs More Information\]

### Summary

Brat is an intuitive web-based tool for text annotation supported by Natural Language Processing (NLP) technology. BRAT has been developed for rich structured annota- tion for a variety of NLP tasks and aims to support manual curation efforts and increase annotator productivity using NLP techniques. brat is designed in particular for structured annotation, where the notes are not free form text but have a fixed form that can be automatically processed and interpreted by a computer.

## Dataset Structure

Dataset annotated with brat format is processed using this script. Annotations created in brat are stored on disk in a standoff format: annotations are stored separately from the annotated document text, which is never modified by the tool. For each text document in the system, there is a corresponding annotation file. The two are associated by the file naming convention that their base name (file name without suffix) is the same: for example, the file DOC-1000.ann contains annotations for the file DOC-1000.txt. More information can be found [here](https://brat.nlplab.org/standoff.html).

### Data Instances

```
{
  "context": ''<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<Document xmlns:gate="http://www.gat...'
  "file_name": "A01"
  "spans": {
    'id': ['T1', 'T2', 'T4', 'T5', 'T6', 'T3', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',...]
    'type': ['background_claim', 'background_claim', 'background_claim', 'own_claim',...]
    'locations': [{'start': [2417], 'end': [2522]}, {'start': [2524], 'end': [2640]},...]
    'text': ['complicated 3D character models...', 'The range of breathtaking realistic...', ...]
   }
  "relations": {
    'id': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',...]
    'type': ['supports', 'supports', 'supports', 'supports', 'contradicts', 'contradicts',...]
    'arguments': [{'type': ['Arg1', 'Arg2'], 'target': ['T4', 'T5']},...]
  }
  "equivalence_relations": {'type': [], 'targets': []},
  "events": {'id': [], 'type': [], 'trigger': [], 'arguments': []},
  "attributions": {'id': [], 'type': [], 'target': [], 'value': []},
  "normalizations": {'id': [], 'type': [], 'target': [], 'resource_id': [], 'entity_id': []},
  "notes": {'id': [], 'type': [], 'target': [], 'note': []},
}
```

### Data Fields

- `context` (`str`): the textual content of the data file
- `file_name` (`str`): the name of the data / annotation file without extension
- `spans` (`dict`): span annotations of the `context` string
  - `id` (`str`): the id of the span, starts with `T`
  - `type` (`str`): the label of the span
  - `locations` (`list`): the indices indicating the span's locations (multiple because of fragments), consisting of `dict`s with
    - `start` (`list` of `int`): the indices indicating the inclusive character start positions of the span fragments
    - `end` (`list` of `int`): the indices indicating the exclusive character end positions of the span fragments
  - `text` (`list` of `str`): the texts of the span fragments
- `relations`: a sequence of relations between elements of `spans`
  - `id` (`str`): the id of the relation, starts with `R`
  - `type` (`str`): the label of the relation
  - `arguments` (`list` of `dict`): the spans related to the relation, consisting of `dict`s with
    - `type` (`list` of `str`): the argument roles of the spans in the relation, either `Arg1` or `Arg2`
    - `target` (`list` of `str`): the spans which are the arguments of the relation
- `equivalence_relations`: contains `type` and `target` (more information needed)
- `events`: contains `id`, `type`, `trigger`, and `arguments` (more information needed)
- `attributions` (`dict`): attribute annotations of any other annotation
  - `id` (`str`): the instance id of the attribution
  - `type` (`str`): the type of the attribution
  - `target` (`str`): the id of the annotation to which the attribution is for
  - `value` (`str`): the attribution's value or mark
- `normalizations` (`dict`): the unique identification of the real-world entities referred to by specific text expressions
  - `id` (`str`): the instance id of the normalized entity
  - `type`(`str`): the type of the normalized entity
  - `target` (`str`): the id of the annotation to which the normalized entity is for
  - `resource_id` (`str`): the associated resource to the normalized entity
  - `entity_id` (`str`): the instance id of normalized entity
- `notes` (`dict`): a freeform text, added to the annotation
  - `id` (`str`): the instance id of the note
  - `type` (`str`): the type of note
  - `target` (`str`): the id of the related annotation
  - `note` (`str`): the text body of the note

### Usage

The `brat` dataset script can be used by calling `load_dataset()` method and passing any arguments that are accepted by the `BratConfig` (which is a special [BuilderConfig](https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/builder_classes#datasets.BuilderConfig)). It requires at least the `url` argument. The full list of arguments is as follows:

- `url` (`str`): the url of the dataset which should point to either a zip file or a directory containing the Brat data (`*.txt`) and annotation (`*.ann`) files

- `description` (`str`, optional): the description of the dataset

- `citation` (`str`, optional): the citation of the dataset

- `homepage` (`str`, optional): the homepage of the dataset

- `split_paths` (`dict`, optional): a mapping of (arbitrary) split names to subdirectories or lists of files (without extension), e.g. `{"train": "path/to/train_directory", "test": "path/to/test_director"}` or `{"train": ["path/to/train_file1", "path/to/train_file2"]}`. In both cases (subdirectory paths or file paths), the paths are relative to the url. If `split_paths` is not provided, the dataset will be loaded from the root directory and all direct subfolders will be considered as splits.

- `file_name_blacklist` (`list`, optional): a list of file names (without extension) that should be ignored, e.g. `["A28"]`. This is useful if the dataset contains files that are not valid brat files.

Important: Using the `data_dir` parameter of the `load_dataset()` method overrides the `url` parameter of the `BratConfig`.

We provide an example of [SciArg](https://aclanthology.org/W18-5206.pdf) dataset below:

```python
from datasets import load_dataset
kwargs = {
"description" :
  """This dataset is an extension of the Dr. Inventor corpus (Fisas et al., 2015, 2016) with an annotation layer containing
  fine-grained argumentative components and relations. It is the first argument-annotated corpus of scientific
  publications (in English), which allows for joint analyses of argumentation and other rhetorical dimensions of
  scientific writing.""",
"citation" :
  """@inproceedings{lauscher2018b,
    title = {An argument-annotated corpus of scientific publications},
    booktitle = {Proceedings of the 5th Workshop on Mining Argumentation},
    publisher = {Association for Computational Linguistics},
    author = {Lauscher, Anne and Glava\v{s}, Goran and Ponzetto, Simone Paolo},
    address = {Brussels, Belgium},
    year = {2018},
    pages = {40–46}
  }""",
"homepage": "https://github.com/anlausch/ArguminSci",
"url": "http://data.dws.informatik.uni-mannheim.de/sci-arg/compiled_corpus.zip",
"split_paths": {
  "train": "compiled_corpus",
},
"file_name_blacklist": ['A28'],
}

dataset = load_dataset('dfki-nlp/brat', **kwargs)
```

## Additional Information

### Licensing Information

\[Needs More Information\]

### Citation Information

```
@inproceedings{stenetorp-etal-2012-brat,
    title = "brat: a Web-based Tool for {NLP}-Assisted Text Annotation",
    author = "Stenetorp, Pontus  and
      Pyysalo, Sampo  and
      Topi{\'c}, Goran  and
      Ohta, Tomoko  and
      Ananiadou, Sophia  and
      Tsujii, Jun{'}ichi",
    booktitle = "Proceedings of the Demonstrations at the 13th Conference of the {E}uropean Chapter of the Association for Computational Linguistics",
    month = apr,
    year = "2012",
    address = "Avignon, France",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/E12-2021",
    pages = "102--107",
}
```
