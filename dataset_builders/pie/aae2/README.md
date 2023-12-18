# PIE Dataset Card for "aae2"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the Argument Annotated Essays v2 (AAE2) dataset ([paper](https://aclanthology.org/J17-3005.pdf) and [homepage](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422)). Since the AAE2 dataset is published in the [BRAT standoff format](https://brat.nlplab.org/standoff.html), this dataset builder is based on the [PyTorch-IE brat dataset loading script](https://huggingface.co/datasets/pie/brat).

Therefore, the `aae2` dataset as described here follows the data structure from the [PIE brat dataset card](https://huggingface.co/datasets/pie/brat).

### Dataset Summary

Argument Annotated Essays Corpus (AAEC) ([Stab and Gurevych, 2017](https://aclanthology.org/J17-3005.pdf)) contains student essays. A stance for a controversial theme is expressed by a `MajorClaim` component as well as `Claim` components, and `Premise` components justify or refute the claims. `Attack` and `Support` labels are defined as relations. The span covers a statement, *which can stand in isolation as a complete sentence*, according to the AAEC annotation guidelines. All components are annotated with minimum boundaries of a clause or sentence excluding so-called "shell" language such as *On the other hand* and *Hence*. (Morio et al., 2022, p. 642)

There are two types of data: essay-level and paragraph-level ([Eger et al., 2017](https://aclanthology.org/P17-1002/)). In other words, a tree structure is complete within each paragraph, and there was no `Premise` that link to another `Premise` or `Claim` in a different paragraph, as seen in **Example** below. Therefore, it is possible to train a model on a paragraph-level which is also less memory-exhaustive (Eger et al., 2017, p. 16).

### Supported Tasks and Leaderboards

`# TODO`

- **Tasks**: Argumentation Mining, Component Identification, Relation Identification
- **Leaderboard:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Languages

The language in the dataset is English (`# TODO`).

### Dataset Variants

See [PIE-Brat Dataset Variants](https://huggingface.co/datasets/pie/brat#dataset-variants).

## Data Schema

See [PIE-Brat Data Schema](https://huggingface.co/datasets/pie/brat#data-schema).

### Usage

```python
from pie_datasets import load_dataset, builders

# load default version
datasets = load_dataset("pie/aae2")
doc = datasets["train"][0]
assert isinstance(doc, builders.brat.BratDocument)

# load version with merged span fragments
dataset_merged_spans = load_dataset("pie/aae2", name="merge_fragmented_spans")
doc_merged_spans = dataset_merged_spans["train"][0]
assert isinstance(doc_merged_spans, builders.brat.BratDocumentWithMergedSpans)
```

## Document Converters

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations`
  - `LabeledSpans`, converted from `BratDocument`'s `spans`
    - labels: `# TODO`
    - if `spans` `# TODO`
  - `BinaryRelations`, converted from `BratDocument`'s `relations`
    - labels: `# TODO`
    - if `relations` `# TODO`
- `pytorch_ie.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions`
  - - `LabeledSpans`, as above
  - `BinaryRelations`, as above
  - `LabeledPartitions`, partitioned `BratDocument`'s `text`, according to the paragraph, using regex.
    - every partition is labeled as `partition`

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.

### Data Splits

`# TODO`

| Statistics      | Train | Test |
| --------------- | ----- | ---: |
| No. of document | 322   |   80 |

### Label Descriptions

`# TODO: recheck the number, and explain the conversion method "connect_first" and "connect_all" `

## Components:

| Components   | Count | Percentage |
| ------------ | ----: | ---------: |
| `MajorClaim` |   751 |       13 % |
| `Claim`      |  1228 |       21 % |
| `Premise`    |  3832 |       66 % |

- `MajorClaim` is the root node of the argumentation structure and represents the author’s standpoint on the topic. Essay bodies either support or attack the author’s standpoint expressed in the major claim.
- `Claim` constitutes the central component of each argument. Each one has at least one premise and take the values "for" or "against"
- `Premise` is the reasons of the argument; either linked to claim or another premise.

**Note that** relations between `MajorClaim` and  `Claim` were not annotated; however, each claim is annotated with `Attribute`: `for` or `against` - which indicates the relation between itself and `MajorClaim`. In addition, when two non-related `claim` 's appear in one paragraph, there is also no relations to one another.

## Relations

| Relations          | Count | Percentage |
| ------------------ | ----: | ---------: |
| support: `Support` |  3613 |     94.3 % |
| attack: `Attack`   |   219 |      5.7 % |

- "Each premise `p` has one **outgoing relation** (i.e., there is a relation that has p as source component) and none or several **incoming relations** (i.e., there can be a relation with `p` as target component)."
- "A `Claim` can exhibit several **incoming relations** but no **outgoing relation**." (S&G, 2017, p. 68)
- "The relations from the claims of the arguments to the major claim are dotted since we will not explicitly annotated them. The relation of each argument to the major claim is indicated by a stance attribute of each claim. This attribute can either be for or against as illustrated in figure 1.4." (Stab & Gurevych, *Guidelines for Annotating Argumentation Structures in Persuasive Essays*, 2015, p. 5)

See further description in Stab & Gurevych 2017, p.627 and the [annotation guideline](https://github.com/ArneBinder/pie-datasets/blob/db94035602610cefca2b1678aa2fe4455c96155d/data/datasets/ArgumentAnnotatedEssays-2.0/guideline.pdf).

## Dataset Creation

### Curation Rationale

`# TODO`

### Source Data

#### Initial Data Collection and Normalization

Persuasive essays were collected from [essayforum.com](https://essayforum.com/) (See essay prompts, along with the essay's `id`'s [here](https://github.com/ArneBinder/pie-datasets/blob/db94035602610cefca2b1678aa2fe4455c96155d/data/datasets/ArgumentAnnotatedEssays-2.0/prompts.csv)).

#### Who are the source language producers?

`# TODO`

### Annotations

#### Annotation process

`# TODO`

The annotation were done using BRAT Rapid Annotation Tool ([Stenetorp et al., 2012](https://aclanthology.org/E12-2021/)).

#### Who are the annotators?

`# TODO`

### Personal and Sensitive Information

`# TODO`

## Considerations for Using the Data

### Social Impact of Dataset

`# TODO`

### Discussion of Biases

`# TODO`

### Other Known Limitations

`# TODO`

## Additional Information

### Dataset Curators

\[More Information Needed\]

### Licensing Information

**License**: [License description by TU Darmstadt](https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2422/arg_annotated_essays_v2_license.pdf?sequence=2&isAllowed=y)

**Funding**: This work has been supported by the
Volkswagen Foundation as part of the
Lichtenberg-Professorship Program under
grant no. I/82806 and by the German Federal
Ministry of Education and Research (BMBF)
as a part of the Software Campus project
AWS under grant no. 01—S12054.

### Citation Information

```
@article{stab2017parsing,
  title={Parsing argumentation structures in persuasive essays},
  author={Stab, Christian and Gurevych, Iryna},
  journal={Computational Linguistics},
  volume={43},
  number={3},
  pages={619--659},
  year={2017},
  publisher={MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info~…}
}
```

```
@misc{https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422,
url = { https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422 },
author = { Stab, Christian and Gurevych, Iryna },
keywords = { Argument Mining, 409-06 Informationssysteme, Prozess- und Wissensmanagement, 004 },
publisher = { Technical University of Darmstadt },
year = { 2017 },
copyright = { License description },
title = { Argument Annotated Essays (version 2) }
}
```

### Contributions

Thanks to [@ArneBinder](https://github.com/ArneBinder) and [@idalr](https://github.com/idalr) for adding this dataset.
