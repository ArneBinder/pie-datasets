# PIE Dataset Card for "sciarg"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the SciArg dataset ([paper](https://aclanthology.org/W18-5206/) and [data repository](https://github.com/anlausch/sciarg_resource_analysis)). Since the SciArg dataset is published in the [BRAT standoff format](https://brat.nlplab.org/standoff.html), this dataset builder is based on the [PyTorch-IE brat dataset loading script](https://huggingface.co/datasets/pie/brat).

Therefore, the `sciarg` dataset as described here follows the data structure from the [PIE brat dataset card](https://huggingface.co/datasets/pie/brat).

### Dataset Summary

The SciArg dataset is an extension of the Dr. Inventor corpus (Fisas et al., [2015](https://aclanthology.org/W15-1605.pdf), [2016](https://aclanthology.org/L16-1492.pdf)) with an annotation layer containing
fine-grained argumentative components and relations, believing that argumentation needs to
be studied in combination with other rhetorical aspects. It is the first publicly-available argument-annotated corpus of scientific publications (in English), which allows for joint analyses of argumentation and other
rhetorical dimensions of scientific writing." ([Lauscher et al., 2018](<(https://aclanthology.org/W18-5206/)>), pp. 40-41)

### Supported Tasks and Leaderboards

- **Tasks**: Argumentation Mining, Component Identification, Relation Identification
- **Leaderboard:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Languages

The language in the dataset is English (scientific academic publications on computer graphics).

### Dataset Variants

See [PIE-Brat Dataset Variants](https://huggingface.co/datasets/pie/brat#dataset-variants).

### Data Schema

See [PIE-Brat Data Schema](https://huggingface.co/datasets/pie/brat#data-schema).

### Usage

```python
from pie_datasets import load_dataset, builders

# load default version
datasets = load_dataset("pie/sciarg")
doc = datasets["train"][0]
assert isinstance(doc, builders.brat.BratDocument)

# load version with merged span fragments
dataset_merged_spans = load_dataset("pie/sciarg", name="merge_fragmented_spans")
doc_merged_spans = dataset_merged_spans["train"][0]
assert isinstance(doc_merged_spans, builders.brat.BratDocumentWithMergedSpans)
```

### Document Converters

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations`
  - `LabeledSpans`, converted from `BratDocument`'s `spans`
    - labels: `background_claim`, `own_claim`, `data`
    - if `spans` contain whitespace at the beginning and/or the end, the whitespace are trimmed out.
  - `BinraryRelations`, converted from `BratDocument`'s `relations`
    - labels: `supports`, `contradicts`, `semantically_same`, `parts_of_same`
    - if the `relations` label is `semantically_same` or `parts_of_same`, they are merged if they are the same arguments after sorting.
- `pytorch_ie.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions`
  - `LabeledSpans`, as above
  - `BinaryRelations`, as above
  - `LabeledPartitions`, partitioned `BratDocument`'s `text`, according to the paragraph, using regex.
    - labels: `title`, `abstract`, `H1`

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.

### Data Splits

The dataset consists of a single `train` split that has 40 documents.

For detailed statistics on the corpus, see Lauscher et al. ([2018](<(https://aclanthology.org/W18-5206/)>), p. 43), and the author's [resource analysis](https://github.com/anlausch/sciarg_resource_analysis).

### Label Descriptions

#### Components

| Components         | Count | Percentage |
| ------------------ | ----: | ---------: |
| `background_claim` |  3291 |     24.2 % |
| `own_claim`        |  6004 |     44.2 % |
| `data`             |  4297 |     31.6 % |

- `own_claim` is an argumentative statement that closely relates to the authors’ own work.
- `background_claim` an argumentative statement relating to the background of authors’ work, e.g., about related work or common practices.
- `data` component represents a fact that serves as evidence for or against a claim. Note that references or (factual) examples can also serve as data.
  (Lauscher et al. 2018, p.41; following and simplified [Toulmin, 2003](https://www.cambridge.org/core/books/uses-of-argument/26CF801BC12004587B66778297D5567C))

#### Relations

| Relations                  | Count | Percentage |
| -------------------------- | ----: | ---------: |
| support: `support`         |  5791 |     74.0 % |
| attack: `contradict`       |   697 |      8.9 % |
| other: `semantically_same` |    44 |      0.6 % |
| other: `parts_of_same`     |  1298 |     16.6 % |

##### Argumentative relations

- `support`:
  - if the assumed veracity of *b* increases with the veracity of *a*
  - "Usually, this relationship exists from data to claim, but in many cases a claim might support another claim. Other combinations are still possible." -  (*Annotation Guidelines*, p. 3)
- `contradict`:
  - if the assumed veracity of *b* decreases with the veracity of *a*
  - It is a **bi-directional**, i.e., symmetric relationship.

##### Non-argumentative relations

- `semantically_same`: between two mentions of effectively the same claim or data component. Can be seen as *argument coreference*, analogous to entity, and *event coreference*. This relation is considered symmetric (i.e., **bidirectional**) and non-argumentative.
  (Lauscher et al. 2018, p.41; following [Dung, 1995](https://www.sciencedirect.com/science/article/pii/000437029400041X?via%3Dihub))
- `parts_of_same`: when a single component is split up in several parts. It is **non-argumentative**, **bidirectional**, but also **intra-component**

(*Annotation Guidelines*, pp. 4-6)

**Important note on label counts**:

There are currently discrepancies in label counts between

- previous report in [Lauscher et al., 2018](https://aclanthology.org/W18-5206/), p. 43),
- current report above here (labels counted in `BratDocument`'s);

possibly since [Lauscher et al., 2018](https://aclanthology.org/W18-5206/) presents the numbers of the real argumentative components, whereas here discontinuous components are still split (marked with the `parts_of_same` helper relation) and, thus, count per fragment.

## Dataset Creation

### Curation Rationale

"\[C\]omputational methods for analyzing scientific writing are becoming paramount...there is no publicly available corpus of scientific publications (in English), annotated with fine-grained argumentative structures. ...\[A\]rgumentative structure of scientific publications should not be studied in isolation, but rather in relation to other rhetorical aspects, such as the
discourse structure.
(Lauscher et al. 2018, p. 40)

### Source Data

#### Initial Data Collection and Normalization

"\[W\]e randomly selected a set of 40 documents, available in PDF format, among a bigger collection provided by experts in the domain, who pre-selected a representative sample of articles in Computer Graphics. Articles were classified into four important subjects in this area: Skinning, Motion Capture, Fluid Simulation and Cloth Simulation. We included in the corpus 10 highly representative articles for each subject." (Fisas et al. 2015, p. 44)

"The Corpus includes 10,789 sentences, with an average of 269.7 sentences per document." (p. 45)

#### Who are the source language producers?

It can be implied from the data source that the language producers were academics in computer graphics and related fields, possibly assisted by other human editors.

### Annotations

#### Annotation process

"We trained the four annotators in a calibration phase, consisting of five iterations, in each of which all annotators annotated one publication. After each iteration we computed the inter-annotator agreement (IAA), discussed the disagreements, and, if needed, adjourned the [annotation guidelines](https://data.dws.informatik.uni-mannheim.de/sci-arg/annotation_guidelines.pdf)."

The detailed evolution of IAA over the five calibration iterations is depicted in Lauscher et al. (2018), p. 42, Figure 1.

The annotation were done using BRAT Rapid Annotation Tool ([Stenetorp et al., 2012](https://aclanthology.org/E12-2021/)).

#### Who are the annotators?

"We hired one expert (a researcher in computational linguistics) and three non-expert annotators (humanities and social sciences scholars)." (Lauscher et al. 2018, p. 42)

### Personal and Sensitive Information

\[More Information Needed\]

## Considerations for Using the Data

### Social Impact of Dataset

"To support learning-based models for automated analysis of scientific publications, potentially leading to better understanding
of the different rhetorical aspects of scientific language (which we dub *scitorics*)." (Lauscher et al. 2018, p. 40)

"The resulting corpus... is, to the best of our knowledge, the first argument-annotated corpus of scientific publications in English, enables (1) computational analysis of argumentation in scientific writing and (2) integrated analysis of argumentation and other rhetorical aspects of scientific text." (Lauscher et al. 2018, p. 44)

### Discussion of Biases

"...not all claims are supported and secondly, claims can be supported by other claims. There are many more supports than contradicts relations."

"While the background claims and own claims are on average of similar length (85 and 87 characters, respectively), they are much longer than data components (average of 25 characters)."

"\[A\]nnotators identified an average of 141 connected component per publication...This indicates that either authors write very short argumentative chains or that our annotators had difficulties noticing long-range argumentative dependencies."

(Lauscher et al. 2018, p.43)

### Other Known Limitations

"Expectedly, we observe higher agreements with more calibration. The agreement on argumentative relations is 23% lower than on the components, which we think is due to the high ambiguity of argumentation structures."

"Additionally, disagreements in component identification are propagated to relations as well, since the agreement on a relation implies the agreement on annotated components at both ends of the relation."

(Lauscher et al. 2018, p. 43)

## Additional Information

### Dataset Curators

- **Repository:** [https://github.com/anlausch/ArguminSci](https://github.com/anlausch/ArguminSci)

### Licensing Information

[MIT License](https://github.com/anlausch/ArguminSci/blob/master/LICENSE)

This research was partly funded by the German Research Foundation (DFG), grant number EC 477/5-1 (LOC-DB).

### Citation Information

```
@inproceedings{lauscher2018b,
  title = {An argument-annotated corpus of scientific publications},
  booktitle = {Proceedings of the 5th Workshop on Mining Argumentation},
  publisher = {Association for Computational Linguistics},
  author = {Lauscher, Anne and Glava\v{s}, Goran and Ponzetto, Simone Paolo},
  address = {Brussels, Belgium},
  year = {2018},
  pages = {40–46}
}
```

```
@inproceedings{lauscher2018a,
  title = {ArguminSci: A Tool for Analyzing Argumentation and Rhetorical Aspects in Scientific Writing},
  booktitle = {Proceedings of the 5th Workshop on Mining Argumentation},
  publisher = {Association for Computational Linguistics},
  author = {Lauscher, Anne and Glava\v{s}, Goran and Eckert, Kai},
  address = {Brussels, Belgium},
  year = {2018},
  pages = {22–28}
}
```

### Contributions

Thanks to [@github-username](https://github.com/%3Cgithub-username%3E) for adding this dataset.
