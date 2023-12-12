# PIE Dataset Card for "abstrct"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the AbstRCT dataset ([paper]() and [data repository]()). Since the AbstRCT dataset is published in the [BRAT standoff format](https://brat.nlplab.org/standoff.html), this dataset builder is based on the [PyTorch-IE brat dataset loading script](https://huggingface.co/datasets/pie/brat).

Therefore, the `abstrct` dataset as described here follows the data structure from the [PIE brat dataset card](https://huggingface.co/datasets/pie/brat).

### Dataset Summary

### Supported Tasks and Leaderboards #TODO

- **Tasks**: Argumentation Mining, Component Identification, Relation Identification
- **Leaderboard:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Languages #TODO

The language in the dataset is English.

### Dataset Variants

See [PIE-Brat Data Variants](https://huggingface.co/datasets/pie/brat#data-variants).

### Data Schema

See [PIE-Brat Data Schema](https://huggingface.co/datasets/pie/brat#data-schema).

### Usage

```python
from pie_datasets import load_dataset, builders

# load default version
datasets = load_dataset("pie/abstrct")
doc = datasets["train"][0]
assert isinstance(doc, builders.brat.BratDocument)

# load version with merged span fragments
dataset_merged_spans = load_dataset("pie/abstrct", name="merge_fragmented_spans")
doc_merged_spans = dataset_merged_spans["train"][0]
assert isinstance(doc_merged_spans, builders.brat.BratDocumentWithMergedSpans)
```

### Document Converters #TODO

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations`
  - `LabeledSpans`, converted from `BratDocument`'s `spans`
    - labels: 
  - `BinraryRelations`, converted from `BratDocument`'s `relations`
    - labels: 

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.

### Data Splits #TODO


### Label Descriptions #TODO

#### Components

#### Relations

## Dataset Creation #TODO

### Curation Rationale

### Source Data

#### Initial Data Collection and Normalization

#### Who are the source language producers?

### Annotations #TODO

#### Annotation process

#### Who are the annotators?

### Personal and Sensitive Information

\[More Information Needed\]

## Considerations for Using the Data #TODO

### Social Impact of Dataset

### Discussion of Biases

### Other Known Limitations

## Additional Information #TODO

### Dataset Curators

### Licensing Information

### Citation Information

```

```

### Contributions

Thanks to [@ArneBinder](https://github.com/ArneBinder) and [@idalr](https://github.com/idalr) for adding this dataset.
