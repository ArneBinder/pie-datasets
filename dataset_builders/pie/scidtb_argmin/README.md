# PIE Dataset Card for "SciDTB Argmin"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[SciDTB ArgMin Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/scidtb_argmin).

## Usage

```python
from pie_datasets import load_dataset
from pie_modules.documents import TextDocumentWithLabeledSpansAndBinaryRelations

# load English variant
dataset = load_dataset("pie/scidtb_argmin")

# if required, normalize the document type (see section Document Converters below)
dataset_converted = dataset.to_document_type(TextDocumentWithLabeledSpansAndBinaryRelations)
assert isinstance(dataset_converted["train"][0], TextDocumentWithLabeledSpansAndBinaryRelations)

# get first relation in the first document
doc = dataset_converted["train"][0]
print(doc.binary_relations[0])
# BinaryRelation(head=LabeledSpan(start=251, end=454, label='means', score=1.0), tail=LabeledSpan(start=455, end=712, label='proposal', score=1.0), label='detail', score=1.0)
print(doc.binary_relations[0].resolve())
# ('detail', (('means', 'We observe , identify , and detect naturally occurring signals of interestingness in click transitions on the Web between source and target documents , which we collect from commercial Web browser logs .'), ('proposal', 'The DSSM is trained on millions of Web transitions , and maps source-target document pairs to feature vectors in a latent space in such a way that the distance between source documents and their corresponding interesting targets in that space is minimized .')))
```

## Data Schema

The document type for this dataset is `SciDTBArgminDocument` which defines the following data fields:

- `tokens` (tuple of string)
- `id` (str, optional)
- `metadata` (dictionary, optional)

and the following annotation layers:

- `units` (annotation type: `LabeledSpan`, target: `tokens`)
- `relations` (annotation type: `BinaryRelation`, target: `units`)

See [here](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/annotations.py) for the annotation type definitions.

## Document Converters

The dataset provides document converters for the following target document types:

- `pie_modules.documents.TextDocumentWithLabeledSpansAndBinaryRelations`
  - `labeled_spans`: `LabeledSpan` annotations, converted from`SciDTBArgminDocument`'s `units`
    - labels: `proposal`, `assertion`, `result`, `observation`, `means`, `description`
    - tuples of `tokens` are joined with a whitespace to create `text` for `LabeledSpans`
  - `binary_relations`: `BinaryRelation` annotations, converted from `SciDTBArgminDocument`'s `relations`
    - labels: `support`, `attack`, `additional`, `detail`, `sequence`

See [here](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/documents.py) for the document type
definitions.

### Collected Statistics after Document Conversion

We use the script `evaluate_documents.py` from [PyTorch-IE-Hydra-Template](https://github.com/ArneBinder/pytorch-ie-hydra-template-1) to generate these statistics.
After checking out that code, the statistics and plots can be generated by the command:

```commandline
python src/evaluate_documents.py dataset=scidtb_argmin_base metric=METRIC
```

where a `METRIC` is called according to the available metric configs in `config/metric/METRIC` (see [metrics](https://github.com/ArneBinder/pytorch-ie-hydra-template-1/tree/main/configs/metric)).

This also requires to have the following dataset config in `configs/dataset/scidtb_argmin_base.yaml` of this dataset within the repo directory:

```commandline
_target_: src.utils.execute_pipeline
input:
  _target_: pie_datasets.DatasetDict.load_dataset
  path: pie/scidtb_argmin
  revision: 335a8e6168919d7f204c6920eceb96745dbd161b
```

For token based metrics, this uses `bert-base-uncased` from `transformer.AutoTokenizer` (see [AutoTokenizer](https://huggingface.co/docs/transformers/v4.37.1/en/model_doc/auto#transformers.AutoTokenizer), and [bert-based-uncased](https://huggingface.co/bert-base-uncased) to tokenize `text` in `TextDocumentWithLabeledSpansAndBinaryRelations` (see [document type](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/documents.py)).

#### Relation argument (outer) token distance per label

The distance is measured from the first token of the first argumentative unit to the last token of the last unit, a.k.a. outer distance.

We collect the following statistics: number of documents in the split (*no. doc*), no. of relations (*len*), mean of token distance (*mean*), standard deviation of the distance (*std*), minimum outer distance (*min*), and maximum outer distance (*max*).
We also present histograms in the collapsible, showing the distribution of these relation distances (x-axis; and unit-counts in y-axis), accordingly.

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=scidtb_argmin_base metric=relation_argument_token_distances
```

</details>

|            | len | max |   mean | min |    std |
| :--------- | --: | --: | -----: | --: | -----: |
| ALL        | 586 | 277 | 75.239 |  21 | 40.312 |
| additional |  54 | 180 | 59.593 |  36 | 29.306 |
| detail     | 258 | 163 |  65.62 |  22 |  29.21 |
| sequence   |  22 |  93 | 59.727 |  38 | 17.205 |
| support    | 252 | 277 | 89.794 |  21 | 48.118 |

<details>
  <summary>Histogram (split: train, 60 documents)</summary>

![rtd-label_scitdb-argmin.png](img%2Frtd-label_scitdb-argmin.png)

</details>

#### Span lengths (tokens)

The span length is measured from the first token of the first argumentative unit to the last token of the particular unit.

We collect the following statistics: number of documents in the split (*no. doc*), no. of spans (*len*), mean of number of tokens in a span (*mean*), standard deviation of the number of tokens (*std*), minimum tokens in a span (*min*), and maximum tokens in a span (*max*).
We also present histograms in the collapsible, showing the distribution of these token-numbers (x-axis; and unit-counts in y-axis), accordingly.

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=scidtb_argmin_base metric=span_lengths_tokens
```

</details>

| statistics |  train |
| :--------- | -----: |
| no. doc    |     60 |
| len        |    353 |
| mean       | 27.946 |
| std        | 13.054 |
| min        |      7 |
| max        |    123 |

<details>
  <summary>Histogram (split: train, 60 documents)</summary>

![slt_scitdb-argmin.png](img%2Fslt_scitdb-argmin.png)

</details>

#### Token length (tokens)

The token length is measured from the first token of the document to the last one.

We collect the following statistics: number of documents in the split (*no. doc*), mean of document token-length (*mean*), standard deviation of the length (*std*), minimum number of tokens in a document (*min*), and maximum number of tokens in a document (*max*).
We also present histograms in the collapsible, showing the distribution of these token lengths (x-axis; and unit-counts in y-axis), accordingly.

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=scidtb_argmin_base metric=count_text_tokens
```

</details>

| statistics |   train |
| :--------- | ------: |
| no. doc    |      60 |
| mean       | 164.417 |
| std        |  64.572 |
| min        |      80 |
| max        |     532 |

<details>
  <summary>Histogram (split: train, 60 documents)</summary>

![tl_scidtb-argmin.png](img%2Ftl_scidtb-argmin.png)

</details>
