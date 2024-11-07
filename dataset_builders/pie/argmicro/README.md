# PIE Dataset Card for "argmicro"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[ArgMicro Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/argmicro).

## Usage

```python
from pie_datasets import load_dataset
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndBinaryRelations

# load English variant
dataset = load_dataset("pie/argmicro", name="en")

# if required, normalize the document type (see section Document Converters below)
dataset_converted = dataset.to_document_type("pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations")
assert isinstance(dataset_converted["train"][0], TextDocumentWithLabeledSpansAndBinaryRelations)

# get first relation in the first document
doc = dataset_converted["train"][0]
print(doc.binary_relations[0])
# BinaryRelation(head=LabeledSpan(start=1769, end=1945, label='Claim', score=1.0), tail=LabeledSpan(start=1, end=162, label='MajorClaim', score=1.0), label='Support', score=1.0)
print(doc.binary_relations[0].resolve())
# ('Support', (('Claim', 'Treatment with mitoxantrone plus prednisone was associated with greater and longer-lasting improvement in several HQL domains and symptoms than treatment with prednisone alone.'), ('MajorClaim', 'A combination of mitoxantrone plus prednisone is preferable to prednisone alone for reduction of pain in men with metastatic, hormone-resistant, prostate cancer.')))
```

## Dataset Variants

The dataset contains two `BuilderConfig`'s:

- `de`: with the original texts collection in German
- `en`: with the English-translated texts

## Data Schema

The document type for this dataset is `ArgMicroDocument` which defines the following data fields:

- `text` (str)
- `id` (str, optional)
- `topic_id` (str, optional)
- `metadata` (dictionary, optional)

and the following annotation layers:

- `stance` (annotation type: `Label`)
  - description: A document may contain one of these `stance` labels: `pro`, `con`, `unclear`, or no label when it is undefined (see [here](https://huggingface.co/datasets/DFKI-SLT/argmicro/blob/main/argmicro.py#L35) for reference).
- `edus` (annotation type: `Span`, target: `text`)
- `adus` (annotation type: `LabeledAnnotationCollection`, target: `edus`)
  - description: each element of `adus` may consist of several entries from `edus`, so we require `LabeledAnnotationCollection` as annotation type. This is originally indicated by `seg` edges in the data.
  - `LabeledAnnotationCollection` has the following fields:
    - `annotations` (annotation type: `Span`, target: `text`)
    - `label` (str, optional), values: `opp`, `pro` (see [here](https://huggingface.co/datasets/DFKI-SLT/argmicro/blob/main/argmicro.py#L36))
- `relations` (annotation type: `MultiRelation`, target: `adus`)
  - description: Undercut (`und`) relations originally target other relations (i.e. edges), but we let them target the `head` of the targeted relation instead. The original state can be deterministically reconstructed by taking the label into account. Furthermore, the head of additional source (`add`) relations are integrated into the head of the target relation (note that this propagates along `und` relations). We model this with `MultiRelation`s whose `head` and `tail` are of type `LabeledAnnotationCollection`.
  - `MultiRelation` has the following fields:
    - `head` (tuple, annotation type: `LabeledAnnotationCollection`, target: `adus`)
    - `tail` (tuple, annotation type: `LabeledAnnotationCollection`, target: `adus`)
    - `label` (str, optional), values: `sup`, `exa`, `reb`, `und` (see [here](https://huggingface.co/datasets/DFKI-SLT/argmicro/blob/main/argmicro.py#L37) for reference, but note that helper relations `seg` and `add` are not there anymore, see above).

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the annotation type definitions.

## Document Converters

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations`
  - `LabeledSpans`, converted from `ArgMicroDocument`'s `adus`
    - labels: `opp`, `pro`
    - if an ADU contains multiple spans (i.e. EDUs), we take the start of the first EDU and the end of the last EDU as the boundaries of the new `LabeledSpan`. We also raise exceptions if any newly created `LabeledSpan`s overlap.
  - `BinraryRelations`, converted from `ArgMicroDocument`'s `relations`
    - labels: `sup`, `reb`, `und`, `joint`, `exa`
    - if the `head` or `tail` consists of multiple `adus`, then we build `BinaryRelation`s with all `head`-`tail` combinations and take the label from the original relation. Then, we build `BinaryRelations`' with label `joint` between each component that previously belongs to the same `head` or `tail`, respectively.
  - `metadata`, we keep the `ArgMicroDocument`'s `metadata`, but `stance` and `topic_id`.

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.

### Collected Statistics after Document Conversion

We use the script `evaluate_documents.py` from [PyTorch-IE-Hydra-Template](https://github.com/ArneBinder/pytorch-ie-hydra-template-1) to generate these statistics.
After checking out that code, the statistics and plots can be generated by the command:

```commandline
python src/evaluate_documents.py dataset=argmicro_base metric=METRIC
```

where a `METRIC` is called according to the available metric configs in `config/metric/METRIC` (see [metrics](https://github.com/ArneBinder/pytorch-ie-hydra-template-1/tree/main/configs/metric)).

This also requires to have the following dataset config in `configs/dataset/argmicro_base.yaml` of this dataset within the repo directory:

```commandline
_target_: src.utils.execute_pipeline
input:
  _target_: pie_datasets.DatasetDict.load_dataset
  path: pie/argmicro
  revision: 28ef031d2a2c97be7e9ed360e1a5b20bd55b57b2
  name: en
```

For token based metrics, this uses `bert-base-uncased` from `transformer.AutoTokenizer` (see [AutoTokenizer](https://huggingface.co/docs/transformers/v4.37.1/en/model_doc/auto#transformers.AutoTokenizer), and [bert-based-uncased](https://huggingface.co/bert-base-uncased) to tokenize `text` in `TextDocumentWithLabeledSpansAndBinaryRelations` (see [document type](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py)).

#### Relation argument (outer) token distance per label

The distance is measured from the first token of the first argumentative unit to the last token of the last unit, a.k.a. outer distance.

We collect the following statistics: number of documents in the split (*no. doc*), no. of relations (*len*), mean of token distance (*mean*), standard deviation of the distance (*std*), minimum outer distance (*min*), and maximum outer distance (*max*).
We also present histograms in the collapsible, showing the distribution of these relation distances (x-axis; and unit-counts in y-axis), accordingly.

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=argmicro_base metric=relation_argument_token_distances
```

</details>

|       |  len | max |   mean | min |    std |
| :---- | ---: | --: | -----: | --: | -----: |
| ALL   | 1018 | 127 | 44.434 |  14 | 21.501 |
| exa   |   18 |  63 | 33.556 |  16 | 13.056 |
| joint |   88 |  48 | 30.091 |  17 |  9.075 |
| reb   |  220 | 127 | 49.327 |  16 | 24.653 |
| sup   |  562 | 124 | 46.534 |  14 | 22.079 |
| und   |  130 |  84 | 38.292 |  17 | 12.321 |

<details>
  <summary>Histogram (split: train, 112 documents)</summary>

![rtd-label_argmicro.png](img%2Frtd-label_argmicro.png)

</details>

#### Span lengths (tokens)

The span length is measured from the first token of the first argumentative unit to the last token of the particular unit.

We collect the following statistics: number of documents in the split (*no. doc*), no. of spans (*len*), mean of number of tokens in a span (*mean*), standard deviation of the number of tokens (*std*), minimum tokens in a span (*min*), and maximum tokens in a span (*max*).
We also present histograms in the collapsible, showing the distribution of these token-numbers (x-axis; and unit-counts in y-axis), accordingly.

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=argmicro_base metric=span_lengths_tokens
```

</details>

| statistics |  train |
| :--------- | -----: |
| no. doc    |    112 |
| len        |    576 |
| mean       | 16.365 |
| std        |  6.545 |
| min        |      4 |
| max        |     41 |

<details>
  <summary>Histogram (split: train, 112 documents)</summary>

![slt_argmicro.png](img%2Fslt_argmicro.png)

</details>

#### Token length (tokens)

The token length is measured from the first token of the document to the last one.

We collect the following statistics: number of documents in the split (*no. doc*), mean of document token-length (*mean*), standard deviation of the length (*std*), minimum number of tokens in a document (*min*), and maximum number of tokens in a document (*max*).
We also present histograms in the collapsible, showing the distribution of these token lengths (x-axis; and unit-counts in y-axis), accordingly.

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=argmicro_base metric=count_text_tokens
```

</details>

| statistics |  train |
| :--------- | -----: |
| no. doc    |    112 |
| mean       | 84.161 |
| std        | 22.596 |
| min        |     36 |
| max        |    153 |

<details>
  <summary>Histogram (split: train, 112 documents)</summary>

![tl_argmicro.png](img%2Ftl_argmicro.png)

</details>
