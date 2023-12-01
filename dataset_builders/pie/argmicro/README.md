# PIE Dataset Card for "ArgMicro"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[ArgMicro Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/argmicro).

## Dataset Variants

The dataset contains two `BuilderConfig`'s:

- `de`: with the original texts collection in German
- `en`: with the English-translated texts

## Data Schema

The document type for this dataset is `ArgMicroDocument` which defines the following data fields:

- `id` (str)
- `text` (str)
- `topic_id` (str, optional)
- `metadata` (dictionary, optional)

and the following annotation layers:

- `stance` (annotation type: `Label`)
  - description: a document may contain one of these `stance` label: `pro`, `con`, `unclear`. or no label when it is undefined. (see reference [here](https://huggingface.co/datasets/DFKI-SLT/argmicro/blob/main/argmicro.py#L35))
- `edus` (annotation type: `Span`, target: `text`)
- `adus` (tuple, annotation type: `LabeledAnnotationCollection`, target: `edus`)
  - description: an `adus` may consist of several `edus`. This special annotation type allows mapping `adus` with those `Span`'s.
  - `LabeledAnnotationCollection` has the following fields:
    - `annotations` (annotation type: `Span`, target: `text`)
    - `label` (str, optional), values: see [here](https://huggingface.co/datasets/DFKI-SLT/argmicro/blob/main/argmicro.py#L36)
- `relations` (annotation type: `MultiRelation`, target: `adus`)
  - description: some relations (e.g. `und`) involve multiple `adus` as `head` or `tail` components. This special annotation type allows `head` and `tail` to contain several `adus`.
  - `MultiRelation` has the following fields:
    - `head` (tuple, annotation type: `LabeledAnnotationCollection`, target: `adus`)
    - `tail` (tuple, annotation type: `LabeledAnnotationCollection`, target: `adus`)
    - `label` (str, optional), values: see [here](https://huggingface.co/datasets/DFKI-SLT/argmicro/blob/main/argmicro.py#L37)

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the annotation type definitions.

## Document Converters

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations`
  - `LabeledSpans`, converted from `ArgMicroDocument`'s `adus`
    - labels: `opp`, `pro`
    - if `adus` contains multiple spans, we take the start of the first `edu` and the end of the last `edu` as the boundaries of `LabeledSpan`. We also raise exceptions when there is an overlapping.
  - `BinraryRelations`, converted from `ArgMicroDocument`'s `relations`
    - labels: `sup`, `reb`, `und`, `joint`, `exa`
    - if `relations` contains multiple `adus` as `head` and/or `tail`, then we build `BinaryRelations` (label taken from `relations`) between each `head`/`tail` to the other component. Then, we build `BinaryRelations`' `joint` between each component that previously belongs to the same `LabeledAnnotationCollection`.
  - `metadata`, we keep the `stance`, `topic_id`, and the rest of `ArgMicroDocument`'s `metadata`.

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.
