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
- `edus` (annotation type: `Span`, target: `text`)
- `adus` (tuple, annotation type: `LabeledAnnotationCollection`, target: `edus`)
  - description: TODO (*why* do we need to have this special annotation type? i.e. why `adus` with multiple `Span`?)
  - `LabeledAnnotationCollection` has the following fields:
    - `annotations` (annotation type: `Span`, target: `text`)
    - `label` (str, optional), values: TODO
- `relations` (annotation type: `MultiRelation`, target: `adus`)
  - description: TODO (*why* do we need to have this special annotation type? i.e. why relations with multiple `head`'s and/or `tail`'s?)
  - `MultiRelation` has the following fields:
    - `head` (tuple, annotation type: `LabeledAnnotationCollection`, target: `adus`)
    - `tail` (tuple, annotation type: `LabeledAnnotationCollection`, target: `adus`)
    - `label` (str, optional), values: TODO

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the annotation type definitions.

## Document Converters

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations`
  - `LabeledSpans`, converted from `ArgMicroDocument`'s `adus`, labels: TODO
    - if `adus` contains multiple spans, we take the start of the first `edu` and the end of the last `edu` as the boundaries of `LabeledSpan`. We also raise exceptions when there is an overlapping.
  - `BinraryRelations`, converted from `ArgMicroDocument`'s `relations`, labels: TODO
    - if `relations` contains multiple `adus` as `head` and/or `tail`, then we build `BinaryRelations` between each `head`/`tail` to the other component (TODO: with which label?). Then, we build `BinaryRelations` between each component that previously belongs to the same `LabeledAnnotationCollection` (TODO: with which label?).
  - `metadata`, we keep the `stance`, `topic_id`, and the rest of `ArgMicroDocument`'s `metadata`.

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.
