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
  - `annotations` (annotation type: `Span`, target: `text`)
  - `label` (str, optional)
- `relations` (annotation type: `MultiRelation`, target: `adus`)
  - `head` (tuple, annotation type: `LabeledAnnotationCollection`, target: `adus`)
  - `tail` (tuple, annotation type: `LabeledAnnotationCollection`, target: `adus`)
  - `label` (str, optional)

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the annotation type definitions.

In addition to common annotation type definitions above, `ArgMicroDocument` contains special annotation types: `LabeledAnnotationCollection` and `MultiRelation`.
Both of which contain a tuple of one `Annotation` or more, as the document allows an `adus` with multiple `Span`'s, as well as a relation with multiple `head`'s and/or `tail`'s.

## Document Converters

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations`
  - `LabeledSpans`, converted from `ArgMicroDocument`'s `adus`
    - if `adus` contains multiple spans, we take the start of the first `edu` and the end of the last `edu` as the boundaries of `LabeledSpan`. We also raise exceptions when there is an overlapping.
  - `BinraryRelations`, converted from `ArgMicroDocument`'s `relations`
    - if `relations` contains multiple `adus` as `head` and/or `tail`, then we build `BinaryRelations` between each `head`/`tail` to the other component. Then, we build `BinaryRelations` between each component that previously belongs to the same `LabeledAnnotationCollection`.
  - `metadata`, we keep the `stance`, `topic_id`, and the rest of `ArgMicroDocument`'s `metadata`.

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.
