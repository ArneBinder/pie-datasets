# PIE Dataset Card for "ArgMicro"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[ArgMicro Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/argmicro).

## Data Schema

The document type for this dataset is `ArgMicroDocument` which defines the following data fields:

- `id` (str)
- `text` (str)
- `topic_id` (str, optional)
- `stance` (str, optional)
- `metadata` (dictionary, optional)

and the following annotation layers:

- `edus` (annotation type: `Span`, target: `text`)
- `adus` (annotation type: `LabeledAnnotationCollection`, target: `edus`)
- `relations` (annotation type: `MultiRelation`, target: `adus`)

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the annotation type definitions.

## Document Converters

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations`

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.
