# PIE Dataset Card for "CDCP"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[CDCP Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/cdcp).

## Data Schema

The document type for this dataset is `CDCPDocument` which defines the following data fields:

- `text` (str)
- `id` (str, optional)
- `metadata` (dictionary, dataclasses)

and the following annotation layers:

- `propositions` (annotation type: `LabeledSpan`, target: `text`)
- `relations` (annotation type: `BinaryRelation`, target: `propositions`)
- `urls` (annotation type: `Attribute`, target: `propositions`)

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the annotation type definitions.

## Document Converters

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations`

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.
