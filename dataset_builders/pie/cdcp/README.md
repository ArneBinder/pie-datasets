# PIE Dataset Card for "cdcp"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[CDCP Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/cdcp).

## Data Schema

The document type for this dataset is `CDCPDocument` which defines the following data fields:

- `text` (str)
- `id` (str, optional)
- `metadata` (dictionary, optional)

and the following annotation layers:

- `propositions` (annotation type: `LabeledSpan`, target: `text`)
- `relations` (annotation type: `BinaryRelation`, target: `propositions`)
- `urls` (annotation type: `Attribute`, target: `propositions`)

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the annotation type definitions.

## Document Converters

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations`
  - `labeled_spans`: `LabeledSpan` annotations, converted from `CDCPDocument`'s `propositions`
    - labels: `fact`, `policy`, `reference`, `testimony`, `value`
    - if `propositions` contain whitespace at the beginning and/or the end, the whitespace are trimmed out.
  - `binary_relations`: `BinaryRelation` annotations, converted from `CDCPDocument`'s `relations`
    - labels:  `reason`, `evidence`

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.
