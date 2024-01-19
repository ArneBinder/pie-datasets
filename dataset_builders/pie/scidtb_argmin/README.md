# PIE Dataset Card for "SciDTB Argmin"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[SciDTB ArgMin Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/scidtb_argmin).

## Data Schema

The document type for this dataset is `SciDTBArgminDocument` which defines the following data fields:

- `tokens` (tuple of string)
- `id` (str, optional)
- `metadata` (dictionary, optional)

and the following annotation layers:

- `units` (annotation type: `LabeledSpan`, target: `tokens`)
- `relations` (annotation type: `BinaryRelation`, target: `units`)

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the annotation type definitions.

## Document Converters

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations`
  - `labeled_spans`: `LabeledSpan` annotations, converted from`SciDTBArgminDocument`'s `units`
    - labels: `proposal`, `assertion`, `result`, `observation`, `means`, `description`
    - tuples of `tokens` are joined with a whitespace to create `text` for `LabeledSpans`
  - `binary_relations`: `BinaryRelation` annotations, converted from `SciDTBArgminDocument`'s `relations`
    - labels: `support`, `attack`, `additional`, `detail`, `sequence`

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.
