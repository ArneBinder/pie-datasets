# PIE Dataset Card for "aae2"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the Argument Annotated Essays v2 dataset.

TODO: Since there is no respective HF dataset card for aae2, we should all respective information here.

TODO: Shortly reference the PIE-Brat dataset card.

## Data Schema

TODO

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the remaining annotation type definitions.

## Document Converters

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations`
  - TODO
- `pytorch_ie.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions`
  - TODO (may reference the above)

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.