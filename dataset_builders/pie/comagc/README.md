# PIE Dataset Card for "CoMAGC"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[CoMAGC Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/CoMAGC).

## Data Schema

The document type for this dataset is `ComagcDocument` which defines the following data fields:

- `text` (str)
- `id` (str, optional)
- `metadata` (dictionary, optional)

and the following annotation layers:

- `entities` (annotation type: `LabeledSpan`, target: `text`)
- `relations` (annotation type: `BinaryRelation`, target: `entities`)

See [here](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/annotations.py) and
[here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the annotation
type definitions.

## Document Converters

The dataset provides predefined document converters for the following target document types:

- `pie_modules.documents.TextDocumentWithLabeledSpansAndBinaryRelations`

See [here](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/documents.py) and
[here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.
