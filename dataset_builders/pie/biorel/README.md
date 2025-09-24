# PIE Dataset Card for "BioRel"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[BioRel Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/BioRel).

## Data Schema

The document type for this dataset is `BioRelDocument` which defines the following data fields:

- `text` (str)

and the following annotation layers:

- `entities` (annotation type: `SpanWithIdAndName`, target: `text`)
- `relations` (annotation type: `BinaryRelation`, target: `entities`)

`SpanWithIdAndName` is a custom annotation type that extends typical `Span` with the following data fields:

- `id` (str, for entity identification)
- `name` (str, entity string between span start and end)

See [here](https://github.com/ArneBinder/pie-documents/blob/main/src/pie_documents/annotations.py) and
[here](https://github.com/ArneBinder/pie-documents/blob/main/src/pie_documents/annotations.py) for the annotation
type definitions.

## Document Converters

The dataset provides predefined document converters for the following target document types:

- `pie_documents.documents.TextDocumentWithLabeledSpansAndBinaryRelations`

See [here](https://github.com/ArneBinder/pie-documents/blob/main/src/pie_documents/documents.py) and
[here](https://github.com/ArneBinder/pie-documents/blob/main/src/pie_documents/documents.py) for the document type
definitions.
