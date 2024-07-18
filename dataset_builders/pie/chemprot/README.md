# PIE Dataset Card for "ChemProt"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[ChemProt Huggingface dataset loading script](https://huggingface.co/datasets/bigbio/chemprot).

## Data Schema

There are three versions of the dataset supported, `chemprot_full_source`, `chemprot_shared_task_eval_source` and `chemprot_bigbio_kb`.

#### `ChemprotDocument` for `chemprot_source` and `chemprot_shared_task_eval_source`

defines following fields:

- `text` (str)
- `id` (str, optional)
- `metadata` (dictionary, optional)

and the following annotation layers:

- `entities` (annotation type: `LabeledSpan`, target: `text`)
- `relations` (annotation type: `BinaryRelation`, target: `entities`)


#### `ChemprotBigbioDocument` for `chemprot_bigbio_kb`

defines following fields:

- `text` (str)
- `id` (str, optional)
- `metadata` (dictionary, optional)

and the following annotation layers:

- `passages` (annotation type: `LabeledSpan`, target: `text`)
- `entities` (annotation type: `LabeledSpan`, target: `text`)
- `relations` (annotation type: `BinaryRelation`, target: `entities`)

See [here](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/annotations.py) for the annotation
type definitions.


## Document Converters

The dataset provides predefined document converters for the following target document types:

- `pie_modules.documents.TextDocumentWithLabeledSpansAndBinaryRelations` for `ChemprotDocument`
- `pie_modules.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions` for `ChemprotBigbioDocument`

See [here](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/documents.py) for the document type
definitions.