# PIE Dataset Card for "squad_v2"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[squad_v2 Huggingface dataset loading script](https://huggingface.co/datasets/squad_v2).

## Data Schema

The document type for this dataset is `SquadV2Document` which defines the following data fields:

- `text` (str)
- `id` (str, optional)
- `metadata` (dictionary, optional)
- `title` (str, optional)

and the following annotation layers:

- `questions` (annotation type: `Question`, target: `None`)
- `answers` (annotation type: `ExtractiveAnswer`, targets: `text` and `questions`)

See [here](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/annotations.py) for the annotation
type definitions.

## Document Converters

The dataset provides predefined document converters for the following target document types:

- `pie_modules.documents.ExtractiveQADocument` (simple cast without any conversion)

See [here](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/documents.py) for the document type
definitions.
