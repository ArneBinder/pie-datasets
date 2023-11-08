# PIE Dataset Card for "SciDTB ArgMin"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[SciDTB ArgMin Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/scidtb_argmin).

## Data Schema

The document type for this dataset is `SciDTBArgminDocument` which defines the following data fields:

- `token` (Tuple, str)
- `id` (str, optional)
- `metadata` (dictionary, optional)

and the following annotation layers:

- `units` (annotation type: `LabeledSpan`, target: `token`)
- `relations` (annotation type: `BinaryRelation`, target: `units`)

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the annotation type definitions.

## Document Converters

[To ne added]
