# PIE Dataset Card for "conll2012_ontonotesv5"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[CoNLL 2012 OntoNotes v.5.0 Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/conll2012_ontonotesv5).

## Dataset Variants

This dataset contains data in three languages and two versions:

- `arabic_v4`
- `chinese_v4`
- `english_v4`
- `english_v12`

## Data Schema

The document type for this dataset is `Conll2012OntonotesV5Document` which defines the following data fields:

*TBA*

and the following annotation layers:

*TBA*

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the annotation type definitions.

## Document Converters

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpansAndLabeledPartitions`
  - *TBA*

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.
