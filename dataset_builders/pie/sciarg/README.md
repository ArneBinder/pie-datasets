# PIE Dataset Card for "sciarg"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the SciArg dataset.

More information about the dataset can be found in the [HF dataset card](https://huggingface.co/datasets/DFKI-SLT/sciarg/blob/main/README.md). However, this is not the wrapper for the [SciArg Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/sciarg), but rather [BRAT Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/brat).

Therefore, the current SciArg dataset follows the data structure described in [PIE-Brat dataset card](https://huggingface.co/datasets/pie/brat).

## Dataset Variants

The dataset provides the following variants:

- `default`: The original dataset. Documents are of type `BratDocument` (with `LabeledMultiSpan` annotations, see below).
- `merge_fragmented_spans`: Documents are of type `BratDocumentWithMergedSpans` (this variant merges spans that are
  fragmented into simple `LabeledSpans`, see below).

## Data Schema

The document type for this dataset is `BratDocument` or `BratDocumentWithMergedSpans`, depending on if the
data was loaded with `merge_fragmented_spans=True` (default: `False`). They define the following data fields:

- `text` (str)
- `id` (str, optional)
- `metadata` (dictionary, optional)

and the following annotation layers:

- `spans` (annotation type: `LabeledMultiSpan` in the case of `BratDocument` and `LabeledSpan` and in the case of `BratDocumentWithMergedSpans`, target: `text`)
- `relations` (annotation type: `BinaryRelation`, target: `spans`)
- `span_attributes` (annotation type: `Attribute`, target: `spans`)
- `relation_attributes` (annotation type: `Attribute`, target: `relations`)

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the common annotation type definitions, and [here](https://huggingface.co/datasets/pie/brat/blob/main/README.md?code=true#L30-L41) for the special-type definitions of `LabeledMultiSpan` and `Attribute`.

## Document Converters

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations`
  - `LabeledSpans`, converted from `BratDocument`'s `spans`
    - labels: `background_claim`, `own_claim`, `data`
    - if `spans` contain whitespace at the beginning and/or the end, the whitespace are trimmed out.
  - `BinraryRelations`, converted from `BratDocument`'s `relations`
    - labels: `supports`, `contradicts`, `semantically_same`, `parts_of_same`
    - if the `relations` label is `semantically_same` or `parts_of_same`, they are merged if they are the same arguments after sorting.
- `pytorch_ie.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions`
  - `LabeledSpans`, as above
  - `BinaryRelations`, as above
  - `LabeledPartitions`, partitioned `BratDocument`'s `text`, according to the paragraph, using regex.
    - labels: `title`, `abstract`, `H1`

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.
