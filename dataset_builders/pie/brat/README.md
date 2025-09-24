# PIE Dataset Card for "brat"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[BRAT Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/brat).

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

The `LabeledMultiSpan` annotation type is defined as follows:

- `slices` (type: `Tuple[Tuple[int, int], ...]`): the slices consisting if start (including) and end (excluding) indices of the spans
- `label` (type: `str`)
- `score` (type: `float`, optional, not included in comparison)

The `Attribute` annotation type is defined as follows:

- `annotation` (type: `Annotation`): the annotation to which the attribute is attached
- `label` (type: `str`)
- `value` (type: `str`, optional)
- `score` (type: `float`, optional, not included in comparison)

See [here](https://github.com/ArneBinder/pie-documents/blob/main/src/pie_documents/annotations.py) for the remaining annotation type definitions.

## Document Converters

The dataset provides no predefined document converters because the BRAT format is very flexible and can be used
for many different tasks.
