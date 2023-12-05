# PIE Dataset Card for "argmicro"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[ArgMicro Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/argmicro).

## Dataset Variants

The dataset contains two `BuilderConfig`'s:

- `de`: with the original texts collection in German
- `en`: with the English-translated texts

## Data Schema

The document type for this dataset is `ArgMicroDocument` which defines the following data fields:

- `text` (str)
- `id` (str, optional)
- `topic_id` (str, optional)
- `metadata` (dictionary, optional)

and the following annotation layers:

- `stance` (annotation type: `Label`)
  - description: A document may contain one of these `stance` labels: `pro`, `con`, `unclear`, or no label when it is undefined (see [here](https://huggingface.co/datasets/DFKI-SLT/argmicro/blob/main/argmicro.py#L35) for reference).
- `edus` (annotation type: `Span`, target: `text`)
- `adus` (tuple, annotation type: `LabeledAnnotationCollection`, target: `edus`)
  - description: each element of `adus` may consist of several entries from `edus`, so we require `LabeledAnnotationCollection` as annotation type. This is originally indicated by `seg` edges in the data.
  - `LabeledAnnotationCollection` has the following fields:
    - `annotations` (annotation type: `Span`, target: `text`)
    - `label` (str, optional), values: `opp`, `pro` (see [here](https://huggingface.co/datasets/DFKI-SLT/argmicro/blob/main/argmicro.py#L36))
- `relations` (annotation type: `MultiRelation`, target: `adus`)
  - description: Undercut (`und`) relations originally target other relations (i.e. edges), but we let them target the `head` of the targeted relation instead. The original state can be deterministically reconstructed by taking the label into account. Furthermore, the head of additional source (`add`) relations are integrated into the head of the target relation (note that this propagates along `und` relations). We model this with `MultiRelation`s whose `head` and `tail` are of type `LabeledAnnotationCollection`.
  - `MultiRelation` has the following fields:
    - `head` (tuple, annotation type: `LabeledAnnotationCollection`, target: `adus`)
    - `tail` (tuple, annotation type: `LabeledAnnotationCollection`, target: `adus`)
    - `label` (str, optional), values: `sup`, `exa`, `reb`, `und` (see [here](https://huggingface.co/datasets/DFKI-SLT/argmicro/blob/main/argmicro.py#L37) for reference, but note that helper relations `seg` and `add` are not there anymore, see above).

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the annotation type definitions.

## Document Converters

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpansAndBinaryRelations`
  - `LabeledSpans`, converted from `ArgMicroDocument`'s `adus`
    - labels: `opp`, `pro`
    - if an ADU contains multiple spans (i.e. EDUs), we take the start of the first EDU and the end of the last EDU as the boundaries of the new `LabeledSpan`. We also raise exceptions if any newly created `LabeledSpan`s overlap.
  - `BinraryRelations`, converted from `ArgMicroDocument`'s `relations`
    - labels: `sup`, `reb`, `und`, `joint`, `exa`
    - if the `head` or `tail` consists of multiple `adus`, then we build `BinaryRelation`s with all `head`-`tail` combinations and take the label from the original relation. Then, we build `BinaryRelations`' with label `joint` between each component that previously belongs to the same `head` or `tail`, respectively.
  - `metadata`, we keep the `ArgMicroDocument`'s `metadata`, but `stance` and `topic_id`.

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.
