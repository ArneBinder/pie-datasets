# PIE Dataset Card for CoNLL2012 shared task data based on OntoNotes 5.0

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) (PIE) wrapper for the
[CoNLL 2012 OntoNotes v.5.0 Huggingface dataset loading script](https://huggingface.co/datasets/conll2012_ontonotesv5).

## Dataset Variants

This dataset contains data in three languages and two versions:

- `arabic_v4`
- `chinese_v4`
- `english_v4`
- `english_v12`

## Data Schema

The document type for this dataset is `Conll2012OntonotesV5Document` which defines the following data fields:

- `id` (str)
- `tokens` (tuple)
- `pos_tags` (list)
- `metadata` (dictionary, optional)

and the following annotation layers:

- `entities` (annotation type: `LabeledSpan`, target: `tokens`)
- `parts` (annotation type: `LabeledSpan`, target: `tokens`)
- `predicates` (annotation type: `Predicate`, target: `tokens`)
- `sentences` (annotation type: `Span`, target: `tokens`)
- `coref_mentions` (annotation type: `Span`, target: `tokens`)
- `coref_clusters` (annotation type: `SpanSet`, target: `coref_mentions`)
- `srl_arguments` (annotation type: `Span`, target: `tokens`)
- `srl_relations` (annotation type: `NaryRelation`, target: `srl_arguments`)
- `word_senses` (annotation type: `LabeledSpan`, target: `tokens`)
- `speakers` (annotation type: `Attribute`, target: `sentences`)
- `parse_trees` (annotation type: `Attribute`, target: `sentences`)

See [here](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/annotations.py) for the annotation type definitions.

## Document Converters

The dataset provides document converters for the following target document types:

- `pie_modules.documents.TextDocumentWithLabeledSpansAndLabeledPartitions`

See [here](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/documents.py) for the document type
definitions.
