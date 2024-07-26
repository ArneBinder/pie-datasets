# PIE Dataset Card for "CoMAGC"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[CoMAGC Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/CoMAGC).

## Data Schema

The document type for this dataset is `ComagcDocument` which defines the following data fields:

- `pmid` (str): unique sentence identifier
- `sentence` (str)
- `cancer_type` (str)
- `cge` (str): change in gene expression
- `ccs` (str): change in cell state
- `pt` (str, optional): proposition type
- `ige` (str, optional): initial gene expression level

and the following annotation layers:

- `gene` (annotation type: `NamedSpan`, target: `sentence`)
- `cancer` (annotation type: `NamedSpan`, target: `sentence`)
- `expression_change_keyword1` (annotation type: `SpanWithNameAndType`, target: `sentence`)
- `expression_change_keyword2` (annotation type: `SpanWithNameAndType`, target: `sentence`)

`NamedSpan` is a custom annotation type that extends typical `Span` with the following data fields:

- `name` (str): entity string between span start and end

`SpanWithNameAndType` is a custom annotation type that extends typical `Span` with the following data fields:

- `name` (str): entity string between span start and end
- `type` (str): entity type classifying the expression

See [here](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/annotations.py) and
[here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the annotation
type definitions.

## Document Converters

The dataset provides predefined document converters for the following target document types:

- `pie_modules.documents.TextDocumentWithLabeledSpansAndBinaryRelations`

See [here](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/documents.py) and
[here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.

Note: The _labels_ for the binary relations are defined using a rule-based approach,
which is described in detail in the `get_relation_label()` method. The label
`"UNIDENTIFIED"` for relations is introduced for the sake of completeness and
is not part of the original dataset.
