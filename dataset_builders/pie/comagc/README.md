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
[here](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/annotations.py) for the annotation
type definitions.

## Document Converters

The dataset provides predefined document converters for the following target document types:

- `pie_modules.documents.TextDocumentWithLabeledSpansAndBinaryRelations`:

  - **labeled_spans**: There are always two labeled spans in each sentence.
    The first one refers to the gene, while the second one refers to the cancer.
    Therefore, the `label` is either `"GENE"` or `"CANCER"`.
  - **binary_relations**: There is always one binary relation in each sentence.
    This relation is always established between the gene as `head` and the cancer as `tail`.
    The specific `label` is the related **gene-class**. It is obtained from inference rules (cf [here](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-323/tables/3)),
    that are based on the values of the columns CGE, CCS, IGE and PT. In case no gene-class can be inferred,
    no binary relation is added to the document. In total to 303 of the 821 examples,
    there is no rule is applicable (cf [here](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-323/tables/7)).

See [here](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/documents.py) and
[here](https://github.com/ArneBinder/pie-modules/blob/main/src/pie_modules/documents.py) for the document type
definitions.
