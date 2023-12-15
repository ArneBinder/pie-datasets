# PIE Dataset Card for "scientific_papers"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[scientific_papers Huggingface dataset loading script](https://huggingface.co/datasets/scientific_papers).

## Data Schema

The document type for this dataset is `ScientificPapersDocument` which defines the following data fields:

- `text` (str)

and the following annotation layers:

- `abstract` (annotation type: `AbstractiveSummary`, target: `None`)
- `section_names` (annotation type: `SectionName`, targets: `None`)
