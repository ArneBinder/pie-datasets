# PIE Dataset Card for "conll2003"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[BRAT Huggingface dataset loading script](https://huggingface.co/datasets/DFKI-SLT/brat).

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

The `Attribute` annotation type is defined as follows:

- `annotation` (type: `Annotation`): the annotation to which the attribute is attached
- `label` (type: `str`)
- `value` (type: `str`, optional)
- `score` (type: `float`, optional, not included in comparison)

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the remaining annotation type definitions.

## Document Converters

The dataset provides no predefined document converters because the BRAT format is very flexible and can be used
for many different tasks. You can add your own document converter by doing the following:

```python
import dataclasses
from typing import Optional

from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument
from pytorch_ie.annotations import LabeledSpan

from pie_datasets import DatasetDict

# define your document class
@dataclasses.dataclass
class MyDocument(TextBasedDocument):
    my_field: Optional[str] = None
    my_span_annotations: AnnotationList[LabeledSpan] = annotation_field(target="text")

# define your document converter
def my_converter(document: BratDocumentWithMergedSpans) -> MyDocument:
    # create your document with the data from the original document.
    # The fields "text", "id" and "metadata" are derived from the TextBasedDocument.
    my_document = MyDocument(id=document.id, text=document.text, metadata=document.metadata, my_field="my_value")

    # create a new span annotation
    new_span = LabeledSpan(label="my_label", start=2, end=10)
    # add the new span annotation to your document
    my_document.my_span_annotations.append(new_span)

    # add annotations from the document to your document
    for span in document.spans:
        # we need to copy the span because an annotation can only be attached to one document
        my_document.my_span_annotations.append(span.copy())

    return my_document


# load the dataset. We use the "merge_fragmented_spans" dataset variant here
# because it provides documents of type BratDocumentWithMergedSpans.
dataset = DatasetDict.load_dataset("pie/brat", name="merge_fragmented_spans", data_dir="path/to/brat/data")

# attach your document converter to the dataset
dataset.register_document_converter(my_converter)

# convert the dataset
converted_dataset = dataset.to_document_type(MyDocument)
```
