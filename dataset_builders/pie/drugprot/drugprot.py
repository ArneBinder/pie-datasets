from dataclasses import dataclass

import datasets
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndBinaryRelations
from pytorch_ie.utils.span import tokens_and_tags_to_text_and_labeled_spans

from pie_datasets import GeneratorBasedBuilder


@dataclass
class DrugprotDocument(TextDocumentWithLabeledSpansAndBinaryRelations):
    pass


class Drugprot(GeneratorBasedBuilder):
    DOCUMENT_TYPE = DrugprotDocument

    BASE_DATASET_PATH = "bigbio/drugprot"
    BASE_DATASET_REVISION = "38ff03d68347aaf694e598c50cb164191f50f61c"

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="drugprot", version=datasets.Version("1.0.0"), description="DrugProt dataset"
        ),
    ]

    # DOCUMENT_CONVERTERS = {
    #     TextDocumentWithLabeledSpans: {
    #         # just rename the layer
    #         "entities": "labeled_spans",
    #     }
    # }

    def _generate_document_kwargs(self, dataset):
        return {"int_to_str": dataset.features["ner_tags"].feature.int2str}

    def _generate_document(self, example):
        doc_id = example["document_id"]
        text = example["text"]
        metadata = {k: example[k] for k in ("title", "abstract")}
        text, ner_spans = tokens_and_tags_to_text_and_labeled_spans(tokens=tokens, tags=ner_tags)
        # spans =
        # relations =

        document = DrugprotDocument(
            text=text,
            id=doc_id,
            metadata=metadata,
            labeled_spans=spans,
            binary_relations=relations,
        )

        for span in sorted(ner_spans, key=lambda span: span.start):
            document.entities.append(span)

        return document
