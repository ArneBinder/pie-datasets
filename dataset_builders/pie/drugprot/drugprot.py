from dataclasses import dataclass

import datasets
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndBinaryRelations

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
            name="drugprot_source",
            version=datasets.Version("1.0.2"),
            description="DrugProt source schema",
        ),
        # datasets.BuilderConfig(
        #     name="drugprot_bigbio_kb", version=datasets.Version("1.0.0"), description="DrugProt BigBio schema"
        # ), # not ready yet
    ]

    # DOCUMENT_CONVERTERS = {
    #     TextDocumentWithLabeledSpans: {
    #         # just rename the layer
    #         "entities": "labeled_spans",
    #     }
    # }

    def _generate_document_kwargs(self, dataset):
        return {}

    def _generate_document(self, example):
        text = example["text"]
        doc_id = example["document_id"]
        metadata = {k: example[k] for k in ("title", "abstract")}
        id2start = {}
        for entity in example["entities"]:
            id2start[entity["id"]] = entity["offset"][0]
        document = DrugprotDocument(
            text=text,
            id=doc_id,
            metadata=metadata,
        )

        for span in sorted(example["entities"], key=lambda span: span["offset"][0]):
            labeled_span = LabeledSpan(
                start=span["offset"][0],
                end=span["offset"][1],
                label=span["type"],
            )
            document.labeled_spans.append(labeled_span)
        for relation in sorted(example["relations"], key=lambda relation: relation["id"]):
            document.binary_relations.append(
                BinaryRelation(
                    head=[
                        span
                        for span in document.labeled_spans
                        if span.start == id2start[relation["arg1_id"]]
                    ][0],
                    tail=[
                        span
                        for span in document.labeled_spans
                        if span.start == id2start[relation["arg2_id"]]
                    ][0],
                    label=relation["type"],
                )
            )
        return document
