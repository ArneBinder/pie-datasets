from pie_documents.documents import TextDocumentWithLabeledSpansAndBinaryRelations

from pie_datasets.builders import BratBuilder, BratConfig
from pie_datasets.builders.brat import BratDocumentWithMergedSpans

URL = "https://gitlab.com/tomaye/abstrct/-/archive/master/abstrct-master.zip"
SPLIT_PATHS = {
    "neoplasm_train": "abstrct-master/AbstRCT_corpus/data/train/neoplasm_train",
    "neoplasm_dev": "abstrct-master/AbstRCT_corpus/data/dev/neoplasm_dev",
    "neoplasm_test": "abstrct-master/AbstRCT_corpus/data/test/neoplasm_test",
    "glaucoma_test": "abstrct-master/AbstRCT_corpus/data/test/glaucoma_test",
    "mixed_test": "abstrct-master/AbstRCT_corpus/data/test/mixed_test",
}


class AbstRCT(BratBuilder):
    BASE_DATASET_PATH = "DFKI-SLT/brat"
    BASE_DATASET_REVISION = "bb8c37d84ddf2da1e691d226c55fef48fd8149b5"

    BUILDER_CONFIGS = [
        BratConfig(name=BratBuilder.DEFAULT_CONFIG_NAME, merge_fragmented_spans=True),
    ]
    DOCUMENT_TYPES = {
        BratBuilder.DEFAULT_CONFIG_NAME: BratDocumentWithMergedSpans,
    }

    # we need to add None to the list of dataset variants to support the default dataset variant
    BASE_BUILDER_KWARGS_DICT = {
        dataset_variant: {"url": URL, "split_paths": SPLIT_PATHS, "trust_remote_code": True}
        for dataset_variant in ["default", None]
    }

    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpansAndBinaryRelations: {
            "spans": "labeled_spans",
            "relations": "binary_relations",
        },
    }
