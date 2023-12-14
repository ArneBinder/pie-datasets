from pytorch_ie.documents import TextDocumentWithLabeledSpansAndBinaryRelations

from pie_datasets.builders import BratBuilder
from pie_datasets.core.dataset import DocumentConvertersType

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

    # we need to add None to the list of dataset variants to support the default dataset variant
    BASE_BUILDER_KWARGS_DICT = {
        dataset_variant: {"url": URL, "split_paths": SPLIT_PATHS}
        for dataset_variant in ["default", "merge_fragmented_spans", None]
    }

    @property
    def document_converters(self) -> DocumentConvertersType:
        if self.config.name == "default":
            return {}
        elif self.config.name == "merge_fragmented_spans":
            return {
                TextDocumentWithLabeledSpansAndBinaryRelations: {
                    "spans": "labeled_spans",
                    "relations": "binary_relations",
                },
            }
        else:
            raise ValueError(f"Unknown dataset variant: {self.config.name}")
