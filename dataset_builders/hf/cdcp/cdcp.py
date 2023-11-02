"""The Cornell eRulemaking Corpus (CDCP) dataset for English Argumentation Mining."""
import glob
import json
from os.path import abspath, isdir
from pathlib import Path

import datasets

_CITATION = """\
@inproceedings{niculae-etal-2017-argument,
    title = "Argument Mining with Structured {SVM}s and {RNN}s",
    author = "Niculae, Vlad  and
      Park, Joonsuk  and
      Cardie, Claire",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1091",
    doi = "10.18653/v1/P17-1091",
    pages = "985--995",
    abstract = "We propose a novel factor graph model for argument mining, designed for settings in which the argumentative relations in a document do not necessarily form a tree structure. (This is the case in over 20{\\%} of the web comments dataset we release.) Our model jointly learns elementary unit type classification and argumentative relation prediction. Moreover, our model supports SVM and RNN parametrizations, can enforce structure constraints (e.g., transitivity), and can express dependencies between adjacent relations and propositions. Our approaches outperform unstructured baselines in both web comments and argumentative essay datasets.",
}
"""

_DESCRIPTION = "The CDCP dataset for English Argumentation Mining"

_HOMEPAGE = ""

_LICENSE = ""


# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URL = "https://facultystaff.richmond.edu/~jpark/data/cdcp_acl17.zip"

_VERSION = datasets.Version("1.0.0")

_SPAN_CLASS_LABELS = ["fact", "policy", "reference", "testimony", "value"]
_RELATION_CLASS_LABELS = ["evidence", "reason"]


class CDCP(datasets.GeneratorBasedBuilder):
    """CDCP is a argumentation mining dataset."""

    BUILDER_CONFIGS = [datasets.BuilderConfig(name="default")]

    DEFAULT_CONFIG_NAME = "default"  # type: ignore

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "text": datasets.Value("string"),
                "propositions": datasets.Sequence(
                    {
                        "start": datasets.Value("int32"),
                        "end": datasets.Value("int32"),
                        "label": datasets.ClassLabel(names=_SPAN_CLASS_LABELS),
                        # urls are replaced with the string "__URL__" in the text. This contains the original url.
                        "url": datasets.Value("string"),
                    }
                ),
                "relations": datasets.Sequence(
                    {
                        "head": datasets.Value("int32"),
                        "tail": datasets.Value("int32"),
                        "label": datasets.ClassLabel(names=_RELATION_CLASS_LABELS),
                    }
                ),
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive

        if dl_manager.manual_dir is not None:
            base_path = abspath(dl_manager.manual_dir)
            if not isdir(base_path):
                base_path = dl_manager.extract(base_path)
        else:
            base_path = dl_manager.download_and_extract(_URL)
        base_path = Path(base_path) / "cdcp"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"path": base_path / "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"path": base_path / "test"}
            ),
        ]

    def _generate_examples(self, path):
        """Yields examples."""
        # This method will receive as arguments the `gen_kwargs` defined in the previous `_split_generators` method.
        # It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        # The key is not important, it's more here for legacy reason (legacy from tfds)

        _id = 0
        text_file_names = sorted(glob.glob(f"{path}/*.txt"))
        for text_file_name in text_file_names:
            txt_fn = Path(text_file_name)
            ann_fn = txt_fn.with_suffix(".ann.json")
            with open(txt_fn, encoding="utf-8") as f:
                text = f.read()
            with open(ann_fn, encoding="utf-8") as f:
                annotations = json.load(f)
            # example content of annotations:
            # {
            #   'evidences': [[[8, 8], 7]],
            #   'prop_labels': ['testimony', 'testimony', 'value'],
            #   'prop_offsets': [[0, 114], [114, 209], [209, 235]],
            #   'reasons': [[[2, 2], 1], [ 0, 0], 2]],
            #   'evidences': [[[2, 2], 1], [ 0, 0], 2]],
            #   'url': {
            #       "3": "http://usa.visa.com/personal/using_visa/checkout_fees/",
            #       "4": "http://usa.visa.com/download/merchants/surcharging-faq-by-merchants.pdf"
            #   }
            # }
            propositions = [
                {
                    "start": start,
                    "end": end,
                    "label": label,
                    "url": annotations["url"].get(str(idx), ""),
                }
                for idx, ((start, end), label) in enumerate(
                    zip(annotations["prop_offsets"], annotations["prop_labels"])
                )
            ]
            relations = []
            for (tail_first_idx, tail_last_idx), head_idx in annotations["evidences"]:
                for tail_idx in range(tail_first_idx, tail_last_idx + 1):
                    relations.append({"head": head_idx, "tail": tail_idx, "label": "evidence"})
            for (tail_first_idx, tail_last_idx), head_idx in annotations["reasons"]:
                for tail_idx in range(tail_first_idx, tail_last_idx + 1):
                    relations.append({"head": head_idx, "tail": tail_idx, "label": "reason"})
            yield _id, {
                "id": txt_fn.stem,
                "text": text,
                "propositions": propositions,
                "relations": relations,
            }
            _id += 1
