import json
import logging
import os
from collections import defaultdict
from copy import copy
from typing import Any, Dict, Iterable, List

import datasets
from datasets import GeneratorBasedBuilder

logger = logging.getLogger(__name__)

_DESCRIPTION = """\
SciFact, a dataset of 1.4K expert-written scientific claims paired with evidence-containing abstracts, and annotated \\
with labels and rationales. This version differs from `allenai/scifact` on HF because we do not have separate splits \\
for claims and a corpus, instead we combine documents with claims that it supports or refutes, note that there are \\
also some documents that do not have any claims associated with them as well as there are some claims that do not \\
have any evidence. In the latter case we assign all such claims to the DUMMY document with ID -1 and without any text \\
(i.e. abstract sentences).
"""

DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SUBDIR = "data"

VARIANT_DOCUMENTS = "as_documents"
VARIANT_CLAIMS = "as_claims"


class ScifactConfig(datasets.BuilderConfig):
    """BuilderConfig for Scifact."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SciFact(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ScifactConfig(
            name=VARIANT_DOCUMENTS,
            description="Documents that serve as evidence for some claims that are split into train, test, dev",
        ),
        ScifactConfig(
            name=VARIANT_CLAIMS,
            description="Documents that serve as evidence for some claims that are split into train, test, dev",
        ),
    ]

    def _info(self):
        # Specifies the datasets.DatasetInfo object
        if self.config.name == VARIANT_DOCUMENTS:
            features = {
                "doc_id": datasets.Value("int32"),  # document ID
                "title": datasets.Value("string"),  # document title
                "abstract": datasets.features.Sequence(
                    datasets.Value("string")
                ),  # document sentences
                "structured": datasets.Value(
                    "bool"
                ),  # whether the abstract is structured, i.e. has OBJECTIVE, CONCLUSION, METHODS marked in the text
                "claims": datasets.features.Sequence(
                    feature={
                        "id": datasets.Value(dtype="int32", id=None),
                        "claim": datasets.Value(dtype="string", id=None),
                        "evidence": datasets.features.Sequence(
                            feature={
                                "label": datasets.Value(dtype="string", id=None),
                                "sentences": datasets.features.Sequence(
                                    datasets.Value(dtype="int32", id=None)
                                ),
                            }
                        ),
                    }
                ),  # list of claims associated with the document
            }
        elif self.config.name == VARIANT_CLAIMS:
            features = {
                "id": datasets.Value("int32"),  # document ID
                "claim": datasets.Value(dtype="string", id=None),
                "cited_docs": datasets.features.Sequence(
                    feature={
                        "doc_id": datasets.Value(dtype="int32", id=None),
                        "title": datasets.Value("string"),  # document title
                        "abstract": datasets.features.Sequence(
                            datasets.Value("string")
                        ),  # document sentences
                        "structured": datasets.Value(
                            "bool"
                        ),  # whether the abstract is structured, i.e. has OBJECTIVE, CONCLUSION, METHODS marked in the text
                        "evidence": datasets.features.Sequence(
                            feature={
                                "label": datasets.Value(dtype="string", id=None),
                                "sentences": datasets.features.Sequence(
                                    datasets.Value(dtype="int32", id=None)
                                ),
                            }
                        ),
                    }
                ),  # list of claims associated with the document
            }
        else:
            raise ValueError(f"unknown dataset variant: {self.config.name}")

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page
            description=_DESCRIPTION,
            features=datasets.Features(features),
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://scifact.apps.allenai.org/",
        )

    def _generate_examples(self, claims_filepath: str, corpus_filepath: str):
        """Yields examples."""
        with open(claims_filepath) as f:
            claim_data = [json.loads(line) for line in f.readlines()]

        with open(corpus_filepath) as f:
            corpus_docs = [json.loads(line) for line in f.readlines()]

        if self.config.name == VARIANT_DOCUMENTS:
            doc_id2claims = defaultdict(list)
            for claim in claim_data:
                cited_doc_ids = claim.pop("cited_doc_ids", [-1])
                evidence = claim.pop("evidence", dict())
                for cited_doc_id in cited_doc_ids:
                    current_claim = claim.copy()
                    current_claim["evidence"] = evidence.get(str(cited_doc_id), [])
                    doc_id2claims[cited_doc_id].append(current_claim)
            dummy_doc = {"doc_id": -1, "title": "", "abstract": [], "structured": False}
            corpus_docs = [dummy_doc] + corpus_docs

            for id_, doc in enumerate(corpus_docs):
                doc = doc.copy()
                doc["claims"] = doc_id2claims.get(doc["doc_id"], [])
                yield id_, doc
        elif self.config.name == VARIANT_CLAIMS:
            doc_id2doc = {doc["doc_id"]: doc for doc in corpus_docs}
            for _id, claim in enumerate(claim_data):
                evidence = claim.pop("evidence", {})
                cited_doc_ids = claim.pop("cited_doc_ids", [])
                claim["cited_docs"] = []
                for cited_doc_id in cited_doc_ids:
                    doc = copy(doc_id2doc[cited_doc_id])
                    doc["evidence"] = evidence.get(str(cited_doc_id), [])
                    claim["cited_docs"].append(doc)
                yield _id, claim
        else:
            raise ValueError(f"unknown dataset variant: {self.config.name}")

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles."""
        if dl_manager.manual_dir is None:
            data_dir = os.path.join(dl_manager.download_and_extract(DATA_URL), SUBDIR)
        else:
            # Absolute path of the manual_dir
            data_dir = os.path.abspath(dl_manager.manual_dir)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "claims_filepath": os.path.join(data_dir, "claims_train.jsonl"),
                    "corpus_filepath": os.path.join(data_dir, "corpus.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "claims_filepath": os.path.join(data_dir, "claims_dev.jsonl"),
                    "corpus_filepath": os.path.join(data_dir, "corpus.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "claims_filepath": os.path.join(data_dir, "claims_test.jsonl"),
                    "corpus_filepath": os.path.join(data_dir, "corpus.jsonl"),
                },
            ),
        ]

    def _convert_to_output_eval_format(
        self, data: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Output should have the format as specified here:

        https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/doc/evaluation.md#submission-format
        Each claim is represented as Dict with:
            "id": int An integer claim ID.
            "evidence": Dict[str, Dict] The evidence for the claim.
            "doc_id": Dict[str, Any] The sentences and label for a single document.
                 "sentences": List[int]
                 "label": str
        """
        if self.config.name == VARIANT_DOCUMENTS:
            # Collect all claim-level annotations from all documents
            claim2doc2sent_with_label = dict()
            for document in data:
                doc_id = document["doc_id"]
                # Skip if document does not have any related claims
                if len(document["claims"]["claim"]) == 0:
                    continue
                for idx in range(len(document["claims"]["claim"])):
                    claim_id = document["claims"]["id"][idx]
                    claim_text = document["claims"]["claim"][idx]
                    claim_evidence = document["claims"]["evidence"][idx]
                    if claim_id not in claim2doc2sent_with_label:
                        claim2doc2sent_with_label[claim_id] = dict()
                    if doc_id not in claim2doc2sent_with_label[claim_id]:
                        if len(claim_evidence["label"]) > 0:
                            ev_label = claim_evidence["label"][0]
                            claim2doc2sent_with_label[claim_id][doc_id] = {
                                "label": ev_label,
                                "sentences": [],
                            }
                            for ev_sentences in claim_evidence["sentences"]:
                                claim2doc2sent_with_label[claim_id][doc_id]["sentences"].extend(
                                    ev_sentences
                                )

            outputs = []
            for claim_id in claim2doc2sent_with_label:
                claim_dict = {"id": claim_id, "evidence": dict()}
                for doc_id in claim2doc2sent_with_label[claim_id]:
                    claim_dict["evidence"][doc_id] = {
                        "sentences": claim2doc2sent_with_label[claim_id][doc_id]["sentences"],
                        "label": claim2doc2sent_with_label[claim_id][doc_id]["label"],
                    }
                outputs.append((int(claim_id), claim_dict.copy()))

            outputs_sorted_by_claim_ids = [
                claim for claim_id, claim in sorted(outputs, key=lambda x: x[0])
            ]

            return outputs_sorted_by_claim_ids

        elif self.config.name == VARIANT_CLAIMS:
            raise NotImplementedError(
                f"_convert_to_output_eval_format is not yet implemented for dataset variant {self.config.name}"
            )
        else:
            raise ValueError(f"unknown dataset variant: {self.config.name}")
