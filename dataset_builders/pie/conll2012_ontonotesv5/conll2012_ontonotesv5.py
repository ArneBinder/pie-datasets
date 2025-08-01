import dataclasses
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import datasets
from pie_core import Annotation, AnnotationLayer, annotation_field
from pie_modules.annotations import LabeledSpan, NaryRelation, Span
from pie_modules.documents import (
    TextDocumentWithLabeledSpansAndLabeledPartitions,
    TokenBasedDocument,
)

from pie_datasets import GeneratorBasedBuilder


@dataclasses.dataclass(eq=True, frozen=True)
class SpanSet(Annotation):
    spans: Tuple[Span, ...]
    score: float = 1.0

    def __post_init__(self) -> None:
        # make the referenced spans unique, sort them and convert to tuples to make everything hashable
        object.__setattr__(
            self,
            "spans",
            tuple(sorted({s for s in self.spans}, key=lambda s: (s.start, s.end))),
        )


@dataclasses.dataclass(eq=True, frozen=True)
class Attribute(Annotation):
    target_annotation: Annotation
    label: str
    value: Optional[str] = None


@dataclasses.dataclass(eq=True, frozen=True)
class Predicate(Span):
    lemma: str
    framenet_id: Optional[str] = None


@dataclasses.dataclass
class Conll2012OntonotesV5Document(TokenBasedDocument):
    pos_tags: Optional[List[str]] = None
    sentences: AnnotationLayer[Span] = annotation_field(target="tokens")
    parse_trees: AnnotationLayer[Attribute] = annotation_field(target="sentences")
    speakers: AnnotationLayer[Attribute] = annotation_field(target="sentences")
    parts: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")
    coref_mentions: AnnotationLayer[Span] = annotation_field(target="tokens")
    coref_clusters: AnnotationLayer[SpanSet] = annotation_field(target="coref_mentions")
    srl_arguments: AnnotationLayer[Span] = annotation_field(target="tokens")
    srl_relations: AnnotationLayer[NaryRelation] = annotation_field(target="srl_arguments")
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")
    predicates: AnnotationLayer[Predicate] = annotation_field(target="tokens")
    word_senses: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")


def bio2spans(bio: List[str], offset: int = 0) -> List[LabeledSpan]:
    """Convert a BIO-encoded sequence of labels to a list of labeled spans.

    Args:
        bio: a BIO-encoded sequence of labels, e.g. ["B-PER", "I-PER", "O", "B-LOC", "I-LOC"]
        offset: offset to add to the start and end indices of the spans

    Returns:
        a list of labeled spans
    """

    spans = []
    prev_start_and_label: Optional[int, str] = None
    for idx, bio_value_and_label in enumerate(bio):
        bio_value = bio_value_and_label[0]
        bio_label = bio_value_and_label[2:] if bio_value != "O" else None
        if bio_value == "B":
            if prev_start_and_label is not None:
                prev_start, prev_label = prev_start_and_label
                spans.append(
                    LabeledSpan(start=prev_start + offset, end=idx + offset, label=prev_label)
                )
            prev_start_and_label = (idx, bio_label)
        elif bio_value == "I":
            if prev_start_and_label is None:
                raise ValueError(f"Invalid BIO encoding: {bio}")
        elif bio_value == "O":
            if prev_start_and_label is not None:
                prev_start, prev_label = prev_start_and_label
                spans.append(
                    LabeledSpan(start=prev_start + offset, end=idx + offset, label=prev_label)
                )
                prev_start_and_label = None

    if prev_start_and_label is not None:
        prev_start, prev_label = prev_start_and_label
        spans.append(
            LabeledSpan(start=prev_start + offset, end=len(bio) + offset, label=prev_label)
        )

    return spans


def example_to_document(
    example: Dict[str, Any],
    entity_labels: datasets.ClassLabel,
    pos_tag_labels: Optional[datasets.ClassLabel] = None,
) -> Conll2012OntonotesV5Document:
    sentences = []
    tokens = []
    pos_tags = []
    parse_trees = []
    speakers = []
    entities = []
    predicates = []
    coref_mentions = []
    coref_clusters = []
    coref_cluster_ids = []
    srl_arguments = []
    srl_relations = []
    word_senses = []
    parts = []

    last_part_id_and_start: Optional[Tuple[int, int]] = None

    for sentence_idx, sentence_dict in enumerate(example["sentences"]):
        sentence_offset = len(tokens)
        current_tokens = sentence_dict["words"]
        current_sentence = Span(start=sentence_offset, end=sentence_offset + len(current_tokens))
        sentences.append(current_sentence)

        if pos_tag_labels is not None:
            pos_tags.extend(
                [pos_tag_labels.int2str(pos_tag_id) for pos_tag_id in sentence_dict["pos_tags"]]
            )
            if pos_tag_labels.int2str is None:
                raise ValueError("pos_tag_labels.int2str is None.")
        else:
            pos_tags.extend(sentence_dict["pos_tags"])
        parse_trees.append(
            Attribute(target_annotation=current_sentence, label=sentence_dict["parse_tree"])
        )
        speakers.append(
            Attribute(target_annotation=current_sentence, label=sentence_dict["speaker"])
        )
        named_entities_bio = [
            entity_labels.int2str(entity_id) for entity_id in sentence_dict["named_entities"]
        ]
        entities.extend(bio2spans(bio=named_entities_bio, offset=len(tokens)))

        for idx, (predicate_lemma_value, predicate_framenet_id) in enumerate(
            zip(sentence_dict["predicate_lemmas"], sentence_dict["predicate_framenet_ids"])
        ):
            token_idx = sentence_offset + idx
            if predicate_lemma_value is not None or predicate_framenet_id is not None:
                predicate = Predicate(
                    start=token_idx,
                    end=token_idx + 1,
                    lemma=predicate_lemma_value,
                    framenet_id=predicate_framenet_id,
                )
                predicates.append(predicate)

        coref_clusters_dict = defaultdict(list)
        for cluster_id, start, end in sentence_dict["coref_spans"]:
            current_coref_mention = Span(
                start=start + sentence_offset, end=end + 1 + sentence_offset
            )
            coref_mentions.append(current_coref_mention)
            coref_clusters_dict[cluster_id].append(current_coref_mention)
        current_coref_clusters = [
            SpanSet(spans=tuple(spans)) for spans in coref_clusters_dict.values()
        ]
        current_coref_cluster_ids = [cluster_id for cluster_id in coref_clusters_dict.keys()]
        coref_clusters.extend(current_coref_clusters)
        coref_cluster_ids.extend(current_coref_cluster_ids)

        # handle srl_frames
        for frame_dict in sentence_dict["srl_frames"]:
            current_srl_arguments_with_roles = bio2spans(
                bio=frame_dict["frames"], offset=sentence_offset
            )
            current_srl_arguments = [
                Span(start=arg.start, end=arg.end) for arg in current_srl_arguments_with_roles
            ]
            current_srl_roles = [arg.label for arg in current_srl_arguments_with_roles]
            current_srl_relation = NaryRelation(
                arguments=tuple(current_srl_arguments),
                roles=tuple(current_srl_roles),
                label="",
            )
            srl_arguments.extend(current_srl_arguments)
            srl_relations.append(current_srl_relation)

        # handle word senses
        for idx, word_sense in enumerate(sentence_dict["word_senses"]):
            token_idx = sentence_offset + idx
            if word_sense is not None:
                word_senses.append(
                    # LabeledSpan(start=token_idx, end=token_idx + 1, label=str(int(word_sense)))
                    LabeledSpan(start=token_idx, end=token_idx + 1, label=str(word_sense))
                )

        # handle parts
        if last_part_id_and_start is not None:
            last_part_id, last_start = last_part_id_and_start
            if last_part_id != sentence_dict["part_id"]:
                parts.append(
                    LabeledSpan(start=last_start, end=sentence_offset, label=str(last_part_id))
                )
                last_part_id_and_start = (sentence_dict["part_id"], sentence_offset)
        else:
            last_part_id_and_start = (sentence_dict["part_id"], sentence_offset)

        tokens.extend(current_tokens)

    if last_part_id_and_start is not None:
        last_part_id, last_start = last_part_id_and_start
        parts.append(LabeledSpan(start=last_start, end=len(tokens), label=str(last_part_id)))

    doc = Conll2012OntonotesV5Document(
        tokens=tuple(tokens),
        id=example["document_id"],
        pos_tags=pos_tags,
    )
    # add the annotations to the document
    doc.parts.extend(parts)
    doc.sentences.extend(sentences)
    doc.parse_trees.extend(parse_trees)
    doc.speakers.extend(speakers)
    doc.entities.extend(entities)
    doc.predicates.extend(predicates)
    doc.coref_mentions.extend(coref_mentions)
    doc.coref_clusters.extend(coref_clusters)
    doc.srl_arguments.extend(srl_arguments)
    doc.srl_relations.extend(srl_relations)
    doc.word_senses.extend(word_senses)
    doc.metadata["coref_cluster_ids"] = coref_cluster_ids

    return doc


def document_to_example(
    document: Conll2012OntonotesV5Document,
    entity_labels: datasets.ClassLabel,
    pos_tag_labels: Optional[datasets.ClassLabel] = None,
) -> Dict[str, Any]:
    example = {
        "document_id": document.id,
        "sentences": [],
    }

    for idx, sentence in enumerate(document.sentences):
        sent_start = sentence.start
        sent_end = sentence.end
        sent_len = sent_end - sent_start

        predicate_lemmas = [None] * sent_len
        predicate_framenet_ids = [None] * sent_len
        for pred in document.predicates:
            if sent_start <= pred.start and pred.end <= sent_end:
                pred_len = pred.end - pred.start
                predicate_lemmas[pred.start - sent_start : pred.end - sent_start] = [
                    pred.lemma
                ] * pred_len
                if pred.framenet_id is not None:
                    predicate_framenet_ids[pred.start - sent_start : pred.end - sent_start] = [
                        pred.framenet_id
                    ] * pred_len

        word_senses = [None] * sent_len
        for sense in document.word_senses:
            if sent_start <= sense.start and sense.end <= sent_end:
                word_senses[sense.start - sent_start : sense.end - sent_start] = [
                    float(sense.label)
                ] * (sense.end - sense.start)

        named_entities = [0] * sent_len
        for ent in document.entities:
            if sent_start <= ent.start and ent.end <= sent_end:
                ent_len = ent.end - ent.start
                named_entities[ent.start - sent_start] = entity_labels.str2int("B-" + ent.label)
                if ent_len > 1:
                    named_entities[ent.start - sent_start + 1 : ent.end - sent_start] = [
                        entity_labels.str2int("I-" + ent.label)
                    ] * (ent_len - 1)

        srl_frames = []
        for srl_rel in document.srl_relations:
            span_start = min([span.start for span in srl_rel.arguments])
            span_end = max([span.end for span in srl_rel.arguments])
            if sent_start <= span_start and span_end <= sent_end:
                verb = None
                frames = ["O"] * sent_len
                for arg, role in zip(srl_rel.arguments, srl_rel.roles):
                    frames[arg.start - sent_start] = "B-" + role
                    if arg.end - arg.start > 1:
                        frames[arg.start - sent_start + 1 : arg.end - sent_start] = [
                            "I-" + role
                        ] * (arg.end - arg.start - 1)
                    # english_v4 and arabic_v4 contain some weird role names (in addition to "V") for the verb
                    if role in [
                        "V",
                        "ARG0(V",
                        "ARG1(V",
                        "C-ARG0(V",
                        "C-ARG1(V",
                        "C-ARG2(V",
                        "R-ARG0(V",
                        "R-ARG1(V",
                    ]:
                        verb = document.tokens[arg.start]
                if verb is None:
                    raise ValueError(f"Verb not found for SRL relation: {srl_rel}")
                srl_frames.append({"verb": verb, "frames": frames})

        coref_spans = []
        for cluster, id in zip(document.coref_clusters, document.metadata["coref_cluster_ids"]):
            span_start = min([span.start for span in cluster.spans])
            span_end = max([span.end for span in cluster.spans])
            if sent_start <= span_start and span_end <= sent_end:
                current_coref = [
                    [id, span.start - sent_start, span.end - sent_start - 1]
                    for span in cluster.spans
                ]
                coref_spans.extend(current_coref)

        for part in document.parts:
            if part.start <= sent_start and sent_end <= part.end:
                part_id = int(part.label)

        pos_tags = []
        if pos_tag_labels is not None:
            pos_tags.extend(
                [
                    pos_tag_labels.str2int(pos_tag)
                    for pos_tag in document.pos_tags[sent_start:sent_end]
                ]
            )
            if pos_tag_labels.int2str is None:
                raise ValueError("pos_tag_labels.str2int is None.")
        else:
            pos_tags = document.pos_tags[sent_start:sent_end]

        example_sentence = {
            "part_id": part_id,
            "words": list(document.tokens[sent_start:sent_end]),
            "pos_tags": pos_tags,
            "parse_tree": document.parse_trees[idx].label,
            "predicate_lemmas": predicate_lemmas,
            "predicate_framenet_ids": predicate_framenet_ids,
            "word_senses": word_senses,
            "speaker": document.speakers[idx].label,
            "named_entities": named_entities,
            "srl_frames": srl_frames,
            "coref_spans": coref_spans,
        }

        example["sentences"].append(example_sentence)

    return example


def convert_to_text_document_with_labeled_spans_and_labeled_partitions(
    doc: Conll2012OntonotesV5Document,
    token_separator: str = " ",
) -> TextDocumentWithLabeledSpansAndLabeledPartitions:
    start = 0
    token_offsets: List[Tuple[int, int]] = []
    for token in doc.tokens:
        end = start + len(token)
        token_offsets.append((start, end))
        # we add the separator after each token
        start = end + len(token_separator)

    text = token_separator.join([token for token in doc.tokens])

    entity_map: Dict[Tuple[int, int], LabeledSpan] = {}

    for entity in doc.entities:
        char_start = token_offsets[entity.start][0]
        char_end = token_offsets[entity.end - 1][1]
        char_offset_entity = LabeledSpan(start=char_start, end=char_end, label=entity.label)
        entity_map[(entity.start, entity.end)] = char_offset_entity

    sentence_map: Dict[Tuple[int, int], LabeledSpan] = {}
    for sentence in doc.sentences:
        char_start = token_offsets[sentence.start][0]
        char_end = token_offsets[sentence.end - 1][1]
        char_offset_sentence = LabeledSpan(start=char_start, end=char_end, label="sentence")
        sentence_map[(sentence.start, sentence.end)] = char_offset_sentence

    new_doc = TextDocumentWithLabeledSpansAndLabeledPartitions(text=text, id=doc.id)
    new_doc.labeled_spans.extend(entity_map.values())
    new_doc.labeled_partitions.extend(sentence_map.values())

    return new_doc


class Conll2012OntonotesV5Config(datasets.BuilderConfig):
    """BuilderConfig for the CoNLL formatted OntoNotes dataset."""

    def __init__(self, language=None, conll_version=None, **kwargs):
        """BuilderConfig for the CoNLL formatted OntoNotes dataset.

        Args:
          language: string, one of the language {"english", "chinese", "arabic"} .
          conll_version: string, "v4" or "v12". Note there is only English v12.
          **kwargs: keyword arguments forwarded to super.
        """
        assert language in ["english", "chinese", "arabic"]
        assert conll_version in ["v4", "v12"]
        if conll_version == "v12":
            assert language == "english"
        super().__init__(
            name=f"{language}_{conll_version}",
            description=f"{conll_version} of CoNLL formatted OntoNotes dataset for {language}.",
            version=datasets.Version("1.0.0"),  # hf dataset script version
            **kwargs,
        )
        self.language = language
        self.conll_version = conll_version


class Conll2012Ontonotesv5(GeneratorBasedBuilder):
    DOCUMENT_TYPE = Conll2012OntonotesV5Document

    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpansAndLabeledPartitions: convert_to_text_document_with_labeled_spans_and_labeled_partitions
    }

    BASE_DATASET_PATH = "conll2012_ontonotesv5"
    BASE_DATASET_REVISION = "1161216f7e7185a4b2f4d0a4e0734dc7919dfa15"

    BUILDER_CONFIGS = [
        Conll2012OntonotesV5Config(
            language=lang,
            conll_version="v4",
        )
        for lang in ["english", "chinese", "arabic"]
    ] + [
        Conll2012OntonotesV5Config(
            language="english",
            conll_version="v12",
        )
    ]

    def _generate_document_kwargs(self, dataset):
        pos_tags_feature = dataset.features["sentences"][0]["pos_tags"].feature
        return dict(
            entity_labels=dataset.features["sentences"][0]["named_entities"].feature,
            pos_tag_labels=(
                pos_tags_feature if isinstance(pos_tags_feature, datasets.ClassLabel) else None
            ),
        )

    def _generate_document(self, example, **document_kwargs):
        return example_to_document(example, **document_kwargs)
