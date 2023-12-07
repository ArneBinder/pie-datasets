# Dataset Card for "ArgMicro"

*Note: this is the edited version from the existing HF dataset card. 05.12.23*

## Dataset Description

- **Homepage:** [https://github.com/anlausch/ArguminSci](https://github.com/anlausch/ArguminSci)
- **Repository:** [https://github.com/anlausch/ArguminSci](https://github.com/anlausch/ArguminSci)
- **Paper:** [An argument-annotated corpus of scientific publications](https://aclanthology.org/W18-5206.pdf)
- **Point of Contact:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Dataset Summary

#TODO

The SciArg dataset is an extension of the Dr. Inventor corpus (Fisas et al., 2015, 2016) with an annotation layer containing
fine-grained argumentative components and relations. It is the first argument-annotated corpus of scientific
publications (in English), which allows for joint analyses of argumentation and other rhetorical dimensions of
scientific writing.

### Supported Tasks and Leaderboards

- **Tasks**: #TODO
- **Leaderboard:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Languages

The language in the dataset is English (academic and scientific).

## Dataset Structure

### Data Instances

```
{
    'document_id': 'A10',
    'text': '<?xml version="1.0" encoding="UTF-8" standalone="no"?> <Document...',
    `text_bound_annotations`: [{
        'offset': [ [ 2452, 2582 ] ],
        'text': "['Skinning is an important part of realistic articulated body animation and is an important topic of computer graphics and animation']",
        'type': "background_claim",
        'id': 'T1' ,
    }, {
        'offset': [ [ 2595, 2684 ] ],
        'text': "['skinning can be categorized into algorithmic, physically-based, and example-based methods']",
        'type': "background_claim",
        'id': 'T2' ,
    },
    ...]
    `relations`: [{
        'id': "R1",
        'head': { "ref_id": "T3", "role": "Arg1" },
        'tail': { "ref_id": "T4", "role": "Arg2" },
        'type': "contradicts",
    }, {
        'id':, "R2"
        'head': { "ref_id": "T3", "role": "Arg1" },
        'tail': { "ref_id": "T5", "role": "Arg2" },
        'type': "supports",
    },
    ...]
}
```

### Data Fields

- `document_id`: the base file name, e.g. "A28", a `string` feature
- `text`: the parsed text of the scientific publication in the XML format, a `string` feature
- `text_bound_annotations`: span annotations that mark argumentative discourse units (ADUs), a `list` of `dictionay` feature. Each entry has the following fields:
  - `offsets`: the indices span of the text with the (inclusive) start and the (exclusive) end, respectively, a `list` of `int` feature
  - `text`: the text that belongs to an ADU, a `string` feature
  - `type`: the label of the ADU, a `string` feature (See [unit type list](#components))
  - `id`: the assigned id to the ADU, a `string` feature
- `relations`: binary relation annotations that mark the argumentative relations that hold between a head and a tail ADU, a `list` of `dictionay` feature. Each entry has the following fields:
  - `id`: the assigned id to a relation, a `string` feature
  - `head`: the first element in a relation, consists of:
    - `ref_id`: the id of the ADU
    - `role`: the ordinal id of the unit in the argument
  - `tail`: the second element in a relation, as `head`, consists of the fields: `ref_id` and `role`.
  - `type`: the label of the relation, a `string` feature (See [relation type list](#relations))

### Data Splits

The dataset consists of a single `train` split that has 40 documents.

### Label Descriptions

#### Components

#### Relations

## Dataset Creation

### Curation Rationale

\[More Information Needed\]

### Source Data

#### Initial Data Collection and Normalization

\[More Information Needed\]

#### Who are the source language producers?

\[More Information Needed\]

### Annotations

#### Annotation process

\[More Information Needed\]

#### Who are the annotators?

\[More Information Needed\]

### Personal and Sensitive Information

\[More Information Needed\]

## Considerations for Using the Data

### Social Impact of Dataset

\[More Information Needed\]

### Discussion of Biases

\[More Information Needed\]

### Other Known Limitations

\[More Information Needed\]

## Additional Information

### Dataset Curators

\[More Information Needed\]

### Licensing Information

\[More Information Needed\]

### Citation Information

```
@inproceedings{lauscher2018b,
  title = {An argument-annotated corpus of scientific publications},
  booktitle = {Proceedings of the 5th Workshop on Mining Argumentation},
  publisher = {Association for Computational Linguistics},
  author = {Lauscher, Anne and Glava\v{s}, Goran and Ponzetto, Simone Paolo},
  address = {Brussels, Belgium},
  year = {2018},
  pages = {40–46}
}
```

### Contributions

Thanks to [@github-username](https://github.com/%3Cgithub-username%3E) for adding this dataset.
