# Dataset Card for "SciDTB Argmin"

### Dataset Summary

Briefly summarize the dataset, its intended use and the supported tasks. Give an overview of how and why the dataset was created. The summary should explicitly mention the languages present in the dataset (possibly in broad terms, e.g. *translations between several pairs of European languages*), and describe the domain, topic, or genre covered.

### Supported Tasks and Leaderboards

- **Tasks:** Argument Mining, Component Classification, Relation Classification
- **Leaderboards:** https://paperswithcode.com/dataset/cdcp

### Languages

The language in the dataset is English (academic).

## Dataset Structure

### Data Instances

- **Size of downloaded dataset files:**  MB

Provide an JSON-formatted example and brief description of a typical instance in the dataset. If available, provide a link to further examples.

```
{
  'example_field': ...,
  ...
}
```

Provide any additional information that is not covered in the other sections about the data here. In particular describe any relationships between data points and if these relationships are made explicit.

### Data Fields

List and describe the fields present in the dataset. Mention their data type, and whether they are used as input or output in any of the tasks the dataset currently supports. If the data has span indices, describe their attributes, such as whether they are at the character level or word level, whether they are contiguous or not, etc. If the datasets contains example IDs, state whether they have an inherent meaning, such as a mapping to other datasets or pointing to relationships between data points.

- `example_field`: description of `example_field`

Note that the descriptions can be initialized with the **Show Markdown Data Fields** output of the [Datasets Tagging app](https://huggingface.co/spaces/huggingface/datasets-tagging), you will then only need to refine the generated descriptions.

### Data Splits

Describe and name the splits in the dataset if there are more than one.

Describe any criteria for splitting the data, if used. If there are differences between the splits (e.g. if the training annotations are machine-generated and the dev and test ones are created by humans, or if different numbers of annotators contributed to each example), describe them here.

Provide the sizes of each split. As appropriate, provide any descriptive statistics for the features, such as average length.  For example:

|                                 | train | validation | test |
| ------------------------------- | ----: | ---------: | ---: |
| Input Sentences                 |       |            |      |
| Span Labels<br/>- A<br/>- B     |       |            |      |
| Relation Labels<br/>- A<br/>- B |       |            |      |

## Dataset Creation

### Curation Rationale

"We propose to tackle the limitations posed by the lack of annotated data for argument mining in the scientific domain by leveraging existing Rhetorical Structure Theory (RST) (Mann et al., 1992) annotations in a corpus of computational linguistics abstracts (SciDTB) (Yang and Li, 2018)." (p. 42)

We introduce a fine-grained annotation
scheme aimed at capturing information that accounts for the specificities of the scientific discourse, including the type of evidence that is offered to support a statement (e.g., background information, experimental data or interpretation of
results). This can provide relevant information, for
instance, to assess the argumentative strength of a
text. (p. 44)

What need motivated the creation of this dataset? What are some of the reasons underlying the major choices involved in putting it together?

### Source Data

This section describes the source data (e.g. news text and headlines, social media posts, translated sentences,...)

#### Initial Data Collection and Normalization

This work is informed by previous research in the areas of argument mining, argumentation quality assessment and the relationship between discourse and argumentative structures and, from the methodological perspective, to transfer learning approaches.

We add a new annotation layer to the Discourse Dependency TreeBank for Scientific Abstracts (SciDTB) (Yang and Li, 2018). SciDTB contains 798 abstracts from the ACL Anthology (Radev et al., 2013) annotated with elementary discourse units (EDUs)
and relations from the RST Framework. (p. 43)

Describe the data collection process. Describe any criteria for data selection or filtering. List any key words or search terms used. If possible, include runtime information for the collection process.

If data was collected from other pre-existing datasets, link to source here and to their [Hugging Face version](https://huggingface.co/datasets/dataset_name).

If the data was modified or normalized after being collected (e.g. if the data is word-tokenized), describe the process and the tools used.

#### Who are the source language producers?

State whether the data was produced by humans or machine generated. Describe the people or systems who originally created the data.

If available, include self-reported demographic or identity information for the source data creators, but avoid inferring this information. Instead state that this information is unknown. See [Larson 2017](https://www.aclweb.org/anthology/W17-1601.pdf) for using identity categories as a variables, particularly gender.

Describe the conditions under which the data was created (for example, if the producers were crowdworkers, state what platform was used, or if the data was found, what website the data was found on). If compensation was provided, include that information here.

Describe other people represented or mentioned in the data. Where possible, link to references for the information.

### Annotations

#### Annotation process

We consider a subset of the SciDTB corpus consisting of 60 abstracts from the Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP) and transformed them into a format suitable for the GraPAT graph annotation tool (Sonntag and Stede, 2014).

The corpus enriched with the argumentation level contains a total of 327 sentences, 8012 tokens, 862 discourse units and 352 argumentative units linked by 292 argumentative relations. (p. 43)

#### Who are the annotators?

\[More Information Needed\]

### Personal and Sensitive Information

\[More Information Needed\]

## Considerations for Using the Data

### Social Impact of Dataset

"The development of automatic systems to support the quality assessment of scientific texts can facilitate the work of editors and referees of scientific publications and, at the same time, be of value for researchers to obtain feedback that can lead to improve the communication of their results...Aspects such as the argumentative structure of the text are key when analyzing its effectiveness with respect to its communication objectives (Walton and Walton, 1989)." (p. 41)

"Being able to extract not only what is being stated by the authors of a text but also the reasons they provide to support it can be useful in multiple applications, ranging from a finegrained analysis of opinions to the generation of abstractive summaries of texts." (p. 41)

### Discussion of Biases

the types of argumentative units are
distributed as follows: 31% of the units are of type
proposal, 25% assertion, 21% result, 18% means,
3% observation, and 2% description. In turn, the
relations are distributed: 45% of type detail, 42%
support, 9% additional, and 4% sequence. No attack relations were identified in the set of currently
annotated texts.

When considering the distance6
of the units to their parent unit in the argumentation tree, we observe that the majority (57%) are
linked to a unit that occurs right before or after it
in the text, while 19% are linked to a unit with a
distance of 1 unit in-between, 12% to a unit with a
distance of 2 units, 6% to a unit with a distance of
3, and 6% to a unit with a distance of 4 or more.

(p. 44)

### Other Known Limitations

\[More Information Needed\]

## Additional Information

### Dataset Curators

This work is (partly) supported by the Spanish
Government under the Mar´ıa de Maeztu Units of
Excellence Programme (MDM-2015-0502). (p. 49)

### Licensing Information

Provide the license and link to the license webpage if available.

### Citation Information

```
@article{yang2018scidtb,
  title={SciDTB: Discourse dependency TreeBank for scientific abstracts},
  author={Yang, An and Li, Sujian},
  journal={arXiv preprint arXiv:1806.03653},
  year={2018}
}
```

Yang & Li (2018)'s [DOI](10.18653/v1/P18-2071)

```
@inproceedings{accuosto2019transferring,
  title={Transferring knowledge from discourse to arguments: A case study with scientific abstracts},
  author={Accuosto, Pablo and Saggion, Horacio},
  booktitle={Proceedings of the 6th Workshop on Argument Mining},
  pages={41--51},
  year={2019}
}
```

### Contributions

Thanks to [@github-username](https://github.com/%3Cgithub-username%3E) for adding this dataset.
