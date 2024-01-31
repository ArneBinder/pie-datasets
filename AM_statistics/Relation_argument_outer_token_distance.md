# Relation argument (outer) token distance

In this document, we collect statistics regarding the token distance between the argumentative units in a relation on our six argument-mining datasets.

For the tokenization, we use `bert-base-uncased` from `transformer.AutoTokenizer` (see [AutoTokenizer](https://huggingface.co/docs/transformers/v4.37.1/en/model_doc/auto#transformers.AutoTokenizer), and [bert-based-uncased](https://huggingface.co/bert-base-uncased))
to tokenize `text` in `TextDocumentWithLabeledSpansAndBinaryRelations` (see [document type](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py)).
The distance is measured from the first token of the first argumentative unit to the last token of the last unit, a.k.a. outer distance.

We collect the following statistics: number of documents in the split (*no. doc*), no. of relations (*len*), mean of token distance (*mean*), standard deviation of the distance (*std*), minimum outer distance (*min*), and maximum outer distance (*max*).
We also present histograms in the collasible, showing the distribution of these relation distances (x-axis; and unit-counts in y-axis), accordingly.

**Remark on statistics collection**:
To manually collect a statistics for each dataset, execute the command provided under each dataset,
using the base variant of the dataset configuration, namely, `*DATASET*_base`.
The script `evaluate_documents.py` comes from [PyTorch-IE-Hydra-Template](https://github.com/ArneBinder/pytorch-ie-hydra-template-1).

## AAE2

| statistics |   train |    test |
| :--------- | ------: | ------: |
| no. doc    |     322 |      80 |
| len        |    9002 |    2372 |
| mean       | 102.582 | 100.711 |
| std        |   93.76 |  92.698 |
| min        |       9 |      10 |
| max        |     514 |     442 |

<details>
  <summary>Histogram (split: train, 322 documents)</summary>

![rtd_aae2_train.png](img%2Frelation_token_distance%2Frtd_aae2_train.png)

</details>
  <details>
  <summary>Histogram (split: test, 80 documents)</summary>

![rtd_aae2_test.png](img%2Frelation_token_distance%2Frtd_aae2_test.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=aae2_base metric=count_relation_argument_distances
```

</details>

## AbsRCT

| statistics | neoplasm_train | neoplasm_dev | neoplasm_test | glaucoma_test | mixed_test |
| :--------- | -------------: | -----------: | ------------: | ------------: | ---------: |
| no. doc    |            350 |           50 |           100 |           100 |        100 |
| len        |           2836 |          438 |           848 |           734 |        658 |
| mean       |        132.903 |      146.393 |       126.731 |       159.166 |    145.067 |
| std        |         80.869 |       98.788 |        75.363 |        83.885 |     77.921 |
| min        |             17 |           24 |            22 |            26 |         23 |
| max        |            511 |          625 |           459 |           488 |        459 |

<details>
  <summary>Histogram (split: neoplasm_train, 350 documents)</summary>

![rtd_abs-neo_train.png](img%2Frelation_token_distance%2Frtd_abs-neo_train.png)

</details>
  <details>
  <summary>Histogram (split: neoplasm_dev, 50 documents)</summary>

![rtd_abs-neo_dev.png](img%2Frelation_token_distance%2Frtd_abs-neo_dev.png)

</details>
  <details>
  <summary>Histogram (split: neoplasm_test, 100 documents)</summary>

![rtd_abs-neo_test.png](img%2Frelation_token_distance%2Frtd_abs-neo_test.png)

</details>
  <details>
  <summary>Histogram (split: glucoma_test, 100 documents)</summary>

![rtd_abs-glu_test.png](img%2Frelation_token_distance%2Frtd_abs-glu_test.png)

</details>
  <details>
  <summary>Histogram (split: mixed_test, 100 documents)</summary>

![rtd_abs-mix_test.png](img%2Frelation_token_distance%2Frtd_abs-mix_test.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=abstrct_base metric=count_relation_argument_distances
```

</details>

## ArgMicro

| statistics |  train |
| :--------- | -----: |
| no. doc    |    112 |
| len        |   1018 |
| mean       | 44.434 |
| std        | 21.501 |
| min        |     14 |
| max        |    127 |

<details>
  <summary>Histogram (split: train, 112 documents)</summary>

![rtd_argmicro.png](img%2Frelation_token_distance%2Frtd_argmicro.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=argmicro_base metric=count_relation_argument_distances
```

</details>

## CDCP

| statistics |  train |   test |
| :--------- | -----: | -----: |
| no. doc    |    580 |    150 |
| len        |   2204 |    648 |
| mean       | 48.839 | 51.299 |
| std        | 31.462 | 31.159 |
| min        |      8 |      8 |
| max        |    240 |    212 |

<details>
  <summary>Histogram (split: train, 580 documents)</summary>

![rtd_cdcp_train.png](img%2Frelation_token_distance%2Frtd_cdcp_train.png)

</details>
  <details>
  <summary>Histogram (split: test, 150 documents)</summary>

![rtd_cdcp_test.png](img%2Frelation_token_distance%2Frtd_cdcp_test.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=cdcp_base metric=count_relation_argument_distances
```

</details>

## SciArg

| statistics |  train |
| :--------- | -----: |
| no. doc    |     40 |
| len        |  15640 |
| mean       | 30.524 |
| std        | 45.351 |
| min        |      3 |
| max        |   2864 |

<details>
  <summary>Histogram (split: train, 40 documents)</summary>

![rtd_sciarg.png](img%2Frelation_token_distance%2Frtd_sciarg.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=sciarg_base metric=count_relation_argument_distances ++metric.tokenize_kwargs.strict_span_conversion=false
```

</details>

## SciDTB_Argmin

| statistics |  train |
| :--------- | -----: |
| no. doc    |     60 |
| len        |    586 |
| mean       | 75.239 |
| std        | 40.312 |
| min        |     21 |
| max        |    277 |

<details>
  <summary>Histogram (split: train, 60 documents)</summary>

![rtd_scidtb-argmin.png](img%2Frelation_token_distance%2Frtd_scidtb-argmin.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=scidtb_argmin_base metric=count_relation_argument_distances
```

</details>
