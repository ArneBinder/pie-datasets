# Span lengths (tokens)

In this document, we collect statistics regarding the token distance between the argumentative units in a relation on our six argument-mining datasets.

For the tokenization, we use `bert-base-uncased` from `transformer.AutoTokenizer` (see [AutoTokenizer](https://huggingface.co/docs/transformers/v4.37.1/en/model_doc/auto#transformers.AutoTokenizer), and [bert-based-uncased](https://huggingface.co/bert-base-uncased))
to tokenize `text` in `TextDocumentWithLabeledSpansAndBinaryRelations` (see [document type](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py)).
The span length is measured from the first token of the first argumentative unit to the last token of the particular unit.

We collect the following statistics: number of documents in the split (*no. doc*), no. of spans (*len*), mean of number of tokens in a span (*mean*), standard deviation of the number of tokens (*std*), minimum tokens in a span (*min*), and maximum tokens in a span (*max*).
We also present histograms in the collasible, showing the distribution of these token-numbers (x-axis; and unit-counts in y-axis), accordingly.

**Remark on statistics collection**:
To manually collect a statistics for each dataset, execute the command provided under each dataset,
using the base variant of the dataset configuration, namely, `*DATASET*_base`.
The script `evaluate_documents.py` comes from [PyTorch-IE-Hydra-Template](https://github.com/ArneBinder/pytorch-ie-hydra-template-1).

## AAE2

| statistics |  train |   test |
| :--------- | -----: | -----: |
| no. doc    |    322 |     80 |
| len        |   4823 |   1266 |
| mean       | 17.157 | 16.317 |
| std        |  8.079 |  7.953 |
| min        |      3 |      3 |
| max        |     75 |     50 |

<details>
  <summary>Histogram (split: train, 322 documents)</summary>

![slt_aae2_train.png](img%2Fspan_len_token%2Fslt_aae2_train.png)

</details>
  <details>
  <summary>Histogram (split: test, 80 documents)</summary>

![slt_aae2_test.png](img%2Fspan_len_token%2Fslt_aae2_test.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=aae2_base metric=span_lengths_tokens
```

</details>

## AbsRCT

| statistics | neoplasm_train | neoplasm_dev | neoplasm_test | glaucoma_test | mixed_test |
| :--------- | -------------: | -----------: | ------------: | ------------: | ---------: |
| no. doc    |            350 |           50 |           100 |           100 |        100 |
| len        |           2267 |          326 |           686 |           594 |        600 |
| mean       |         34.303 |       37.135 |        32.566 |        38.997 |     38.507 |
| std        |         22.425 |       29.941 |        20.264 |        22.604 |     24.036 |
| min        |              5 |            5 |             6 |             6 |          7 |
| max        |            250 |          288 |           182 |           169 |        159 |

<details>
  <summary>Histogram (split: neoplasm_train, 350 documents)</summary>

![slt_abs-neo_train.png](img%2Fspan_len_token%2Fslt_abs-neo_train.png)

</details>
  <details>
  <summary>Histogram (split: neoplasm_dev, 50 documents)</summary>

![slt_abs-neo_dev.png](img%2Fspan_len_token%2Fslt_abs-neo_dev.png)

</details>
  <details>
  <summary>Histogram (split: neoplasm_test, 100 documents)</summary>

![slt_abs-neo_test.png](img%2Fspan_len_token%2Fslt_abs-neo_test.png)

</details>
  <details>
  <summary>Histogram (split: glucoma_test, 100 documents)</summary>

![slt_abs-glu_test.png](img%2Fspan_len_token%2Fslt_abs-glu_test.png)

</details>
  <details>
  <summary>Histogram (split: mixed_test, 100 documents)</summary>

![slt_abs-mix_test.png](img%2Fspan_len_token%2Fslt_abs-mix_test.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=abstrct_base metric=span_lengths_tokens
```

</details>

## ArgMicro

| statistics |  train |
| :--------- | -----: |
| no. doc    |    112 |
| len        |    576 |
| mean       | 16.365 |
| std        |  6.545 |
| min        |      4 |
| max        |     41 |

<details>
  <summary>Histogram (split: train, 112 documents)</summary>

![slt_argmicro.png](img%2Fspan_len_token%2Fslt_argmicro.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=argmicro_base metric=span_lengths_tokens
```

</details>

## CDCP

| statistics |  train |   test |
| :--------- | -----: | -----: |
| no. doc    |    580 |    150 |
| len        |   3901 |   1026 |
| mean       | 19.441 | 18.758 |
| std        |  11.71 | 10.388 |
| min        |      2 |      3 |
| max        |    142 |     83 |

<details>
  <summary>Histogram (split: train, 580 documents)</summary>

![slt_cdcp_train.png](img%2Fspan_len_token%2Fslt_cdcp_train.png)

</details>
  <details>
  <summary>Histogram (split: test, 150 documents)</summary>

![slt_cdcp_test.png](img%2Fspan_len_token%2Fslt_cdcp_test.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=cdcp_base metric=span_lengths_tokens
```

</details>

## SciArg

| statistics |  train |
| :--------- | -----: |
| no. doc    |     40 |
| len        |  13586 |
| mean       | 11.677 |
| std        |  8.731 |
| min        |      1 |
| max        |    138 |

<details>
  <summary>Histogram (split: train, 40 documents)</summary>

![slt_sciarg.png](img%2Fspan_len_token%2Fslt_sciarg.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=sciarg_base metric=span_lengths_tokens ++metric.tokenize_kwargs.strict_span_conversion=false
```

</details>

## SciDTB_Argmin

| statistics |  train |
| :--------- | -----: |
| no. doc    |     60 |
| len        |    353 |
| mean       | 27.946 |
| std        | 13.054 |
| min        |      7 |
| max        |    123 |

<details>
  <summary>Histogram (split: train, 60 documents)</summary>

![slt_scitdb-argmin.png](img%2Fspan_len_token%2Fslt_scitdb-argmin.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=scidtb_argmin_base metric=span_lengths_tokens
```

</details>
