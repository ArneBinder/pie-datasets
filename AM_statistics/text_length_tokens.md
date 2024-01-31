# Text length (tokens)

In this document, we collect statistics regarding the token distance between the argumentative units in a relation on our six argument-mining datasets.

For the tokenization, we use `bert-base-uncased` from `transformer.AutoTokenizer` (see [AutoTokenizer](https://huggingface.co/docs/transformers/v4.37.1/en/model_doc/auto#transformers.AutoTokenizer), and [bert-based-uncased](https://huggingface.co/bert-base-uncased))
to tokenize `text` in `TextDocumentWithLabeledSpansAndBinaryRelations` (see [document type](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py)).
The text (document) length is measured from the first token of the document to the last one.

We collect the following statistics: number of documents in the split (*no. doc*), mean of document token-length (*mean*), standard deviation of the length (*std*), minimum number of tokens in a document (*min*), and maximum number of tokens in a document (*max*).
We also present histograms in the collasible, showing the distribution of these text lengths (x-axis; and unit-counts in y-axis), accordingly.

**Remark on statistics collection**:
To manually collect a statistics for each dataset, execute the command provided under each dataset,
using the base variant of the dataset configuration, namely, `*DATASET*_base`.
The script `evaluate_documents.py` comes from [PyTorch-IE-Hydra-Template](https://github.com/ArneBinder/pytorch-ie-hydra-template-1).

## AAE2

| statistics |   train |   test |
| :--------- | ------: | -----: |
| no. doc    |     322 |     80 |
| mean       | 377.686 |  378.4 |
| std        |  64.534 | 66.054 |
| min        |     236 |    269 |
| max        |     580 |    532 |

<details>
  <summary>Histogram (split: train, 322 documents)</summary>

![tl_aae2_train.png](img%2Ftoken_len%2Ftl_aae2_train.png)

</details>
  <details>
  <summary>Histogram (split: test, 80 documents)</summary>

![tl_aae2_test.png](img%2Ftoken_len%2Ftl_aae2_test.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=aae2_base metric=count_text_tokens
```

</details>

## AbsRCT

| statistics | neoplasm_train | neoplasm_dev | neoplasm_test | glaucoma_test | mixed_test |
| :--------- | -------------: | -----------: | ------------: | ------------: | ---------: |
| no. doc    |            350 |           50 |           100 |           100 |        100 |
| mean       |        447.291 |       481.66 |        442.79 |        456.78 |     450.29 |
| std        |         91.266 |      116.239 |        89.692 |       115.535 |     87.002 |
| min        |            301 |          329 |           292 |           212 |        268 |
| max        |            843 |          952 |           776 |          1022 |        776 |

<details>
  <summary>Histogram (split: neoplasm_train, 350 documents)</summary>

![tl_abs-neo_train.png](img%2Ftoken_len%2Ftl_abs-neo_train.png)

</details>
  <details>
  <summary>Histogram (split: neoplasm_dev, 50 documents)</summary>

![tl_abs-neo_dev.png](img%2Ftoken_len%2Ftl_abs-neo_dev.png)

</details>
  <details>
  <summary>Histogram (split: neoplasm_test, 100 documents)</summary>

![tl_abs-neo_test.png](img%2Ftoken_len%2Ftl_abs-neo_test.png)

</details>
  <details>
  <summary>Histogram (split: glucoma_test, 100 documents)</summary>

![tl_abs-glu_test.png](img%2Ftoken_len%2Ftl_abs-glu_test.png)

</details>
  <details>
  <summary>Histogram (split: mixed_test, 100 documents)</summary>

![tl_abs-mix_test.png](img%2Ftoken_len%2Ftl_abs-mix_test.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=abstrct_base metric=count_text_tokens
```

</details>

## ArgMicro

| statistics |  train |
| :--------- | -----: |
| no. doc    |    112 |
| mean       | 84.161 |
| std        | 22.596 |
| min        |     36 |
| max        |    153 |

<details>
  <summary>Histogram (split: train, 112 documents)</summary>

![tl_argmicro.png](img%2Ftoken_len%2Ftl_argmicro.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=argmicro_base metric=count_text_tokens
```

</details>

## CDCP

| statistics |   train |    test |
| :--------- | ------: | ------: |
| no. doc    |     580 |     150 |
| mean       | 130.781 | 128.673 |
| std        | 101.121 |  98.708 |
| min        |      13 |      15 |
| max        |     562 |     571 |

<details>
  <summary>Histogram (split: train, 580 documents)</summary>

![tl_cdcp_train.png](img%2Ftoken_len%2Ftl_cdcp_train.png)

</details>
  <details>
  <summary>Histogram (split: test, 150 documents)</summary>

![tl_cdcp_test.png](img%2Ftoken_len%2Ftl_cdcp_test.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=cdcp_base metric=count_text_tokens
```

</details>

## SciArg

| statistics |   train |
| :--------- | ------: |
| no. doc    |      40 |
| mean       | 10521.1 |
| std        |  2472.2 |
| min        |    6452 |
| max        |   16421 |

<details>
  <summary>Histogram (split: train, 40 documents)</summary>

![tl_sciarg.png](img%2Ftoken_len%2Ftl_sciarg.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=sciarg_base metric=count_text_tokens
```

</details>

## SciDTB_Argmin

| statistics |   train |
| :--------- | ------: |
| no. doc    |      60 |
| mean       | 164.417 |
| std        |  64.572 |
| min        |      80 |
| max        |     532 |

<details>
  <summary>Histogram (split: train, 60 documents)</summary>

![tl_scidtb-argmin.png](img%2Ftoken_len%2Ftl_scidtb-argmin.png)

</details>

<details>
<summary>Command</summary>

```
python src/evaluate_documents.py dataset=scidtb_argmin_base metric=count_text_tokens
```

</details>
