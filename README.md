# Chinese Financial Sentiment Analysis

## Overview

This is a repository for using Large Language Models (LLMs) to analysis Chinese financial news.

- We allow three types of sentiment classes: 正面 (positive), 负面 (negative), 中立 (neutral).

- We use [GLM-10B-Chinese](https://huggingface.co/THUDM/glm-10b-chinese) as the base language model for fine-tuning.
- We use [financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank) as dataset and translate the content into Chinese using apis.
- We perform LoRA + instruct tuning on this dataset. The instruct template is:

```
[NEWS CONTENT] 请给出上述文本的情感分类（正面、负面或中立）：[SENTIMENT]
```

- The fine-tuned model can achieve ~88% prediction accuracy on the validation dataset.
- All the process is done on a single A100. No distributed training are supported yet ,since the training process is fast (~20 min).

## Quick Start

### Install

Use ``pip install -r requirements.txt`` .

### Dataset Inspection

- The full dataset in stored in ``data/FinancialPhraseBank-v1.0`` . All the raw dataset files, preprocessing python scripts and final dataset files are included in this directory.

- For dataset inspection, run:

  ```shell
  python data/FinancialPhraseBank-v1.0/inspect_data.py
  ```

### Instruct Tuning

- We perform instruct tuning on this dataset.
- We use Low-Rank-Adaptation (LoRA) for efficient fine-tuning.
- We use ``transformers `` library to perform all the processes.
- Detailed hyper-parameters are listed in ``instruct_tuning_glm_10b.py``
- To run your own instruct tuning, run:

```
python instruct_tuning_glm_10b.py
```

**Note:** You should download the GLM model from huggingface hub first, and specify the download path in this script before fine-tuning.

**Hyper-parameter Tuning Experiments:**

| Method                                            | Validation Accuracy |
| ------------------------------------------------- | ------------------- |
| target_module="query_key_value", r=1, alpha=2     | 87.4%               |
| **target_module="query_key_value", r=2, alpha=4** | **88.6%**           |
| target_module="query_key_value", r=4, alpha=8     | 86.8%               |
| target_module="dense", r=1, alpha=2               | 85.9%               |
| target_module="dense", r=4, alpha=8               | 88.2%               |

### Interact

- To interact with the pretrained model as a demonstration, run:

```
python interact_glm_10b.py
```

- Or if you want to use CPU to inference, run (extremely slow):

```
python interact_glm_10b.py --cpu
```

### Inference

- To use the pretrained model to evaluate given dataset, run:
```
python inference_glm_10b.py
```

## Citation

```latex
@misc{FinSentiment,
    title={Chinese Financial Sentiment Analysis},
    author={kxzxvbk},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/kxzxvbk/FinSentiment}},
    year={2024},
}
```
