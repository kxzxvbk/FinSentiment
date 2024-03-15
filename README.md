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

- The fine-tuned model can achieve ~87% prediction accuracy on the validation dataset.
- All the process is done on a single A100. No distributed training are supported yet (since the training process is super fast).

## Quick Start

### Dataset Inspection

- The full dataset in stored in ``data/FinancialPhraseBank-v1.0`` . All the raw dataset files, preprocessing python scripts and final dataset files are included in this directory.

- For dataset inspection, run:

  ```shell
  python data/FinancialPhraseBank-v1.0/inspect_data.py
  ```

### Instruct Tuning

- We perform instruct tuning on this dataset.
- We use LoRA (r=1, alpha=2) for efficient fine-tuning.
- We use ``transformers `` library to perform all the processes.
- Detailed hyper-parameters are listed in ``instruct_tuning_glm_10b.py``
- To run your own instruct tuning, run:

```
python instruct_tuning_glm_10b.py
```

**Note:** You should download the GLM model from huggingface hub first, and specify the download path in this script before fine-tuning.

### Inference

- To interact with the pretrained model, run:

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

