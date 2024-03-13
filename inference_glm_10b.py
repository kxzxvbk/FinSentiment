import os
import pickle
import argparse

import torch
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training, PeftModel

from instruct_tuning_glm_10b import label2zh
from modeling_glm import GLMForConditionalGeneration
from tokenization_glm import GLMChineseTokenizer


def generate_txt(model: nn.Module, tokenizer: GLMChineseTokenizer, prompt: str, instruct_prompt: str) -> str:
    """
    Given a prompt, generate the result by pure LLMs.
    """
    # Encode the prompt.
    prompt_ids = tokenizer.encode(
        prompt + instruct_prompt,
        max_length=2048,
        truncation=True,
        add_special_tokens=True
    )
    prompt_ids = torch.tensor([prompt_ids]).to(args.device)
    # Generate the results.
    generate_ids = model.generate(inputs=prompt_ids, max_length=4096, do_sample=False)
    total_res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # Remove the prompt part, only return the response.
    return total_res[len(prompt + instruct_prompt):]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_mode', type=str, default='dataset')
    args = parser.parse_args()

    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    assert args.test_mode in ['dataset', 'manual'], "The test mode should be either dataset or manual."

    # The pretrained model path.
    ckpt = '/mnt/nfs/whl/FinSentiment/instruct_tuning_10b/checkpoint-1600/pytorch_model.bin'

    # Prepare the lora model.
    base_model = '/mnt/nfs/whl/LLM/glm-10b-chinese'
    peft_config = LoraConfig(
        target_modules=['query_key_value'],
        task_type=TaskType.CAUSAL_LM, inference_mode=True, r=1, lora_alpha=2, lora_dropout=0.1
    )
    tokenizer = GLMChineseTokenizer.from_pretrained(base_model)
    print('Initializing model ... This may take quite a few minutes.')
    model = GLMForConditionalGeneration.from_pretrained(base_model, revision='main', torch_dtype=torch.float16)
    model = get_peft_model(model, peft_config)

    # Resume from checkpoint.
    sd = torch.load(ckpt)
    model.load_state_dict(sd)
    model = model.to(args.device)
    model.eval()

    if args.test_mode == 'dataset':
        # Get the evaluate dataset.
        with open('./data/FinancialPhraseBank-v1.0/dataset_en_split50.pkl', 'rb') as f:
            test_data = pickle.load(f)

        # Loop over the test dataset.
        test_idx = 0
        while test_idx < len(test_data['sentence']):
            data_sample = test_data['sentence'][test_idx]
            answer = label2zh(test_data['label'][test_idx])
            pred = generate_txt(model, tokenizer, data_sample, instruct_prompt='上述文本的情感分类为（正面、负面或中立）：')
            print('文本：' + data_sample)
            print('预测结果：' + pred)
            print('正确答案' + answer)
            print('####################################################################################')
            input('输入回车以继续。')
            test_idx += 1
    elif args.test_mode == 'manual':
        while True:
            query = input('请输入文本内容：')
            pred = generate_txt(model, tokenizer, query, instruct_prompt='上述文本的情感分类为（正面、负面或中立）：')
            print('预测结果：' + pred)
    else:
        raise ValueError("The test mode should be either dataset or manual.")
