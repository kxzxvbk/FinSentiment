import torch
from glm_model.modeling_glm import GLMForConditionalGeneration
from glm_model.tokenization_glm import GLMTokenizer
from peft import get_peft_model, LoraConfig, TaskType


def generate_response(query, device='cuda:0'):
    # Encode the prompt.
    prompt = query
    instruct_prompt = '上述文本的情感分类为（正面、负面或中立）：'

    prompt_ids = tokenizer.encode(
        prompt + instruct_prompt,
        max_length=2048,
        truncation=True,
        add_special_tokens=True
    )

    prompt_ids = torch.tensor([prompt_ids]).to(device)
    # Generate the results.
    generate_ids = model.generate(inputs=prompt_ids, max_new_tokens=10, do_sample=False)
    total_res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # Remove the prompt part, only return the response.
    res = total_res[len(prompt + instruct_prompt):].strip()
    return res[:2]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    # Determine the inference device.
    device = 'cpu' if args.cpu else 'cuda:0'

    # Specify your base model path.
    base_model = '/mnt/nfs/whl/LLM/glm-10b-chinese'

    # Prepare tokenizer.
    tokenizer = GLMTokenizer.from_pretrained(base_model)
    # Prepare model.
    print('Initializing model ... This may take quite a few minutes.')
    model = GLMForConditionalGeneration.from_pretrained(base_model, revision='main', torch_dtype=torch.float16)

    # Prepare lora model.
    peft_config = LoraConfig(
        target_modules=['query_key_value'],
        task_type=TaskType.CAUSAL_LM, inference_mode=True, r=2, lora_alpha=4, lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    # Resume from checkpoint.
    ckpt = '/mnt/nfs/whl/FinSentiment/instruct_tuning_10b/final.pth.tar'
    sd = torch.load(ckpt)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()

    while True:
        news = input('请输入需要分析的语句：')
        print('结果：' + generate_response(news, device=device))
