import torch
from modeling_glm import GLMForConditionalGeneration
from tokenization_glm import GLMTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training, PeftModel


def generate_response(query):
    # Encode the prompt.
    prompt = query
    instruct_prompt = '上述文本的情感分类为（正面、负面或中立）：'

    prompt_ids = tokenizer.encode(
        prompt + instruct_prompt,
        max_length=2048,
        truncation=True,
        add_special_tokens=True
    )

    prompt_ids = torch.tensor([prompt_ids]).to('cuda:0')
    # Generate the results.
    generate_ids = model.generate(inputs=prompt_ids, max_new_tokens=10, do_sample=False)
    total_res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # Remove the prompt part, only return the response.
    res = total_res[len(prompt + instruct_prompt):].strip()
    return res[:2]


if __name__ == '__main__':
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
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=1, lora_alpha=2, lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config).cuda()

    # Resume from checkpoint.
    ckpt = '/mnt/nfs/whl/FinSentiment/instruct_tuning_10b/final.pth.tar'
    sd = torch.load(ckpt)
    model.load_state_dict(sd)
    model = model.to('cuda:0')
    model.eval()

    while True:
        news = input('请输入需要分析的语句：')
        print('结果：' + generate_response(news))
