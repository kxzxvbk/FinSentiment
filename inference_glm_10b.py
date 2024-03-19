import os
import torch
from typing import List, Union
import pandas as pd
import time

from glm_model.modeling_glm import GLMForConditionalGeneration
from glm_model.tokenization_glm import GLMTokenizer
from peft import get_peft_model, LoraConfig, TaskType


def read_csv(path):
    print(f'Loading data file: {path}...')
    with open(path, 'r') as f:
        header = f.readline()
        res = {}
        while True:
            line = f.readline()
            if line is None or len(line.strip()) == 0:
                break
            line = line.split(',"')
            date = line[0].strip()
            try:
                title = line[2][:-1]
            except:
                continue
            if date in res:
                res[date].append(title)
            else:
                res[date] = [title]
    for date in res.keys():
        tmp = list(set(res[date]))
        tmp.sort(reverse=True, key=lambda x: len(x))
        res[date] = tmp[:1000]

    final = {"日期": [], "标题": [], "情感": []}
    for date in res.keys():
        for title in res[date]:
            final['日期'].append(date)
            final['标题'].append(title)
            final['情感'].append('')
    print(f'Loading finished. Total samples number: {len(final["日期"])}.')
    return final


def pad_sequence(sequences: List[torch.Tensor], batch_first: bool = True, padding_value: Union[float, int] = 0.0):
    sequences = [sequences[i][0] for i in range(len(sequences))]
    max_len = max([len(sequences[i]) for i in range(len(sequences))])
    if batch_first:
        out_dims = (len(sequences), max_len)
    else:
        out_dims = (max_len, len(sequences))
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, -length:, ...] = tensor
        else:
            out_tensor[-length:, i, ...] = tensor

    return out_tensor


def batch_generate_response(model: GLMForConditionalGeneration, tokenizer: GLMTokenizer,
                            queries: List[str], device: str = 'cuda:0'):
    bs = len(queries)
    instruct_prompt = '上述文本的情感分类为（正面、负面或中立）：'
    queries = [tokenizer.encode(
        queries[i].strip() + instruct_prompt,
        max_length=2048,
        truncation=True,
        add_special_tokens=True,
        return_tensors='pt'
    ) for i in range(bs)]

    queries = pad_sequence(queries, batch_first=True).to(device)
    generate_ids = model.generate(inputs=queries, max_new_tokens=3, do_sample=False)
    total_res = tokenizer.batch_decode(generate_ids[:, queries.shape[1]:], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
    return [total_res[i].strip()[:2] for i in range(bs)]


def inference_file(model, tokenizer, src_path, dst_path, file_name):
    df = read_csv(os.path.join(src_path, file_name))
    df_len = len(df['标题'])
    df['情感'] = [''] * df_len
    batch_size = 128
    start_idx = 0
    t0 = time.time()
    while start_idx < df_len:
        inputs = df['标题'][start_idx: start_idx + batch_size]
        df["情感"][start_idx: start_idx + batch_size] = batch_generate_response(model, tokenizer, queries=inputs)
        start_idx += batch_size
        if start_idx % (50 * batch_size) == 0:
            tmp_df = pd.DataFrame(df)
            tmp_df.to_csv(os.path.join(dst_path, file_name))
            print('Processing: {} / {}. Time: {:.2f}s'.format(start_idx, df_len, time.time() - t0))
            t0 = time.time()
    tmp_df = pd.DataFrame(df)
    tmp_df.to_csv(os.path.join(dst_path, file_name))
    print('Processing: {} / {}. Time: {:.2f}s'.format(start_idx, df_len, time.time() - t0))


if __name__ == '__main__':
    inference_data_path = './data/InferenceData'
    files = [f"RESSET_NEWSDAILY_EXT_2023_{i}.csv" for i in range(1, 28)]
    output_data_path = './data/InferenceResult'

    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)

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

    with torch.no_grad():
        for file in files:
            inference_file(model=model, tokenizer=tokenizer, src_path=inference_data_path,
                           dst_path=output_data_path, file_name=file)
