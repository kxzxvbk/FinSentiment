import torch
import evaluate
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training, PeftModel
import pickle
import random
from torch.utils.data import Dataset
from transformers import TrainerCallback

random.seed(0)


def create_inputs_and_labels(tokenizer, question, answer):
    eop = tokenizer.eos_token_id
    prompt = tokenizer.encode(
        question,
        max_length=2048,
        truncation=True,
        add_special_tokens=True
    )
    completion = tokenizer.encode(
        answer,
        max_length=2048,
        truncation=True,
        add_special_tokens=False
    )

    inputs = prompt + completion + [eop]
    labels = [-100] * len(prompt) + completion + [eop]

    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    return inputs, labels


def label2zh(lab: str) -> str:
    lab = lab.strip()
    if lab.lower() == 'positive':
        return '正面'
    elif lab.lower() == 'negative':
        return '负面'
    elif lab.lower() == 'neutral':
        return '中立'
    else:
        raise ValueError(f'Unrecognized label type: {lab.lower()}.')


class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item_data = self.data[index]
        tokenizer = self.tokenizer
        input_ids, labels = create_inputs_and_labels(
            tokenizer,
            question=item_data['sentence']+"请给出上述文本的情感分类（正面、负面或中立）：",
            answer=label2zh(item_data['answer'])
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_fn(batch):
    # Let's assume that each element in "batch" is a tuple (data, label).
    # Sort the batch in the descending order
    sorted_batch = sorted(batch, key=lambda x: x['input_ids'].shape[0], reverse=True)
    # Get each sequence and pad it
    sequences = [x['input_ids'] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    # Also need to store the length of each sequence
    # This is later needed in order to unpad the sequences
    lengths = torch.LongTensor([len(x) for x in sequences])
    # Don't forget to grab the labels of the *sorted* batch
    labels = [x['labels'] for x in sorted_batch]
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return {
        'input_ids': sequences_padded,
        'labels': labels_padded,
    }


class ModifiedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs:
            outputs = model(
                input_ids=inputs["input_ids"],
                labels=inputs["labels"],
            )
            return outputs.loss, outputs
        outputs = model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        )
        return outputs.loss


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


def _compute_base_metric(pred_str, label_str, prefix):
    rouge = evaluate.load('rouge')

    rouge_output = rouge.compute(
        predictions=pred_str,
        references=label_str,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
    )

    acc_count = 0
    for pred, label in zip(pred_str, label_str):
        if pred == label:
            acc_count += 1

    res_dict = {
        f"{prefix}_R1": round(rouge_output["rouge1"], 4),
        f"{prefix}_R2": round(rouge_output["rouge2"], 4),
        f"{prefix}_RL": round(rouge_output["rougeL"], 4),
        f"{prefix}_RLsum": round(rouge_output["rougeLsum"], 4),
        f"{prefix}_acc": round(acc_count / len(label_str), 4)
    }

    return res_dict


def compute_metrics(pred):
    labels_ids = pred.label_ids[..., 1:]
    pred_ids = pred.predictions[0][..., :-1]
    for id, pred in enumerate(pred_ids):
        pred_ids[id][labels_ids[id] == -100] = 2
        pred_ids[id][pred_ids[id] == -100] = 2
        labels_ids[id][labels_ids[id] == -100] = 2

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    res_dict = {}
    res_dict.update(_compute_base_metric(pred_str, label_str, prefix='sentiment'))

    return res_dict


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def prepare_dataset(tokenizer):
    with open(f'./data/FinancialPhraseBank-v1.0/dataset_en_split50.pkl', 'rb') as f:
        data = pickle.load(f)
    collated_data = []
    for i in range(len(data['sentence'])):
        collated_data.append({'sentence': data['sentence'][i], 'label': data['label'][i]})
    data = collated_data
    random.shuffle(data)
    eval_num = len(data) * 0.1
    train_data = data[:-eval_num]
    test_data = data[-eval_num:]

    return SentimentDataset(train_data, tokenizer), SentimentDataset(test_data, tokenizer)


if __name__ == '__main__':
    # Initialize base model.
    base_model = '/mnt/nfs/whl/LLM/glm-10b-chinese'

    # Initialize base model.
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    print('Initializing model ... This may take quite a few minutes.')
    model = AutoModel.from_pretrained(base_model, trust_remote_code=True, revision='main', torch_dtype=torch.float16)

    # Initialize lora model.
    peft_config = LoraConfig(
        target_modules=['query_key_value'],
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=1, lora_alpha=2, lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    print(model)
    model.print_trainable_parameters()

    # Prepare dataset.
    train_dataset, test_dataset = prepare_dataset(tokenizer)

    # Training arguments.
    training_args = TrainingArguments(
        output_dir=f'./instruct_tuning_10b',
        num_train_epochs=10,
        evaluation_strategy="steps",
        eval_steps=400,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to=None,
        remove_unused_columns=False,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        group_by_length=False,
        dataloader_pin_memory=False,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        gradient_accumulation_steps=2,
        warmup_steps=3000,
    )
    trainer = ModifiedTrainer(
        model=model,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # Resume from the checkpoint
    trainer.add_callback(EvaluateFirstStepCallback())
    trainer.train()
