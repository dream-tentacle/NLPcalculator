from datasets import load_dataset, DatasetDict, Dataset
import random
from tokenizer import CharTokenizer
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch


def make_datasets(tokenizer, Context_length=128, size=10):
    def tokenize(element):
        tokenized = tokenizer(
            element["full"],
            return_length=True,
            truncation=True,
            max_length=Context_length,
            padding="max_length",
        )
        return {"input_ids": tokenized["input_ids"]}

    print("Creating datasets... ")
    # 创建数据集字典
    raw_datasets = []
    max_num_len = 4
    for i in range(size):
        num_len = random.randint(1, max_num_len)
        a = random.randint(int(10 ** (num_len - 1)), int(10**num_len))
        b = random.randint(int(10 ** (num_len - 1)), int(10**num_len))
        x = str(a) + "+" + str(b) + "="
        y = str(a + b)
        raw_datasets.append(
            {"question": x, "answer": y, "full": x + y, "question_len": len(x)}
        )
    raw_datasets = Dataset.from_list(raw_datasets)
    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets.column_names
    )
    labels_column = []
    for i in range(0, len(tokenized_datasets)):
        tokenized_datasets[i]["input_ids"] = [
            tokenizer.eos_token_id
        ] + tokenized_datasets[i]["input_ids"]
        # 生成labels
        outputs = (
            ([-100] * len(raw_datasets[i]["question"]))
            + tokenized_datasets[i]["input_ids"][len(raw_datasets[i]["question"]) + 1 :]
            + [tokenizer.eos_token_id]
        )
        labels_column.append(outputs)
    tokenized_datasets = tokenized_datasets.add_column(
        "labels",
        labels_column,
    )
    print("Done")
    return tokenized_datasets


def pretrain_model():
    config = AutoConfig.from_pretrained("./download")
    model = GPT2LMHeadModel.from_pretrained("./download", config=config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
    return model


args = TrainingArguments(
    report_to="none",
    output_dir="calculator",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=200,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    use_cpu=True,
)


def train(model, tokenizer, args, data_collator, tokenized_datasets):
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=tokenized_datasets,
    )
    trainer.train()
    model.save_pretrained("model/test")
    return trainer


def evaluate(model, tokenized_datasets, tokenizer, Context_length):
    right = 0
    wrong = []
    for i in range(0, len(tokenized_datasets)):
        input_ids = tokenized_datasets[i]["input_ids"]
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        outputs = model.generate(
            input_ids,
            max_length=Context_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        output_str = tokenizer.decode(outputs[0])
        if outputs == tokenized_datasets[i]["input_ids"]:
            right += 1
        else:
            wrong.append(
                [tokenizer.decode(tokenized_datasets[i]["input_ids"]), output_str]
            )
    print("Accuracy: ", right / len(tokenized_datasets))

    if len(wrong) >= 10:
        print(wrong[:10])
    else:
        print(wrong)


# 加载tokenizer
tokenizer = CharTokenizer()
tokenizer.pad_token = tokenizer.eos_token
Context_length = 128
data_collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=False,
)  # 它默认是进行mlm，设为False则进行clm

model = pretrain_model()
for i in range(0, 10):
    print("Round: ", i)
    tokenized_datasets = make_datasets(tokenizer, Context_length, size=10000)
    trainer = train(model, tokenizer, args, data_collator, tokenized_datasets)
    if i == 9:
        trainer.evaluate()
    print("Done\n")


# model = GPT2LMHeadModel.from_pretrained("model/test")
# tokenized_datasets = make_datasets(tokenizer, Context_length, size=10)


# CUDA_VISIBLE_DEVICES=7 python fintune.py
