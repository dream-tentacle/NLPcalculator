from datasets import load_dataset, DatasetDict, Dataset
import random
from tokenizer import CharTokenizer
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
from tqdm import tqdm


def make_datasets(
    tokenizer, Context_length=128, size=10, max_max_num=15, set_num=None, to_print=False
):
    def create_one(max_num):
        x = []
        for i in range(max_num):
            if i != 0:
                x.append(["+", "-", "*"][random.randint(0, 2)])
                # x.append(choose_from("+", "-"))
            x.append(random.randint(0, 9))
        y = ""
        x_str = "".join([str(i) for i in x])
        for i in range(max_num - 1):
            y += "="
            has_multi = -1
            for j in range(len(x)):
                if x[j] == "*":
                    has_multi = j
                    break
            if has_multi != -1:
                combined = x[has_multi - 1] * x[has_multi + 1]
                x[has_multi - 1] = combined
                x.pop(has_multi)
                x.pop(has_multi)
            else:
                if x[1] == "+":
                    combined = x[0] + x[2]
                else:
                    combined = x[0] - x[2]
                x[0] = combined
                x.pop(1)
                x.pop(1)
            y += str("".join([str(k) for k in x]))
        # print(x_str, y)
        x = x_str + "="
        y = y[1:]
        return {"question": x, "answer": y, "full": x + y, "question_len": len(x)}

    def tokenize(element):
        tokenized = tokenizer(
            element["full"],
            return_length=True,
            truncation=True,
            max_length=Context_length,
            padding="max_length",
        )
        return tokenized

    if to_print:
        print("Creating datasets... ")
    # 创建数据集字典
    raw_datasets = []
    if set_num is not None:
        for i in range(size):
            raw_datasets.append(create_one(set_num))
    else:
        for i in range(size):
            raw_datasets.append(create_one(random.randint(2, max_max_num)))
    raw_datasets = Dataset.from_list(raw_datasets)
    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets.column_names
    )
    labels_column = []
    for i in range(0, len(tokenized_datasets)):
        # 生成labels
        outputs = ([-100] * len(raw_datasets[i]["question"])) + tokenized_datasets[i][
            "input_ids"
        ][len(raw_datasets[i]["question"]) :]
        labels_column.append(outputs)
    tokenized_datasets = tokenized_datasets.add_column(
        "labels",
        labels_column,
    )
    tokenized_datasets = tokenized_datasets.add_column(
        "question",
        raw_datasets["question"],
    )
    tokenized_datasets = tokenized_datasets.add_column(
        "answer",
        raw_datasets["answer"],
    )
    tokenized_datasets = tokenized_datasets.add_column(
        "full",
        raw_datasets["full"],
    )
    if to_print:
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
    per_device_eval_batch_size=32,
    logging_steps=200,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-3,
    save_steps=3000,
)


def train(model, tokenizer, args, data_collator, tokenized_datasets):
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=tokenized_datasets,
    )
    trainer.train()
    return trainer


def evaluate(model, tokenized_datasets, tokenizer, Context_length):
    right = 0
    wrong = []

    pbar = tqdm(range(len(tokenized_datasets)))
    model.eval()
    model.to("cuda")
    for i in pbar:
        outputs = model(
            input_ids=torch.tensor(tokenized_datasets[i]["input_ids"])
            .unsqueeze(0)
            .to("cuda"),
        )
        outputs = outputs.logits
        outputs = torch.argmax(outputs, dim=-1)
        outputs = outputs[0].cpu().detach().numpy().tolist()
        outputs = outputs[
            len(tokenized_datasets[i]["question"])
            - 1 : len(tokenized_datasets[i]["full"])
            - 1
        ]
        if tokenizer.decode(outputs) == tokenized_datasets[i]["answer"]:
            right += 1
        else:
            wrong.append(
                [
                    tokenizer.decode(outputs),
                    tokenized_datasets[i]["answer"],
                ]
            )

    for i in range(min(10, len(wrong))):
        print(f"{wrong[i][0]}, len: {len(wrong[i][0])}")
        print(f"{wrong[i][1]}, len: {len(wrong[i][1])}")
    print(f"accuracy: {right/len(tokenized_datasets)}")


def my_token_fn(self, text):
    return list(text)


# 加载tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./download")
tokenizer.tokenize = my_token_fn.__get__(tokenizer, GPT2Tokenizer)

tokenizer.pad_token = tokenizer.eos_token
Context_length = 512
data_collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=False,
)  # 它默认是进行mlm，设为False则进行clm


# model = pretrain_model()
# tokenized_datasets = make_datasets(tokenizer, Context_length, size=1000000)
# train(model, tokenizer, args, data_collator, tokenized_datasets)
# model.save_pretrained("model/1")
# print("Done\n")


model = GPT2LMHeadModel.from_pretrained("model/1")
for i in range(3, 17):
    print(f"set_num: {i}")
    tokenized_datasets = make_datasets(tokenizer, Context_length, size=1000, set_num=i)
    evaluate(model, tokenized_datasets, tokenizer, Context_length)
# tokenized_datasets = make_datasets(tokenizer, Context_length, size=10000)

# evaluate(model, tokenized_datasets, tokenizer, Context_length)


# CUDA_VISIBLE_DEVICES=7 python fintune.py
