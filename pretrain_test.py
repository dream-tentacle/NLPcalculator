import random
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", pad_token_id=tokenizer.eos_token_id
).to(torch_device)


class myDataset(Dataset):
    def __init__(self, length, max_num_len) -> None:
        super().__init__()
        self.length = length
        self.max_num_len = max_num_len

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        max_num = random.randint(1, self.max_num_len)
        a = max_num
        b = max_num
        a = random.randint(0, int(pow(10, a)))
        b = random.randint(0, int(pow(10, b)))
        x = str(a) + "+" + str(b) + "="
        y = tokenizer.bos_token + str(a + b) + tokenizer.eos_token
        input_ids = tokenizer.encode(x, return_tensors="pt").squeeze()
        labels = tokenizer.encode(y, return_tensors="pt").squeeze()
        return {"input_ids": input_ids, "labels": labels}


dataset = myDataset(5, max_num_len=5)

from transformers import Trainer, TrainingArguments

train_args = TrainingArguments(
    output_dir="./results",
)
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=dataset,
)
trainer.train()
