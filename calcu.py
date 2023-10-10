import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import tqdm
import math
from utils import *


class Seq2seqTransformer(nn.Module):
    def __init__(
        self, enc_layer, dec_layer, embedding_dim, hidden_size, dic, head, dropout=0.1
    ) -> None:
        super().__init__()
        self.dic = dic
        self.head = head
        self.d_model = embedding_dim
        self.embed = nn.Embedding(
            num_embeddings=len(dic.dic),
            embedding_dim=embedding_dim,
            padding_idx=dic.pad_id,
        )
        self.embed2 = nn.Embedding(
            num_embeddings=len(dic.dic),
            embedding_dim=embedding_dim,
            padding_idx=dic.pad_id,
        )
        self.position_encoding = PositionalEncoding(embedding_dim, dropout=dropout)
        self.encdec = nn.Transformer(
            d_model=embedding_dim,
            nhead=head,
            num_encoder_layers=enc_layer,
            num_decoder_layers=dec_layer,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, len(dic.dic)),
        )

    def forward(
        self, src, tgt, src_key_padding=None, tgt_key_padding=None, tgt_mask=None
    ):
        if src_key_padding is None:
            src_key_padding = get_key_padding_mask(src, self.dic.pad_id).to(src.device)
        if tgt_key_padding is None:
            tgt_key_padding = get_key_padding_mask(tgt, self.dic.pad_id).to(tgt.device)
        if tgt_mask is None:
            tgt_mask = get_subsequent_mask(tgt).to(tgt.device)
        src = self.position_encoding(self.embed(src))
        tgt = self.position_encoding(self.embed2(tgt))
        out = self.encdec(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding,
            tgt_key_padding_mask=tgt_key_padding,
        )
        out = self.fc(out)
        return out


class myDataset(Dataset):
    def __init__(self, length, max_num_len) -> None:
        super().__init__()
        self.length = length
        self.max_num = max_num_len

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        a = self.max_num
        b = self.max_num
        a = random.randint(0, int(pow(10, a)))
        b = random.randint(0, int(pow(10, b)))
        x = str(a) + "+" + str(b) + "="
        y = "<" + str(a + b) + ">"
        return x, y


word2num = {"-": 0, "+": 11, "=": 12, "<": 13, ">": 14, ",": 15}
for i in range(10):
    word2num[str(i)] = i + 1

dic = dictionary(pad_id=0, pad_letter="-", dic=word2num)


def collate_fn(data):
    dataset.max_num = random.randint(1, 10)
    x = []
    y = []
    for i in data:
        i_x = [dic.dic[j] for j in i[0]]
        i_y = [dic.dic[j] for j in i[1]]
        i_x = torch.tensor(i_x)
        i_y = torch.tensor(i_y)
        x.append(i_x)
        y.append(i_y)
    x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=dic.pad_id)
    y = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=dic.pad_id)
    return x, y


dataset = myDataset(50000, max_num_len=20)
dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)

model = Seq2seqTransformer(
    enc_layer=3,
    dec_layer=3,
    embedding_dim=128,
    hidden_size=256,
    dic=dic,
    head=8,
    dropout=0,
)


# 初始化model
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


model.apply(init_weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()


def train(epoch, save_model_name):
    pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))  # 进度条
    model.train()
    sum = 0
    total = 0
    correct = 0
    total_loss = 0
    total_correct = 0
    for i, (x, y) in pbar:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        out = model(x, y[..., :-1])
        y = y[..., 1:]
        loss = loss_fn(out.reshape(-1, out.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        sum += x.size(0)
        total += (y != 0).sum().item()
        out = out.argmax(dim=-1)
        correct += ((out == y) & (y != 0)).sum().item()
        total_correct += [torch.equal(i, j) for i, j in zip(out, y)].count(True)
        total_loss += loss.item()
        if i % 10 == 0:
            pbar.set_description(
                f"epoch:{epoch}, loss:{total_loss/(i+1):.4f}, acc:{correct/total:.4f}, acc2:{total_correct/sum:.4f}"
            )
    torch.save(model.state_dict(), save_model_name)
    # 输出一个例子便于观察
    model.eval()
    x, y = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)
    out = model(x, y[:, :-1])
    out = out.argmax(dim=-1)
    print("".join([dic.rev_dic[i.item()] for i in x[0]]))
    print("".join([dic.rev_dic[i.item()] for i in y[0, 1:]]))
    print("".join([dic.rev_dic[i.item()] for i in out[0]]))


def test(total=1000, x=None, print_right=False):
    model.eval()
    correct = 0
    pbar = tqdm.tqdm(range(total))
    wrong = []
    if x is not None:
        x = torch.tensor([dic.dic[i] for i in x]).to(device)
        out = beam_search(model, x)
        out = "".join([dic.rev_dic[i] for i in out])
        print(out)
        return
    for i in pbar:
        dataset.max_num = random.randint(21, 25)
        x, y = dataset[i]
        x = torch.tensor([dic.dic[i] for i in x]).to(device)
        out = beam_search(model, x)
        out = "".join([dic.rev_dic[i] for i in out])
        if out == y:
            correct += 1
            if print_right:
                x = "".join([dic.rev_dic[i.item()] for i in x])
                wrong.append(f"{x}{out}(true:{y})")

        elif not print_right:
            x = "".join([dic.rev_dic[i.item()] for i in x])
            wrong.append(f"{x}{out}(true:{y})")
        pbar.set_description(f"acc:{correct/(i+1):.4f}")
    for i in wrong:
        print(i)


model_file = "model"
load_model_name = "calcu_slob.pt"
save_model_name = "calcu_slob.pt"  # same len on batch
# model.load_state_dict(torch.load(model_file + "\\" + load_model_name))


for i in range(100):
    train(i, model_file + "\\" + save_model_name)

test(100)
exit()
