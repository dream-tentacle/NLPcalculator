import torch
import torch.nn as nn


class dictionary:
    def __init__(self, pad_id: int, pad_letter: str, dic: dict) -> None:
        self.dic = dic
        self.pad_id = pad_id
        self.pad_letter = pad_letter
        self.rev_dic = dict(zip(dic.values(), dic.keys()))


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        import math

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        from torch.autograd import Variable

        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


def get_key_padding_mask(tokens, pad_id):
    key_padding_mask = torch.zeros(tokens.size())
    key_padding_mask[tokens == pad_id] = -torch.inf
    return key_padding_mask


def get_subsequent_mask(tokens):
    return nn.Transformer.generate_square_subsequent_mask(tokens.size(-1))


def masked_loss(loss_fn, predict, target, pad_id):
    """Masked loss for transformer"""
    # target: [batch_size, seq_len]
    # predict: [batch_size, seq_len, vocab_size]
    predict = predict.reshape(-1, predict.size(-1))
    target = target.reshape(-1)  # [batch_size * seq_len]
    # 对于target中的pad部分，不计算loss, 因此删去这部分的target和predict
    mask = target != pad_id
    predict = predict[mask, :]
    target = target[mask]
    return loss_fn(predict, target)


import math


def PE(pos, embedding_dim):
    # possition encoding
    re = torch.zeros(embedding_dim)
    for i in range(embedding_dim):
        if i % 2 == 0:
            re[i] = math.sin(pos / (math.pow(10000, 2 * i / embedding_dim)))
        else:
            re[i] = math.cos(pos / (math.pow(10000, 2 * i / embedding_dim)))
    return re


def greedy_search(model, input, max_len=100, start_id=13, end_id=14):
    # input should be only one sentence
    # input (torch): [src_len]
    # output (torch): [tgt_len]
    # model.eval()
    predict = [start_id]
    with torch.no_grad():
        while predict[-1] != end_id and len(predict) < max_len:
            tgt = torch.tensor(predict).unsqueeze(0)
            output = model(input.unsqueeze(0), tgt.to(input.device))
            output = output[0]
            predict.append(output.argmax(-1)[-1].item())
    return predict


def beam_search(
    model, input, beam_size=3, max_len=100, start_id=13, end_id=14, toprint=False
):
    # input should be only one sentence
    # input (torch): [src_len]
    # output (torch): [tgt_len]
    # model.eval()
    def get_score(predict):
        return predict[1] / len(predict[0])

    def topk(predicts, k):
        predicts = sorted(predicts, key=get_score, reverse=True)
        return predicts[:k]

    predicts = [[[start_id], 0]]
    ended_predicts = []
    with torch.no_grad():
        for _ in range(max_len):
            new_predicts = []
            for predict in predicts:
                tgt = torch.tensor(predict[0]).unsqueeze(0)
                output = model(input.unsqueeze(0), tgt.to(input.device))
                output = output[0, -1].softmax(-1)
                output = [
                    (math.log(score) if score > 0.01 else -100, number)
                    for number, score in enumerate(output)
                ]
                output = sorted(output, key=lambda x: x[0], reverse=True)
                output = output[:beam_size]
                for i in range(beam_size):
                    make_predict = (
                        predict[0] + [output[i][1]],
                        predict[1] + output[i][0],
                    )
                    if output[i][1] == end_id:
                        ended_predicts.append(make_predict)
                    elif make_predict[1] > -100:
                        new_predicts.append(make_predict)
                    if toprint:
                        print(make_predict)
            predicts = topk(new_predicts, beam_size)
            if len(predicts) == 0:
                break
    if len(ended_predicts) == 0:
        print(predicts)
        return predicts[0][0]
    return sorted(ended_predicts, key=get_score, reverse=True)[0][0]


def choose_from(*args):
    import random

    return args[random.randint(0, len(args) - 1)]
