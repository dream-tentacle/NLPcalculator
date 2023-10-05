from typing import Dict
from transformers import PreTrainedTokenizer, GPT2Tokenizer


class CharTokenizer(PreTrainedTokenizer):
    """
    由于GPT-2的分词器是基于词的，因此我们需要自定义一个分词器，将输入的文本分割成单个字符
    这样才能让GPT-2模型学习到加法
    """

    def __init__(self, *args, **kwargs):
        # 初始化GPT-2的分词器
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("./download")
        super().__init__(*args, **kwargs)
        self.pad_token = self.gpt2_tokenizer.eos_token
        self.eos_token = self.gpt2_tokenizer.eos_token
        self.pad_token_id = self.gpt2_tokenizer.eos_token_id
        self.eos_token_id = self.gpt2_tokenizer.eos_token_id

    def get_vocab(self) -> Dict[str, int]:
        """返回词表"""
        return self.gpt2_tokenizer.get_vocab()

    def _tokenize(self, text):
        """将输入文本分割成单个字符"""
        return list(text)

    def _convert_token_to_id(self, token):
        """将字符转换为其对应的ID"""
        return self.gpt2_tokenizer.convert_tokens_to_ids(token)

    def _convert_id_to_token(self, id):
        """将ID转换为对应的字符"""
        return self.gpt2_tokenizer.convert_ids_to_tokens(id)

    def _decode(self, token_ids, **kwargs):
        """将ID列表转换为对应的文本"""
        re = ""
        for id in token_ids:
            re += self._convert_id_to_token(id)
        return re


if __name__ == "__main__":
    custom_tokenizer = CharTokenizer()

    # 要编码的文本
    text = "12+12=24"

    # 使用自定义分词器进行编码
    input_ids = custom_tokenizer.encode(text)

    # 打印编码后的结果
    print(input_ids)
