import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
from tokenization import ChineseTokenizer
from transformers import GPT2Tokenizer


path = "tokenizer.model"

chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model.Load(path)
chinese_spm = sp_pb2_model.ModelProto()
chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())
output_dir = './transformers_tokenizer/chinese/'
os.makedirs(output_dir, exist_ok=True)
with open(output_dir + 'chinese.model', 'wb') as f:
    f.write(chinese_spm.SerializeToString())
tokenizer = ChineseTokenizer(vocab_file=output_dir + 'chinese.model')

tokenizer.save_pretrained(output_dir)
print(f"Chinese tokenizer has been saved to {output_dir}")

# Test
chinese_tokenizer = ChineseTokenizer.from_pretrained(output_dir)
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)
text = '''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
The primary use of LLaMA is research on large language models, including'''
print("Test text:\n", text)
print(f"Tokenized by Chinese-LLaMA tokenizer:{chinese_tokenizer.tokenize(text)}")
# GPT2Tokenizer.from_pretrained("gpt2").save_pretrained("tokenizer")

# spm.SentencePieceTrainer.Train(
#     input='3.txt',
#     model_prefix='tokenizer',
#     vocab_size=50000,
#     character_coverage=1.0,
#     model_type='bpe',
#     num_threads=6,
# )
