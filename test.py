# import os
# import tqdm
# from tokenizers import Tokenizer
# import re
# tokenizer = Tokenizer.from_file("tokenizer.json")
# def remove_punctuation(text):
#     return re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)

# with open('2.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     lines = [remove_punctuation(str(line)) for line in lines]
#     lines = [line.replace('\n', '[SEP]') for line in lines]
# all_len = len(lines)
# print(all_len)
# num_piecs = 100
# min_length = 5
# for i in tqdm.tqdm(range(num_piecs)):
#     sublines = lines[all_len // num_piecs * i: all_len // num_piecs * (i + 1)]
#     if i == num_piecs - 1:
#         sublines.extend(lines[all_len // num_piecs * (i + 1):])
    
#     sublines = [tokenizer.encode(line).tokens for line in sublines if  len(line) > min_length]
#     # sublines = [tokenizer.encode(line).ids for line in sublines if len(line) > min_length ]
#     full_line = []
#     for subline in sublines:
#         full_line.append(tokenizer.encode("[MASK]").ids)
#         full_line.extend(subline)
#         full_line.append(tokenizer.encode("[CLS]").ids)
#     with open('vocab.txt','w') as f:
#         for id in full_line:
#             f.write(str(id) + '\n')

# from tokenizers import Tokenizer

# tokenizer = Tokenizer.from_file("tokenizer.json")

# text = "你\n好"
# encoded = tokenizer.encode(text).tokens
# print(encoded)
from transformers import BertTokenizer

BertTokenizer.from_pretrained(,do_lower_case=True)