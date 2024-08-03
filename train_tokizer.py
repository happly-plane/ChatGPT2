import sentencepiece as spm
import re


spm.SentencePieceTrainer.Train(
    input='3.txt',
    model_prefix='tokenizer',
    vocab_size=50000,
    character_coverage=1.0,
    model_type='bpe',
    num_threads=6,
)








# zs = re.compile(r"第[0-9]|[1-9][0-9]|1[0-3][0-6][0-6]章")
# with open('2.txt','r',encoding='utf-8') as f:
#     text = f.readlines()
#     for d in text :
#         if re.match(zs,d):
#             print(d)
#         # if re.findall(r"第[0-9]|[1-9][0-9]|1[0-3][0-6][0-6]章",d):
#         #     print(d)



