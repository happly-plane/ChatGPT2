import sentencepiece as spm
import re

with open('2.txt','r',encoding='utf-8') as f:
    text = f.read().strip().strip('\n')
sentences = []
a = 0
for d in text:
    d = text.strip()
    if "==" in d or len(d) == 0 :
        a += 1
        print(a)
        continue
    sentences.append(d)
with open('3.txt','w',encoding='utf-8') as f:
    f.write("\n".join(sentences))