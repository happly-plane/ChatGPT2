from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# 初始化一个空的BPE模型
tokenizer = Tokenizer(models.BPE())

# 使用 `Whitespaces` 作为预处理器
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 设置 PostProcessor


# 加载数据
files = ["2.txt"]

# 训练BPE模型
trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]"])
tokenizer.train(files, trainer)


# 设置解码器
tokenizer.decoder = decoders.BPEDecoder()
tokenizer.post_processor = processors.BertProcessing(
    sep = tokenizer.token_to_id("[SEP]"),
    cls = tokenizer.token_to_id("[CLS]")
)

# 保存分词器
tokenizer.save("tokenizer.json")
