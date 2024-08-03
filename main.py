import datetime
import random
import numpy as np
import transformers
import torch
import os
from tqdm import tqdm
from torch.nn import DataParallel

model_config = transformers.GPT2Config.from_json_file("config/model.json")
model = transformers.GPT2LMHeadModel(config=model_config)

model.train()
# model.to(device)

warmup_steps = 1000
total_steps = 100000
epochs = 100
num_pieces = 1000
num_parameters = 0
batch_size = 10


parameters = model.parameters()
for parameter in parameters:
    num_parameters += parameter.numel()
print(num_parameters)

optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
                                                          t_total=total_steps)

print('starting training')
overall_step = 0
running_loss = 0

for epoch in range(epochs):
    print('epoch {}'.format(epoch + 1))
    now = datetime.now()
    print('time: {}'.format(now))
    x = np.linspace(0, num_pieces - 1, num_pieces, dtype=np.int32)
    random.shuffle(x)
    piece_num = 0
    for i in x:
        with open('vocab.txt'.format(i), 'r') as f:
            line = f.read().strip()
        tokens = line.split()
        tokens = [int(token) for token in tokens]
        start_point = 0
        samples = []
        while start_point < len(tokens) - n_ctx:
            samples.append(tokens[start_point: start_point + n_ctx])
            start_point += stride
        if start_point < len(tokens):
            samples.append(tokens[len(tokens)-n_ctx:])
        random.shuffle(samples)
        
        for step in range(len(samples) // batch_size):  # drop last

            #  prepare data
            batch = samples[step * batch_size: (step + 1) * batch_size]
            batch_inputs = []
            for ids in batch:
                int_ids = [int(x) for x in ids]
                batch_inputs.append(int_ids)
            batch_inputs = torch.tensor(batch_inputs).long().to(device)

            #  forward pass
            outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
            loss, logits = outputs[:2]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            #  optimizer step
            if (overall_step + 1) % gradient_accumulation == 0:
                running_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            if (overall_step + 1) % log_step == 0:
                print('now time: {}:{}. Step {} of piece {} of epoch {}, loss {}'.format(
                    datetime.now().hour,
                    datetime.now().minute,
                    step + 1,
                    piece_num,
                    epoch + 1,
                    running_loss * gradient_accumulation / (log_step / gradient_accumulation)))
                running_loss = 0
            overall_step += 1
        piece_num += 1

    print('saving model for epoch {}'.format(epoch + 1))
    if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
        os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
    # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
    # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
    print('epoch {} finished'.format(epoch + 1))

    then = datetime.now()
    print('time: {}'.format(then))
    print('time for one epoch: {}'.format(then - now))

print('training finished')
if not os.path.exists(output_dir + 'final_model'):
    os.mkdir(output_dir + 'final_model')
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir + 'final_model')