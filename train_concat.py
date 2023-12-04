import datasets
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, Trainer, TrainingArguments
from itertools import chain
from torch.utils.data import DataLoader
import sys
sys.path.append('/home/ubuntu/stl/models--THUDM--chatglm3-6b/snapshots/e46a14881eae613281abbd266ee918e93a56018f/')
from modeling_chatglm import ChatGLMForConditionalGeneration, RMSNorm
import torch
from transformers import PrinterCallback as callback_fun
import numpy as np 


batch_size = 8
vocab_size = 22
hidden_size = 1000

cfg = AutoConfig.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, cache_dir='.')
cfg.ffn_hidden_size=2*hidden_size
cfg.hidden_size=hidden_size
cfg.padded_vocab_size=vocab_size
cfg.seq_length=100
cfg.vocab_size=vocab_size
cfg.torch_dtype=torch.float #使用fp32
model = ChatGLMForConditionalGeneration._from_config(cfg)
for m in model.modules():
    if hasattr(m, 'weight'):
        if isinstance(m, RMSNorm):
            torch.nn.init.kaiming_uniform_(m.weight.unsqueeze(0))
        else:
            torch.nn.init.kaiming_uniform_(m.weight)
        

pad_token_id = 0

freq = [0 for _ in range(22)]
def tokenizer(input_list):
    def _tokenize(x):
        x = round(float(x)) + 11
        if x > 21:
            return 0
        ratio = freq[x] / (sum(freq)+1e-3)
        thr = (ratio - 0.08) * 50
        if np.random.rand() > thr:
            freq[x] += 1
            return x
        else: return 0
        
    input_ids = []
    for line in input_list:
        y = [_tokenize(x) for x in line.split()] 
        if len(y) > 1:
            input_ids.append([0] + y[1:])
    attention_mask = [[1]*len(i) for i in input_ids]
    position_ids = [list(range(len(i))) for i in input_ids]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids, 'labels': input_ids}

dataset = load_dataset("text", data_files={"train": ["text.txt"]})
tokenized_examples = dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)['train']
concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in ['input_ids', 'attention_mask', 'position_ids', 'labels']}
total_length = len(concatenated_examples['input_ids'])
total_length = (total_length // cfg.seq_length) * cfg.seq_length
result = {
            k: [t[i: i + cfg.seq_length] for i in range(0, total_length, cfg.seq_length)]
            for k, t in concatenated_examples.items()
        }


training_args = TrainingArguments(output_dir='output_concat')
training_args.logging_steps = 10
training_args.num_train_epochs = 5
training_args.save_steps = 1000
training_args.save_total_limit = 1
trainer = Trainer(model=model, args=training_args, train_dataset=datasets.Dataset.from_dict(result), callbacks=[callback_fun()])
# trainer.train()
trainer.train(resume_from_checkpoint=True)
trainer.save_state()
trainer.save_model()