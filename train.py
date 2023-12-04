from datasets import load_dataset
from transformers import AutoConfig, AutoModel, Trainer, TrainingArguments
# from itertools import chain
from torch.utils.data import DataLoader
import sys
sys.path.append('/home/ubuntu/stl/models--THUDM--chatglm3-6b/snapshots/e46a14881eae613281abbd266ee918e93a56018f/')
from modeling_chatglm import ChatGLMForConditionalGeneration, RMSNorm
import torch
from transformers import PrinterCallback as callback_fun


batch_size = 32
vocab_size = 118
hidden_size = 1000
pad_token_id = 0
log_step = 10

cfg = AutoConfig.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, cache_dir='.')
cfg.ffn_hidden_size=hidden_size*2
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
        

def pad(x, padding):
    difference = cfg.seq_length - len(x)
    return [padding] * difference + x[:cfg.seq_length]

def tokenizer(input_list):
    input_ids = []
    # position_ids = []
    for line in input_list:
        y = [round(float(x)) + 11 if round(float(x)) != 0 else pad_token_id for x in line.split()]
        # p = [i if x != 11 else 0 for i,x in enumerate(y)]
        if len(y) > 1:
            input_ids.append(y)
            # position_ids.append(p)
    attention_mask = [[1]*len(i) for i in input_ids]
    position_ids = [list(range(len(x))) for x in input_ids]
    # pad
    input_ids = [pad(x, pad_token_id) for x in input_ids]
    position_ids = [pad(x, 0) for x in position_ids]
    attention_mask = [pad(x, 0) for x in attention_mask]
    # labels = [pad(x[-5:], 0) for x in input_ids]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids, 'labels': input_ids}

dataset = load_dataset("text", data_files={"train": ["text.txt"]})
tokenized_examples = dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)['train']
# concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in ['input_ids', 'attention_mask', 'position_ids', 'labels']}
# total_length = len(concatenated_examples['input_ids'])
# total_length = (total_length // cfg.seq_length) * cfg.seq_length
# result = {
#             k: [t[i: i + cfg.seq_length] for i in range(0, total_length, cfg.seq_length)]
#             for k, t in concatenated_examples.items()
#         }
# dataloader_params = {
#             "batch_size": batch_size,
#             "collate_fn": self._get_collator_with_removed_columns(data_collator, description="training"),
#             "num_workers": 0,
#             "pin_memory": True,
#             "sampler" : self._get_train_sampler(),
#             'drop_last': False, 
#             'worker_init_fn': seed_worker
#         }

# DataLoader(tokenized_examples, **dataloader_params)

training_args = TrainingArguments(output_dir='output')
training_args.logging_steps = log_step
training_args.num_train_epochs = 6
training_args.save_steps = 1000
training_args.save_total_limit = 1
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_examples, callbacks=[callback_fun()])
trainer.train()
# trainer.train(resume_from_checkpoint=True)
trainer.save_state()
trainer.save_model()