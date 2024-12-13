import logging
from typing import Optional, Dict, Sequence
import argparse
import pdb

import torch
import transformers
from transformers import Trainer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from dataset import load_dataset, SupervisedDataset, DataCollator
from metric import Metric, preprocess_logits_for_metrics
from utils import parse_lora_config

def train(args):
    # Model
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_weight)
    model = GPT2LMHeadModel.from_pretrained(args.pretrained_weight)
    tokenizer.pad_token = tokenizer.eos_token
    
    if args.enable_lora:
        lora_config = parse_lora_config(args, fan_in_fan_out = True) # For GPT2 only
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Dataset
    dataset = load_dataset(tokenizer=tokenizer, data_path=args.dataset, split = args.split)
    train_dataset, val_dataset = dataset['train_dataset'], dataset['val_dataset']
    print(f'Training dataset: {len(train_dataset)}')
    print(f'Val dataset: {len(val_dataset) if val_dataset else 0}')
    data_collator = DataCollator(tokenizer=tokenizer)

    # Metric
    metric = Metric(tokenizer)

    # Train
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=transformers.TrainingArguments(
                          per_device_train_batch_size = args.batch_size,
                          gradient_accumulation_steps = args.batch_accumulation,
                          per_device_eval_batch_size = args.batch_size,
                          warmup_steps = args.warmup,
                          num_train_epochs = args.epochs,
                          learning_rate = args.lr,
                          fp16 = args.fp16,
                          optim = args.optimizer,
                          save_strategy = 'steps',
                          save_steps = 5000,
                          logging_steps = 5000,
                          # logging_dir = ,
                          output_dir = args.output_dir,
                          save_total_limit = 3,
                          evaluation_strategy = 'steps' if val_dataset else 'no',
                          eval_steps = 5000 if val_dataset else None,
                          eval_accumulation_steps = 1,
                          load_best_model_at_end = True if val_dataset else False,
                      ),
                      train_dataset = train_dataset,
                      eval_dataset = val_dataset,
                      data_collator = data_collator,
                      preprocess_logits_for_metrics = preprocess_logits_for_metrics,
                      compute_metrics = metric,
                      )
    print('Start training...')
    trainer.train()
    if val_dataset:
        trainer.evaluate()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_weight', type=str, default='./exp/checkpoint-9700/', help='./gpt2_pretrained/gpt2_124M/')
    parser.add_argument('--dataset', type=str, default='dataset/chatdoctor200k.json', help='dataset/alpaca_data_52k.json')
    parser.add_argument('--split', type=bool, default=True, help='False')
    parser.add_argument('--epochs', type=int, default=3, help='total training epochs, 3')
    parser.add_argument('--warmup', type=int, default=500, help='warmup steps')
    parser.add_argument('--batch_size', type=int, default=4, help='total batch size for all GPUs')
    parser.add_argument('--batch_accumulation', type=int, default=4, help='batch accumulation')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw_torch', help='')
    parser.add_argument('--fp16', type=bool, default=True, help='enable fp16 training')
    parser.add_argument('--output_dir', type=str, default='exp2', help='')

    # LoRA config
    parser.add_argument('--enable_lora', action='store_true', help='enable lora')
    parser.add_argument('--lora_r', type=int, default=8, help='lora matrix rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='lora alpha')
    parser.add_argument('--lora_target_modules', type=list, default=None, help='module name for lora')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='lora dropout')
    parser.add_argument('--lora_bias', type=str, default='none', help='lora bias')
    parser.add_argument('--lora_task_type', type=str, default='CAUSAL_LM', help='lora task type')
    
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parse_args()
    train(args)
