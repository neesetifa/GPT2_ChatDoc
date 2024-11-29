#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
from typing import Optional, Dict, Sequence

import torch
import transformers

from transformers import Trainer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextGenerationPipeline
from dataset import SupervisedDataset, DataCollator

import utils
import pdb


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def train():
    tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_pretrained/gpt2_124M/')
    model = GPT2LMHeadModel.from_pretrained('./gpt2_pretrained/gpt2_124M/')
    tokenizer.pad_token = tokenizer.eos_token
    # print(tokenizer.pad_token_id)

    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path='dataset/alpaca_data_52k.json')
    data_collator = DataCollator(tokenizer=tokenizer)
    
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=transformers.TrainingArguments(
                          per_device_train_batch_size = 4,
                          gradient_accumulation_steps = 4,
                          warmup_steps = 500,
                          num_train_epochs = 3,
                          learning_rate = 5e-5,
                          fp16 = True,
                          logging_steps = 100,
                          optim = "adamw_torch",
                          evaluation_strategy = "no", #"steps",
                          save_strategy = "steps",
                          eval_steps = None,
                          save_steps = 100,
                          output_dir = 'exp',
                          save_total_limit = 3,
                          load_best_model_at_end = False,
                      ),
                      train_dataset=train_dataset,
                      eval_dataset=None,
                      data_collator=data_collator)
    print('Start training...')
    trainer.train()


if __name__ == "__main__":
    train()
