import torch
from torch.utils.data import Dataset
from typing import Union, Optional, Dict, List, Sequence
import utils
import copy
import random
import pdb

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    ),
}

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Collate examples for supervised fine-tuning."""
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # 以下填充方式是普遍的标准做法
        # input使用pad/eos填充, 模型的embedding层也不接受-100的index
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        # label使用IGNORE_INDEX填充
        # 设置为IGNORE_INDEX是为了不计入损失
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX) 
        return dict(input_ids=input_ids,
                    labels=labels,
                    attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                    )

    
def _tokenize_fn(strings: Sequence[str], tokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [tokenizer(text,
                                return_tensors="pt", # None:返回List(int), 'pt':torch.Tensor, 'tf':tf.Tensor, 'np':ndarray
                                # padding="longest", # 由于preprocess是单独一条条数据处理的, 所以此处没有意义
                                truncation=True, # 最大长度截断
                                max_length=tokenizer.model_max_length,
                                )
                      for text in strings
                      ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(input_ids=input_ids,
                labels=labels,
                input_ids_lens=input_ids_lens,
                labels_lens=labels_lens,
                )

IGNORE_INDEX = -100 # torch.nn.CrossEntropyLoss的默认忽略值(ignore_index=-100)
"""
(Pdb) self.input_ids[0]
tensor([21106,   318,   281, 12064,   326,  8477,   257,  4876,    13, 19430,
          257,  2882,   326, 20431, 32543,   262,  2581,    13,   198,   198,
        21017, 46486,    25,   198, 23318,  1115,  9040,   329, 10589,  5448,
           13,   198,   198, 21017, 18261,    25,   198,    16,    13, 47659,
          257, 12974,  5496,   290,   787,  1654,   284,  2291,  6088,   286,
        15921,   290, 13701,    13,   220,   198,    17,    13, 32900,  7987,
          284,  1394,   534,  1767,  4075,   290,  1913,    13,   220,   198,
           18,    13,  3497,  1576,  3993,   290,  5529,   257,  6414,  3993,
         7269,    13, 50256])
(Pdb) self.labels[0]
tensor([ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,    16,    13, 47659,
          257, 12974,  5496,   290,   787,  1654,   284,  2291,  6088,   286,
        15921,   290, 13701,    13,   220,   198,    17,    13, 32900,  7987,
          284,  1394,   534,  1767,  4075,   290,  1913,    13,   220,   198,
           18,    13,  3497,  1576,  3993,   290,  5529,   257,  6414,  3993,
         7269,    13, 50256])
即前半部分忽略, 不计入loss, 只有response部分需要预测
"""
def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX  # 前半部分设置为忽略index
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data: Union[str, List], tokenizer, prefix = ''):
        super(SupervisedDataset, self).__init__()
        # logging.warning("Loading data...")
        print(f'Loading {prefix} data...')
        if isinstance(data, str):
            list_data_dict = utils.jload(data_path)
        elif isinstance(data, list):
            list_data_dict = data
        else:
            raise ValueError(f'Unkown data type {type(data)}')

        # logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources, targets  = [], []
        for example in list_data_dict:
            if example.get("input", "") != "":
                ss = prompt_input.format_map(example)
            else:
                ss = prompt_no_input.format_map(example)
            tt = f"{example['output']}{tokenizer.eos_token}" 
            sources.append(ss)
            targets.append(tt)
        # sources = [
        #     prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        #     for example in list_data_dict
        # ]
        # targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        # logging.warning("Tokenizing inputs... This may take some time...")
        print(f'Tokenizing {prefix} data... This may take some time...')
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def load_dataset(tokenizer, data_path, split = False, split_ratio = 0.9, seed=42):
    list_data_dict = utils.jload(data_path)
    if split:
        random.seed(seed)
        random.shuffle(list_data_dict)
        split_num = int(len(list_data_dict)*split_ratio)
        train_data = list_data_dict[:split_num]
        val_data = list_data_dict[split_num:]
        train = SupervisedDataset(train_data, tokenizer, 'training')
        val = SupervisedDataset(val_data, tokenizer, 'validation')
    else:
        train = SupervisedDataset(list_data_dict, tokenizer)
        val = None

    return dict(train_dataset = train, val_dataset = val)

    
if __name__ == "__main__":
    # unit test
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_pretrained/gpt2_124M/')
    tokenizer.pad_token = tokenizer.eos_token
    # 1.
    # dataset = SupervisedDataset(data_path = 'dataset/chatdoctor200k.json', tokenizer = tokenizer)
    
    # 2.
    dataset = load_dataset(tokenizer, 'dataset/chatdoctor200k.json', True)
    train_dataset, val_dataset = dataset['train_dataset'], dataset['val_dataset']

    pdb.set_trace()
