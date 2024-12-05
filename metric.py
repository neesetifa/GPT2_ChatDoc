import pdb

import torch
from metrics.bleu import bleu
from metrics.rouge import rouge

def preprocess_logits_for_metrics(logits, labels):
    # logits shape = [batch, seq_len_pred, vocab_size]
    # labels shape = [batch, seq_len_label]
    # technically seq_len_pred = seq_len_label
    pred_ids = logits.argmax(dim=-1)  # use greedy search during training
    return pred_ids
    

# TODO
# This function is called after evaluating on the whole dataset in transformers.Trainer
# if used as compute_metrics function, 
# For beam search and topkp, need to rewrite trainer prediction_step() and add this Metric.
class Metric:
    def __init__(self, tokenizer, decoding_strategy:str = 'greedy', model = None):
        assert decoding_strategy in ['greedy', 'beam', 'topkp'], \
            f'Got unknown decoding_strategy: {decoding_strategy}'

        # assert not model and decoding_strategy != 'greedy', \
        #    f'For strategy like beam search or top sampling, you must pass model'

        self.tokenizer = tokenizer
        self.model = model
        self.decoding_strategy = decoding_strategy
        
    def __call__(self, eval_pred, debug_mode = False):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        # predss shape = [whole_val_dataset, max_seq_len_pred = 1024]
        # labels shape = [whole_val_dataset, max_seq_len_label = 1024]

        # shift preds and labels
        shift_preds = preds[..., :-1]
        shift_labels = labels[..., 1:]
        ignore_idx = shift_labels == -100
        shift_preds[ignore_idx] = self.tokenizer.pad_token_id
        shift_labels[ignore_idx] = self.tokenizer.pad_token_id
        
        # evaluate
        print('Decoding...')
        if self.decoding_strategy == 'greedy':
            decoded_preds = self.tokenizer.batch_decode(shift_preds, skip_special_tokens=True)
            
        elif self.decoding_strategy == 'beam':
            batch_size = logits.shape[0]
            decoded_preds = []
            for i in range(batch_size):
                input_ids = logits[i].unsqueeze(0) # shape = [1, seq_len_pred, vocab_size]
                outputs = model.generate(input_ids,
                                         max_len = self.tokenizer.model_max_length,
                                         num_beams = 5,
                                         no_repeat_ngram = 2,
                                         early_stopping = True)
                decoded_pred.append(self.tokenizer.decode(outputs[0], skip_special_tokens = True))
        elif self.decoding_strategy == 'topkp':
            batch_size = logits.shape[0]
            decoded_preds = []
            for i in range(batch_size):
                input_ids = logits[i].unsqueeze(0) # shape = [1, seq_len_pred, vocab_size]
                outputs = model.generate(input_ids,
                                         max_len = self.tokenizer.model_max_length,
                                         do_sample = True,
                                         top_k = 50,
                                         top_p = 0.95,
                                         temperature = 1.0)
                decoded_pred.append(self.tokenizer.decode(outputs[0], skip_special_tokens = True))
            
        # label
        decoded_labels = self.tokenizer.batch_decode(shift_labels, skip_special_tokens = True)
        references = [[ref] for ref in decoded_labels]

        # score
        if debug_mode:
            print(decoded_preds, decoded_labels)
        print('Calculating scores...')
        bleu_score = bleu(predictions = decoded_preds, references = references)
        rouge_score = rouge(predictions = decoded_preds, references = decoded_labels, use_aggregator = True)
        return dict(bleu = bleu_score['bleu'],
                    rouge1 = rouge_score['rouge1'],
                    rougeL = rouge_score['rougeL'])


if __name__ == "__main__":
    # unit test
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_pretrained/gpt2_124M/')
    tokenizer.pad_token = tokenizer.eos_token

    metric = Metric(tokenizer)
    logits = torch.randint(low=1000, high=1005, size=(1,3))
    labels = torch.randint(low=1000, high=1005, size=(1,3))
    result = metric((logits, labels), debug_mode = True)
    print(result)
    
    pdb.set_trace()
