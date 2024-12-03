from .rouge_scorer import RougeScorer
from .scoring import BootstrapAggregator

def rouge(predictions, references, rouge_types=None, use_aggregator=False, use_stemmer=False, tokenizer=None):
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    multi_ref = isinstance(references[0], list)
    
    scorer = RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer, tokenizer=tokenizer)
    if use_aggregator:
        aggregator = BootstrapAggregator()
    else:
        scores = []

    for ref, pred in zip(references, predictions):
        if multi_ref:
            score = scorer.score_multi(ref, pred)
        else:
            score = scorer.score(ref, pred)
            
        if use_aggregator:
            aggregator.add_scores(score)
        else:
            scores.append(score)
            
    if use_aggregator:
        result = aggregator.aggregate()
        for key in result:
            result[key] = result[key].mid.fmeasure

    else:
        result = {}
        for key in scores[0]:
            result[key] = list(score[key].fmeasure for score in scores)

    return result

"""
result = rouge(predictions = ["hello there", "general kenobi"],
               references = ["hello there", "general kenobi"])
"""
