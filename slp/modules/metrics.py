from nltk.translate.bleu_score import sentence_bleu

def calc_sentence_bleu_score(reference,hypothesis,n=4):
    """
    This function receives as input a reference sentence(list) and a
    hypothesis(list) and
    returns bleu score.

    Bleu score formula: https://leimao.github.io/blog/BLEU-Score/
    """
    reference = [reference]
    weights = [1/n for _ in range(n)]
    return sentence_bleu(reference, hypothesis, weights)

