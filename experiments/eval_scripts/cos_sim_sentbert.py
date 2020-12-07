import pickle
import argparse
import os
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
"""
model possible use-cases:
bert-base-nli-stsb-mean-tokens: Performance: STSbenchmark: 85.14
bert-large-nli-stsb-mean-tokens: Performance: STSbenchmark: 85.29
roberta-base-nli-stsb-mean-tokens: Performance: STSbenchmark: 85.44
roberta-large-nli-stsb-mean-tokens: Performance: STSbenchmark: 86.39
distilbert-base-nli-stsb-mean-tokens: Performance: STSbenchmark: 84.38
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_semantic_emb(sentences, version):
    model = SentenceTransformer(version, device=DEVICE)
    model.to(DEVICE)
    sentence_embeddings = model.encode(sentences)
    mydict = dict(zip(sentences, sentence_embeddings))
    return mydict


def make_arg_parser():
    parser = argparse.ArgumentParser(description='Main options')
    parser.add_argument('-inpickle', type=str, help='Pickle file with '
                                                      'sentences',
                        required=True)
    parser.add_argument('-out', type=str, help='outputfolder to store '
                                                     'sentence embeddings',
                        default=None)
    parser.add_argument('-version', type=str, help='Model version to be used',
                        default='bert-base-nli-stsb-mean-tokens')
    return parser


if __name__ == "__main__":

    parser = make_arg_parser()
    options = parser.parse_args()

    with open(options.inpickle, 'rb') as handle:
        test_data = pickle.load(handle)
    handle.close()

    sentences = []
    for inp, answer, target in test_data:
        sentences.append(inp)
        sentences.append(answer)
        sentences.append(target)

    #encode all sentences (for speed up)
    sent_embeddings = get_semantic_emb(sentences, options.version)

    cos_sims = []
    for inp, answer, target in test_data:
        tensor1 = torch.tensor(sent_embeddings[answer]).unsqueeze(0)
        tensor2 = torch.tensor(sent_embeddings[target]).unsqueeze(0)
        out = cosine_similarity(tensor1, tensor2)
        cos_sims.append(out)
    print(sum(cos_sims)/len(cos_sims))
