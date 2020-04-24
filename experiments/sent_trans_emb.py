import pickle
import argparse
import os
import torch
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


def get_semantic_emb(sentences, version):
    model = SentenceTransformer(version)
    sentence_embeddings = model.encode(sentences)
    return sentence_embeddings

def get_sentences(picklefile):
    with open(picklefile, 'rb') as handle:
        sentences = pickle.load(handle)
    handle.close()
    return sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main options')
    parser.add_argument('-inpickle', type=str, help='Pickle file with '
                                                      'sentences',
                        required=True)
    parser.add_argument('-out', type=str, help='outputfolder to store '
                                                     'sentence embeddings',
                        required=True)
    parser.add_argument('-version', type=str, help='Model version to be used',
                        default='bert-base-nli-stsb-mean-tokens')

    options = parser.parse_args()
    # read sentences
    #sentences = get_sentences(options.sentpickle)
    # get sentence embeddings
    sentences = ["Hi how are you today ?", "Hi, what's up?"]

    sent_embeddings = get_semantic_emb(sentences, options.version)
    mydict = dict(zip(sentences, sent_embeddings))
    # store sentence embeddings
    if not os.path.exists(options.out):
        os.makedirs(options.out)

    with open(os.path.join(options.out, "sent_emb.pkl"), 'wb') as handle:
        pickle.dump(mydict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    for sentence, embedding in zip(sentences, sent_embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")
    import ipdb;ipdb.set_trace()

    tensor1 = torch.tensor(sent_embeddings[0]).unsqueeze(0)
    tensor2 = torch.tensor(sent_embeddings[1]).unsqueeze(0)
    out = cosine_similarity(tensor1, tensor2)
    print(out)
