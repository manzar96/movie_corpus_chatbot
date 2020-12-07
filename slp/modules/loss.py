import torch.nn as nn
import torch


class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self, pad_idx=0):
        super(SequenceCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, y_pred, targets):
        y_pred = y_pred.contiguous().view(-1, y_pred.size(-1))
        targets = targets.contiguous().view(-1)
        return self.criterion(y_pred, targets)


class Perplexity(nn.Module):
    def __init__(self, pad_idx=0):
        super(Perplexity, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, y_pred, targets):
        y_pred = y_pred.contiguous().view(-1, y_pred.size(-1))
        targets = targets.contiguous().view(-1)
        return torch.exp(self.criterion(y_pred, targets))

from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch.nn import CosineSimilarity
class Sequence_Sentence_Embeddings_Cosine_Similarity(nn.Module):
    def __init__(self,batch_size,pad_idx=0,idx2word={}):
        super(Sequence_Sentence_Embeddings_Cosine_Similarity, self).__init__()
        self.cos_sim = CosineSimilarity()
        self.pad_idx = pad_idx
        self.batch_size = batch_size
        self.trans_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        for param in self.trans_model.parameters():
            param.requires_grad = False
        self.idx2word=idx2word
        train_numparams = sum([p.numel() for p in self.trans_model.parameters() if
                               p.requires_grad])
        print("Transformer train parameteres:       ",train_numparams)

    def forward(self, y_pred, targets):
        sentences_ref = []
        sentences_pred = []
        targets = targets.reshape(self.batch_size,-1)
        voc_dim = y_pred.shape[1]
        y_pred = y_pred.reshape(self.batch_size,-1,voc_dim)
        tensors1_batched = []
        tensors2_batched = []
        for i in range(self.batch_size):
            prediction = y_pred[i]
            top_index = F.log_softmax(prediction, dim=1)
            _, topi = top_index.topk(1, dim=-1)
            target = targets[i]
            ziped = [ (pred,tgt) for pred,tgt in zip(topi, target) if  \
                    tgt.item()!=self.pad_idx]
            preds_idx = [pred.item() for pred,tgt in ziped]
            tgt_idx = [tgt.item() for pred,tgt in ziped]
            pred_words = [self.idx2word[pred] for pred in preds_idx]
            tgt_words = [self.idx2word[tgt] for tgt in tgt_idx]
            pred_sent = " ".join(pred_words)
            tgt_sent = " ".join(tgt_words)
            sentences = [pred_sent,tgt_sent]
            sent_embeddings = self.trans_model.encode(sentences,
                                                      show_progress_bar=False)
            tensors1_batched.append(torch.tensor(sent_embeddings[0]))
            tensors2_batched.append(torch.tensor(sent_embeddings[1]))

        tensors1_batched = torch.stack(tensors1_batched)
        tensors2_batched = torch.stack(tensors2_batched)
        out = self.cos_sim(tensors1_batched, tensors2_batched)
        return sum(out) / len(out)
