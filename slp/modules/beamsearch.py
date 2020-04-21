import torch


class BeamNode:
    def __init__(self, decoder_hidden, last_idx, sentence_idxes,
                 sentence_scores, device='cpu'):

        if (len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes = sentence_idxes
        self.sentence_scores = sentence_scores
        self.device = device

    def getScore(self, mode='avg', gamma=0.2):
        if len(self.sentence_scores) == 0:
            print("sentence of length 0")
            if self.device=='cuda':
                return torch.FloatTensor(-999).to(self.device)

        if mode == 'avg':
            res = sum(self.sentence_scores) / len(self.sentence_scores)
        else:
            res = sum(self.sentence_scores) + gamma * len(self.sentence_scores)
        return res


class BeamGraph:

    def __init__(self, beam_size, eos_token,sos_token, device='cuda'):
        self.terminal_nodes = []
        self.prev_top_nodes = []
        self.next_top_nodes = []
        self.beam_size = beam_size
        self.eos_token = eos_token
        self.sos_token = sos_token
        self.device = device

    def create_node(self, decoderhidden, last_idx=None, sent_indexes=None,
                    sent_scores=None):
        if last_idx is None:
            return BeamNode(decoderhidden, self.sos_token, [], [], self.device)
        else:
            return BeamNode(decoderhidden, last_idx, sent_indexes,
                            sent_scores, self.device)

    def addTopk(self, topi, topv, decoder_hidden, node):
        terminates, sentences = [], []
        topi = topi.squeeze(0)  # we have only one batch so squeeze first dim
        topv = topv.squeeze(0)

        for i in range(self.beam_size):
            if topi[0][i] == self.eos_token:
                terminates.append(
                    ([int(idx) for idx in node.sentence_idxes] + [
                        self.eos_token],
                     node.getScore()))
                continue
            idxes = node.sentence_idxes[:]  # pass by value
            scores = node.sentence_scores[:]  # pass by value
            idxes.append(topi[0][i])
            scores.append(topv[0][i])
            sentences.append(BeamNode(decoder_hidden, topi[0][i], idxes,
                                      scores))

        self.terminal_nodes.extend(terminates)
        self.next_top_nodes.extend(sentences)
        self.next_top_nodes.sort(key=lambda s: s.getScore(), reverse=True)
        self.prev_top_nodes = self.next_top_nodes[:self.beam_size]
        self.next_top_nodes = []

    def toWordScore(self, node):
        words = []
        for i in range(len(node.sentence_idxes)):
            if node.sentence_idxes[i] == self.eos_token:
                words.append(self.eos_token)
            else:
                words.append(int(node.sentence_idxes[i]))
        if node.sentence_idxes[-1] != self.eos_token:
            words.append(self.eos_token)
        return words, node.getScore()
