import numpy as np
import torch
import collections
import os
import random
import torch.nn as nn
import unicodedata
import re
import torch.nn.functional as f

from ignite.metrics import Loss
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import Compose

from slp.util.embeddings import EmbeddingsLoader
from slp.data.moviecorpus_tut import MovieCorpusDatasetv2
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.data.collators import Seq2SeqCollator
from slp.trainer.trainer import Seq2SeqTrainer
from slp.config.moviecorpus import SPECIAL_TOKENS
from slp.modules.loss_tut import SequenceCrossEntropyLoss
from slp.modules.seq2seq_moviecorpus_tutorial import EncoderLSTM, DecoderLSTMv2,  \
    EncoderDecoder_SeqCrossEntropy


def dataloaders_from_indices(dataset, train_indices, val_indices, batch_train,
                             batch_val):
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_train,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=COLLATE_FN)
    val_loader = DataLoader(
        dataset,
        batch_size=batch_val,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=COLLATE_FN)
    return train_loader, val_loader


def train_test_split(dataset, batch_train, batch_val,
                     test_size=0, shuffle=True, seed=None):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_size * dataset_size))
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices = indices[test_split:]
    val_indices = indices[:test_split]
    return dataloaders_from_indices(dataset, train_indices, val_indices,
                                    batch_train, batch_val)


def create_emb_file(new_emb_file, old_emb_file, words2counts,
                    most_freq=None):
    freq_words_file = './cache/freq_words.txt'
    sorted_voc = sorted(words2counts.items(), key=lambda kv: kv[1])

    with open(freq_words_file, "w") as file:
        if most_freq is not None:
            for item in sorted_voc[-most_freq:]:
                file.write(item[0]+'\n')
        else:
            for item in sorted_voc:
                file.write(item[0]+'\n')
    file.close()

    os.system("awk 'FNR==NR{a[$1];next} ($1 in a)' " + freq_words_file +
              " " + old_emb_file + ">" + new_emb_file)


def train(input_variable, lengths, target_variable,model,
          model_optimizer, clip):

    # Zero gradients
    model_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    all_outputs = model(input_variable,lengths,target_variable)
    loss = criterion(all_outputs, target_variable)

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)


    # Adjust model weights
    model_optimizer.step()

    return loss.item()


def training_mini_batches(train_loader,model, model_optimizer, n_iteration, clip):
    train_list = list(train_loader)
    training_batches = [random.choice(train_list)for _ in range(
        n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    # if loadFilename:
    #     start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, input_lengths, target_variable,lengths_target = \
            training_batch

        # Run a training iteration with batch
        loss = train(input_variable, input_lengths, target_variable, model,
                     model_optimizer, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # # Save checkpoint
        # if (iteration % save_every == 0):
        #     directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
        #     if not os.path.exists(directory):
        #         os.makedirs(directory)
        #     torch.save({
        #         'iteration': iteration,
        #         'en': encoder.state_dict(),
        #         'de': decoder.state_dict(),
        #         'model_opt': model_optimizer.state_dict(),
        #         'loss': loss,
        #         'embedding': embedding.state_dict()
        #     }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))



def indexesFromSentence(word2index, sentence,eos_token):
    return [word2index[word] for word in sentence.split(' ')] + [eos_token]

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def evaluate(searcher, word2idx,idx2word, sentence, max_length):
    # Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(word2idx, sentence, word2idx[SPECIAL_TOKENS.EOS.value])]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    # input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = torch.LongTensor(indexes_batch)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [idx2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(searcher, word2idx,idx2word, max_length):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(searcher, word2idx, idx2word,
                                    input_sentence, max_length)
            # Format and print response sentence
            print(output_words)
            output_words[:] = [x for x in output_words if not (x == '[EOS]' or
                                                               x == '[PAD]')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


class MyGreedySearchDecoder(nn.Module):
    def __init__(self, model,device):
        super(MyGreedySearchDecoder, self).__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.SOS_token = model.bos_indx
        self.device=device

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.num_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long)\
                        * self.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # Obtain most likely word token and its softmax score
            current_output = torch.squeeze(decoder_output, dim=1)
            decoder_output = f.softmax(current_output, dim=1)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COLLATE_FN = Seq2SeqCollator(device='cpu')
BATCH_TRAIN_SIZE = 64
BATCH_VAL_SIZE = 32


dataset = MovieCorpusDatasetv2('./data/', transforms=None)
# Preprocess dataset
MIN_COUNT= 3
MAX_LENGTH = 10
dataset.normalize_data()
dataset.threshold_data(MAX_LENGTH, tokenizer=SpacyTokenizer())
dataset.trim_words(MIN_COUNT, tokenizer=SpacyTokenizer())

# Load embeddings
emb_file = './cache/glove.6B.300d.txt'
new_emb_file = './cache/new_embs.txt'

create_emb_file(new_emb_file, emb_file, dataset.word2count)
loader = EmbeddingsLoader(new_emb_file, 300, extra_tokens=SPECIAL_TOKENS)
word2idx, idx2word, embeddings = loader.load()


# receive sos,eos and pad tokens
pad_index = word2idx[SPECIAL_TOKENS.PAD.value]
bos_index = word2idx[SPECIAL_TOKENS.BOS.value]
eos_index = word2idx[SPECIAL_TOKENS.EOS.value]

# apply transforms to dataset
tokenizer = SpacyTokenizer(append_eos=True, specials=SPECIAL_TOKENS)
to_token_ids = ToTokenIds(word2idx)
to_tensor = ToTensor(device='cpu')
transforms = Compose([tokenizer, to_token_ids, to_tensor])
dataset.apply_transforms(transforms)

# load batches for training
train_loader, _ = train_test_split(dataset, BATCH_TRAIN_SIZE,
                                            BATCH_VAL_SIZE)

# Making seq2seq model
embedding = nn.Embedding(embeddings.shape[0],embeddings.shape[1])
hidden_size=512
encoder_n_layers=2
dropout=0.2
device = DEVICE
decoder_n_layers=2
out_size = embedding.num_embeddings
teacher_forcing_ratio=0.9

encoder = EncoderLSTM(embedding, weights_matrix=None,
                      hidden_size=hidden_size,
                      num_layers=encoder_n_layers, dropout=dropout,
                      bidirectional=True, rnn_type='gru', batch_first=True,
                      emb_train=True, device=device)
decoder = DecoderLSTMv2(embedding, weights_matrix=None,
                        hidden_size=hidden_size,
                        output_size=out_size, max_target_len=10,
                        num_layers=decoder_n_layers, dropout=dropout,
                        bidirectional=False, batch_first=True,
                        rnn_type='gru',
                        device=device)

model = EncoderDecoder_SeqCrossEntropy(encoder, decoder, bos_index,
                                       teacher_forcing_ratio, device)
model = model.to(device)

criterion = SequenceCrossEntropyLoss()

clip = 50.0
learning_rate = 0.001
n_iteration = 4000
print_every = 1

# Initialize optimizers
print('Building optimizer ...')
model_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Run training iterations
print("Starting Training!")
model.train()
training_mini_batches(train_loader,model, model_optimizer, n_iteration,clip)


# Set dropout layers to eval mode
model.eval()

searcher = MyGreedySearchDecoder(model,device)
evaluateInput(searcher,word2idx,idx2word,MAX_LENGTH)
print("end chatbot")