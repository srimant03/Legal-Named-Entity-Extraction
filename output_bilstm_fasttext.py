import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

import gensim
from gensim.models import FastText
from gensim.models import KeyedVectors

#word2vec_model = KeyedVectors.load_word2vec_format('/home/srimant_ubuntu/ml/protein-interaction/wiki-news-300d-1M-subword.vec')
word2vec_output_file = 'fasttext_word2vec.txt'
fasttext_embed = KeyedVectors.load_word2vec_format(word2vec_output_file)

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, glove_embedding):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}
        
#         self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
#                             num_layers=1, bidirectional=True)
#         self.word_embeds = Word2VecEmbedding(word_to_vec_map, len(word_to_vec_map.index_to_key), word_to_vec_map.vector_size)
#         self.lstm = nn.LSTM(word_to_vec_map.vector_size, hidden_dim // 2,
#                             num_layers=1, bidirectional=True)
        #self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(word_to_vec_map.vectors))
        #self.word_embeds.weight.requires_grad = False  # freeze the embedding layer

        #self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(glove_embedding.vectors))
        #self.word_embeds.weight.requires_grad = False  # freeze the embedding layer

        self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(fasttext_embed.vectors))
        self.word_embeds.weight.requires_grad = False  # freeze the embedding layer

        self.lstm = nn.LSTM(fasttext_embed.vector_size, hidden_dim // 2, num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

#     def forward(self, sentence):  # dont confuse this with _forward_alg above.
#         # Get the emission scores from the BiLSTM
#         lstm_feats = self._get_lstm_features(sentence)

#         # Find the best path, given the features.
#         score, tag_seq = self._viterbi_decode(lstm_feats)
#         return score, tag_seq
    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_indices = self._viterbi_decode(lstm_feats)
        
        # Convert tag indices to actual tags
        tags = [self.ix_to_tag[idx] for idx in tag_indices]
        return score, tags

import json

filename = '/home/srimant_ubuntu/ml/nlp_A2/Datasets&Embeddings/ATE_train.json'

with open(filename, 'r') as file:
    data_train = json.load(file)

filename = '/home/srimant_ubuntu/ml/nlp_A2/Datasets&Embeddings/ATE_test.json'

with open(filename, 'r') as file:
    data_test = json.load(file)
    
filename = '/home/srimant_ubuntu/ml/nlp_A2/Datasets&Embeddings/ATE_val.json'

with open(filename, 'r') as file:
    data_val = json.load(file)

data = data_val
tokenized_texts = []
labels = []

# Iterate over the items in the parsed JSON object
for key, value in data.items():
    # Extract text and labels for each item
    text = value['text']
    label_seq = value['labels']
    
    # Tokenize the text (if needed)
    tokenized_text = text.split()  # Using simple split for illustration
    
#     Store the tokenized text
    tokenized_texts.append(tokenized_text)
#     tokenized_texts.append(text)
    # Store the labels
    labels.append(label_seq)
    
val_data = []

for i in range(len(labels)):
    pair = [tokenized_texts[i], labels[i]]
    val_data.append(pair)

data = data_test
tokenized_texts = []
labels = []

# Iterate over the items in the parsed JSON object
for key, value in data.items():
    # Extract text and labels for each item
    text = value['text']
    label_seq = value['labels']
    
    # Tokenize the text (if needed)
    tokenized_text = text.split()  # Using simple split for illustration
    
#     Store the tokenized text
    tokenized_texts.append(tokenized_text)
#     tokenized_texts.append(text)
    # Store the labels
    labels.append(label_seq)
    
test_data = []

for i in range(len(labels)):
    pair = [tokenized_texts[i], labels[i]]
    test_data.append(pair)

data = data_train
tokenized_texts = []
labels = []

# Iterate over the items in the parsed JSON object
for key, value in data.items():
    # Extract text and labels for each item
    text = value['text']
    label_seq = value['labels']
    
    # Tokenize the text (if needed)
    tokenized_text = text.split()  # Using simple split for illustration
    
#     Store the tokenized text
    tokenized_texts.append(tokenized_text)
#     tokenized_texts.append(text)
    # Store the labels
    labels.append(label_seq)
    
train_data = []

for i in range(len(labels)):
    pair = [tokenized_texts[i], labels[i]]
    train_data.append(pair)

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def calculate_f1_score(model, dataset):
    all_predictions = []
    all_targets = []
    for sentence, tags in dataset:
        sentence_in = []
        for word in sentence:
            if word in fasttext_embed.key_to_index:
                sentence_in.append(torch.tensor(fasttext_embed.key_to_index[word], dtype=torch.long))
            else:
                sentence_in.append(torch.tensor(0, dtype=torch.long))
        sentence_in = torch.stack(sentence_in)
        _, predicted_tages = model(sentence_in)
        all_predictions.extend(predicted_tages)
        all_targets.extend(tags)
    print("Accuracy: ", accuracy_score(all_targets, all_predictions))
    return f1_score(all_targets, all_predictions, average='macro')

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {}
for tag_sen in labels:
    for tag in tag_sen:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
            
tag_to_ix[START_TAG] = 3
tag_to_ix[STOP_TAG] = 4
print(tag_to_ix)

EMBEDDING_DIM = 50
HIDDEN_DIM = 40

#train_set = train_data
#val_set = val_data
test_set = test_data
train_losses = []
val_losses = []
train_f1_scores = []
val_f1_scores = []

# Precompute embeddings for all words in the vocabulary
embedding_cache = {}
for word, idx in fasttext_embed.key_to_index.items():
    embedding_cache[word] = torch.tensor(idx, dtype=torch.long)

#model = BiLSTM_CRF(len(fasttext_embed.key_to_index), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, fasttext_embed)
#optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
"""
for epoch in range(10):
    epoch_train_loss = []
    for sentence, tags in train_set:
        model.zero_grad()
        sentence_in = []
        targets = []
        for word in sentence:
            if word in embedding_cache:
                sentence_in.append(embedding_cache[word])
            else:
                sentence_in.append(torch.tensor(0, dtype=torch.long))
        sentence_in = torch.stack(sentence_in)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        loss = model.neg_log_likelihood(sentence_in, targets)
        epoch_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
    train_loss = sum(epoch_train_loss) / len(train_set)
    train_losses.append(train_loss)

    val_loss = 0
    for sentence, tags in val_set:
        sentence_in = []
        for word in sentence:
            if word in embedding_cache:
                sentence_in.append(embedding_cache[word])
            else:
                sentence_in.append(torch.tensor(0, dtype=torch.long))
        sentence_in = torch.stack(sentence_in)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        loss = model.neg_log_likelihood(sentence_in, targets)
        val_loss += loss.item()

    val_loss /= len(val_set)
    val_losses.append(val_loss)

    train_f1 = calculate_f1_score(model, train_set)
    val_f1 = calculate_f1_score(model, val_set)
    train_f1_scores.append(train_f1)
    val_f1_scores.append(val_f1)

    print(f'Epoch {epoch+1}/{10}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')"""

# Plot Loss and F1 scores
'''epochs = range(1, 11)  # Assuming 10 epochs
plt.figure(figsize=(12, 6))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.legend()

# F1 Score Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_f1_scores, label='Train F1 Score')
plt.plot(epochs, val_f1_scores, label='Val F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score Plot')
plt.legend()

plt.tight_layout()
plt.savefig('bilstm_ate_fasttext_again.pdf')
plt.show()'''

#torch.save(model.state_dict(), 'bilstm_ate_fasttext_again.pt')

#load the model for testing
model1 = BiLSTM_CRF(len(fasttext_embed.key_to_index), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, fasttext_embed)
device = torch.device('cpu')
model_path = 'bilstm_ate_fasttext.pt'
model1.load_state_dict(torch.load(model_path, map_location=device))

test_f1 = calculate_f1_score(model1, test_data)
print(f'Test F1 Score: {test_f1:.4f}')