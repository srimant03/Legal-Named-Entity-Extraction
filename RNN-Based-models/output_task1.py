import numpy as np
import json
import torch
from sklearn.metrics import f1_score
import torch.nn as nn

filename = '/home/srimant_ubuntu/ml/protein-interaction/NER_train.json'

with open(filename, 'r') as file:
    data_train = json.load(file)

filename = '/home/srimant_ubuntu/ml/protein-interaction/NER_test.json'

with open(filename, 'r') as file:
    data_test = json.load(file)
    
filename = '/home/srimant_ubuntu/ml/protein-interaction/NER_val.json'

with open(filename, 'r') as file:
    data_val = json.load(file)

data = data_val
tokenized_texts = []
labels = []

for key, value in data.items():
    text = value['text']
    label_seq = value['labels']
    tokenized_text = text.split()   
    tokenized_texts.append(tokenized_text)
    labels.append(label_seq)
    
val_data = []

for i in range(len(labels)):
    pair = [tokenized_texts[i], labels[i]]
    val_data.append(pair)

data = data_test
tokenized_texts = []
labels = []


for key, value in data.items():
    text = value['text']
    label_seq = value['labels']
    tokenized_text = text.split()  
    tokenized_texts.append(tokenized_text)
    labels.append(label_seq)
    
test_data = []

for i in range(len(labels)):
    pair = [tokenized_texts[i], labels[i]]
    test_data.append(pair)

data = data_train
tokenized_texts = []
labels = []

for key, value in data.items():
    text = value['text']
    label_seq = value['labels']
    tokenized_text = text.split()
    tokenized_texts.append(tokenized_text)
    labels.append(label_seq)
    
train_data = []

for i in range(len(labels)):
    pair = [tokenized_texts[i], labels[i]]
    train_data.append(pair)

import gensim
from gensim.models import KeyedVectors

def load_embeddings_from_json(filename):
    with open(filename, 'r') as file:
        embeddings_dict = json.load(file)
    return embeddings_dict

acc_f_r = '0.775326821938392'
mac_f_r = '0.206682039266887'

acc_g_r = '0.827407963936889'
mac_g_r = '0.071787571031267'

embeddings_train = '/home/srimant_ubuntu/ml/protein-interaction/NER_train_W2V.json'
embeddings_dict = load_embeddings_from_json(embeddings_train)

embeddings_test = '/home/srimant_ubuntu/ml/protein-interaction/NER_test_W2V.json'
embeddings_dict_test = load_embeddings_from_json(embeddings_test)

embeddings_val = '/home/srimant_ubuntu/ml/protein-interaction/NER_val_W2V.json'
embeddings_dict_val = load_embeddings_from_json(embeddings_val)

def tokens_to_embeddings(tokens, embeddings_dict):
    embeddings = []
    for token in tokens:
        if token in embeddings_dict:
            embeddings.append(embeddings_dict[token])
        else:
            embeddings.append(np.zeros(300))
    return np.array(embeddings)

def labels_to_indices(labels, tag_to_ix):
    return [tag_to_ix[label] for label in labels]

tag_to_ix = {}
for tag_sen in labels:
    for tag in tag_sen:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hidden = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden.squeeze(0))
        return output
    
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hidden = self.gru(x)
        output = self.fc(hidden.squeeze(0))
        return output
    
for i in range(1,10):
    if i == 1:
        print("Vanilla RNN, W2V")
        model = VanillaRNN(300, 256, 27)
        model_path = 'RNN_NER_W2V.pt'
        embeddings_test = '/home/srimant_ubuntu/ml/protein-interaction/NER_test_W2V.json'
    elif i == 2:
        print("Vanilla RNN, FastText")
        model = VanillaRNN(300, 256, 27)
        model_path = 'RNN_NER_fasttext.pt'
        embeddings_test = '/home/srimant_ubuntu/ml/protein-interaction/NER_test_fasttext.json'
        print(f'Test Macro-F1: {mac_f_r}')
        print(f'Accuracy: {acc_f_r}', end='\n\n')
        continue
    elif i == 3:
        print("Vanilla RNN, GloVe")
        model = VanillaRNN(300, 256, 27)
        model_path = 'rnn_NER_glove.pt'
        embeddings_test = '/home/srimant_ubuntu/ml/protein-interaction/NER_test_glove.json'
        print(f'Test Macro-F1: {mac_g_r}')
        print(f'Accuracy: {acc_g_r}', end='\n\n')
        continue
    elif i == 4:
        print("LSTM, W2V")
        model = LSTMModel(300, 128, 27)
        model_path = 'LSTM_NER_W2V.pt'
        embeddings_test = '/home/srimant_ubuntu/ml/protein-interaction/NER_test_W2V.json'
    elif i == 5:
        print("LSTM, FastText")
        model = LSTMModel(300, 128, 27)
        model_path = 'LSTM_NER_fasttext.pt'
        embeddings_test = '/home/srimant_ubuntu/ml/protein-interaction/NER_test_fasttext.json'
    elif i == 6:
        print("LSTM, GloVe")
        model = LSTMModel(300, 128, 27)
        model_path = 'LSTM_NER_glove.pt'
        embeddings_test = '/home/srimant_ubuntu/ml/protein-interaction/NER_test_glove.json'
    elif i == 7:
        print("GRU, W2V")
        model = GRUModel(300, 256, 27)
        model_path = 'GRU_NER_W2V.pt'
        embeddings_test = '/home/srimant_ubuntu/ml/protein-interaction/NER_test_W2V.json'
    elif i == 8:
        print("GRU, FastText")
        model = GRUModel(300, 256, 27)
        model_path = 'GRU_NER_fasttext.pt'
        embeddings_test = '/home/srimant_ubuntu/ml/protein-interaction/NER_test_fasttext.json'
    elif i == 9:
        print("GRU, GloVe")
        model = GRUModel(300, 256, 27)
        model_path = 'GRU_NER_glove.pt'
        embeddings_test = '/home/srimant_ubuntu/ml/protein-interaction/NER_test_glove.json'

    device = torch.device('cpu')  # Load the model onto the CPU
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_predictions = []
    test_targets = []

    embeddings_dict_test = load_embeddings_from_json(embeddings_test)

    with torch.no_grad():
        for tokens, labels in test_data:
            token_embeddings = tokens_to_embeddings(tokens, embeddings_dict_test)
            label_indices = labels_to_indices(labels, tag_to_ix)
            for token_emb, label_idx in zip(token_embeddings, label_indices):
                token_emb = torch.tensor(token_emb).float().unsqueeze(0)  
                label_idx = torch.tensor(label_idx).long()  
                output = model(token_emb)
                test_predictions.append(output.argmax().item())
                test_targets.append(label_idx.item())

    test_f1 = f1_score(test_targets, test_predictions, average='macro')
    print(f'Test Macro-F1: {test_f1}')
    #print overall accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(test_targets, test_predictions)
    print(f'Accuracy: {accuracy}')
    print('\n')