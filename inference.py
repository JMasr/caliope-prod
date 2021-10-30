from __future__ import unicode_literals, print_function, division

import time
import pickle
from io import open

import torch
import torch.nn as nn
import torch.nn.functional as function


def save_obj(obj, name=None):
    """Save a Python object into a `pickle` file on disk.

        Parameters
        ----------
        obj  : the Python object to be saved.
            Input Python object
        name : path where the Python object will be saved, without the .pkl extension.
            Input string
        Returns
        -------
        Nothing
    """
    if name is None:
        with open('obj/' + "obj_saved_" + str(time.time()) + '.pkl', 'wb') as f:
            pickle.dump(obj, f, 0)
    else:
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, 0)


def load_obj(name):
    """Load a `pickle` object from disk.

        Parameters
        ----------
        name : path to the object without the .pkl extension.
            Input string
        Returns
        -------
        The Python object store in disk.
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def evaluate_init_0(input_text):
    """
    Parameters
    ----------
    input_text: lista donde cada elemento debe ser una cadena de strings sin signos de puntuación o mayúsculas.

    Returns
    -------
    Una lista de igual tamaño que **input_text** donde cada elemento corresponde a la predicción del modelo.

    Examples::
        >>> lista = ['en galicia anunciouse reformas no consello de vigo', 'galicia', 'consello de vigo']
        >>> salida = evaluate_init_0(lista)
    """

    class Lang:
        def __init__(self, name):
            self.name = name
            self.word2index = {}
            self.word2count = {}
            self.index2word = {0: "SOS", 1: "EOS"}
            self.n_words = 2  # Count SOS and EOS
            self.sentence_leng = {}  # Length in Words : Number of utterances

        @staticmethod
        def printDict(path, dicc, sort=True):

            file_w = open(path, "w+")
            if sort:
                dicc = sorted(dicc.items(), key=lambda x: x[1], reverse=True)

            for allKeys in dicc:
                printed_line = str(allKeys[0]) + "\t" + str(allKeys[1]) + "\n"
                file_w.writelines(printed_line)
            file_w.close()

        def printAllDicts(self, path, sort=True):
            self.printDict(path + "/word2index.txt", self.word2index, sort=sort)
            self.printDict(path + "/word2count.txt", self.word2count, sort=sort)
            self.printDict(path + "/index2word.txt", self.index2word, sort=sort)
            self.printDict(path + "/sentence_leng.txt", self.sentence_leng, sort=sort)

        def addSentence(self, sentence):
            # Counting length of every sentence
            sentence_len = len(sentence.split(' '))
            if sentence_len not in self.sentence_leng:
                self.sentence_leng[sentence_len] = 1
            else:
                self.sentence_leng[sentence_len] += 1

            for word in sentence.split(' '):
                self.addWord(word)

        def addWord(self, word):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1

        def save(self, path):
            lang_obj = [self.name, self.word2index, self.word2count, self.index2word, self.n_words, self.sentence_leng]
            with open(path + '.pkl', 'wb') as f:
                pickle.dump(lang_obj, f, 0)

        def load(self, path):
            with open(path + '.pkl', 'rb') as f:
                lang_obj = pickle.load(f)

            self.name = lang_obj[0]
            self.word2index = lang_obj[1]
            self.word2count = lang_obj[2]
            self.index2word = lang_obj[3]
            self.n_words = lang_obj[4]
            self.sentence_leng = lang_obj[5]
            return self

    def get_network_predictions(in_text, in_n_words, in_word2index, out_index2word, out_n_words):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sos_token = 0
        eos_token = 1

        class EncoderRNN(nn.Module):
            def __init__(self, input_size, p_hidden_size=128, p_num_layers=4):
                super(EncoderRNN, self).__init__()
                self.num_layers = p_num_layers
                self.hidden_size = p_hidden_size

                self.embedding = nn.Embedding(input_size, p_hidden_size)
                self.gru = nn.GRU(p_hidden_size, p_hidden_size, p_num_layers)

            def forward(self, p_input, hidden):
                embedded = self.embedding(p_input).view(1, 1, -1)
                output = embedded
                output, hidden = self.gru(output, hidden)
                return output, hidden

            def init_hidden(self):
                return torch.zeros(1, 1, self.hidden_size, device=device)

        class DecoderRNN(nn.Module):
            def __init__(self, p_hidden_size, output_size):
                super(DecoderRNN, self).__init__()
                self.hidden_size = p_hidden_size

                self.embedding = nn.Embedding(output_size, p_hidden_size)
                self.gru = nn.GRU(p_hidden_size, p_hidden_size)
                self.out = nn.Linear(p_hidden_size, output_size)
                self.softmax = nn.LogSoftmax(dim=1)

            def forward(self, p_input, hidden):
                output = self.embedding(p_input).view(1, 1, -1)
                output = function.relu(output)
                output, hidden = self.gru(output, hidden)
                output = self.softmax(self.out(output[0]))
                return output, hidden

            def initHidden(self):
                return torch.zeros(1, 1, self.hidden_size, device=device)

        class AttnDecoderRNN(nn.Module):
            def __init__(self, p_hidden_size, output_size, p_max_length=40, p_num_layers=1, dropout_p=0.1):
                super(AttnDecoderRNN, self).__init__()
                self.num_layers = p_num_layers
                self.hidden_size = p_hidden_size
                self.output_size = output_size
                self.dropout_p = dropout_p
                self.max_length = p_max_length

                self.embedding = nn.Embedding(self.output_size, self.hidden_size)
                self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
                self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
                self.dropout = nn.Dropout(self.dropout_p)
                self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=p_num_layers)
                self.out = nn.Linear(self.hidden_size, self.output_size)

            def forward(self, p_input, hidden, encoder_outputs):
                embedded = self.embedding(p_input).view(1, 1, -1)
                embedded = self.dropout(embedded)

                attn_weights = function.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
                attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

                output = torch.cat((embedded[0], attn_applied[0]), 1)
                output = self.attn_combine(output).unsqueeze(0)

                output = function.relu(output)
                output, hidden = self.gru(output, hidden)

                output = function.log_softmax(self.out(output[0]), dim=1)
                return output, hidden, attn_weights

            def init_hidden(self):
                return torch.zeros(1, 1, self.hidden_size, device=device)

        def indexesFromSentence(word2index, sentence):
            return [word2index[word] for word in sentence.split(' ') if word in word2index.keys()]

        def tensorFromSentence(word2index, sentence):
            indexes = indexesFromSentence(word2index, sentence)
            indexes.append(eos_token)
            return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

        def evaluate(encoder, decoder, sentence, p_max_length=40):
            with torch.no_grad():
                input_tensor = tensorFromSentence(in_word2index, sentence)
                input_length = input_tensor.size()[0]
                encoder_hidden = encoder.init_hidden()

                encoder_outputs = torch.zeros(p_max_length, encoder.hidden_size, device=device)

                for ei in range(input_length):
                    encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
                    encoder_outputs[ei] += encoder_output[0, 0]

                decoder_input = torch.tensor([[sos_token]], device=device)  # SOS

                decoder_hidden = encoder_hidden

                decoded_conf = []
                decoded_words = []
                decoder_attentions = torch.zeros(p_max_length, p_max_length)

                for di in range(input_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                                encoder_outputs)
                    logits = torch.nn.functional.softmax(decoder_output, dim=1)
                    decoded_conf.append(torch.max(logits).item())

                    decoder_attentions[di] = decoder_attention.data
                    topv, topi = decoder_output.data.topk(1)

                    if topi.item() == eos_token:
                        break
                    else:
                        decoded_words.append(out_index2word[topi.item()])

                    decoder_input = topi.squeeze().detach()

                return decoded_words, decoded_conf[:-1], decoder_attentions[:di + 1]

        # +=========================+
        # |    Evaluation Phase     |

        encoder1_loaded = EncoderRNN(in_n_words, 128, 1).to(device)
        attn_decoder1_loaded = AttnDecoderRNN(128, out_n_words, 40, 1, dropout_p=0.1).to(device)

        # +=======Load-Models=======+
        encoder1_loaded.load_state_dict(
            torch.load("model/encoder.pt", map_location=device))
        attn_decoder1_loaded.load_state_dict(
            torch.load("model/decoder.pt", map_location=device))

        # +====Doing-Predictions====+
        network_prediction = []
        network_confidence = []
        if isinstance(in_text, str):
            network_prediction, confidence, attentions_load = evaluate(encoder1_loaded, attn_decoder1_loaded, in_text)
        else:
            for texto in in_text:
                output_words_load, confidence, attentions_load = evaluate(encoder1_loaded, attn_decoder1_loaded, texto)
                network_prediction.append(output_words_load)
                network_confidence.append(confidence)

        return network_prediction, network_confidence

    def translate_prediction(pairs, assume_utt4line=False):
        if len(pairs[0]) == 1:
            print('a')
        elif len(pairs[0]) == 0:
            print('b')
        results = []
        for true, prediction in zip(pairs[0], pairs[1]):
            if prediction == "lower":
                prediction = true.lower()
            elif prediction == "lower_dot":
                prediction = true.lower() + "."
            elif prediction == "lower_comma":
                prediction = true.lower() + ","
            elif prediction == "all-upper":
                prediction = true.upper()
            elif prediction == "upper":
                prediction = true[0].upper() + true[1:].lower()
            elif prediction == "upper_dot":
                prediction = true[0].upper() + true[1:].lower() + "."
            elif prediction == "upper_comma":
                prediction = true[0].upper() + true[1:].lower() + ","
            results.append(prediction)

        if assume_utt4line:
            if results[-1][-1] != ".":
                results[-1] = results[-1] + "."
            if results[0][0] != results[0][0].upper():
                results[0] = results[0][0].upper() + results[0][1:]

        return " ".join(results)

    input_lang = Lang("examples")
    output_lang = Lang("labels")
    input_lang.load("model/input_lang")
    output_lang.load("model/output_lang")

    predictions, predictions_conf = get_network_predictions(input_text, input_lang.n_words, input_lang.word2index,
                                                            output_lang.index2word, output_lang.n_words)
    if isinstance(input_text, list):
        input_text = [i.split(" ") for i in input_text]
        final_results = [translate_prediction([t, p]) for t, p in zip(input_text, predictions)]
    elif isinstance(input_text, str):
        input_text = input_text.split(" ")
        final_results = translate_prediction([input_text, predictions])
    return [final_results, predictions_conf]
