import time
import pickle
from io import open
from multiprocessing import Pool, cpu_count

import numpy as np
import regex as re
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import *

# probability of applying substitution operation on tokens selected for augmentation
alpha_sub = 0.40
# probability of applying delete operation on tokens selected for augmentation
alpha_del = 0.40

tokenizer = None
# substitution strategy: 'unk' -> replace with unknown tokens, 'rand' -> replace with random tokens from vocabulary
sub_style = 'unk'

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
        with open(name + '.pkl', 'wb') as f:
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


def augment_none(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    """
    apply no augmentation
    """
    x_aug.append(x[i])
    y_aug.append(y[i])
    y_mask_aug.append(y_mask[i])


def augment_substitute(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    """
    replace a token with a random token or the unknown token
    """
    if sub_style == 'rand':
        x_aug.append(np.random.randint(tokenizer.vocab_size))
    else:
        x_aug.append(TOKEN_IDX[token_style]['UNK'])
    y_aug.append(y[i])
    y_mask_aug.append(y_mask[i])


def augment_insert(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    """
    insert the unknown token before this token
    """
    x_aug.append(TOKEN_IDX[token_style]['UNK'])
    y_aug.append(0)
    y_mask_aug.append(1)
    x_aug.append(x[i])
    y_aug.append(y[i])
    y_mask_aug.append(y_mask[i])


def augment_delete(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    """
    remove this token i..e, not add in augmented tokens
    """
    return


def augment_all(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    """
    apply substitution with alpha_sub probability, deletion with alpha_sub probability and insertion with
    1-(alpha_sub+alpha_sub) probability
    """
    r = np.random.rand()
    if r < alpha_sub:
        augment_substitute(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)
    elif r < alpha_sub + alpha_del:
        augment_delete(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)
    else:
        augment_insert(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)


# supported augmentation techniques
AUGMENTATIONS = {
    'none': augment_none,
    'substitute': augment_substitute,
    'insert': augment_insert,
    'delete': augment_delete,
    'all': augment_all
}

# special tokens indices in different models available in transformers
TOKEN_IDX = {
    'bert': {
        'START_SEQ': 101,
        'PAD': 0,
        'END_SEQ': 102,
        'UNK': 100
    },
    'xlm': {
        'START_SEQ': 0,
        'PAD': 2,
        'END_SEQ': 1,
        'UNK': 3
    },
    'roberta': {
        'START_SEQ': 0,
        'PAD': 1,
        'END_SEQ': 2,
        'UNK': 3
    },
    'albert': {
        'START_SEQ': 2,
        'PAD': 0,
        'END_SEQ': 3,
        'UNK': 1
    },
}

# 'O' -> No punctuation
punctuation_dict = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3, 'ALL_CAPITAL': 4, 'FRITS_CAPITAL': 5,
                    'ALL_CAPITAL+COMMA': 6, 'ALL_CAPITAL+PERIOD': 7, 'ALL_CAPITAL+QUESTION': 8,
                    'FRITS_CAPITAL+COMMA': 9, 'FRITS_CAPITAL+PERIOD': 10, 'FRITS_CAPITAL+QUESTION': 11}

transformation_dict = {0: lambda x: x.lower(), 1: (lambda x: x + ','), 2: (lambda x: x + '.'),
                       3: (lambda x: x + '?'),
                       4: lambda x: x.upper(), 5: (lambda x: x[0].upper() + x[1:]), 6: (lambda x: x.upper() + ','),
                       7: (lambda x: x.upper() + '.'), 8: (lambda x: x.upper() + '?'),
                       9: (lambda x: x[0].upper() + x[1:] + ','), 10: (lambda x: x[0].upper() + x[1:] + '.'),
                       11: (lambda x: x[0].upper() + x[1:] + '?')}

# pretrained model name: (model class, model tokenizer, output dimension, token style)
MODELS = {
    'bert-base-uncased': (BertModel, BertTokenizer, 768, 'bert'),
    'bert-large-uncased': (BertModel, BertTokenizer, 1024, 'bert'),
    'bert-base-multilingual-cased': (BertModel, BertTokenizer, 768, 'bert'),
    'bert-base-multilingual-uncased': (BertModel, BertTokenizer, 768, 'bert'),
    'xlm-mlm-en-2048': (XLMModel, XLMTokenizer, 2048, 'xlm'),
    'xlm-mlm-100-1280': (XLMModel, XLMTokenizer, 1280, 'xlm'),
    'roberta-base': (RobertaModel, RobertaTokenizer, 768, 'roberta'),
    'roberta-large': (RobertaModel, RobertaTokenizer, 1024, 'roberta'),
    'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer, 768, 'bert'),
    'distilbert-base-multilingual-cased': (DistilBertModel, DistilBertTokenizer, 768, 'bert'),
    'xlm-roberta-base': (XLMRobertaModel, XLMRobertaTokenizer, 768, 'roberta'),
    'xlm-roberta-large': (XLMRobertaModel, XLMRobertaTokenizer, 1024, 'roberta'),
    'albert-base-v1': (AlbertModel, AlbertTokenizer, 768, 'albert'),
    'albert-base-v2': (AlbertModel, AlbertTokenizer, 768, 'albert'),
    'albert-large-v2': (AlbertModel, AlbertTokenizer, 1024, 'albert'),
    'bertinho-gl-base-cased': (BertModel.from_pretrained('model/bertinho/'),
                               AutoTokenizer.from_pretrained('model/bertinho/'), 768, 'bert')
}

#
KNOW_WORDS = load_obj('model/dict')


def seq_transformation(raw_data):
    data_output = ''
    if len(raw_data) > 2:
        for word in raw_data.split(" "):
            label = ''
            if word == word.lower():
                if word.isalnum():
                    label = "\t" + "O"
                elif word[-1] == ",":
                    label = "\t" + "COMMA"
                elif word[-1] == ".":
                    label = "\t" + "PERIOD"
                elif word[-1] == "?":
                    label = "\t" + "QUESTION"
            elif word == word.upper():
                if word.isalnum():
                    label = "\t" + "ALL_CAPITAL"
                elif word[-1] == ",":
                    label = "\t" + "ALL_CAPITAL+COMMA"
                elif word[-1] == ".":
                    label = "\t" + "ALL_CAPITAL+PERIOD"
                elif word[-1] == "?":
                    label = "\t" + "ALL_CAPITAL+QUESTION"
            elif word[0] == word[0].upper():
                if word.isalnum():
                    label = "\t" + 'FRITS_CAPITAL'
                elif word[-1] == ",":
                    label = "\t" + "FRITS_CAPITAL+COMMA"
                elif word[-1] == ".":
                    label = "\t" + "FRITS_CAPITAL+PERIOD"
                elif word[-1] == "?":
                    label = "\t" + "FRITS_CAPITAL+QUESTION"
            if label:
                word = re.sub(r"[.,]", "", word).lower()
                data_output += word + label + '\n'
    return data_output


def transform_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data_raw = f.read()
    data_raw = re.sub(r" \.", ".", data_raw)
    data_raw = re.sub(r" ,", ",", data_raw)
    data_raw = data_raw.split("\n")
    pool = Pool(processes=cpu_count())
    data_raw = pool.map(seq_transformation, data_raw)
    pool.close()
    pool.join()

    with open(file_path + "_GL.txt", 'w', encoding="utf-8") as f:
        f.write("".join(data_raw))
    return data_raw


def parse_data(lines, tokenizer_data, sequence_len, token_style):
    """

    :param lines: list that contains tokens and punctuations separated by tab in lines
    :param tokenizer_data: tokenizer that will be used to further tokenize word for BERT like models
    :param sequence_len: maximum length of each sequence
    :param token_style: For getting index of special tokens in config.TOKEN_IDX
    :return: list of [tokens_index, punctuation_index, attention_masks, punctuation_mask], each having sequence_len
    punctuation_mask is used to ignore special indices like padding and intermediate sub-word token during evaluation
    """
    data_items = []

    # loop until end of the entire text
    idx = 0
    while idx < len(lines):
        x = [TOKEN_IDX[token_style]['START_SEQ']]
        y = [0]
        y_mask = [1]  # which positions we need to consider while evaluating i.e., ignore pad or sub tokens

        # loop until we have required sequence length
        # -1 because we will have a special end of sequence token at the end
        while len(x) < sequence_len - 1 and idx < len(lines):
            word, punc = lines[idx], 'O'
            tokens = tokenizer_data.tokenize(word)
            # if taking these tokens exceeds sequence length we finish current sequence with padding
            # then start next sequence from this token
            if len(tokens) + len(x) >= sequence_len:
                break
            else:
                for i in range(len(tokens) - 1):
                    x.append(tokenizer_data.convert_tokens_to_ids(tokens[i]))
                    y.append(0)
                    y_mask.append(0)
                if len(tokens) > 0:
                    x.append(tokenizer_data.convert_tokens_to_ids(tokens[-1]))
                else:
                    x.append(TOKEN_IDX[token_style]['UNK'])
                y.append(punctuation_dict[punc])
                y_mask.append(1)
                idx += 1
        x.append(TOKEN_IDX[token_style]['END_SEQ'])
        y.append(0)
        y_mask.append(1)
        if len(x) < sequence_len:
            x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
            y = y + [0 for _ in range(sequence_len - len(y))]
            y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
        attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]
        data_items.append([x, y, attn_mask, y_mask])
    return data_items


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_txt, tokenizer_c, sequence_len, token_style, is_train=False, augment_rate=0.1,
                 augment_type='substitute'):
        """

        :param input_txt: list containing tokens and punctuations separated by tab in lines
        :param tokenizer_c: tokenizer that will be used to further tokenize word for BERT like models
        :param sequence_len: length of each sequence
        :param token_style: For getting index of special tokens in config.TOKEN_IDX
        :param augment_rate: token augmentation rate when preparing data
        :param is_train: if false do not apply augmentation
        """
        self.data = parse_data(input_txt, tokenizer_c, sequence_len, token_style)
        self.sequence_len = sequence_len
        self.augment_rate = augment_rate
        self.token_style = token_style
        self.is_train = is_train
        self.augment_type = augment_type

    def __len__(self):
        return len(self.data)

    def _augment(self, x, y, y_mask):
        x = []
        y_aug = []
        y_mask_aug = []
        for i in range(len(x)):
            r = np.random.rand()
            if r < self.augment_rate:
                AUGMENTATIONS[self.augment_type](x, y, y_mask, x, y_aug, y_mask_aug, i, self.token_style)
            else:
                x.append(x[i])
                y_aug.append(y[i])
                y_mask_aug.append(y_mask[i])

        if len(x) > self.sequence_len:
            # len increased due to insert
            x = x[0:self.sequence_len]
            y_aug = y_aug[0:self.sequence_len]
            y_mask_aug = y_mask_aug[0:self.sequence_len]
        elif len(x) < self.sequence_len:
            # len decreased due to delete
            x = x + [TOKEN_IDX[self.token_style]['PAD'] for _ in range(self.sequence_len - len(x))]
            y_aug = y_aug + [0 for _ in range(self.sequence_len - len(y_aug))]
            y_mask_aug = y_mask_aug + [0 for _ in range(self.sequence_len - len(y_mask_aug))]

        attn_mask = [1 if token != TOKEN_IDX[self.token_style]['PAD'] else 0 for token in x]
        return x, y_aug, attn_mask, y_mask_aug

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        attn_mask = self.data[index][2]
        y_mask = self.data[index][3]

        if self.is_train and self.augment_rate > 0:
            x, y, attn_mask, y_mask = self._augment(x, y, y_mask)

        x = torch.tensor(x)
        y = torch.tensor(y)
        attn_mask = torch.tensor(attn_mask)
        y_mask = torch.tensor(y_mask)

        return x, y, attn_mask, y_mask


class DeepPunctuation(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1):
        super(DeepPunctuation, self).__init__()
        self.output_dim = len(punctuation_dict)
        if 'bertinho' in pretrained_model:
            self.bert_layer = MODELS[pretrained_model][0].from_pretrained('model/bertinho/')
        else:
            self.bert_layer = MODELS[pretrained_model][0].from_pretrained(pretrained_model)
        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        bert_dim = MODELS[pretrained_model][2]
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim
        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=len(punctuation_dict))

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)
        x = self.bert_layer(x, attention_mask=attn_masks)[0]
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)
        return x


class DeepPunctuationCRF(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1):
        super(DeepPunctuationCRF, self).__init__()
        self.bert_lstm = DeepPunctuation(pretrained_model, freeze_bert, lstm_dim)
        self.crf = CRF(len(punctuation_dict), batch_first=True)

    def log_likelihood(self, x, attn_masks, y):
        x = self.bert_lstm(x, attn_masks)
        attn_masks = attn_masks.byte()
        return -self.crf(x, y, mask=attn_masks, reduction='token_mean')

    def forward(self, x, attn_masks, y):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        x = self.bert_lstm(x, attn_masks)
        attn_masks = attn_masks.byte()
        dec_out = self.crf.decode(x, mask=attn_masks)
        y_pred = torch.zeros(y.shape).long().to(y.device)
        for i in range(len(dec_out)):
            y_pred[i, :len(dec_out[i])] = torch.tensor(dec_out[i]).to(y.device)
        return y_pred
