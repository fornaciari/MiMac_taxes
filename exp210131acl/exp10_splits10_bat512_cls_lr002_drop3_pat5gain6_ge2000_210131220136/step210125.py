# coding=latin-1
import util201217 as ut
import os, re, sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
import pickle
from scipy.stats import ttest_rel
from scipy.stats import linregress
from scipy.sparse import csr_matrix, save_npz, load_npz
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel, Whitespace, CharDelimiterSplit
from tokenizers.trainers import BpeTrainer
from transformers import BertTokenizer, BertConfig, BertModel, AutoModel, AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification, FlaubertTokenizer, FlaubertModel, FlaubertConfig
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from unidecode import unidecode
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_fscore_support, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class EarlyStopping():
    """https://rguigoures.github.io/word2vec_pytorch/"""
    def __init__(self, patience=5, min_percent_gain=1):
        self.patience = patience
        self.min_percent_gain = min_percent_gain / 100.
        self.loss_list = []

    def update_loss(self, loss):
        self.loss_list.append(loss)
        if len(self.loss_list) > self.patience:
            del self.loss_list[0]

    def stop_training(self, epoch):
        if len(self.loss_list) == 1:
            line = f"Epoch {epoch:<3} - Loss {self.loss_list[-1]:<12.4f}"
            print(line)
            return False
        norm_losses = np.array(self.loss_list) / np.linalg.norm(self.loss_list)  # altrimenti il gain salta quando passa da valor1 > 1 a < 1 - https://www.kite.com/python/answers/how-to-normalize-an-array-in-numpy-in-python#:~:text=Use%20numpy.,()%20to%20normalize%20an%20array&text=norm(arr)%20to%20find%20the,norm%20to%20normalize%20the%20array.&text=Further%20Reading%20Normalizing%20a%20dataset,to%20%5B0%2C%201%5D%20.
        gain = (max(norm_losses) - min(norm_losses)) / max(norm_losses) #if max(self.loss_list) > 1 else 0
        line = f"Epoch {epoch:<3} - Loss {self.loss_list[-1]:<12.4f} - Gain {gain * 100:.2f}%"
        print(line)
        # if (epoch >= 20) and (epoch % 10 == 0):
        if epoch > 20:
            self.min_percent_gain += .01
            print(f"min percent gain updated to {self.min_percent_gain * 100:.2f}")
        if (len(self.loss_list) == self.patience) and (gain < self.min_percent_gain):
            return True
        elif np.isnan(self.loss_list[-1]):
            print(f"{ut.bcolors.red}loss to nan!!{ut.bcolors.reset}")
            return True
        else:
            return False


class Processing():
    def __init__(self, dir_root, device):
        self.dir_root = dir_root
        self.device = device
        self.path_fastext_en = '/home//fastext/crawl-300d-2M-subword.vec'
        self.path_fastext_sw = '/home//fastext/cc.sv.300.vec'

    @staticmethod
    def plotexp_stl(trn, dev, tst, path_pdf):
        fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(18, 5))
        x_range = range(1, trn.shape[0]+1)
        ax = plt.subplot(131)
        line1, = ax.plot(x_range, tst.loss, label='tst hard loss')
        line2, = ax.plot(x_range, dev.loss, label='dev hard loss')
        line3, = ax.plot(x_range, trn.loss, label='trn hard loss')
        ax.legend()
        ax = plt.subplot(132)
        line1, = ax.plot(x_range, tst.acc, label='tst acc')
        line2, = ax.plot(x_range, dev.acc, label='dev acc')
        line3, = ax.plot(x_range, trn.acc, label='trn acc')
        ax.legend()
        ax = plt.subplot(133)
        line1, = ax.plot(x_range, tst.f1, label='tst f1')
        line2, = ax.plot(x_range, dev.f1, label='dev f1')
        line3, = ax.plot(x_range, trn.f1, label='trn f1')
        ax.legend()
        plt.savefig(path_pdf)
        plt.close()
        return 1

    @staticmethod
    def plotexp_mtl(trn, dev, tst, path_pdf):
        x_range = range(1, trn.shape[0]+1)
        plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 10))
        ax = plt.subplot(221)
        ax.plot(x_range, tst.acc, label='tst acc')
        ax.plot(x_range, dev.acc, label='dev acc')
        ax.plot(x_range, trn.acc, label='trn acc')
        ax.legend()
        ax = plt.subplot(222)
        ax.plot(x_range, tst.f1, label='tst f1')
        ax.plot(x_range, dev.f1, label='dev f1')
        ax.plot(x_range, trn.f1, label='trn f1')
        ax.legend()
        ax = plt.subplot(223)
        ax.plot(x_range, tst.loss_hard, label='tst hard loss')
        ax.plot(x_range, dev.loss_hard, label='dev hard loss')
        ax.plot(x_range, trn.loss_hard, label='trn hard loss')
        ax.legend()
        ax = plt.subplot(224)
        ax.plot(x_range, tst.loss_soft, label='tst soft loss')
        ax.plot(x_range, dev.loss_soft, label='dev soft loss')
        ax.plot(x_range, trn.loss_soft, label='trn soft loss')
        ax.legend()
        plt.savefig(path_pdf)
        plt.close()
        return 1

    @staticmethod
    def save_x(X, path_noextension, lasttrn=None, lastdev=None):
        """from_numpy() automatically inherits input array dtype. On the other hand, torch.Tensor is an alias for torch.FloatTensor.
           Therefore, if you pass int64 array to torch.Tensor, output tensor is float tensor and they wouldn't share the storage"""
        assert type(lasttrn) == type(lastdev)
        path = path_noextension if not re.search("\.pt$", path_noextension) else path_noextension[:-3]
        if lasttrn is None:
            if type(X) is np.ndarray:
                torch.save(torch.from_numpy(X), f"{path}.pt")
            else:
                torch.save(X, f"{path}.pt")
            print(f"{path.split('/')[-1]:<60}{X.shape}")
        else:
            if type(X) is np.ndarray:
                torch.save(torch.from_numpy(X[:lasttrn]), f"{path}_trn.pt")
                torch.save(torch.from_numpy(X[lasttrn:lastdev]), f"{path}_dev.pt")
                torch.save(torch.from_numpy(X[lastdev:]), f"{path}_tst.pt")
            else:
                torch.save(X[:lasttrn], f"{path}_trn.pt")
                torch.save(X[lasttrn:lastdev], f"{path}_dev.pt")
                torch.save(X[lastdev:], f"{path}_tst.pt")
            print(f"{path.split('/')[-1] + '_trn':<60}{X[:lasttrn].shape}\n"
                  f"{path.split('/')[-1] + '_dev':<60}{X[lasttrn:lastdev].shape}\n"
                  f"{path.split('/')[-1] + '_tst':<60}{X[lastdev:].shape}")
        return 1

    def fastext(self, X, lang='en', minfreq=0, nobelow=.01, noabove=.9, padrate=99.5, splits=(None, None), name=''):
        assert 0 <= nobelow <= 1, 'nobelow è una percentuale sul numero dei documenti e deve essere compresa tra 0 e 1'
        assert 0 <= noabove <= 1, 'noabove è una percentuale sul numero dei documenti e deve essere compresa tra 0 e 1'
        lang2path = {'en': self.path_fastext_en, 'sw': self.path_fastext_sw}
        conditiunderscore = '_' if name != '' else ''
        winfo = defaultdict(lambda: {'freq': 0, 'docs': 0})
        nr_docs = 0
        for row in X:
            nr_docs += 1
            seen = set()
            for word in row.split():
                winfo[word]['freq'] += 1
                if word not in seen:
                    winfo[word]['docs'] += 1
                seen.add(word)
        # ut.printdict(winfo, 5)
        dic = {w: {'idx': None, 'freq': winfo[w]['freq'], 'docs': winfo[w]['docs']} for w in winfo if (winfo[w]['freq'] >= minfreq) and (winfo[w]['docs'] / nr_docs >= nobelow) and (winfo[w]['docs'] / nr_docs <= noabove)}
        print(f"{'#'*50}\nfasttext embeddings {name}\n"
              f"{'min freq:':<60}{minfreq}\n"
              f"{'no below ' + str(nobelow) + ':':<60}{int(nobelow * nr_docs)}/{nr_docs}\n"
              f"{'no above ' + str(noabove) + ':':<60}{int(noabove * nr_docs)}/{nr_docs}\n"
              f"{'dictionary size before filtering:':<60}{len(winfo)}\n"
              f"{'dictionary size after filtering:':<60}{len(dic)}")
        with open(lang2path[lang], 'r', encoding='utf-8', newline='\n', errors='ignore') as file_handler: # è già un generator
            fastext_vocsize, fastext_embsize = map(int, file_handler.__next__().split()) # la prima riga del file contiene il nr delle righe e la dim degli emb
            idx = -1
            embs = list()
            print("traversing fasttext embeddings...")
            while True:
                try:
                    row = file_handler.__next__().split() # le righe successive alla prima contengono una stringa con parola e valori dell'embedding, separati da spazio
                    word = row[0]
                    if word in dic:
                        idx += 1
                        dic[word]['idx'] = idx + 2 # 0 per pad e 1 per oov
                        emb = list(map(float, row[1:]))
                        embs.append(emb)
                        ut.say_progress(idx, 1000)
                except StopIteration:
                    break
        print(f"\n{'iteration ended at index:':<60}{idx}\n{'coverage:':<60}{(idx + 1)/len(dic) * 100:.2f}%")
        dic = {w: dic[w] for w in dic if dic[w]['idx'] is not None}
        print(f"{'dictionary size after fasttext selection:':<60}{len(dic)}")
        ut.writejson(dic, f"{self.dir_root}fastext_{name}{conditiunderscore}word2info.json")
            
        embmatrix = np.array(embs)
        zeros = np.zeros((1, embmatrix.shape[1])) # pad
        meanvector = np.reshape(embmatrix.mean(axis=0), (1, embmatrix.shape[1])) # oov
        embmatrix = np.concatenate((zeros, meanvector, embmatrix), axis=0)  # 0: pad, 1: oov
        print(f"{'fasttext embmatrix shape:':<60}{embmatrix.shape} (0: pad, 1: oov)")
        torch.save(torch.from_numpy(embmatrix), f"{self.dir_root}fastext_{name}{conditiunderscore}embmatrix.pt")
    
        lens = [len(text.split()) for text in X]
        padsize = int(round(np.percentile(lens, padrate), -1))
        print(f"{'longest text:':<60}{max(lens)}\n{'pad size:':<60}{padsize} ({padrate:.2f}%)")
        unpad_ids = [[dic[word]['idx'] if word in dic else 1 for word in text.split()] for text in X] # index 1 for oov. lo uso anche come parola unica nelle frasi mancanti, [1]
        pad_ids  = np.array([row + [0] * (padsize - len(row)) if len(row) < padsize else row[-padsize:] for row in unpad_ids]) # index 0 for pad
        pad_mask = np.array([[1] * len(row) + [0] * (padsize - len(row)) if len(row) < padsize else [1] * padsize for row in unpad_ids])
        self.save_x(pad_ids, path_noextension=f"{self.dir_root}fastext_{name}{conditiunderscore}ids", lasttrn=splits[0], lastdev=splits[1])
        self.save_x(pad_mask, path_noextension=f"{self.dir_root}fastext_{name}{conditiunderscore}mask", lasttrn=splits[0], lastdev=splits[1])
        return padsize

    def bert_lookup(self, lang, hiddenlayer, X1, X2=None, padsize=510, matrix=False, special_token=True, pad_mode='max_length', returnmask=True, truncstrategy='only_second', splits=(None, None), name=''):
        '''vedi step201106printmemory.py per la corrispondenza del CLS e la stampa della memoria
        con AutoModel:
        azz[0] = last hidden layer
        azz[1] = pooler output
        azz[2] = all hidden layers
        per prove:
        azz = bert(input_ids=texts_ids[0].unsqueeze(0), attention_mask=texts_masks[0].unsqueeze(0))
        ass = torch.stack([bert(input_ids=text_ids.unsqueeze(0), attention_mask=text_masks.unsqueeze(0))[0][0][0].detach() for text_ids, text_masks in zip(texts_ids[:16], texts_masks)]) # [2][hiddenlayer][0][0] = [itupla][ilayer][idoc][iword]
        hiddenlayer = len(azz[2])-1
        assert torch.equal(azz[0][0][0], azz[2][hiddenlayer][0][0]) # [itupla][idoc][iword] [itupla][ilayer][idoc][iword]
        assert torch.equal(azz[0], azz[2][hiddenlayer]) # [itupla] [itupla][ilayer]
        '''
        startime = ut.start()
        kindbert = 'single' if X2 is None else 'paired'
        conditiunderscore = '_' if name != '' else ''
        condition = f"{lang}_{kindbert}{conditiunderscore}{name}"
        print(f"{condition}")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")     if lang == 'ml' else \
                    AutoTokenizer.from_pretrained('bert-base-cased')                  if lang == 'en' else \
                    AutoTokenizer.from_pretrained("bert-large-cased")                 if lang == 'enlarge' else \
                    AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-cased')    if lang == 'it' else \
                    AutoTokenizer.from_pretrained('dbmdz/bert-base-german-cased')     if lang == 'de' else \
                    AutoTokenizer.from_pretrained('camembert-base')                   if lang == 'fr' else \
                    AutoTokenizer.from_pretrained('pdelobelle/robbert-v2-dutch-base') if lang == 'nl' else \
                    AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")       if lang == 'sw' else sys.exit('bert language non presente')
        vocsize = tokenizer.vocab_size
        # vocab = tokenizer.get_vocab()
        # ut.printdict(vocab, 5)
        # ut.writejson(vocab, f"{self.dir_root}vocab{condition}.json")

        if X2 is None:
            texts_ids = torch.cat([tokenizer.encode_plus(text, # Sentence to encode.
                                   add_special_tokens=special_token, # Add '[CLS]' and '[SEP]'
                                   truncation=True,
                                   padding=pad_mode,
                                   max_length=padsize, # Pad & truncate all sentences.
                                   return_attention_mask=returnmask, # Construct attn. masks.
                                   return_tensors='pt')['input_ids'] for text in X1], dim=0).to(device=self.device)
            texts_masks = torch.cat([tokenizer.encode_plus(text, # Sentence to encode.
                                     add_special_tokens=special_token, # Add '[CLS]' and '[SEP]'
                                     truncation=True,
                                     padding=pad_mode,
                                     max_length=padsize, # Pad & truncate all sentences.
                                     return_attention_mask=returnmask, # Construct attn. masks.
                                     return_tensors='pt')['attention_mask'] for text in X1], dim=0).to(device=self.device)
        else:
            assert len(X1) == len(X2)
            texts_ids = torch.cat([tokenizer.encode_plus(text1, text2, # Sentence to encode.
                                   add_special_tokens=special_token, # Add '[CLS]' and '[SEP]'
                                   truncation=True,
                                   padding=pad_mode,
                                   max_length=padsize, # Pad & truncate all sentences.
                                   return_attention_mask=returnmask, # Construct attn. masks.
                                   truncation_strategy=truncstrategy, # only_first only_second longest_first
                                   return_tensors='pt')['input_ids'] for text1, text2 in zip(X1, X2)], dim=0).to(device=self.device)
            texts_masks = torch.cat([tokenizer.encode_plus(text1, text2, # Sentence to encode.
                                     add_special_tokens=special_token, # Add '[CLS]' and '[SEP]'
                                     truncation=True,
                                     padding=pad_mode,
                                     max_length=padsize, # Pad & truncate all sentences.
                                     return_attention_mask=returnmask, # Construct attn. masks.
                                     truncation_strategy=truncstrategy, # only_first only_second longest_first
                                     return_tensors='pt')['attention_mask'] for text1, text2 in zip(X1, X2)], dim=0).to(device=self.device)
        self.save_x(texts_ids, path_noextension=f"{self.dir_root}bert_{condition}_ids", lasttrn=splits[0], lastdev=splits[1])
        self.save_x(texts_masks, path_noextension=f"{self.dir_root}bert_{condition}_masks", lasttrn=splits[0], lastdev=splits[1])

        config = BertConfig.from_pretrained("bert-base-multilingual-cased",     output_hidden_states=True) if lang == 'ml' else \
                 BertConfig.from_pretrained("bert-base-cased",                  output_hidden_states=True) if lang == 'en' else \
                 BertConfig.from_pretrained("bert-large-cased",                 output_hidden_states=True) if lang == 'enlarge' else \
                 BertConfig.from_pretrained("dbmdz/bert-base-italian-cased",    output_hidden_states=True) if lang == 'it' else \
                 BertConfig.from_pretrained("dbmdz/bert-base-german-cased",     output_hidden_states=True) if lang == 'de' else \
                 BertConfig.from_pretrained("camembert-base",                   output_hidden_states=True) if lang == 'fr' else \
                 BertConfig.from_pretrained("pdelobelle/robbert-v2-dutch-base", output_hidden_states=True) if lang == 'nl' else \
                 BertConfig.from_pretrained("KB/bert-base-swedish-cased",       output_hidden_states=True) if lang == 'sw' else sys.exit('bert language non presente')
        config.to_json_file(f"{self.dir_root}bert_{condition}_config.json")
        bert   = AutoModel.from_pretrained("bert-base-multilingual-cased",     config=config).to(device=self.device, dtype=torch.float32) if lang == 'ml' else \
                 AutoModel.from_pretrained('bert-base-cased',                  config=config).to(device=self.device, dtype=torch.float32) if lang == 'en' else \
                 AutoModel.from_pretrained("bert-large-cased",                 config=config).to(device=self.device, dtype=torch.float32) if lang == 'enlarge' else \
                 AutoModel.from_pretrained('dbmdz/bert-base-italian-cased',    config=config).to(device=self.device, dtype=torch.float32) if lang == 'it' else \
                 AutoModel.from_pretrained('dbmdz/bert-base-german-cased',     config=config).to(device=self.device, dtype=torch.float32) if lang == 'de' else \
                 AutoModel.from_pretrained('camembert-base',                   config=config).to(device=self.device, dtype=torch.float32) if lang == 'fr' else \
                 AutoModel.from_pretrained("pdelobelle/robbert-v2-dutch-base", config=config).to(device=self.device, dtype=torch.float32) if lang == 'nl' else \
                 AutoModel.from_pretrained("KB/bert-base-swedish-cased",       config=config).to(device=self.device, dtype=torch.float32) if lang == 'sw' else sys.exit('bert language non presente')
        embsize = bert.config.to_dict()['hidden_size'] # 768
        print(f"{'voc size':<60}{vocsize}")

        def make_pooler():
            lookup_pooler = torch.cat([bert(input_ids=text_ids.unsqueeze(0), attention_mask=text_masks.unsqueeze(0))[1].detach() for text_ids, text_masks in zip(texts_ids, texts_masks)])
            self.save_x(lookup_pooler, path_noextension=f"{self.dir_root}bert_{condition}_lookup_pooler", lasttrn=splits[0], lastdev=splits[1])
            torch.cuda.empty_cache()
            return 1

        def make_cls():
            lookup_cls = torch.stack([bert(input_ids=text_ids.unsqueeze(0), attention_mask=text_masks.unsqueeze(0))[2][hiddenlayer][0][0].detach().cpu() for text_ids, text_masks in zip(texts_ids, texts_masks)]) # [2][hiddenlayer][0][0] = [itupla][ilayer][idoc][iword]
            self.save_x(lookup_cls, path_noextension=f"{self.dir_root}bert_{condition}_lookup_cls", lasttrn=splits[0], lastdev=splits[1])
            torch.cuda.empty_cache()
            return 1

        def make_meanmat():
            lookup_meanmat = torch.cat([bert(input_ids=text_ids.unsqueeze(0), attention_mask=text_masks.unsqueeze(0))[2][hiddenlayer].mean(1).detach().cpu() for text_ids, text_masks in zip(texts_ids, texts_masks)])
            self.save_x(lookup_meanmat, path_noextension=f"{self.dir_root}bert_{condition}_lookup_meanmat", lasttrn=splits[0], lastdev=splits[1])
            torch.cuda.empty_cache()
            return 1

        def make_mat():
            lookup_mat = torch.cat([bert(input_ids=text_ids.unsqueeze(0), attention_mask=text_masks.unsqueeze(0))[2][hiddenlayer].detach().cpu() for text_ids, text_masks in zip(texts_ids, texts_masks)])
            self.save_x(lookup_mat, path_noextension=f"{self.dir_root}bert_{condition}_lookup_mat", lasttrn=splits[0], lastdev=splits[1])
            torch.cuda.empty_cache()
            return 1

        try:
            make_cls()
        except RuntimeError as err:
            print(f"{err}\n{'lookup_cls':<60} not done")
        try:
            make_meanmat()
        except RuntimeError as err:
            print(f"{err}\n{'lookup_meanmat':<60} not done")
        if matrix:
            try:
                make_mat()
            except RuntimeError as err:
                print(f"{err}\n{'lookup_mat':<60} not done")

        ut.end(startime)
        return vocsize, embsize

    @staticmethod
    def token_preproc(X, vocsize, padsize):
        tokenizer = Tokenizer(BPE.empty()) # empty Byte-Pair Encoding model
        tokenizer.normalizer = Sequence([NFKC(), Lowercase()]) # ordered Sequence of normalizers: unicode-normalization then lower-casing
        tokenizer.pre_tokenizer = ByteLevel()  # pre-tokenizer converting the input to a ByteLevel representation.
        tokenizer.decoder = ByteLevelDecoder() # plug a decoder so we can recover from a tokenized input to the original one
        trainer = BpeTrainer(vocab_size=vocsize, show_progress=True, initial_alphabet=ByteLevel.alphabet(), special_tokens=["<pad>"]) # trainer initialization
        ut.list2file([text for text in X], 'temporary.txt') # stupidamente, il trainer vuole come input solo un(a lista di) file
        tokenizer.train(trainer, ['temporary.txt'])
        os.system('rm temporary.txt')
        print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))
        texts_ids = list()
        texts_masks = list()
        tokenizer.enable_padding(max_length=padsize) # https://huggingface.co/transformers/_modules/transformers/tokenization_utils.html
        tokenizer.enable_truncation(max_length=padsize)
        for text in tqdm(X):
            encoded = tokenizer.encode(text) # Return pytorch tensors.
            texts_ids.append(encoded.ids)
            texts_masks.append(encoded.attention_mask)
        texts_ids = np.array(texts_ids)
        texts_masks = np.array(texts_masks)
        return texts_ids, texts_masks, tokenizer.get_vocab_size()

    @staticmethod
    def realigned_token_preproc(X, vocsize, padsize):
        pad_token = "[PAD]" # se non dai qualcosa al trainer, s'inkazza. dovrei dargli gli special tokens già presenti nel mio corpus. [PAD] viene dato automaticamente per il pad: più che altro lo uso dopo, per raccogliere il suo id
        tokenizer = Tokenizer(BPE.empty()) # empty Byte-Pair Encoding model
        tokenizer.normalizer = Sequence([NFKC(), Lowercase()]) # ordered Sequence of normalizers: unicode-normalization then lower-casing
        tokenizer.pre_tokenizer = CharDelimiterSplit(' ') # deve dividere per spazi perché si mantenga la corrispondenza token->label: in questo caso, la punteggiatura è già separata in input
        # tokenizer.pre_tokenizer = ByteLevel()  # pre-tokenizer converting the input to a ByteLevel representation.
        tokenizer.decoder = ByteLevelDecoder() # plug a decoder so we can recover from a tokenized input to the original one
        ut.list2file([text for text in X], 'temporary.txt') # stupidamente, train vuole come input solo un(a lista di) file
        trainer = BpeTrainer(vocab_size=vocsize, show_progress=True, initial_alphabet=ByteLevel.alphabet(), continuing_subword_prefix=subword_prefix, special_tokens=[pad_token]) # trainer initialization
        tokenizer.train(trainer, files=['temporary.txt'])
        os.system('rm temporary.txt')
        # tokenizer.model.save(".") # basta la dir, il nome è di default (si può usare il campo successivo come prefix)
        texts_ids, texts_masks, texts_lens = list(), list(), list()
        tokenizer.enable_padding(max_length=padsize) # https://huggingface.co/transformers/_modules/transformers/tokenization_utils.html
        tokenizer.enable_truncation(max_length=padsize)
        for text in tqdm(X, desc='text to ids', ncols=80):
            encoded = tokenizer.encode(text)
            subword_mask = [1 if ( # 1 se è un token singolo o finale, 0 se è inziale o successivo ma non ultimo di un token multiplo
                                 re.match(subword_prefix, encoded.tokens[i]) and # se il token è una subword e
                                   ((i+1 == len(encoded.tokens)) # è l'ultimo della frase
                                   or # o
                                   (not re.match(subword_prefix, encoded.tokens[i + 1])))) # il token successivo non è un'altra subword
                              or ( # oppure
                                 (not re.match(subword_prefix, encoded.tokens[i])) and # il token è iniziale e
                                   ((i+1 == len(encoded.tokens)) # è l'ultimo della frase
                                   or # o
                                   (not re.match(subword_prefix, encoded.tokens[i + 1])))) # il token successivo non è una subword
                        else 0 for i in range(len(encoded.tokens))]
            text_ids = [id for mask, id in zip(subword_mask, encoded.ids) if mask] + [tokenizer.token_to_id(pad_token)] * (len(encoded.ids) - sum(subword_mask))
            text_mask = [1 if id != tokenizer.token_to_id(pad_token) else 0 for id in text_ids]
            texts_ids.append(text_ids)
            texts_masks.append(text_mask)
            texts_lens.append(sum(text_mask))
            assert len(text.split()[:padsize]) == sum(text_mask) # il nr di parole all'inizio e alla fine deve essere uguale. ma se la frase è più lunga di padsize, non si sa se dentro ci sono casini
        texts_ids = np.array(texts_ids)
        texts_masks = np.array(texts_masks)
        texts_lens = np.array(texts_lens)
        print(f"{'words vocab size':.<25} {tokenizer.get_vocab_size()}\n{'words shape':.<25} {texts_ids.shape}")
        return texts_ids, texts_masks, texts_lens, tokenizer.get_vocab_size()

    @staticmethod
    def char_preproc(X, padsize):
        X = [unidecode(str(x)).lower() for x in X] # str perché ci sono alcuni numeri visti come int. alcuni non ascii char vengono sostituiti con ''
        char2id = {c for x in X for c in x} # set of the seen ascii chars
        char2id = {c: i + 1 for i, c in enumerate(sorted(char2id))} # lasciio lo 0 per pad
        chars_ids = np.array([[char2id[c] for c in x][:padsize] if len(x) > padsize else
                              [char2id[c] for c in x] + [0] * (padsize - len(x))
                              for x in X])
        chars_masks = np.array([[1] * padsize if len(x) > padsize else
                                [1] * len(x) + [0] * (padsize - len(x))
                                for x in X])
        vocsize = len(char2id) + 1 # per pad token
        print(f"{'chars vocab size':.<25} {vocsize}\n{'chars shape':.<25} {chars_ids.shape}")
        return chars_ids, chars_masks, vocsize

    @staticmethod
    def performance(targs, preds, title=''):
        print(f"{'#'*40}\n{title if title != '' else 'performance:'}")
        conf_matrix = confusion_matrix(targs, preds)
        acc  = round(accuracy_score(targs, preds) * 100, 2)
        prec = round(precision_score(targs, preds, average='macro') * 100, 2)
        rec  = round(recall_score(targs, preds, average='macro') * 100, 2)
        f1   = round(f1_score(targs, preds, average='macro') * 100, 2)
        micro_measures = precision_recall_fscore_support(targs, preds, average='micro')
        macro_measures = precision_recall_fscore_support(targs, preds, average='macro')
        arrs = precision_recall_fscore_support(targs, preds)
        countpreds = Counter(preds)
        counttargs = Counter(targs)
        countpreds = [f"class {tup[0]} freq {tup[1]} perc {tup[1] / len(preds) * 100:.2f}%" for tup in sorted({k: countpreds[k] for k in countpreds}.items(), key=lambda item: item[0])]
        counttargs = [f"class {tup[0]} freq {tup[1]} perc {tup[1] / len(targs) * 100:.2f}%" for tup in sorted({k: counttargs[k] for k in counttargs}.items(), key=lambda item: item[0])]
        print(f"confusion matrix:\n{conf_matrix}")
        print(f"preds count: {countpreds}")
        print(f"targs count: {counttargs}")
        print(f"micro precision_recall_fscore_support:{micro_measures}\nmacro precision_recall_fscore_support:{macro_measures}")
        for arr, met in zip(arrs, ['precisions', 'recalls', 'F-measures', 'supports']): print(f"{met:.<15} {arr} mean {round(arr.mean() * 100, 4)}")
        print(f"{'accuracy':.<12} {acc}\n{'precision':.<12} {prec}\n{'recall':.<12} {rec}\n{'F-measure':.<12} {f1:<10}\n{'#'*40}")
        return 1


    def exp(self, model,
                  optimizer,
                  lossfuncs,
                  x_inputs        = None, # trn per holdout, all per crossval
                  x_inputs_dev    = None,
                  x_inputs_tst    = None,
                  x_dtypes        = None,
                  y_inputs        = None,
                  y_inputs_dev    = None,
                  y_inputs_tst    = None,
                  y_dtypes        = None,
                  batsize         = 128,
                  patience        = 4,
                  min_gain        = 4,
                  n_splits        = 10, # solo crossval, nr folds
                  save            = False,
                  additional_tsts = (),
                  str_info        = ''):
        validation = 'ho' if x_inputs_tst[0] is not None else 'cv'
        dir_exp = f"{self.dir_root}{str_info}_{validation}_{model.__class__.__name__}"
        os.mkdir(dir_exp)
        print(f"{validation} {'dir out':.<25} {dir_exp}")
        assert len(x_inputs) == len(x_dtypes)
        assert len(y_inputs) == len(y_dtypes) == len(lossfuncs)

        def batches(step, x_step, y_step):
            preds = list()
            losss = list()
            model.train() if step == 'trn' else model.eval()
            for ifir_bat in tqdm(range(0, len(y_step[0]), batsize), desc=step, ncols=80): # desc='training' # prefix
                nlas_bat = ifir_bat + batsize
                x_bat = [x[ifir_bat: nlas_bat].to(device=self.device, dtype=dt) if torch.is_tensor(x) else torch.from_numpy(x[ifir_bat: nlas_bat]).to(device=self.device, dtype=dt) for x, dt in zip(x_step, x_dtypes)]
                y_bat = [y[ifir_bat: nlas_bat].to(device=self.device, dtype=dt) if torch.is_tensor(y) else torch.from_numpy(y[ifir_bat: nlas_bat]).to(device=self.device, dtype=dt) for y, dt in zip(y_step, y_dtypes)]
                # for x in x_bat: print(x.shape)
                # for y in y_bat: print(y.shape)
                pred_bat = model(*x_bat)
                if len(lossfuncs) > 1:
                    # for x in x_bat: print('input', x, x.shape)
                    # for lossf, pred, y in zip(lossfuncs, pred_bat, y_bat): print('loss', lossf, "\npreds", pred, "\npreds.shape", pred.shape, "\ntargs", y, "\ntargs.shape", y.shape)
                    loss_bat = [lossf(pred, y) for lossf, pred, y in zip(lossfuncs, pred_bat, y_bat)]
                    pred_bat = pred_bat[0].argmax(1).data.tolist() if len(pred_bat[0].shape) > 1 else [0 if v < .5 else 1 for v in pred_bat[0].data.tolist()] # se le pred hanno righe e colonne, le trasformo in int. la tupla diventa un solo vettore, ma tanto il secondo che perdo non viene usato per misurare la performance, e l'ho già usato nella lossfunc
                else:
                    # print('loss', lossfuncs[0], "\npreds", pred_bat, "\npreds.shape", pred_bat.shape, "\ntargs", y_bat, "\ntargs.shape", y_bat[0].shape, "$$$")
                    loss_bat = lossfuncs[0](pred_bat, y_bat[0])
                    pred_bat = pred_bat.argmax(1).data.tolist() if len(pred_bat.shape) > 1 else [0 if v < .5 else 1 for v in pred_bat.data.tolist()]
                if step == 'trn':
                    if isinstance(loss_bat, list):
                        # distinct back propagations
                        # optimizer.zero_grad()
                        # for i, loss in enumerate(loss_bat):
                        #     loss.backward(retain_graph=True) # retain_graph sarebbe sufficiente solo per la prima back, ma pare che la seconda non faccia danni. E serve sole per nn.KLDivLoss(): le mie loss, con require_grad, potrebbero farne a meno
                        #     optimizer.step()
                        # overall back propagation
                        optimizer.zero_grad()
                        torch.mean(torch.stack(loss_bat)).backward()
                        optimizer.step()
                    else:
                        optimizer.zero_grad()
                        loss_bat.backward()
                        optimizer.step()
                    # torch.nn.utils.clip_grad_norm_(mod.parameters(), 0.5)
                preds.extend(pred_bat)
                losss.append([loss.item() for loss in loss_bat] if isinstance(loss_bat, list) else [loss_bat.item()])
            y_perf = y_step[0].argmax(1).data.tolist() if len(y_step[0].shape) > 1 else y_step[0] # solo per la performance. normalmente y è una list di int, ma se è prob dist (list di list) la devo portare a vec
            accu = round(accuracy_score(y_perf, preds) * 100, 2)
            fmea = round(f1_score(y_perf, preds, average='macro') * 100, 2)
            mean_losss = [round(np.array(losss)[:, i].mean(), 4) for i in range(len(losss[0]))]
            dictout = {f"loss{i}": mean_losss[i] for i in range(len(mean_losss))}
            dictout['acc'] = round(accu, 2)
            dictout['f1'] = round(fmea, 2)
            serie_out = pd.Series(dictout, name=step)
            print(dictout)
            return serie_out, preds

        def data_loop(x_trn, x_dev, x_tst, y_trn, y_dev, y_tst):
            df_trn, df_dev, df_tst = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()                
            stopping = EarlyStopping(patience=patience, min_percent_gain=min_gain)
            nepoch = 0
            while True:
                nepoch += 1
                print(f"epoch {nepoch}")
                serie_trn, _,        = batches('trn', x_trn, y_trn)
                serie_dev, _,        = batches('dev', x_dev, y_dev)
                serie_tst, preds_tst = batches('tst', x_tst, y_tst)
                df_trn = df_trn.append(serie_trn)
                df_dev = df_dev.append(serie_dev)
                df_tst = df_tst.append(serie_tst.append(pd.Series({'n_epochs': nepoch}, name='tst')))
                stopping.update_loss(serie_dev.loss0)

                if stopping.stop_training(nepoch):
                    if save:
                        torch.save(model.state_dict(), f"{dir_exp}model.pt")
                        print('model saved')
                        if len(lossfuncs) == 1:
                            self.plotexp_stl(df_trn, df_dev, df_tst, f"{dir_exp}results_fold{fold}.pdf")
                        elif len(lossfuncs) == 2:
                            self.plotexp_mtl(df_trn, df_dev, df_tst, f"{dir_exp}results_fold{fold}.pdf")
                    break
            return preds_tst, y_tst[0].astype(int).tolist(), df_tst.iloc[-1, :]
        
        if x_inputs_tst[0] is not None: # holdout
            assert len(x_inputs) == len(x_inputs_dev) == len(x_inputs_tst)
            assert len(y_inputs) == len(y_inputs_dev) == len(y_inputs_tst)
            shuffled_indices = torch.randperm(len(x_inputs[0])) # mescolare il trn?
            x_inputs = [x[shuffled_indices] for x in x_inputs]
            y_inputs = [y[shuffled_indices] for y in y_inputs]
            preds, targs, df = data_loop(x_inputs, x_inputs_dev, x_inputs_tst, y_inputs, y_inputs_dev, y_inputs_tst)
            ut.list2file(preds, f"{dir_exp}/preds.txt")
            ut.list2file(targs, f"{dir_exp}/targs.txt")
            df = df.to_frame().transpose()
            i_tst = None
        else: # crossval
            path_virgin = f"{self.dir_root}virgin_model.pt"
            torch.save(model.state_dict(), path_virgin)
            i_tst, preds, targs = list(), list(), list()
            df = pd.DataFrame()
            fold = 0
            skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
            for i_trn_dev_fold, i_tst_fold in skf.split(x_inputs[0], y_inputs[0]):
                fold += 1
                # max_metric = 0
                print(f"{'#'*80}\nfold {fold}")
                dev_rate = int(len(y_inputs[0]) * (1 / n_splits))
                shuffled_indices = torch.randperm(len(i_trn_dev_fold)) # skf mescola, ma anche ordina: rimescolo
                i_dev_fold = i_trn_dev_fold[shuffled_indices][-dev_rate:]
                i_trn_fold = i_trn_dev_fold[shuffled_indices][:-dev_rate]
                x_shuf_trn = [x[i_trn_fold] for x in x_inputs]
                x_shuf_dev = [x[i_dev_fold] for x in x_inputs]
                x_shuf_tst = [x[i_tst_fold] for x in x_inputs]
                y_shuf_trn = [y[i_trn_fold] for y in y_inputs]
                y_shuf_dev = [y[i_dev_fold] for y in y_inputs]
                y_shuf_tst = [y[i_tst_fold] for y in y_inputs]
                # for x in x_shuf_trn: print(x.shape)
                # for y in y_shuf_trn: print(y.shape)
                virgin = torch.load(path_virgin)
                model.load_state_dict(virgin)
                preds_fold, targs_fold, df_fold = data_loop(x_shuf_trn, x_shuf_dev, x_shuf_tst, y_shuf_trn, y_shuf_dev, y_shuf_tst)
                i_tst.extend(i_tst_fold)
                preds.extend(preds_fold)
                targs.extend(targs_fold)
                df = df.append(df_fold)
            ut.list2file(i_tst, f"{dir_exp}/i_tst.txt")
            ut.list2file(preds, f"{dir_exp}/preds.txt")
            ut.list2file(targs, f"{dir_exp}/targs.txt")
            os.system(f"rm {path_virgin}")

        m_epochs = df.n_epochs.mean()
        print(f"{'#'*40}\n{df}\nmean df\n{df.mean()}")
        self.performance(targs, preds, dir_exp)

        # if len(additional_tsts) > 0:
        #     """additional_tsts: lista di (x, y, name)
        #     x e y, a loro volta, sono liste anche se contengono un solo elemento, come solito"""
        #     for additional_x, additional_y, name in additional_tsts:
        #         serie_tst, preds_tst = self.batches('tst', model, optimizer, lossfuncs, additional_x, x_dtypes, additional_y, y_dtypes, batsize)
        #         preds = preds_tst
        #         targs = y_inputs_tst[0].astype(int).tolist()
        #         ut.list2file(preds, f"{dir_exp}/{name}_preds.txt")
        #         ut.list2file(targs, f"{dir_exp}/{name}_targs.txt")
        #         self.performance(targs, preds)
        #         print(Counter(preds))
        
        dir_out_results = f"{dir_exp}_f{str(round(f1_score(targs, preds, average='macro'), 4))[2:]}/" # non da df, o in crossval fa la media delle fold
        os.rename(dir_exp, dir_out_results)
        ut.sendslack(f"{dir_out_results} done\n{df.mean().to_string()}")
        return dir_out_results, preds, targs, i_tst, m_epochs



class Bootstrap:
    def __init__(self, dirout):
        self.dirout = dirout
        self.data = defaultdict(lambda: {'dirs':   list(),
                                    'preds':  list(),
                                    'targs':  list(),
                                    'epochs': list(),
                                    'treatment': defaultdict(lambda: {'dirs':   list(),
                                                                      'preds':  list(),
                                                                      'targs':  list(),
                                                                      'epochs': list()})})

    def feed(self, control, treatment=None, fold=None, preds=None, targs=None, epochs=None):
        assert len(preds) == len(targs), 'preds e targs di lunghezza diversa!'
        if treatment:
            self.data[control]['treatment'][treatment]['dirs'].append(fold)
            self.data[control]['treatment'][treatment]['preds'].append(preds)
            self.data[control]['treatment'][treatment]['targs'].append(targs)
            self.data[control]['treatment'][treatment]['epochs'].append(epochs)
        else:
            self.data[control]['dirs'].append(fold)
            self.data[control]['preds'].append(preds)
            self.data[control]['targs'].append(targs)
            self.data[control]['epochs'].append(epochs)
        return 1

    def run(self, n_loops, perc_sample=.1, verbose=False):
        """
        :param data:
                defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list(), 'epochs': list(),
                                     'treatment': defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list(), 'epochs': list()})}
        """
        startime = ut.start()

        def metrics(targs, control_preds, treatment_preds):
            rounding_value = 2
            control_acc    = round(accuracy_score(targs, control_preds) * 100, rounding_value)
            control_f1     = round(f1_score(targs, control_preds, average='macro') * 100, rounding_value)
            control_prec   = round(precision_score(targs, control_preds, average='macro') * 100, rounding_value)
            control_rec    = round(recall_score(targs, control_preds, average='macro') * 100, rounding_value)
            treatment_acc  = round(accuracy_score(targs, treatment_preds) * 100, rounding_value)
            treatment_f1   = round(f1_score(targs, treatment_preds, average='macro') * 100, rounding_value)
            treatment_prec = round(precision_score(targs, treatment_preds, average='macro') * 100, rounding_value)
            treatment_rec  = round(recall_score(targs, treatment_preds, average='macro') * 100, rounding_value)
            control_conf_matrix = confusion_matrix(targs, control_preds)
            treatment_conf_matrix = confusion_matrix(targs, treatment_preds)
            diff_acc  = round(treatment_acc - control_acc, rounding_value)
            diff_f1   = round(treatment_f1  - control_f1, rounding_value)
            diff_prec = round(treatment_prec - control_prec, rounding_value)
            diff_rec  = round(treatment_rec  - control_rec, rounding_value)
            return control_acc, treatment_acc, diff_acc, control_f1, treatment_f1, diff_f1, control_prec, treatment_prec, diff_prec, control_rec, treatment_rec, diff_rec

        df = pd.DataFrame(columns="mean_epochs acc diff_acc sign_acc prec diff_prec sign_prec rec diff_rec sign_rec f1 diff_f1 sign_f1".split())
        for control_cond in self.data:
            print('#'*80)
            control_preds_all, control_targs_all = list(), list()
            for dire, preds, targs in zip(self.data[control_cond]['dirs'], self.data[control_cond]['preds'], self.data[control_cond]['targs']):
                acc = round(accuracy_score(targs, preds) * 100, 2)
                f1  = round(f1_score(targs, preds, average='macro') * 100, 2)
                control_preds_all.extend(preds)
                control_targs_all.extend(targs)
                if verbose:  print(f"{''.join(dire.split('/')[2:]):<100} acc {acc:<7} F {f1}")
            for treatment_cond in self.data[control_cond]['treatment']:
                print(f"{'#'*80}\n{control_cond}   vs   {treatment_cond}")
                treatment_preds_all, treatment_targs_all, = list(), list()
                for dire, preds, targs in zip(self.data[control_cond]['treatment'][treatment_cond]['dirs'],
                                              self.data[control_cond]['treatment'][treatment_cond]['preds'],
                                              self.data[control_cond]['treatment'][treatment_cond]['targs']):
                    acc = round(accuracy_score(targs, preds) * 100, 2)
                    f1  = round(f1_score(targs, preds, average='macro') * 100, 2)
                    treatment_preds_all.extend(preds)
                    treatment_targs_all.extend(targs)
                    if verbose: print(f"{''.join(dire.split('/')[2:]):<100} acc {acc:<7} F {f1}")
                # print(len(control_targs_all), len(treatment_targs_all), control_targs_all[:7], treatment_targs_all[:7])
                assert control_targs_all == treatment_targs_all
                targs_all = control_targs_all
                tot_control_acc, tot_treatment_acc, tot_diff_acc, tot_control_f1, tot_treatment_f1, tot_diff_f1, tot_control_prec, tot_treatment_prec, tot_diff_prec, tot_control_rec, tot_treatment_rec, tot_diff_rec = metrics(targs_all, control_preds_all, treatment_preds_all)
                control_countpreds   = Counter(control_preds_all)
                treatment_countpreds = Counter(treatment_preds_all)
                counttargs           = Counter(control_targs_all)
                control_countpreds   = [f"class {tup[0]} freq {tup[1]} perc {tup[1] / len(control_preds_all) * 100:.2f}%" for tup in sorted({k: control_countpreds[k] for k in control_countpreds}.items(), key=lambda item: item[0])]
                treatment_countpreds = [f"class {tup[0]} freq {tup[1]} perc {tup[1] / len(treatment_preds_all) * 100:.2f}%" for tup in sorted({k: treatment_countpreds[k] for k in treatment_countpreds}.items(), key=lambda item: item[0])]
                counttargs           = [f"class {tup[0]} freq {tup[1]} perc {tup[1] / len(control_targs_all) * 100:.2f}%" for tup in sorted({k: counttargs[k] for k in counttargs}.items(), key=lambda item: item[0])]
                print(f"{control_cond + ' preds count:':<80} {control_countpreds}")
                print(f"{treatment_cond + ' preds count:':<80} {treatment_countpreds}")
                print(f"{'targs count:':<80} {counttargs}")
                print(f"{'control total F-measure':.<25} {tot_control_f1:<8} {'treatment total F-measure':.<30} {tot_treatment_f1:<8} {'diff':.<7} {tot_diff_f1}")
                print(f"{'control total accuracy':.<25} {tot_control_acc:<8} {'treatment total accuracy':.<30} {tot_treatment_acc:<8} {'diff':.<7} {tot_diff_acc}")
                print(f"{'control total precision':.<25} {tot_control_prec:<8} {'treatment total precision':.<30} {tot_treatment_prec:<8} {'diff':.<7} {tot_diff_prec}")
                print(f"{'control total recall':.<25} {tot_control_rec:<8} {'treatment total recall':.<30} {tot_treatment_rec:<8} {'diff':.<7} {tot_diff_rec}")

                tst_overall_size = len(targs_all)
                # estraggo l'equivalente di un esperimento. Più è piccolo il numero, più è facile avere significatività. In altre parole, più esperimenti si fanno più è facile
                samplesize = int(len(targs_all) * perc_sample)
                print(f"{'tot experiments size':.<25} {tst_overall_size}\n{'sample size':.<25} {samplesize}")
                twice_diff_acc  = 0
                twice_diff_f1   = 0
                twice_diff_prec = 0
                twice_diff_rec  = 0
                for loop in tqdm(range(n_loops), desc='bootstrap', ncols=80):
                    i_sample = np.random.choice(range(tst_overall_size), size=samplesize, replace=True)
                    sample_control_preds   = [control_preds_all[i]   for i in i_sample]
                    sample_treatment_preds = [treatment_preds_all[i] for i in i_sample]
                    sample_targs           = [targs_all[i]           for i in i_sample]
                    _, _, sample_diff_acc, _, _, sample_diff_f1, _, _, sample_diff_prec, _, _, sample_diff_rec = metrics(sample_targs, sample_control_preds, sample_treatment_preds)
                    if sample_diff_acc   > 2 * tot_diff_acc:   twice_diff_acc  += 1
                    if sample_diff_f1    > 2 * tot_diff_f1:    twice_diff_f1   += 1
                    if sample_diff_prec  > 2 * tot_diff_prec:  twice_diff_prec += 1
                    if sample_diff_rec   > 2 * tot_diff_rec:   twice_diff_rec  += 1
                sign_f1   = '**' if twice_diff_f1   / n_loops < 0.01 else '*' if twice_diff_f1   / n_loops < 0.05 else ''
                sign_acc  = '**' if twice_diff_acc  / n_loops < 0.01 else '*' if twice_diff_acc  / n_loops < 0.05 else ''
                sign_prec = '**' if twice_diff_prec / n_loops < 0.01 else '*' if twice_diff_prec / n_loops < 0.05 else ''
                sign_rec  = '**' if twice_diff_rec  / n_loops < 0.01 else '*' if twice_diff_rec  / n_loops < 0.05 else ''
                str_out = f"{'count sample diff f1   is twice tot diff f1':.<50} {twice_diff_f1:<5}/ {n_loops:<8}p < {round((twice_diff_f1 / n_loops), 4):<6} {ut.bcolors.red}{sign_f1  }{ut.bcolors.reset}\n" \
                          f"{'count sample diff acc  is twice tot diff acc':.<50} {twice_diff_acc:<5}/ {n_loops:<8}p < {round((twice_diff_acc / n_loops), 4):<6} {ut.bcolors.red}{sign_acc }{ut.bcolors.reset}\n" \
                          f"{'count sample diff prec is twice tot diff prec':.<50} {twice_diff_prec:<5}/ {n_loops:<8}p < {round((twice_diff_prec / n_loops), 4):<6} {ut.bcolors.red}{sign_prec}{ut.bcolors.reset}\n" \
                          f"{'count sample diff rec  is twice tot diff rec ':.<50} {twice_diff_rec:<5}/ {n_loops:<8}p < {round((twice_diff_rec / n_loops), 4):<6} {ut.bcolors.red}{sign_rec }{ut.bcolors.reset}"
                print(str_out)
                if control_cond not in df.index:
                    df = df.append(pd.Series({"mean_epochs": round(np.mean(self.data[control_cond]['epochs']), 2),                              "prec": tot_control_prec,   "diff_prec": None,          "sign_prec": None,      "rec": tot_control_rec,   "diff_rec": None,         "sign_rec": None,     "acc": tot_control_acc,   "diff_acc": None,         "sign_acc": None,     "f1": tot_control_f1,   "diff_f1": None,        "sign_f1": None}, name=control_cond))
                if treatment_cond not in df.index:
                    df = df.append(pd.Series({"mean_epochs": round(np.mean(self.data[control_cond]['treatment'][treatment_cond]['epochs']), 2), "prec": tot_treatment_prec, "diff_prec": tot_diff_prec, "sign_prec": sign_prec, "rec": tot_treatment_rec, "diff_rec": tot_diff_rec, "sign_rec": sign_rec, "acc": tot_treatment_acc, "diff_acc": tot_diff_acc, "sign_acc": sign_acc, "f1": tot_treatment_f1, "diff_f1": tot_diff_f1, "sign_f1": sign_f1}, name=treatment_cond))
            df.to_csv(f"{self.dirout}bootresults.csv")
            ut.writejson(self.data, f"{self.dirout}bootstrap.json")
            # df = pd.read_csv(f"{self.dirout}bootresults.csv", index_col=0)
            print(df)
            ut.sendslack(df.to_string())
            ut.end(startime)
        return df


