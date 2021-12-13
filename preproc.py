# coding=latin-1
import util201217 as ut
import step210125 as st
import argparse, os, re, sys, time
from unidecode import unidecode
import demoji
import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from collections import defaultdict, Counter
from scipy import special
from scipy.sparse import csr_matrix, save_npz, load_npz
import pandas as pd
import random
import torch
import torch.nn as nn
from torch import optim
###################################################################################################
parser = argparse.ArgumentParser()
# inputs
parser.add_argument("-path",        type=str,   default='0_source/jupyter_xsl_preproc_210130170501_min20char/all210126.xlsx')
parser.add_argument("-device",      type=str,   default='cuda:1')
parser.add_argument("-bertlang",    type=str,   default='enlarge')
parser.add_argument("-bertlayer",   type=int,   default=24)
parser.add_argument("-min_freq",    type=int,   default=0)
parser.add_argument("-no_below",    type=float, default=.0005, help='min word freq')
parser.add_argument("-no_above",    type=float, default=.9,    help='max % of docs where the word is found')
parser.add_argument("-pad_rate",    type=float, default=99.9,  help='rate pad wrt maxlen, per fasttext')
parser.add_argument("-max_context", type=int,  default=2)
# parser.add_argument("-pad_size", type=int,  default = 510,  help='per bert')
args = parser.parse_args()
sys.stdout = sys.stderr = log = ut.log(__file__, f"")
os.system(f"cp {__file__} {ut.__file__} {log.pathtime}")
startime = ut.start()
align_size = ut.print_args(args)
print(f"{'dirout':.<{align_size}} {log.pathtime}")
###################################################################################################
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"{'GPU in use':.<{align_size}} {device}\n{'#'*80}" if torch.cuda.is_available() else f"No GPU available, using the CPU.\n{'#'*80}")
###################################################################################################
df_all = pd.read_excel(args.path)
print(df_all)
print(Counter(df_all.binary_pledge))
print(Counter(df_all.party))
print(Counter(df_all.year))
pad_size = int(round(np.percentile(df_all.len, args.pad_rate), -1))

proc = st.Processing(log.pathtime, device)
proc.bert_lookup(lang='ml', hiddenlayer=12, X1=df_all.clean_text, padsize=pad_size, matrix=False, splits=(None, None), name='both')
for context in range(1, args.max_context + 1):
    all_context = [' '.join(df_all.clean_text[i-sum(df_all.manifesto_id[max(0, i-context): i] < df_all.manifesto_id[i]): i].tolist() + ['[PAD]'] * (context - len(df_all.clean_text[i-sum(df_all.manifesto_id[max(0, i-context): i] < df_all.manifesto_id[i]): i].tolist()))) for i in range(df_all.shape[0])]
    _, _ = proc.bert_lookup(lang='ml',      hiddenlayer=12, X1=df_all.clean_text, X2=all_context, padsize=pad_size * (context + 1), matrix=False, name=f'cont{context}_both')

df_swe = df_all[df_all.country == 'Sweden']
print(df_swe)
print(Counter(df_swe.binary_pledge))
print(Counter(df_swe.party))
print(Counter(df_swe.year))
df_ind = df_all[df_all.country == 'India']
print(df_ind)
print(Counter(df_ind.binary_pledge))
print(Counter(df_ind.party))
print(Counter(df_ind.year))


def single_proc(df, fastlang, bertlang, bertlayer, name):
    padsize = proc.fastext(df.clean_text, lang=fastlang, minfreq=args.min_freq, nobelow=args.no_below, noabove=args.no_above, padrate=args.pad_rate, splits=(None, None), name=name)
    proc.bert_lookup(lang=bertlang, hiddenlayer=bertlayer, X1=df.clean_text, padsize=padsize, matrix=False, splits=(None, None), name=name)
    for cont in range(1, args.max_context + 1):
        df_context = [' '.join(df.clean_text[i-sum(df.manifesto_id.values[max(0, i-cont): i] < df.manifesto_id.values[i]): i].tolist() + ['[PAD]'] * (cont - len(df.clean_text[i-sum(df.manifesto_id.values[max(0, i-cont): i] < df.manifesto_id.values[i]): i].tolist()))) for i in range(df.shape[0])]
        _, _ = proc.bert_lookup(lang=bertlang, hiddenlayer=bertlayer, X1=df.clean_text, X2=df_context, padsize=padsize * (cont + 1), matrix=False, name=f'cont{cont}_{name}')
    
    print(df.binary_pledge.shape)
    np.save(f"{log.pathtime}{name}_pledge", df.binary_pledge)
    
    labencoder = LabelEncoder()
    
    party = labencoder.fit_transform(df.party)
    id2party = {i: l for i, l in enumerate(labencoder.classes_)}
    ut.writejson(id2party, f"{log.pathtime}id2party.json")
    print(id2party)
    print(party.shape)
    np.save(f"{log.pathtime}{name}_party", party)

    year = labencoder.fit_transform(df.year)
    id2year = {i: str(l) for i, l in enumerate(labencoder.classes_)} # al json non piace l'anno come int
    ut.writejson(id2year, f"{log.pathtime}id2year.json")
    print(id2year)
    print(year.shape)
    np.save(f"{log.pathtime}{name}_year", year)
    return 1
    
    
single_proc(df_swe, 'sw', 'sw', 12, 'swed')
single_proc(df_swe, 'sw', 'ml', 12, 'swed')
single_proc(df_ind, 'en', 'enlarge', 24, 'indi')
single_proc(df_ind, 'en', 'ml', 12, 'indi')

ut.end(startime)