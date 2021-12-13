# coding=latin-1
import util201217 as ut
import step210125 as st
# import models201114 as mod
import models210116 as mod
import argparse, re, sys, os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_fscore_support, log_loss, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, Counter
import pandas as pd
import random
import torch
import torch.nn as nn
from torch import optim
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
# simplefilter(action='ignore', category=(FutureWarning, UserWarning))
parser = argparse.ArgumentParser()
# torch settings
parser.add_argument("-seed",   type=int, default=1234)
parser.add_argument("-device", type=str, default='cuda:1')
parser.add_argument("-dtype",  type=int, default=32, choices=[32, 64])
# inputs
# parser.add_argument("-path_xls", type=str, default='0_source/jupyter_xsl_preproc_210129084738_min10char/all210126.xlsx')
# parser.add_argument("-dir_data", type=str, default='1_preproc/210129085128_to_min10char/')
# parser.add_argument("-path_xls", type=str, default='0_source/jupyter_xsl_preproc_210130170501_min20char/all210126.xlsx')
# parser.add_argument("-dir_data", type=str, default='1_preproc/210130171552_to_min20char/')
parser.add_argument("-path_xls", type=str, default='2_preproc_subset/ge2000_210131181851/all210126.xlsx')
parser.add_argument("-dir_data", type=str, default='2_preproc_subset/ge2000_210131181851/')
# parser.add_argument("-path_xls", type=str, default='2_preproc_subset/ge2010_210131184405/all210126.xlsx')
# parser.add_argument("-dir_data", type=str, default='2_preproc_subset/ge2010_210131184405/')
parser.add_argument("-bert_lookup_kind",   type=str, default='cls', choices=['cls', 'meanmat', 'mat'])
parser.add_argument("-bertcontexts", type=str, nargs='+', default=['single', 'paired_cont1'], help="['single', 'paired_cont1', 'paired_cont2']")
parser.add_argument("-contexts", type=int, nargs='+', default=[0, 1])
# preproc
# # model settings
parser.add_argument("-patience",    type=int,   default = 5)
parser.add_argument("-min_gain",    type=int,   default = 6)
parser.add_argument("-save",        type=bool,  default = False)
parser.add_argument("-nr_exps",     type=int,   default = 10)
parser.add_argument("-splits",      type=int,   default = 10, help='almeno 3 o dà un errore, credo dovuto all\'output dello stratified')
parser.add_argument("-big_batsize",     type=int,   default = 512)
parser.add_argument("-small_batsize",   type=int,   default = 512)
parser.add_argument("-high_learate",    type=float, default = 0.002)
parser.add_argument("-low_learate",     type=float, default = 0.002)
parser.add_argument("-droprob",     type=float, default = 0.3)
parser.add_argument("-trainable",   type=bool,  default = False)
# # attention
parser.add_argument("-att_heads",      type=int, default = 1)
parser.add_argument("-att_layers",     type=int, default = 1)
parser.add_argument("-txt_fc_outsize", type=int, default = 10)
parser.add_argument("-doc_fc_outsize", type=int, default = 10)
# fc settings
parser.add_argument("-fc_layers", type=int, default=1)
# conv settings
parser.add_argument("-conv_channels",     type=int, nargs='+', default=[32, 64], help="nr of channels conv by conv")
parser.add_argument("-conv_filter_sizes", type=int, nargs='+', default=[2, 3],   help="sizes of filters: window, in each conv")
parser.add_argument("-conv_stridesizes",  type=int, nargs='+', default=[1, 1],   help="conv stride size, conv by conv")
parser.add_argument("-pool_filtersizes",  type=int, nargs='+', default=[2, 2],   help="pool filter size, conv by conv. in order to have a vector as output, the last value will be substituted with the column size of the last conv, so that the last column size will be 1, then squeezed")
parser.add_argument("-pool_stridesizes",  type=int, nargs='+', default=[1, 1],   help="pool stride size, conv by conv")
# bootstrap
parser.add_argument("-n_short_loops", type=int,   default=100)
parser.add_argument("-n_loops",       type=int,   default=1000)
parser.add_argument("-perc_sample",   type=float, default=.3)

args = parser.parse_args()
sys.stdout = sys.stderr = log = ut.log(__file__, f"exp{args.nr_exps}_splits{args.splits}_bat{args.big_batsize}_{args.bert_lookup_kind}_lr{str(args.high_learate)[2:]}_drop{str(args.droprob)[2:]}_pat{args.patience}gain{args.min_gain}_ge2000")
os.system(f"cp {__file__} {ut.__file__} {st.__file__} {mod.__file__} {log.pathtime}")
startime = ut.start()
align_size = ut.print_args(args)
print(f"{'dirout':.<{align_size}} {log.pathtime}")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
dtype_float = torch.float64 if args.dtype == 64 else torch.float32
dtype_int = torch.int64 # if args.dtype == 64 else torch.int32 # o 64 o s'inkazza
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"{'GPU in use':.<{align_size}} {device}\n{'#'*80}" if torch.cuda.is_available() else f"No GPU available, using the CPU.\n{'#'*80}")
# swed sw cv (fastext)
# indi enlarge cv (fastext)
# both ml cv
# both ml ho swed vs indi
# both ml ho indi vs swe

# trn swed sw/sw      swed cv         fastext
#          ml         swed cv
#                     indi ho (0shot)
#     indi enlarge/en indi cv         fastext
#          ml         indi cv
#                     swed ho (0shot)
#     both ml         both cv
short2country = {'swed': 'Sweden', 'indi': 'India'}
# trn2zeroshotst = {'swed': 'indi', 'indi': 'swed', 'both': 'both'}
bce  = nn.BCELoss().to(device=device)
ce   = nn.CrossEntropyLoss().to(device=device) # non ammette target float
df_all = pd.read_excel(args.path_xls)
print(df_all)


def build_context_targets_tensor(matrix, empty_row, context_size, categories, name=''):
    context_targets = torch.stack([
                          torch.stack([empty_row if (icontext < 0) or (categories[icontext] != categories[i]) else # empty_row se siamo all'inizio del dataset o di un doc
                                       matrix[icontext] # altrimenti il vettore di lookup
                          for icontext in range(i - context_size, i+1)]) # dal primo contesto al target
                      for i in range(matrix.shape[0])]) # per ogni target
    print(f"{name:<50}{context_targets.shape}")
    return context_targets


def run(short_trn, bertlang, trainlang):
    print(f"{'#'*80}\n{short_trn} {bertlang}")
    condition = f"{short_trn}_{bertlang}{trainlang}"
    
    pledge_all = np.load(f"{args.dir_data}{short_trn}_pledge.npy") if short_trn != 'both' else np.concatenate((np.load(f"{args.dir_data}indi_pledge.npy"), np.load(f"{args.dir_data}swed_pledge.npy")), axis=0)
    party_all = np.load(f"{args.dir_data}{short_trn}_party.npy") if short_trn != 'both' else np.concatenate((np.load(f"{args.dir_data}indi_pledge.npy"), np.load(f"{args.dir_data}swed_pledge.npy")), axis=0)
    year_all = np.load(f"{args.dir_data}{short_trn}_year.npy") if short_trn != 'both' else np.concatenate((np.load(f"{args.dir_data}indi_pledge.npy"), np.load(f"{args.dir_data}swed_pledge.npy")), axis=0)
    print(f"{short_trn} {'pledge_all':<50}{pledge_all.shape}")
    print(f"{short_trn} {'party_all':<50}{party_all.shape}")
    print(f"{short_trn} {'year_all':<50}{year_all.shape}")
    party_size_all = len(set(party_all))
    year_size_all = len(set(year_all))
    
    if (short_trn != 'both') and (bertlang == 'ml') and (trainlang == 'both'): # 0shot
        short_tst = 'indi' if short_trn == 'swed' else 'swed'
        print(f"{'#'*60}\n{short_trn} vs. {short_tst}")
        df_trn = df_all[df_all.country == short2country[short_trn]]
        df_trn.reset_index(inplace=True, drop=True)
        party_size_trn = len(set(df_trn.party))
        year_size_trn = len(set(df_trn.year))

        i_trn = df_trn.index[df_trn.set == 'trn']
        i_dev = df_trn.index[df_trn.set == 'dev']
        pledge_trn = pledge_all[i_trn]
        pledge_dev = pledge_all[i_dev]
        pledge_tst = np.load(f"{args.dir_data}{short_tst}_pledge.npy") # serve solo per la lunghezza di y, dal momento che il test potrebbe contenere un numero diverso di labels (e deve essere < del trn)
        print(f"{short_trn} {'pledge_trn':<50}{pledge_trn.shape}")
        print(f"{short_trn} {'pledge_dev':<50}{pledge_dev.shape}")
        print(f"{short_tst} {'pledge_tst':<50}{pledge_tst.shape}")

        party_trn = party_all[i_trn]
        party_dev = party_all[i_dev]
        party_tst = np.array([0] * pledge_tst.shape[0])
        # party_tst = np.load(f"{args.dir_data}{short_tst}_party.npy")
        # party_tst = np.where(party_tst >= party_size_trn, 0, party_tst)
        print(f"{short_trn} {'party_trn':<50}{party_trn.shape}")
        print(f"{short_trn} {'party_dev':<50}{party_dev.shape}")
        print(f"{short_tst} {'party_tst':<50}{party_tst.shape}")

        year_trn = year_all[i_trn]
        year_dev = year_all[i_dev]
        year_tst = np.array([0] * pledge_tst.shape[0])
        # year_tst = np.load(f"{args.dir_data}{short_tst}_year.npy")
        # year_tst = np.where(year_tst >= year_size_trn, 0, year_tst)
        print(f"{short_trn} {'year_trn':<50}{year_trn.shape}")
        print(f"{short_trn} {'year_dev':<50}{year_dev.shape}")
        print(f"{short_tst} {'year_tst':<50}{year_tst.shape}")


        def bertvec_trans_mtl_cross():
            dirout = f"{log.pathtime}{condition}_mtl_bert{args.bert_lookup_kind}_trans_0shot2{short_tst}/"
            os.mkdir(dirout)
            print(f"{'#'*80}\n{dirout}")
            proc = st.Processing(dirout, device)
            boot = st.Bootstrap(dirout)

            file_lookup = f"bert_{bertlang}_single_{trainlang}_lookup_{args.bert_lookup_kind}.pt"
            lookup = torch.load(f"{args.dir_data}{file_lookup}").to(device=device)
            lookup_trn = lookup[-df_trn.shape[0]:][i_trn] if short_trn == 'swed' else lookup[:df_trn.shape[0]][i_trn]
            lookup_dev = lookup[-df_trn.shape[0]:][i_dev] if short_trn == 'swed' else lookup[:df_trn.shape[0]][i_dev]
            lookup_tst = lookup[:-df_trn.shape[0]] if short_trn == 'swed' else lookup[df_trn.shape[0]:]
            print(f"{file_lookup[:-3]:<50}{lookup.shape}")
            print(f"{short_trn + ' trn':<50}{lookup_trn.shape}")
            print(f"{short_trn + ' dev':<50}{lookup_dev.shape}")
            print(f"{short_tst + ' tst':<50}{lookup_tst.shape}")
            # bert_ids = torch.load(f"{args.dir_data}bert_{bertlang}_{bertcontext}_{short_trn}_ids.pt").to(device=device, dtype=dtype_float)
            # print(f"{short_trn} {'bert_ids':<50}{bert_ids.shape}")
            # bert_mask = torch.load(f"{args.dir_data}bert_{bertlang}_{bertcontext}_{short_trn}_masks.pt").to(device=device, dtype=dtype_float)
            # print(f"{short_trn} {'bert_mask':<50}{bert_mask.shape}")
            
            for exp in range(1, args.nr_exps + 1):
                direxp  = f"exp{exp}"
                model = mod.StlVecTransFc(emb_size   = lookup.shape[1],
                                          y0_size    = 1,
                                          att_heads  = args.att_heads,
                                          att_layers = args.att_layers,
                                          fc_layers  = args.fc_layers,
                                          droprob    = args.droprob,
                                          device     = device)
                optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                             optimizer       = optimizer,
                                                             lossfuncs       = [bce],
                                                             x_inputs        = [lookup_trn],
                                                             x_inputs_dev    = [lookup_dev],
                                                             x_inputs_tst    = [lookup_tst],
                                                             x_dtypes        = [dtype_float],
                                                             y_inputs        = [pledge_trn],
                                                             y_inputs_dev    = [pledge_dev],
                                                             y_inputs_tst    = [pledge_tst],
                                                             y_dtypes        = [dtype_float],
                                                             batsize         = args.big_batsize,
                                                             patience        = args.patience,
                                                             min_gain        = args.min_gain,
                                                             n_splits        = args.splits, # solo crossval, nr folds
                                                             save            = args.save,
                                                             additional_tsts = (),
                                                             str_info        = direxp)
                control = f"{model.__class__.__name__}"
                boot.feed(control=control, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)

                info = f"party"
                direxp  = f"exp{exp}_{info}"
                model = mod.Mtl1VecTransFc(emb_size   = lookup.shape[1],
                                           y0_size    = 1,
                                           y1_size    = party_size_trn,
                                           att_heads  = args.att_heads,
                                           att_layers = args.att_layers,
                                           fc_layers  = args.fc_layers,
                                           droprob    = args.droprob,
                                           device     = device)
                optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                             optimizer       = optimizer,
                                                             lossfuncs       = [bce, ce],
                                                             x_inputs        = [lookup_trn],
                                                             x_inputs_dev    = [lookup_dev],
                                                             x_inputs_tst    = [lookup_tst],
                                                             x_dtypes        = [dtype_float],
                                                             y_inputs        = [pledge_trn, party_trn],
                                                             y_inputs_dev    = [pledge_dev, party_dev],
                                                             y_inputs_tst    = [pledge_tst, party_tst],
                                                             y_dtypes        = [dtype_float, dtype_int],
                                                             batsize         = args.big_batsize,
                                                             patience        = args.patience,
                                                             min_gain        = args.min_gain,
                                                             n_splits        = args.splits, # solo crossval, nr folds
                                                             save            = args.save,
                                                             additional_tsts = (),
                                                             str_info        = direxp)
                treatment = f"{model.__class__.__name__}_{info}"
                boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)

                info = f"year"
                direxp  = f"exp{exp}_{info}"
                model = mod.Mtl1VecTransFc(emb_size   = lookup.shape[1],
                                           y0_size    = 1,
                                           y1_size    = year_size_trn,
                                           att_heads  = args.att_heads,
                                           att_layers = args.att_layers,
                                           fc_layers  = args.fc_layers,
                                           droprob    = args.droprob,
                                           device     = device)
                optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                             optimizer       = optimizer,
                                                             lossfuncs       = [bce, ce],
                                                             x_inputs        = [lookup_trn],
                                                             x_inputs_dev    = [lookup_dev],
                                                             x_inputs_tst    = [lookup_tst],
                                                             x_dtypes        = [dtype_float],
                                                             y_inputs        = [pledge_trn, year_trn],
                                                             y_inputs_dev    = [pledge_dev, year_dev],
                                                             y_inputs_tst    = [pledge_tst, year_tst],
                                                             y_dtypes        = [dtype_float, dtype_int],
                                                             batsize         = args.big_batsize,
                                                             patience        = args.patience,
                                                             min_gain        = args.min_gain,
                                                             n_splits        = args.splits, # solo crossval, nr folds
                                                             save            = args.save,
                                                             additional_tsts = (),
                                                             str_info        = direxp)
                treatment = f"{model.__class__.__name__}_{info}"
                boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)

                info = f"partyear"
                direxp  = f"exp{exp}_{info}"
                model = mod.Mtl2VecTransFc(emb_size   = lookup.shape[1],
                                           y0_size    = 1,
                                           y1_size    = party_size_trn,
                                           y2_size    = year_size_trn,
                                           att_heads  = args.att_heads,
                                           att_layers = args.att_layers,
                                           fc_layers  = args.fc_layers,
                                           droprob    = args.droprob,
                                           device     = device)
                optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                             optimizer       = optimizer,
                                                             lossfuncs       = [bce, ce, ce],
                                                             x_inputs        = [lookup_trn],
                                                             x_inputs_dev    = [lookup_dev],
                                                             x_inputs_tst    = [lookup_tst],
                                                             x_dtypes        = [dtype_float],
                                                             y_inputs        = [pledge_trn, party_trn, year_trn],
                                                             y_inputs_dev    = [pledge_dev, party_dev, year_dev],
                                                             y_inputs_tst    = [pledge_tst, party_tst, year_tst],
                                                             y_dtypes        = [dtype_float, dtype_int, dtype_int],
                                                             batsize         = args.big_batsize,
                                                             patience        = args.patience,
                                                             min_gain        = args.min_gain,
                                                             n_splits        = args.splits, # solo crossval, nr folds
                                                             save            = args.save,
                                                             additional_tsts = (),
                                                             str_info        = direxp)
                treatment = f"{model.__class__.__name__}_{info}"
                boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)

            boot.run(args.n_loops, args.perc_sample)
            return 1
    
        bertvec_trans_mtl_cross()

        def bertvec_trans_bertpair_cross():
            dirout = f"{log.pathtime}{condition}_pair_bert{args.bert_lookup_kind}_trans_0shot2{short_tst}/"
            os.mkdir(dirout)
            print(f"{'#'*80}\n{dirout}")
            proc = st.Processing(dirout, device)
            boot = st.Bootstrap(dirout)

            for bertcontext in args.bertcontexts:
                file_lookup = f"bert_ml_{bertcontext}_{trainlang}_lookup_{args.bert_lookup_kind}.pt"
                lookup = torch.load(f"{args.dir_data}{file_lookup}").to(device=device)
                lookup_trn = lookup[-df_trn.shape[0]:][i_trn] if short_trn == 'swed' else lookup[:df_trn.shape[0]][i_trn]
                lookup_dev = lookup[-df_trn.shape[0]:][i_dev] if short_trn == 'swed' else lookup[:df_trn.shape[0]][i_dev]
                lookup_tst = lookup[:-df_trn.shape[0]] if short_trn == 'swed' else lookup[df_trn.shape[0]:]
                print(f"{file_lookup[:-3]:<50}{lookup.shape}")
                print(f"{short_trn + ' trn':<50}{lookup_trn.shape}")
                print(f"{short_trn + ' dev':<50}{lookup_dev.shape}")
                print(f"{short_tst + ' tst':<50}{lookup_tst.shape}")
                control = f"single_StlVecTransFc"
                # bert_ids = torch.load(f"{args.dir_data}bert_{bertlang}_{bertcontext}_{short_trn}_ids.pt").to(device=device, dtype=dtype_float)
                # print(f"{short_trn} {'bert_ids':<50}{bert_ids.shape}")
                # bert_mask = torch.load(f"{args.dir_data}bert_{bertlang}_{bertcontext}_{short_trn}_masks.pt").to(device=device, dtype=dtype_float)
                # print(f"{short_trn} {'bert_mask':<50}{bert_mask.shape}")

                for exp in range(1, args.nr_exps + 1):
                    direxp  = f"exp{exp}_{bertcontext}"
                    model = mod.StlVecTransFc(emb_size   = lookup.shape[1],
                                              y0_size    = 1,
                                              att_heads  = args.att_heads,
                                              att_layers = args.att_layers,
                                              fc_layers  = args.fc_layers,
                                              droprob    = args.droprob,
                                              device     = device)
                    optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                    dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                                 optimizer       = optimizer,
                                                                 lossfuncs       = [bce],
                                                                 x_inputs        = [lookup_trn],
                                                                 x_inputs_dev    = [lookup_dev],
                                                                 x_inputs_tst    = [lookup_tst],
                                                                 x_dtypes        = [dtype_float],
                                                                 y_inputs        = [pledge_trn],
                                                                 y_inputs_dev    = [pledge_dev],
                                                                 y_inputs_tst    = [pledge_tst],
                                                                 y_dtypes        = [dtype_float],
                                                                 batsize         = args.big_batsize,
                                                                 patience        = args.patience,
                                                                 min_gain        = args.min_gain,
                                                                 n_splits        = args.splits, # solo crossval, nr folds
                                                                 save            = args.save,
                                                                 additional_tsts = (),
                                                                 str_info        = direxp)
                    if bertcontext == 'single':
                        boot.feed(control=control, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
                    else:
                        treatment = f"{bertcontext}_{model.__class__.__name__}"
                        boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)

            boot.run(args.n_short_loops, args.perc_sample)
            return 1
    
        # bertvec_trans_bertpair_cross()

        def bertvec_trans_context_cross():
            dirout = f"{log.pathtime}{condition}_context_bert{args.bert_lookup_kind}_trans_0shot2{short_tst}/"
            os.mkdir(dirout)
            print(f"{'#'*80}\n{dirout}")
            proc = st.Processing(dirout, device)
            boot = st.Bootstrap(dirout)

            file_lookup = f"bert_ml_single_{trainlang}_lookup_{args.bert_lookup_kind}.pt"
            lookup = torch.load(f"{args.dir_data}{file_lookup}").to(device=device)
            print(f"{file_lookup[:-3]:<50}{lookup.shape}")

            mean_tensor = lookup.mean(0)
            for context in args.contexts:
                context_targets = build_context_targets_tensor(lookup, mean_tensor, context, df_all.file, 'context_targets')
                context_targets_trn = context_targets[-df_trn.shape[0]:][i_trn] if short_trn == 'swed' else context_targets[:df_trn.shape[0]][i_trn]
                context_targets_dev = context_targets[-df_trn.shape[0]:][i_dev] if short_trn == 'swed' else context_targets[:df_trn.shape[0]][i_dev]
                context_targets_tst = context_targets[:-df_trn.shape[0]] if short_trn == 'swed' else context_targets[df_trn.shape[0]:]
                print(f"{'context_targets':<50}{context_targets.shape}")
                print(f"{'context_targets_trn':<50}{context_targets_trn.shape}")
                print(f"{'context_targets_dev':<50}{context_targets_dev.shape}")
                print(f"{'context_targets_tst':<50}{context_targets_tst.shape}")

                control = f"StlVecHierTransCat_cont0"
                # bert_ids = torch.load(f"{args.dir_data}bert_{bertlang}_{bertcontext}_{short_trn}_ids.pt").to(device=device, dtype=dtype_float)
                # print(f"{short_trn} {'bert_ids':<50}{bert_ids.shape}")
                # bert_mask = torch.load(f"{args.dir_data}bert_{bertlang}_{bertcontext}_{short_trn}_masks.pt").to(device=device, dtype=dtype_float)
                # print(f"{short_trn} {'bert_mask':<50}{bert_mask.shape}")

                for exp in range(1, args.nr_exps + 1):
                    direxp  = f"exp{exp}_{context}"
                    model = mod.StlVecHierTransCat(cont_size  = context_targets.shape[1],
                                                   emb_size   = lookup.shape[1],
                                                   y0_size    = 1,
                                                   att_heads  = args.att_heads,
                                                   att_layers = args.att_layers,
                                                   fc_layers  = args.fc_layers,
                                                   droprob    = args.droprob,
                                                   device     = device)
                    optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                    dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                                 optimizer       = optimizer,
                                                                 lossfuncs       = [bce],
                                                                 x_inputs        = [context_targets_trn],
                                                                 x_inputs_dev    = [context_targets_dev],
                                                                 x_inputs_tst    = [context_targets_tst],
                                                                 x_dtypes        = [dtype_float],
                                                                 y_inputs        = [pledge_trn],
                                                                 y_inputs_dev    = [pledge_dev],
                                                                 y_inputs_tst    = [pledge_tst],
                                                                 y_dtypes        = [dtype_float],
                                                                 batsize         = args.big_batsize,
                                                                 patience        = args.patience,
                                                                 min_gain        = args.min_gain,
                                                                 n_splits        = args.splits, # solo crossval, nr folds
                                                                 save            = args.save,
                                                                 additional_tsts = (),
                                                                 str_info        = direxp)
                    if context == 0:
                        boot.feed(control=control, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
                    else:
                        treatment = f"{model.__class__.__name__}_cont{context}"
                        boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)

            boot.run(args.n_short_loops, args.perc_sample)
            return 1
    
        # bertvec_trans_context_cross()


    else:

        def bertvec_trans_mtl():
            dirout = f"{log.pathtime}{condition}_mtl_bert{args.bert_lookup_kind}_trans/"
            os.mkdir(dirout)
            print(f"{'#'*80}\n{dirout}")
            proc = st.Processing(dirout, device)
            boot = st.Bootstrap(dirout)
    
            file_lookup = f"bert_{bertlang}_single_{trainlang}_lookup_{args.bert_lookup_kind}.pt"
            lookup = torch.load(f"{args.dir_data}{file_lookup}").to(device=device)
            print(f"{file_lookup[:-3]:<50}{lookup.shape}")
            # bert_ids = torch.load(f"{args.dir_data}bert_{bertlang}_{bertcontext}_{short_trn}_ids.pt").to(device=device, dtype=dtype_float)
            # print(f"{short_trn} {'bert_ids':<50}{bert_ids.shape}")
            # bert_mask = torch.load(f"{args.dir_data}bert_{bertlang}_{bertcontext}_{short_trn}_masks.pt").to(device=device, dtype=dtype_float)
            # print(f"{short_trn} {'bert_mask':<50}{bert_mask.shape}")
    
            for exp in range(1, args.nr_exps + 1):
                direxp  = f"exp{exp}"
                model = mod.StlVecTransFc(emb_size   = lookup.shape[1],
                                          y0_size    = 1,
                                          att_heads  = args.att_heads,
                                          att_layers = args.att_layers,
                                          fc_layers  = args.fc_layers,
                                          droprob    = args.droprob,
                                          device     = device)
                optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                             optimizer       = optimizer,
                                                             lossfuncs       = [bce],
                                                             x_inputs        = [lookup],
                                                             x_inputs_dev    = [None],
                                                             x_inputs_tst    = [None],
                                                             x_dtypes        = [dtype_float],
                                                             y_inputs        = [pledge_all],
                                                             y_inputs_dev    = [None],
                                                             y_inputs_tst    = [None],
                                                             y_dtypes        = [dtype_float],
                                                             batsize         = args.big_batsize,
                                                             patience        = args.patience,
                                                             min_gain        = args.min_gain,
                                                             n_splits        = args.splits, # solo crossval, nr folds
                                                             save            = args.save,
                                                             additional_tsts = (),
                                                             str_info        = direxp)
                control = f"{model.__class__.__name__}"
                boot.feed(control=control, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
    
                info = f"party"
                direxp  = f"exp{exp}_{info}"
                model = mod.Mtl1VecTransFc(emb_size   = lookup.shape[1],
                                           y0_size    = 1,
                                           y1_size    = party_size_all,
                                           att_heads  = args.att_heads,
                                           att_layers = args.att_layers,
                                           fc_layers  = args.fc_layers,
                                           droprob    = args.droprob,
                                           device     = device)
                optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                             optimizer       = optimizer,
                                                             lossfuncs       = [bce, ce],
                                                             x_inputs        = [lookup],
                                                             x_inputs_dev    = [None],
                                                             x_inputs_tst    = [None],
                                                             x_dtypes        = [dtype_float],
                                                             y_inputs        = [pledge_all, party_all],
                                                             y_inputs_dev    = [None],
                                                             y_inputs_tst    = [None],
                                                             y_dtypes        = [dtype_float, dtype_int],
                                                             batsize         = args.big_batsize,
                                                             patience        = args.patience,
                                                             min_gain        = args.min_gain,
                                                             n_splits        = args.splits, # solo crossval, nr folds
                                                             save            = args.save,
                                                             additional_tsts = (),
                                                             str_info        = direxp)
                treatment = f"{model.__class__.__name__}_{info}"
                boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
    
                info = f"year"
                direxp  = f"exp{exp}_{info}"
                model = mod.Mtl1VecTransFc(emb_size   = lookup.shape[1],
                                           y0_size    = 1,
                                           y1_size    = year_size_all,
                                           att_heads  = args.att_heads,
                                           att_layers = args.att_layers,
                                           fc_layers  = args.fc_layers,
                                           droprob    = args.droprob,
                                           device     = device)
                optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                             optimizer       = optimizer,
                                                             lossfuncs       = [bce, ce],
                                                             x_inputs        = [lookup],
                                                             x_inputs_dev    = [None],
                                                             x_inputs_tst    = [None],
                                                             x_dtypes        = [dtype_float],
                                                             y_inputs        = [pledge_all, year_all],
                                                             y_inputs_dev    = [None],
                                                             y_inputs_tst    = [None],
                                                             y_dtypes        = [dtype_float, dtype_int],
                                                             batsize         = args.big_batsize,
                                                             patience        = args.patience,
                                                             min_gain        = args.min_gain,
                                                             n_splits        = args.splits, # solo crossval, nr folds
                                                             save            = args.save,
                                                             additional_tsts = (),
                                                             str_info        = direxp)
                treatment = f"{model.__class__.__name__}_{info}"
                boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
    
                info = f"partyear"
                direxp  = f"exp{exp}_{info}"
                model = mod.Mtl2VecTransFc(emb_size   = lookup.shape[1],
                                           y0_size    = 1,
                                           y1_size    = party_size_all,
                                           y2_size    = year_size_all,
                                           att_heads  = args.att_heads,
                                           att_layers = args.att_layers,
                                           fc_layers  = args.fc_layers,
                                           droprob    = args.droprob,
                                           device     = device)
                optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                             optimizer       = optimizer,
                                                             lossfuncs       = [bce, ce, ce],
                                                             x_inputs        = [lookup],
                                                             x_inputs_dev    = [None],
                                                             x_inputs_tst    = [None],
                                                             x_dtypes        = [dtype_float],
                                                             y_inputs        = [pledge_all, party_all, year_all],
                                                             y_inputs_dev    = [None],
                                                             y_inputs_tst    = [None],
                                                             y_dtypes        = [dtype_float, dtype_int, dtype_int],
                                                             batsize         = args.big_batsize,
                                                             patience        = args.patience,
                                                             min_gain        = args.min_gain,
                                                             n_splits        = args.splits, # solo crossval, nr folds
                                                             save            = args.save,
                                                             additional_tsts = (),
                                                             str_info        = direxp)
                treatment = f"{model.__class__.__name__}_{info}"
                boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
    
            boot.run(args.n_loops, args.perc_sample)
            return 1
    
        bertvec_trans_mtl()
    
        def bertvec_trans_bertpair():
            dirout = f"{log.pathtime}{condition}_pair_bert_trans/"
            os.mkdir(dirout)
            print(f"{'#'*80}\n{dirout}")
            proc = st.Processing(dirout, device)
            boot = st.Bootstrap(dirout)
    
            for bertcontext in args.bertcontexts:
                file_lookup = f"bert_{bertlang}_{bertcontext}_{trainlang}_lookup_{args.bert_lookup_kind}.pt"
                lookup = torch.load(f"{args.dir_data}{file_lookup}").to(device=device)
                print(f"{file_lookup[:-3]:<50}{lookup.shape}")
                control = f"single_StlVecTransFc"
                # bert_ids = torch.load(f"{args.dir_data}bert_{bertlang}_{bertcontext}_{short_trn}_ids.pt").to(device=device, dtype=dtype_float)
                # print(f"{short_trn} {'bert_ids':<50}{bert_ids.shape}")
                # bert_mask = torch.load(f"{args.dir_data}bert_{bertlang}_{bertcontext}_{short_trn}_masks.pt").to(device=device, dtype=dtype_float)
                # print(f"{short_trn} {'bert_mask':<50}{bert_mask.shape}")
    
                for exp in range(1, args.nr_exps + 1):
                    direxp  = f"exp{exp}_{bertcontext}"
                    model = mod.StlVecTransFc(emb_size   = lookup.shape[1],
                                              y0_size    = 1,
                                              att_heads  = args.att_heads,
                                              att_layers = args.att_layers,
                                              fc_layers  = args.fc_layers,
                                              droprob    = args.droprob,
                                              device     = device)
                    optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                    dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                                 optimizer       = optimizer,
                                                                 lossfuncs       = [bce],
                                                                 x_inputs        = [lookup],
                                                                 x_inputs_dev    = [None],
                                                                 x_inputs_tst    = [None],
                                                                 x_dtypes        = [dtype_float],
                                                                 y_inputs        = [pledge_all],
                                                                 y_inputs_dev    = [None],
                                                                 y_inputs_tst    = [None],
                                                                 y_dtypes        = [dtype_float],
                                                                 batsize         = args.big_batsize,
                                                                 patience        = args.patience,
                                                                 min_gain        = args.min_gain,
                                                                 n_splits        = args.splits, # solo crossval, nr folds
                                                                 save            = args.save,
                                                                 additional_tsts = (),
                                                                 str_info        = direxp)
                    if bertcontext == 'single':
                        boot.feed(control=control, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
                    else:
                        treatment = f"{bertcontext}_{model.__class__.__name__}"
                        boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
    
            boot.run(args.n_short_loops, args.perc_sample)
            return 1
    
        # bertvec_trans_bertpair()
    
        def bertvec_trans_context():
            dirout = f"{log.pathtime}{condition}_context_bert_trans/"
            os.mkdir(dirout)
            print(f"{'#'*80}\n{dirout}")
            proc = st.Processing(dirout, device)
            boot = st.Bootstrap(dirout)
    
            file_lookup = f"bert_{bertlang}_single_{trainlang}_lookup_{args.bert_lookup_kind}.pt"
            lookup = torch.load(f"{args.dir_data}{file_lookup}").to(device=device)
            print(f"{file_lookup[:-3]:<50}{lookup.shape}")
            mean_tensor = lookup.mean(0)
            for context in args.contexts:
                context_targets = build_context_targets_tensor(lookup, mean_tensor, context, df_all.file, 'context_targets')
                print(f"{'context_targets':<50}{context_targets.shape}")
                control = f"StlVecHierTransCat_cont0"
                # bert_ids = torch.load(f"{args.dir_data}bert_{bertlang}_{bertcontext}_{short_trn}_ids.pt").to(device=device, dtype=dtype_float)
                # print(f"{short_trn} {'bert_ids':<50}{bert_ids.shape}")
                # bert_mask = torch.load(f"{args.dir_data}bert_{bertlang}_{bertcontext}_{short_trn}_masks.pt").to(device=device, dtype=dtype_float)
                # print(f"{short_trn} {'bert_mask':<50}{bert_mask.shape}")
    
                for exp in range(1, args.nr_exps + 1):
                    direxp  = f"exp{exp}_{context}"
                    model = mod.StlVecHierTransCat(cont_size  = context_targets.shape[1],
                                                   emb_size   = lookup.shape[1],
                                                   y0_size    = 1,
                                                   att_heads  = args.att_heads,
                                                   att_layers = args.att_layers,
                                                   fc_layers  = args.fc_layers,
                                                   droprob    = args.droprob,
                                                   device     = device)
                    optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                    dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                                 optimizer       = optimizer,
                                                                 lossfuncs       = [bce],
                                                                 x_inputs        = [context_targets],
                                                                 x_inputs_dev    = [None],
                                                                 x_inputs_tst    = [None],
                                                                 x_dtypes        = [dtype_float],
                                                                 y_inputs        = [pledge_all],
                                                                 y_inputs_dev    = [None],
                                                                 y_inputs_tst    = [None],
                                                                 y_dtypes        = [dtype_float],
                                                                 batsize         = args.big_batsize,
                                                                 patience        = args.patience,
                                                                 min_gain        = args.min_gain,
                                                                 n_splits        = args.splits, # solo crossval, nr folds
                                                                 save            = args.save,
                                                                 additional_tsts = (),
                                                                 str_info        = direxp)
                    if context == 0:
                        boot.feed(control=control, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
                    else:
                        treatment = f"{model.__class__.__name__}_cont{context}"
                        boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
    
            boot.run(args.n_short_loops, args.perc_sample)
            return 1
    
        # bertvec_trans_context()
    
    
            if (bertlang != 'ml') and (short_trn != 'both'): #
                fastext_embs = torch.load(f"{args.dir_data}fastext_{short_trn}_embmatrix.pt").to(device=device, dtype=dtype_float)
                print(f"{short_trn} {'fastext_embs':<50}{fastext_embs.shape}")
                fastext_ids = torch.load(f"{args.dir_data}fastext_{short_trn}_ids.pt").to(device=device, dtype=dtype_float)
                print(f"{short_trn} {'fastext_ids':<50}{fastext_ids.shape}")
                fastext_mask = torch.load(f"{args.dir_data}fastext_{short_trn}_mask.pt").to(device=device, dtype=dtype_float)
                print(f"{short_trn} {'fastext_mask':<50}{fastext_mask.shape}")
        
        
                def fastext_conv_mtl():
                    dirout = f"{log.pathtime}{condition}_mtl_fastext_conv/"
                    os.mkdir(dirout)
                    print(f"{'#'*80}\n{dirout}")
                    proc = st.Processing(dirout, device)
                    boot = st.Bootstrap(dirout)
                    for exp in range(1, args.nr_exps + 1):
                        direxp  = f"exp{exp}"
                        model = mod.StlIds4lookupConvFc(emb              = fastext_embs,
                                                        y0_size          = 1,
                                                        trainable        = args.trainable,
                                                        conv_channels    = args.conv_channels,
                                                        filter_sizes     = args.conv_filter_sizes,
                                                        conv_stridesizes = args.conv_stridesizes,
                                                        pool_filtersizes = args.pool_filtersizes,
                                                        pool_stridesizes = args.pool_stridesizes,
                                                        fc_layers        = args.fc_layers,
                                                        droprob          = args.droprob,
                                                        device           = device)
                        optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                        dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                                     optimizer       = optimizer,
                                                                     lossfuncs       = [bce],
                                                                     x_inputs        = [fastext_ids],
                                                                     x_inputs_dev    = [None],
                                                                     x_inputs_tst    = [None],
                                                                     x_dtypes        = [dtype_int],
                                                                     y_inputs        = [pledge_all],
                                                                     y_inputs_dev    = [None],
                                                                     y_inputs_tst    = [None],
                                                                     y_dtypes        = [dtype_float],
                                                                     batsize         = args.big_batsize,
                                                                     patience        = args.patience,
                                                                     min_gain        = args.min_gain,
                                                                     n_splits        = args.splits, # solo crossval, nr folds
                                                                     save            = args.save,
                                                                     additional_tsts = (),
                                                                     str_info        = direxp)
                        control = f"{model.__class__.__name__}"
                        boot.feed(control=control, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
        
                        info = f"party"
                        direxp  = f"exp{exp}_{info}"
                        model = mod.Mtl1Ids4lookupConvFc(emb              = fastext_embs,
                                                         y0_size          = 1,
                                                         y1_size          = party_size_all,
                                                         trainable        = args.trainable,
                                                         conv_channels    = args.conv_channels,
                                                         filter_sizes     = args.conv_filter_sizes,
                                                         conv_stridesizes = args.conv_stridesizes,
                                                         pool_filtersizes = args.pool_filtersizes,
                                                         pool_stridesizes = args.pool_stridesizes,
                                                         fc_layers        = args.fc_layers,
                                                         droprob          = args.droprob,
                                                         device           = device)
                        optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                        dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                                     optimizer       = optimizer,
                                                                     lossfuncs       = [bce, ce],
                                                                     x_inputs        = [fastext_ids],
                                                                     x_inputs_dev    = [None],
                                                                     x_inputs_tst    = [None],
                                                                     x_dtypes        = [dtype_int],
                                                                     y_inputs        = [pledge_all, party_all],
                                                                     y_inputs_dev    = [None],
                                                                     y_inputs_tst    = [None],
                                                                     y_dtypes        = [dtype_float, dtype_int],
                                                                     batsize         = args.big_batsize,
                                                                     patience        = args.patience,
                                                                     min_gain        = args.min_gain,
                                                                     n_splits        = args.splits, # solo crossval, nr folds
                                                                     save            = args.save,
                                                                     additional_tsts = (),
                                                                     str_info        = direxp)
                        treatment = f"{model.__class__.__name__}_{info}"
                        boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
        
                        info = f"year"
                        direxp  = f"exp{exp}_{info}"
                        model = mod.Mtl1Ids4lookupConvFc(emb              = fastext_embs,
                                                         y0_size          = 1,
                                                         y1_size          = year_size_all,
                                                         trainable        = args.trainable,
                                                         conv_channels    = args.conv_channels,
                                                         filter_sizes     = args.conv_filter_sizes,
                                                         conv_stridesizes = args.conv_stridesizes,
                                                         pool_filtersizes = args.pool_filtersizes,
                                                         pool_stridesizes = args.pool_stridesizes,
                                                         fc_layers        = args.fc_layers,
                                                         droprob          = args.droprob,
                                                         device           = device)
                        optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                        dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                                     optimizer       = optimizer,
                                                                     lossfuncs       = [bce, ce],
                                                                     x_inputs        = [fastext_ids],
                                                                     x_inputs_dev    = [None],
                                                                     x_inputs_tst    = [None],
                                                                     x_dtypes        = [dtype_int],
                                                                     y_inputs        = [pledge_all, year_all],
                                                                     y_inputs_dev    = [None],
                                                                     y_inputs_tst    = [None],
                                                                     y_dtypes        = [dtype_float, dtype_int],
                                                                     batsize         = args.big_batsize,
                                                                     patience        = args.patience,
                                                                     min_gain        = args.min_gain,
                                                                     n_splits        = args.splits, # solo crossval, nr folds
                                                                     save            = args.save,
                                                                     additional_tsts = (),
                                                                     str_info        = direxp)
                        treatment = f"{model.__class__.__name__}_{info}"
                        boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
        
                        info = f"partyear"
                        direxp  = f"exp{exp}_{info}"
                        model = mod.Mtl2Ids4lookupConvFc(emb              = fastext_embs,
                                                         y0_size          = 1,
                                                         y1_size          = party_size_all,
                                                         y2_size          = year_size_all,
                                                         trainable        = args.trainable,
                                                         conv_channels    = args.conv_channels,
                                                         filter_sizes     = args.conv_filter_sizes,
                                                         conv_stridesizes = args.conv_stridesizes,
                                                         pool_filtersizes = args.pool_filtersizes,
                                                         pool_stridesizes = args.pool_stridesizes,
                                                         fc_layers        = args.fc_layers,
                                                         droprob          = args.droprob,
                                                         device           = device)
                        optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                        dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                                     optimizer       = optimizer,
                                                                     lossfuncs       = [bce, ce, ce],
                                                                     x_inputs        = [fastext_ids],
                                                                     x_inputs_dev    = [None],
                                                                     x_inputs_tst    = [None],
                                                                     x_dtypes        = [dtype_int],
                                                                     y_inputs        = [pledge_all, party_all, year_all],
                                                                     y_inputs_dev    = [None],
                                                                     y_inputs_tst    = [None],
                                                                     y_dtypes        = [dtype_float, dtype_int, dtype_int],
                                                                     batsize         = args.big_batsize,
                                                                     patience        = args.patience,
                                                                     min_gain        = args.min_gain,
                                                                     n_splits        = args.splits, # solo crossval, nr folds
                                                                     save            = args.save,
                                                                     additional_tsts = (),
                                                                     str_info        = direxp)
                        treatment = f"{model.__class__.__name__}_{info}"
                        boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
        
                    boot.run(args.n_loops, args.perc_sample)
                    return 1
        
                # fastext_conv_mtl()
        
                def fastext_trans_mtl():
                    dirout = f"{log.pathtime}{condition}_mtl_fastext_trans/"
                    os.mkdir(dirout)
                    print(f"{'#'*80}\n{dirout}")
                    proc = st.Processing(dirout, device)
                    boot = st.Bootstrap(dirout)
                    for exp in range(1, args.nr_exps + 1):
                        direxp  = f"exp{exp}"
                        model = mod.StlIds4lookupTransFc(emb        = fastext_embs,
                                                         y0_size    = 1,
                                                         trainable  = args.trainable,
                                                         att_heads  = args.att_heads,
                                                         att_layers = args.att_layers,
                                                         fc_layers  = args.fc_layers,
                                                         droprob    = args.droprob,
                                                         device     = device)
                        optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                        dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                                     optimizer       = optimizer,
                                                                     lossfuncs       = [bce],
                                                                     x_inputs        = [fastext_ids, fastext_mask],
                                                                     x_inputs_dev    = [None],
                                                                     x_inputs_tst    = [None],
                                                                     x_dtypes        = [dtype_int, dtype_int],
                                                                     y_inputs        = [pledge_all],
                                                                     y_inputs_dev    = [None],
                                                                     y_inputs_tst    = [None],
                                                                     y_dtypes        = [dtype_float],
                                                                     batsize         = args.big_batsize,
                                                                     patience        = args.patience,
                                                                     min_gain        = args.min_gain,
                                                                     n_splits        = args.splits, # solo crossval, nr folds
                                                                     save            = args.save,
                                                                     additional_tsts = (),
                                                                     str_info        = direxp)
                        control = f"{model.__class__.__name__}"
                        boot.feed(control=control, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
        
                        info = f"party"
                        direxp  = f"exp{exp}_{info}"
                        model = mod.Mtl1Ids4lookupTransFc(emb        = fastext_embs,
                                                          y0_size    = 1,
                                                          y1_size    = party_size_all,
                                                          trainable  = args.trainable,
                                                          att_heads  = args.att_heads,
                                                          att_layers = args.att_layers,
                                                          fc_layers  = args.fc_layers,
                                                          droprob    = args.droprob,
                                                          device     = device)
                        optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                        dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                                     optimizer       = optimizer,
                                                                     lossfuncs       = [bce, ce],
                                                                     x_inputs        = [fastext_ids, fastext_mask],
                                                                     x_inputs_dev    = [None],
                                                                     x_inputs_tst    = [None],
                                                                     x_dtypes        = [dtype_int, dtype_int],
                                                                     y_inputs        = [pledge_all, party_all],
                                                                     y_inputs_dev    = [None],
                                                                     y_inputs_tst    = [None],
                                                                     y_dtypes        = [dtype_float, dtype_int],
                                                                     batsize         = args.big_batsize,
                                                                     patience        = args.patience,
                                                                     min_gain        = args.min_gain,
                                                                     n_splits        = args.splits, # solo crossval, nr folds
                                                                     save            = args.save,
                                                                     additional_tsts = (),
                                                                     str_info        = direxp)
                        treatment = f"{model.__class__.__name__}_{info}"
                        boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
        
                        info = f"year"
                        direxp  = f"exp{exp}_{info}"
                        model = mod.Mtl1Ids4lookupTransFc(emb        = fastext_embs,
                                                          y0_size    = 1,
                                                          y1_size    = year_size_all,
                                                          trainable  = args.trainable,
                                                          att_heads  = args.att_heads,
                                                          att_layers = args.att_layers,
                                                          fc_layers  = args.fc_layers,
                                                          droprob    = args.droprob,
                                                          device     = device)
                        optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                        dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                                     optimizer       = optimizer,
                                                                     lossfuncs       = [bce, ce],
                                                                     x_inputs        = [fastext_ids, fastext_mask],
                                                                     x_inputs_dev    = [None],
                                                                     x_inputs_tst    = [None],
                                                                     x_dtypes        = [dtype_int, dtype_int],
                                                                     y_inputs        = [pledge_all, year_all],
                                                                     y_inputs_dev    = [None],
                                                                     y_inputs_tst    = [None],
                                                                     y_dtypes        = [dtype_float, dtype_int],
                                                                     batsize         = args.big_batsize,
                                                                     patience        = args.patience,
                                                                     min_gain        = args.min_gain,
                                                                     n_splits        = args.splits, # solo crossval, nr folds
                                                                     save            = args.save,
                                                                     additional_tsts = (),
                                                                     str_info        = direxp)
                        treatment = f"{model.__class__.__name__}_{info}"
                        boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
        
                        info = f"partyear"
                        direxp  = f"exp{exp}_{info}"
                        model = mod.Mtl2Ids4lookupTransFc(emb        = fastext_embs,
                                                          y0_size    = 1,
                                                          y1_size    = party_size_all,
                                                          y2_size    = year_size_all,
                                                          trainable  = args.trainable,
                                                          att_heads  = args.att_heads,
                                                          att_layers = args.att_layers,
                                                          fc_layers  = args.fc_layers,
                                                          droprob    = args.droprob,
                                                          device     = device)
                        optimizer = optim.Adam(model.parameters(), lr=args.high_learate)
                        dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                                     optimizer       = optimizer,
                                                                     lossfuncs       = [bce, ce, ce],
                                                                     x_inputs        = [fastext_ids, fastext_mask],
                                                                     x_inputs_dev    = [None],
                                                                     x_inputs_tst    = [None],
                                                                     x_dtypes        = [dtype_int, dtype_int],
                                                                     y_inputs        = [pledge_all, party_all, year_all],
                                                                     y_inputs_dev    = [None],
                                                                     y_inputs_tst    = [None],
                                                                     y_dtypes        = [dtype_float, dtype_int, dtype_int],
                                                                     batsize         = args.big_batsize,
                                                                     patience        = args.patience,
                                                                     min_gain        = args.min_gain,
                                                                     n_splits        = args.splits, # solo crossval, nr folds
                                                                     save            = args.save,
                                                                     additional_tsts = (),
                                                                     str_info        = direxp)
                        treatment = f"{model.__class__.__name__}_{info}"
                        boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
        
                    boot.run(args.n_loops, args.perc_sample)
                    return 1
        
                # fastext_trans_mtl()
        
        # bertvec_trans_context()
        
        def bertmat_trans_mtl():
            dirout = f"{log.pathtime}{condition}_mtl_bertmat_trans/"
            os.mkdir(dirout)
            print(f"{'#'*80}\n{dirout}")
            proc = st.Processing(dirout, device)
            boot = st.Bootstrap(dirout)
    
            file_lookup = f"bert_{bertlang}_single_{trainlang}_lookup_mat.pt"
            lookup = torch.load(f"{args.dir_data}{file_lookup}").to(device=device)
            print(f"{file_lookup[:-3]:<50}{lookup.shape}")
            # bert_ids = torch.load(f"{args.dir_data}bert_{bertlang}_{bertcontext}_{short_trn}_ids.pt").to(device=device, dtype=dtype_float)
            # print(f"{short_trn} {'bert_ids':<50}{bert_ids.shape}")
            # bert_mask = torch.load(f"{args.dir_data}bert_{bertlang}_{bertcontext}_{short_trn}_masks.pt").to(device=device, dtype=dtype_float)
            # print(f"{short_trn} {'bert_mask':<50}{bert_mask.shape}")
        
            for exp in range(1, args.nr_exps + 1):
                direxp  = f"exp{exp}"
                model = mod.StlLookupTransFc(emb_size   = lookup.shape[2],
                                             y0_size    = 1,
                                             att_heads  = args.att_heads,
                                             att_layers = args.att_layers,
                                             fc_layers  = args.fc_layers,
                                             droprob    = args.droprob,
                                             device     = device)
                optimizer = optim.Adam(model.parameters(), lr=args.low_learate)
                dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                             optimizer       = optimizer,
                                                             lossfuncs       = [bce],
                                                             x_inputs        = [lookup],
                                                             x_inputs_dev    = [None],
                                                             x_inputs_tst    = [None],
                                                             x_dtypes        = [dtype_float],
                                                             y_inputs        = [pledge_all],
                                                             y_inputs_dev    = [None],
                                                             y_inputs_tst    = [None],
                                                             y_dtypes        = [dtype_float],
                                                             batsize         = args.small_batsize,
                                                             patience        = args.patience,
                                                             min_gain        = args.min_gain,
                                                             n_splits        = args.splits, # solo crossval, nr folds
                                                             save            = args.save,
                                                             additional_tsts = (),
                                                             str_info        = direxp)
                control = f"{model.__class__.__name__}"
                boot.feed(control=control, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
    
                info = f"party"
                direxp  = f"exp{exp}_{info}"
                model = mod.Mtl1LookupTransFc(emb_size   = lookup.shape[2],
                                              y0_size    = 1,
                                              y1_size    = party_size_all,
                                              att_heads  = args.att_heads,
                                              att_layers = args.att_layers,
                                              fc_layers  = args.fc_layers,
                                              droprob    = args.droprob,
                                              device     = device)
                optimizer = optim.Adam(model.parameters(), lr=args.low_learate)
                dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                             optimizer       = optimizer,
                                                             lossfuncs       = [bce, ce],
                                                             x_inputs        = [lookup],
                                                             x_inputs_dev    = [None],
                                                             x_inputs_tst    = [None],
                                                             x_dtypes        = [dtype_float],
                                                             y_inputs        = [pledge_all, party_all],
                                                             y_inputs_dev    = [None],
                                                             y_inputs_tst    = [None],
                                                             y_dtypes        = [dtype_float, dtype_int],
                                                             batsize         = args.small_batsize,
                                                             patience        = args.patience,
                                                             min_gain        = args.min_gain,
                                                             n_splits        = args.splits, # solo crossval, nr folds
                                                             save            = args.save,
                                                             additional_tsts = (),
                                                             str_info        = direxp)
                treatment = f"{model.__class__.__name__}_{info}"
                boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
    
                info = f"year"
                direxp  = f"exp{exp}_{info}"
                model = mod.Mtl1LookupTransFc(emb_size   = lookup.shape[2],
                                              y0_size    = 1,
                                              y1_size    = year_size_all,
                                              att_heads  = args.att_heads,
                                              att_layers = args.att_layers,
                                              fc_layers  = args.fc_layers,
                                              droprob    = args.droprob,
                                              device     = device)
                optimizer = optim.Adam(model.parameters(), lr=args.low_learate)
                dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                             optimizer       = optimizer,
                                                             lossfuncs       = [bce, ce],
                                                             x_inputs        = [lookup],
                                                             x_inputs_dev    = [None],
                                                             x_inputs_tst    = [None],
                                                             x_dtypes        = [dtype_float],
                                                             y_inputs        = [pledge_all, year_all],
                                                             y_inputs_dev    = [None],
                                                             y_inputs_tst    = [None],
                                                             y_dtypes        = [dtype_float, dtype_int],
                                                             batsize         = args.small_batsize,
                                                             patience        = args.patience,
                                                             min_gain        = args.min_gain,
                                                             n_splits        = args.splits, # solo crossval, nr folds
                                                             save            = args.save,
                                                             additional_tsts = (),
                                                             str_info        = direxp)
                treatment = f"{model.__class__.__name__}_{info}"
                boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
    
                info = f"partyear"
                direxp  = f"exp{exp}_{info}"
                model = mod.Mtl2LookupTransFc(emb_size   = lookup.shape[2],
                                              y0_size    = 1,
                                              y1_size    = party_size_all,
                                              y2_size    = year_size_all,
                                              att_heads  = args.att_heads,
                                              att_layers = args.att_layers,
                                              fc_layers  = args.fc_layers,
                                              droprob    = args.droprob,
                                              device     = device)
                optimizer = optim.Adam(model.parameters(), lr=args.low_learate)
                dirres, preds, targs, _, m_epochs = proc.exp(model           = model,
                                                             optimizer       = optimizer,
                                                             lossfuncs       = [bce, ce, ce],
                                                             x_inputs        = [lookup],
                                                             x_inputs_dev    = [None],
                                                             x_inputs_tst    = [None],
                                                             x_dtypes        = [dtype_float],
                                                             y_inputs        = [pledge_all, party_all, year_all],
                                                             y_inputs_dev    = [None],
                                                             y_inputs_tst    = [None],
                                                             y_dtypes        = [dtype_float, dtype_int, dtype_int],
                                                             batsize         = args.small_batsize,
                                                             patience        = args.patience,
                                                             min_gain        = args.min_gain,
                                                             n_splits        = args.splits, # solo crossval, nr folds
                                                             save            = args.save,
                                                             additional_tsts = (),
                                                             str_info        = direxp)
                treatment = f"{model.__class__.__name__}_{info}"
                boot.feed(control=control, treatment=treatment, fold=dirres, preds=preds, targs=targs, epochs=m_epochs)
    
            boot.run(args.n_loops, args.perc_sample)
            return 1
    
        # bertmat_trans_mtl()

    return 1


run('swed', 'sw',      'swed')
run('swed', 'ml',      'swed')
run('swed', 'ml',      'both')
run('indi', 'enlarge', 'indi')
run('indi', 'ml',      'indi')
run('indi', 'ml',      'both')
run('both', 'ml',      'both')


ut.end(startime)
