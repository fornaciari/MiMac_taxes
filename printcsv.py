# coding=latin-1
import util201217 as ut
import argparse, os, re
import pandas as pd
###################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-dir",   type=str, default='')
parser.add_argument("-latex", type=int, default=0)
parser.add_argument("-cols",  type=str, default='acc prec rec f1', help='acc prec sign_prec rec sign_rec f1 sign_f1')
args = parser.parse_args()
###################################################################################################


def traverse(foldin, level=''):
    level += '%'
    foldin = foldin + '/' if not re.search("/$", foldin) else foldin
    folds = sorted([elem for elem in os.listdir(foldin) if os.path.isdir(f"{foldin}{elem}")])
    csvs  = sorted([elem for elem in os.listdir(foldin) if re.search("sv$", elem)])
    if len(csvs) > 0:
        print(f"{level} {foldin}")
        for csv in csvs:
            if args.latex:
                df = pd.read_csv(f"{foldin}{csv}", index_col=0)
                for metric in 'acc prec rec f1'.split():
                    df[metric] = [metric if type(signif) is float else f"\textbf{{{metric} {signif}}}" for metric, signif in zip(df[metric], df[f"sign_{metric}"])]
                df = df[args.cols.split()]
                df.fillna('', inplace=True)
                print(df.to_latex(escape=False))
            else:
                os.system(f"sed 's/,,/, ,/g;s/,,/, ,/g' {foldin}{csv} | column -s, -t")
        print()
    for fold in folds:
        traverse(f"{foldin}{fold}/", level)


traverse(args.dir)

