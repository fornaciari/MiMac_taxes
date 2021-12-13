# coding=latin-1
import os
import sys
import re
import time
import pickle
import json
import numpy as np
import psutil
import torch
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError


###############################################################################
def time0():
    print("***** t0 *****")
    return time.time()


def time1(t0, dec=6):
    t1 = time.time()
    print(f"***** t1 *****\n   {round(t1 - t0, dec)}\n**************")
    return 1


def stringtime(n):
    h = str(int(n / 3600))
    m = str(int((n % 3600) / 60))
    s = str(int((n % 3600) % 60))
    if len(h) == 1: h = '0' + h
    if len(m) == 1: m = '0' + m
    if len(s) == 1: s = '0' + s
    return h + ':' + m + ':' + s


def steptime(start):
    now = time.time()
    dur = stringtime(now - start)
    print("time elapsed:", dur)
    return now


def start(sep=True):
    start = time.time()
    now = time.strftime("%Y/%m/%d %H:%M:%S")
    if sep: print('#'*80)
    print('start:', now)
    return start


def begin():
    start = time.time()
    return start


def end(start, sep=True):
    end = time.time()
    dur = end - start
    str_dur = stringtime(end - start)
    now = time.strftime("%Y/%m/%d %H:%M:%S")
    if sep:
        print('#'*80 + "\nend:", now, " - time elapsed:", str_dur + "\n" + '#'*80)
    else:
        print("end:", now, " - time elapsed:", str_dur)
    return dur


def now():
    now = time.strftime("%Y/%m/%d %H:%M:%S")
    return now


def timefrom(start):
    now = time.time()
    return stringtime(now - start)


def printstep(string, index, step):
    n = index + 1
    if n % step == 0:
        print(string, 'step', n)
    return 1


###############################################################################
def say(*anything):
    printstr = ''
    for elem in list(anything): printstr += str(elem) + ' '
    printstr = re.sub(' $', '', printstr)
    sys.stdout.write("\r" + printstr)
    sys.stdout.flush()


###############################################################################


def tsv2npmatrix(pathfile, sep="\t", encoding='utf-8', elemtype=int):
    '''
    input: tsv file
    output: list of lists 
    '''
    with open(pathfile, 'r', encoding=encoding) as input_file: str_file = input_file.read()
    str_file = re.sub("\n+$", '', str_file) # via il o gli ultimi \n, se ci sono righe vuote alla fine del file
    if   elemtype == int:   out = np.array([[int(elem)   for elem in row.split(sep)] for row in str_file.split("\n")])
    elif elemtype == float: out = np.array([[float(elem) for elem in row.split(sep)] for row in str_file.split("\n")])
    else:                   out = np.array([[str(elem)   for elem in row.split(sep)] for row in str_file.split("\n")])
    return out


def tsv2matrix(pathfile, sep="\t", encoding='utf-8', elemtype=int):
    '''
    input: tsv file
    output: list of lists
    '''
    with open(pathfile, 'r', encoding=encoding) as input_file: str_file = input_file.read()
    str_file = re.sub("\n+$", '', str_file) # via il o gli ultimi \n, se ci sono righe vuote alla fine del file
    if   elemtype == int:   out = [[int(elem)   for elem in row.split(sep)] for row in str_file.split("\n")]
    elif elemtype == float: out = [[float(elem) for elem in row.split(sep)] for row in str_file.split("\n")]
    else:                   out = [[str(elem)   for elem in row.split(sep)] for row in str_file.split("\n")]
    return out


def file2list(pathfile, encoding='utf-8', elemtype=int, sep="\n", emptyend=False):
    with open(pathfile, 'r', encoding=encoding) as input_file: str_file = input_file.read()
    if emptyend: str_file = re.sub("\n+$", '', str_file) # via il o gli ultimi \n, se ci sono righe vuote alla fine del file
    if elemtype == int:
        out = [float(x) for x in str_file.split(sep)] # se il format è float, non lo traduce direttamente
        out = [int(x) for x in out]
    elif elemtype == float:
        out = [float(x) for x in str_file.split(sep)]
    else:
        out = [x for x in str_file.split(sep)]
    return out


def file2dict(pathfile, keytype=str, valtype=str, encoding='utf-8'):
    with open(pathfile, 'r', encoding=encoding) as input_file: str_file = input_file.read()
    fileraw = re.sub("\n+$", '', str_file)  # via il o gli ultimi \n, se ci sono righe vuote alla fine del file
    out = {}
    rows = fileraw.split("\n")
    for row in rows:
        cols = row.split("\t")
        if keytype == str and valtype == int:
            out[cols[0]] = int(cols[1])
        elif keytype == str and valtype == float:
            out[cols[0]] = float(cols[1])
        elif keytype == int and valtype == int:
            out[int(cols[0])] = int(cols[1])
        elif keytype == float and valtype == float:
            out[float(cols[0])] = float(cols[1])
        else:
            out[cols[0]] = cols[1]
    #for k in out: print(k, out[k])
    return out


def file2dictlist(pathfile, keytype=int, listype=int, encoding='utf-8'):
    with open(pathfile, 'r', encoding=encoding) as input_file: str_file = input_file.read()
    fileraw = re.sub("\n+$", '', str_file)  # via il o gli ultimi \n, se ci sono righe vuote alla fine del file
    out = {}
    rows = fileraw.split("\n")
    for row in rows:
        cols = row.split("\t")
        if keytype == int and listype == int:
            out[int(cols[0])] = [int(i) for i in cols[1:]]
        elif keytype == float and listype == float:
            out[float(cols[0])] = [float(i) for i in cols[1:]]
        elif keytype == str and listype == float:
            out[cols[0]] = [float(i) for i in cols[1:]]
        else:
            out[cols[0]] = [i for i in cols[1:]]
    return out


def file2dictset(pathfile, keytype=int, setype=int, encoding='utf-8'):
    with open(pathfile, 'r', encoding=encoding) as input_file: str_file = input_file.read()
    fileraw = re.sub("\n+$", '', str_file)  # via il o gli ultimi \n, se ci sono righe vuote alla fine del file
    out = {}
    rows = fileraw.split("\n")
    for row in rows:
        cols = row.split("\t")
        if keytype == int and setype == int:
            out[int(cols[0])] = {int(i) for i in cols[1:]}
        elif keytype == float and setype == float:
            out[float(cols[0])] = {float(i) for i in cols[1:]}
        else:
            out[cols[0]] = {i for i in cols[1:]}
    return out


###############################################################################
def list2file(lis, fileout, sepline="\n", wra='w'):
    with open(fileout, wra) as fileout: [fileout.write(str(x) + sepline) for x in lis]
    return 1


def tuple2file(tup, fileout, wra='w'):
    with open(fileout, wra) as f_out:
        for item in tup: f_out.write(str(item[0]) + "\t" + str(item[1]) + "\n")
    return 1


def dict2file(dic, fileout, wra='w'):
    with open(fileout, wra) as f_out:
        for k in dic: f_out.write(str(k) + "\t" + str(dic[k]) + "\n")
    return 1


def docs4words2tsv(matrix, fileout, wra='w'):
    with open(fileout, wra) as f_out:
        # f_out.write("\n".join(["\t".join(row) for row in matrix])) # scrivo invece riga per riga, o i file troppo grandi saltano
        for row in matrix[:-1]:
            f_out.write("\t".join(row) + "\n")
        f_out.write("\t".join(matrix[-1]))
    return 1


def dictlist2file(dic, fileout, wra='w'):
    with open(fileout, wra) as f_out:
        for k in dic:
            stringrow = str(k) + "\t"
            for v in dic[k]:
                stringrow += str(v) + "\t"
            stringrow = re.sub("\t$", "\n", stringrow)
            f_out.write(stringrow)
    return 1


def setuple2file(setuple, fileout, wra='w'):
    with open(fileout, wra) as f_out:
        for tup in setuple:
            stringrow = ''
            for elem in tup:
                stringrow += str(elem) + "\t"
            #print(stringrow)
            stringrow = re.sub("\t$", "\n", stringrow)
            f_out.write(stringrow)
    return 1


def writebin(data, f_out):
    out = open(f_out, "wb")
    pickle.dump(data, out)
    out.close()
    return 1


def readbin(f_in, enc="Latin-1"):
    inp = open(f_in, "rb")
    out = pickle.load(inp, encoding=enc)
    inp.close()
    return out


###############################################################################


def get_lines(path, maxrow=-1):
    """the function traverses a generator until maxrow or StopIteration.
    the output is the another generator, having the wanted nr of lines"""
    generator = open(path)
    row_counter = 0
    while maxrow != 0:
        try:
            line = next(generator)
            maxrow -= 1
            row_counter += 1
            yield line
        except StopIteration:
            print(f"{'nr rows in generator:':.<40} {row_counter}")
            maxrow = 0


def read_file(filename, code='utf-8'):
    with open(filename, 'r', encoding=code) as f_in:
        out = f_in.read()
        return out


def readjson(pathname):
    with open(pathname) as f_in: out = json.load(f_in)
    return out


def writejson(data, pathname):
    with open(pathname, 'w') as out: json.dump(data, out)
    return 1


def printjson(data, stop=None):
    for i, k in enumerate(data):
        if i == stop: break
        print(f"{k}: {json.dumps(data[k], indent=4, ensure_ascii=False)}") # ensure ascii false permette di stampare i caratteri utf-8 invece di \usblindo
    return 1


def get_methods(obj):
    for method in dir(obj):
        if callable(getattr(obj, method)):
            print(method)
    return 1


def get_attributes(obj):
    for attribute in dir(obj):
        if not callable(getattr(obj, attribute)):
            print(attribute)
    return 1


def get_parameters(fun):
    print(fun.__doc__)
    return 1


def print_memory(device=None):
    if device is not None:
        print(f"cuda allocated memory: {torch.cuda.memory_allocated(device=device)}")
        print(f"cuda max allocated memory:  {torch.cuda.max_memory_allocated(device=device)}")
    print(f"cpu count: {psutil.cpu_count()}")
    print(f"cpu %: {psutil.cpu_percent()}")
    print(f"cpu stats: {psutil.stats()}")
    return 1


def traverse(k2v, level=''):
    """
    modello di funzione ricorsiva, che attraversa completamente un albero
    racchiuso in un dict.
    se dico:
    print(level, k, k2v[k])
    non salta, per il try ereditato dalla precedente iterazione,
    ma non stampa l'ultimo livello, dove k2v[k] non esiste
    Dove esiste, k2v[k] contiene tutto il blocco a valle.
    per provare:
    asss = {'a':
                {'aa':
                     ['aaa', 'aab', 'aac'],
                 'ab':
                     ['aba', 'abb', 'abc'],
                 'ac':
                     {'aca': 0, 'acb': 0, 'acc': ['accd', 'acce']}},
            'b':
                {'ba':
                     ['baa', 'bab', 'bac'],
                 'bb':
                     ['bba', 'bbb', 'bbc'],
                 'bc':
                 ['bca', 'bcb', 'bcc']},
            'c':
                {'ca':
                     ['caa', 'cab', 'cac'],
                 'cb':
                     ['cba', 'cbb', 'cbc'],
                 'cc':
                     ['cca', 'ccb', 'ccc']}
           }
    """
    level += '-'
    for k in k2v:
        print(level, k)
        try:
            traverse(k2v[k], level)
        except:
            pass
            
###############################################################################


def print_matrix(matrix, lastrow=None, lastcol=None):
    for index, row in enumerate(matrix):
        if index == lastrow: break
        print(index, "-> ", end='')
        for elem in row[:lastcol]:
            print(elem, "\t", end='')
        print()
    return 1


def printdict(k2v, last=-1):
    for i, k in enumerate(k2v):
        if i == last: break
        print("{} \t->\t{}".format(k, k2v[k]))
    return 1


def print_dictlist(dictionary, lastrow=None, lastcol=None):
    if lastrow == None: lastrow = len(dictionary)
    ir = 0
    for index in dictionary:
        ir += 1
        if ir > lastrow: break
        print(index, "\t->\t", end='')
        ic = 0
        if lastcol == None:
            thiscol = len(dictionary[index])
        else:
            thiscol = lastcol
        for elem in dictionary[index]:
            ic += 1
            if ic > thiscol: break
            print(elem, "\t", end='')
        print()
    return 1


def print_dictset(dictionary, lastrow=None, lastcol=None):
    if lastrow == None: lastrow = len(dictionary)
    ir = 0
    for index in dictionary:
        ir += 1
        if ir > lastrow: break
        print(index, "\t->\t", end='')
        ic = 0
        if lastcol == None:
            thiscol = len(dictionary[index])
        else:
            thiscol = lastcol
        for elem in dictionary[index]:
            ic += 1
            if ic > thiscol: break
            print(elem, "\t", end='')
        print()
    return 1


def printshape(x, n='', level=''):
    if hasattr(x, 'shape'):
        print(f"{level}{' ' if level != '' else ''}{n + ' shape' if n != '' else 'shape:'} {x.shape}")
    elif isinstance(x, list):
        print(f"{level}{' ' if level != '' else ''}{n + ' shape' if n != '' else 'shape:'} {np.shape(x)} ({type(x)})")
    elif isinstance(x, tuple):
        level += '-'
        for e in x:
            printshape(e, n=n, level=level)
    elif isinstance(x, dict):
        level += '-'
        for k in x:
            printshape(x[k], n=k, level=level)

###############################################################################


def print_args(args, align_size=15): # default value per i print che non appartengono ad args, tipo 'GPU in use'
    print('#'*80)
    maxlen = max([len(arg) for arg in vars(args)])
    align_size = maxlen + 3 if maxlen > align_size - 3 else align_size # garantisco ci siano sempre almeno 3 punti e lo spazio tra arg e relativo valore
    for arg in vars(args): print(f"{arg:.<{align_size}} {getattr(args, arg)}")
    return align_size


def str_or_none(value):
    if value == 'None':
        return None
    return value


def timeprogress(startime, i, step=10000, end=10000):
    i += 1
    if i % step == 0 or i == end: print(i, stringtime(startime), flush=True)
    return 1


def say_progress(n, step=10000):
    if n % step == 0: say('step', n, 'done')
    return 1


###############################################################################


def bip():
    duration = 3  # seconds
    freq = 440  # Hz
    # os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    os.system('say "Bernarda, sto facendo la doccia."')
    return 1


###############################################################################
import smtplib, socket, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def sendmail(dest="", sub='nerd-mail', body='nerd-mail'):
    try:
        mitt = ""
        mess = MIMEMultipart()
        mess['From'] = mitt
        mess['To'] = dest
        mess['Subject'] = sub
        mess.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('', 587)
        server.starttls()
        server.login(mitt, "")
        text = mess.as_string()
        server.sendmail(mitt, dest, text)
        server.quit()
    except (HTTPError, URLError, socket.error, ssl.SSLError, smtplib.SMTPException) as e:
        print("{}:\nmail non inviata\nerror:".format(time.strftime("%Y/%m/%d %H:%M:%S"), str(e)))


def sendslack(text='ping!', blocks=None, attachments=None, thread_ts=None, mrkdwn=True):
    webhook_url = "https://hooks.slack.com/services/"
    mess_payload = json.dumps({'text': text, 'blocks': blocks, 'attachments': attachments, 'thread_ts': thread_ts, 'mrkdwn': mrkdwn})
    os.system(f"curl -X POST -H 'Content-type: application/json' --data '{mess_payload}' {webhook_url} &> /dev/null") # &> /dev/null evita di stampare "ok" ad ogni mess
    return 1


###############################################################################


class log:
    def __init__(self, filename, details=''):
        self.timestamp = time.strftime("%y%m%d%H%M%S")
        log_name = re.sub('.py', '.log', filename)
        self.root = re.sub('.py', '', filename)
        self.pathtime = f"{self.root}/{details}{'_' if details != '' else ''}{self.timestamp}/"
        os.system(f"mkdir -p {self.pathtime}")
        self.terminal = sys.stdout
        self.log_out = open(self.pathtime + log_name, 'a')

    def cp(self, *args):
        # print(self, "$$$")
        # for arg in args: os.system(f"cp {arg} {self.pathtime}")
        pass

    def write(self, message):
        self.terminal.write(message)
        self.log_out.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        # sendslack('azz')
        pass


def yupdir(name=None):
    dirout = 'jupyter_' + name + time.strftime("%y%m%d%H%M%S/") if name else 'jupyter' + time.strftime("%y%m%d%H%M%S/")
    os.system('mkdir -p ' + dirout)
    print('created dirout:', dirout)
    return dirout


class bcolors:
    reset     = '\033[0m'
    bold      = '\033[1m'
    underline = '\033[4m'
    reversed  = '\033[7m'

    white     = '\033[38;5;0m'
    cyan      = '\033[38;5;14m'
    magenta   = '\033[38;5;13m'
    blue      = '\033[38;5;12m'
    yellow    = '\033[38;5;11m'
    green     = '\033[38;5;10m'
    red       = '\033[38;5;9m'
    grey      = '\033[38;5;8m'
    black     = '\033[38;5;0m'

    cleargrey  = '\033[38;5;7m'
    darkyellow = '\033[38;5;3m'
    darkred    = '\033[38;5;88m'
    darkcyan   = '\033[38;5;6m'
    pink       = '\033[38;5;207m'
    clearpink  = '\033[38;5;218m'
    cyangreen  = '\033[38;5;85m'
    cleargreen = '\033[38;5;192m'
    olivegreen = '\033[38;5;29m'
    
    CEND      = '\33[0m'
    CBOLD     = '\33[1m'
    CITALIC   = '\33[3m'
    CURL      = '\33[4m'
    CBLINK    = '\33[5m'
    CBLINK2   = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK    = '\33[30m'
    CRED      = '\33[31m'
    CGREEN    = '\33[32m'
    CYELLOW   = '\33[33m'
    CBLUE     = '\33[34m'
    CVIOLET   = '\33[35m'
    CBEIGE    = '\33[36m'
    CWHITE    = '\33[37m'

    CBLACKBG  = '\33[40m'
    CREDBG    = '\33[41m'
    CGREENBG  = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG   = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG  = '\33[46m'
    CWHITEBG  = '\33[47m'

    CGREY     = '\33[90m'
    CRED2     = '\33[91m'
    CGREEN2   = '\33[92m'
    CYELLOW2  = '\33[93m'
    CBLUE2    = '\33[94m'
    CVIOLET2  = '\33[95m'
    CBEIGE2   = '\33[96m'
    CWHITE2   = '\33[97m'

    CGREYBG    = '\33[100m'
    CREDBG2    = '\33[101m'
    CGREENBG2  = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2   = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2  = '\33[106m'
    CWHITEBG2  = '\33[107m'

    i2red = {0:  ([255/255, 204/255, 204/255]), # quasi bianco
             1:  ([255/255, 153/255, 153/255]),
             2:  ([255/255, 102/255, 102/255]),
             3:  ([255/255, 51/255,  51/255]),
             4:  ([255/255, 0/255,   0/255]), # rosso
             5:  ([204/255, 0/255,   0/255]),
             6:  ([153/255, 0/255,   0/255]),
             7:  ([102/255, 0/255,   0/255]),
             8:  ([51/255,  0/255,   0/255])} # quasi nero

    def examples(self):
        for i in range(0, 16):
            for j in range(0, 16):
                code = str(i * 16 + j)
                sys.stdout.write(u"\u001b[38;5;" + code + "m " + code.ljust(4))
        for i in range(0, 16):
            for j in range(0, 16):
                code = str(i * 16 + j)
                sys.stdout.write(u"\u001b[48;5;" + code + "m " + code.ljust(4))


