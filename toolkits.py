#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import pickle
import random
import jieba.posseg as pseg
from collections import defaultdict

def build_bw_dataset():
    f = open('/home/zyma/work/data_daily_punct/test.ids110000.in')
    f1 = open('/home/zyma/work/data_daily_punct/bw_data/test.ids110000.in','w')
    q = f.readline()
    while q:
        f1.write(q)
        r = f.readline().strip().split()
        l = len(r)
        p = random.randint(1,l)
        r = r[:p]
        r.reverse()
        r = ' '.join(r)+'\n'
        f1.write(r)
        q = f.readline()

def build_ntable():
    d = defaultdict(int)
    f = open('/home/zyma/work/data_daily_punct/train.in')
    f1 = open('/home/zyma/work/data_daily_punct/nouns2500.in','w')
    s = f.readline()
    while s:
        words = pseg.cut(s.strip())
        for ele in words:
            if ele.flag[0]=='n':
                d[ele.word] += 1
        s = f.readline()
    nl = d.items()
    nl = sorted(nl,key=lambda x:x[1],reverse=True)
    for ele in nl[:2500]:
        f1.write(ele[0]+'\n')
    f.close()
    f1.close()
def build_wqtable():
    d = defaultdict(int)
    f = open('/home/zyma/work/data_daily_punct/train.in')
    s = f.readline()
    while s:
        s = s.decode('utf-8').replace(' ','')
        s = ' '.join(list(s)).encode('utf-8')
        s = s.split()
        for w in s:
            d[w]+=1
        f.readline()
        s = f.readline()
    f.close()
    f = open('q_table.pkl','w')
    total = sum(d.values())
    d = dict([(k,1.*v/total) for k,v in d.items()])
    pickle.dump(d,f)
    f.close()
       
def build_table(data,vocab,nouns):
    tmp = []
    for w in vocab:
        for w_ in nouns:
            tmp.append(w+'|'+w_)
    cotable = dict([(ele,1) for ele in tmp])
    #ntable = dict([(ele,1) for ele in nouns])
    #qtable = dict([(ele,1) for ele in vocab])
    for ele in data:
        q,a = ele
        aws = a.strip().split()
        qws = list(q.decode('utf-8').replace(' ',''))
        qws = ' '.join(qws)
        qws = qws.encode('utf-8')
        #print('%d:%s'%(i,qws))
        qws = qws.split()
        for ind,w in enumerate(aws):
            for w_ in qws:
                try:
                    cotable[w_+'|'+w] += 1
                except:
                    pass
            #try:
               # ntable[w] += 1
            #except:
               # pass
    #total = sum(cotable.values())
    ntable = defaultdict(int)
    qtable = defaultdict(int)
    for k,ele in cotable.items():
        ntable[k.split('|')[1]] += ele
        qtable[k.split('|')[0]] += ele
        
    cotable = dict([(k,1.*v/ntable[k.split('|')[1]]) for k,v in cotable.items()])
    total = sum(qtable.values())
    qtable = dict([(k,1.*v/total) for k,v in qtable.items()])
    f = open('co_table.pkl','w')
    pickle.dump(cotable,f)
    f.close()
    f = open('n_table.pkl','w')
    pickle.dump(ntable,f)
    f.close()
    f = open('q_table.pkl','w')
    pickle.dump(qtable,f)
    f.close()
def sentencePMI(sentence,cotable,qtable,nouns):
    sentence = sentence.decode('utf-8').replace(' ','')
    #print(sentence)
    sentence = ' '.join(list(sentence)).encode('utf-8')
    #print(sentence)
    ws = sentence.split()
    def pmi(wq):
        v = np.zeros(len(nouns))
        for ind,ele in enumerate(nouns):
          try:
            v[ind] = np.log((cotable[wq+'|'+ele])/(qtable[wq]))
          except:
            pass
        #print('middle result:%s,pmi:%f'%(nouns[np.argmax(v)],np.max(v)))
        return v
    vs = np.zeros(len(nouns))
    for w in ws:
        vs += pmi(w)
    #d = [(ind,v) for ind,v in enumerate(list(vs))]
    #d = sorted(d,key=lambda x:x[1],reverse=True)
    #d = d[:10]
    #for ele in d:
    #    print('word:%s,pmi:%f'%(nouns[ele[0]],ele[1]))
    return nouns[np.argmax(vs)]


if __name__ == '__main__':
    #build_bw_dataset()
     
    f = open('/home/zyma/work/train.in.w')
    d = f.readlines()
    f.close()
    f = open('/home/zyma/work/data_daily_punct/vocab_char5000.in')
    vocab = f.readlines()
    vocab = [ele.strip() for ele in vocab]
    f.close()
    f = open('/home/zyma/work/data_daily_punct/nouns2500.in')
    nouns = f.readlines()
    nouns = [ele.strip() for ele in nouns]
    f.close()
    data = []
    for i in range(len(d)//2):
        data.append((d[2*i],d[2*i+1]))
    print('build table')
    build_table(data,vocab,nouns)
    #test_pmi
    '''
    f = open('pkl_file/co_table.pkl')
    cotable = pickle.load(f)
    f.close()
    print('cotable loaded')
    f = open('pkl_file/q_table.pkl')
    qtable = pickle.load(f)
    f.close()
    print('qtable loaded')
    #f = open('pkl_file/n_table.pkl')
    #ntable = pickle.load(f)
    #f.close()
    #print('ntable loaded')
    f = open('/home/zyma/work/data_daily_punct/nouns2500.in')
    nouns = f.readlines()
    nouns = [ele.strip() for ele in nouns]
    f.close()
    print('nouns loaded')
    sentence = '烟台 哪里 买 滑雪板'
    print(sentencePMI(sentence,cotable,qtable,nouns))
    '''
