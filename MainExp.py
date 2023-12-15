import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from chapydette import cp_estimation
from scipy.stats import ortho_group
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
from scipy.linalg import eigh
from scipy.stats import chi2
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.signal import periodogram
from numpy.fft import rfft
from sklearn.metrics import roc_auc_score
from numpy.polynomial.legendre import Legendre
import pandas as pd
import time
import tqdm
import pickle
import lzma
import sys
import json
import logging
logging.basicConfig(filename='logging.log',level=logging.DEBUG)
logging.captureWarnings(True)

############################################################################################


def jump(n, s=0, lin=False, resol=1000):
    n1 = int(resol/2)
    t = np.zeros(resol) if not lin else np.linspace(-1,1,resol)
    t[:n1] -= s/2
    t[n1:] += s/2
    return t + np.random.normal(size=(n,resol))

def multi_jump(n, s=0, d=1, lin=False, resol=1000):
    t = np.linspace(-1,1,resol)
    U = 0.5-np.random.random(size=n)
    V = np.random.choice([-1,1],size=n)
    
    c = int(resol/2)
    t[c:] *= d
    t[c:] += s
    
    m = ((2.5*t[None,:]+U[:,None])%1>0.5).astype(float)*V[:,None] 
    if lin:
        m += np.linspace(-1,1,resol)[None,:]
    m += np.random.normal(size=m.shape)/2
    return m

def lemniscate(n, s=0,d=1, lin=False, resol=1000):
    U = 0.5-np.random.random(size=n)
    V = 1+(0.5-np.random.random(size=n))
    t = np.linspace(0,2*np.pi*8,resol)
    t -= t[-1]/2
    
    c = int(resol/2)
    t[c:] += (d-1)*(t[c:]-t[c])
    t[c:] += s
    
    T = V[:,None]*t[None,:]+2*np.pi*U[:,None]
    
    x = np.sin(2* T)/2 
    y = np.cos(   T)
    z = (T-T.mean())/t.std() 
    
    if lin:
        X = np.hstack((x[:,None,:],y[:,None,:],z[:,None,:]))
    else:
        X = np.hstack((x[:,None,:],y[:,None,:]))
    return np.array([ (ortho_group.rvs(x.shape[0]) @ x)[0,:] for x in X]) + np.random.normal(size=(n,resol))


############################################################################################

def gen_window_matrix(l1,l2, n_perm, chage=dict()):
    if (l1,l2, n_perm) not in chage.keys():
        w = np.array(l1*[1./l1]+(l2)*[-1./(l2)])
        W = np.array([w] + [np.random.permutation(w) for _ in range(n_perm)])
        chage[(l1,l2,n_perm)] = W
    return chage[(l1,l2,n_perm)]
def mmd(X, s=None, n_perm=2500):
    X = X.reshape(-1,1)
    K = apply_kernel(X, metric="linear")
    if s is None:
        s = int(X.shape[0]/2)
    
    W = gen_window_matrix(s,K.shape[0]-s, n_perm)
    s = np.einsum('ij,ij->i', np.dot(W, K), W)
    p = (s[0] < s).sum() / n_perm
    
    return {"p-val":p, "val": -p, "method":"MMD"}




def kfdr(X1,X2 = None, d=10, metric="rbf"):
    if len(X1.shape) == 1:
        X1 = X1.reshape(-1,1)
    if X2 is None:
        n = int(X1.shape[0]/2)
        X1,X2 = X1[:n],X1[n:]
    n1,n2 = X1.shape[0], X2.shape[0]
    X = np.vstack((X1,X2))
    K = apply_kernel(X, metric=metric)
    
    P = np.eye(n1+n2)
    P[:n1,:n1] -= 1/n1
    P[n1:,n1:] -= 1/n2
    
    N = (P @ K @ P)/(n1+n2-1)
    L,V = eigh(N, subset_by_index=[n1+n2-d, n1+n2-1])
    
    m = np.array( n1*[1/n1] + n2*[-1/n2] )
    
    stat = m @ K @ V @ np.diag(1/L) @ V.T @ K.T @ m
    p = 1-chi2.cdf(stat, df=d)
    
    return {"p-val":p, "val": -p, "method":"KRFD"}




def create_projection(n,k=111,lam=1, mem=dict()):
    if (n,k,lam) not in mem.keys():
        t = np.linspace(-1,1, n)
        A = [Legendre(i*[0]+[1])(t)/(i+1) for i in range(15)]
        A += [np.sin(4*np.pi*d*t+o)/(d) for o in [0,np.pi/2] for d in np.linspace(1,5,150)]
        A = np.array(A)

        P = A.T @ np.linalg.solve( A @ A.T - lam*np.eye(A.shape[0]) , A )
        P = np.eye(P.shape[0]) - P

        K = np.abs(t[:,None]-t[None,:]) < (k/2)/t.shape[0]
        K = K/K.sum(axis=0)[None,:]
        
        mem[(n,k,lam)] = K @ P
    return mem[(n,k,lam)]
def glob_mod(X):
    P = create_projection(X.shape[0])
    p = P @ X
    return {"lMSE": ((p)**2).mean(), "val": ((p)**2).mean(), "method":"Global model"}




def kcpd(X):
    cp = cp_estimation.mkcpe(X=X.reshape(-1,1), n_cp=(0, 10), alpha=2, bw=0.1, kernel_type = 'gaussian-euclidean', est_ncp=True)
    return {"cp": cp, "val": cp.shape[0], "method":"KCpD"}



def ft(X,cor=True):
    if cor:
        t = np.linspace(-1,1,X.shape[0]); t = np.array([Legendre(i*[0]+[1])(t)/(i+1) for i in range(3)]).T
        X -= Ridge().fit(t,X).predict(t)
    return {"ft": rfft(X), "method":"FT", "correction":cor}



def period(X,cor=True):
    if cor:
        t = np.linspace(-1,1,X.shape[0]); t = np.array([Legendre(i*[0]+[1])(t)/(i+1) for i in range(3)]).T
        X -= Ridge().fit(t,X).predict(t)
    f,Pxx = periodogram(X)
    return {"f": f, "Pxx":Pxx, "method":"Periodogram", "correction":cor}




def ADF(X):
    p = adfuller(X,regression="ctt")[1]
    return {"p-val":p, "val": -p, "method":"ADF"}
def KPSS(X):
    p = kpss(X,regression="ct")[1]
    return {"p-val":p, "val": -p, "method":"KPSS"}




def AR_err(X, model=Ridge(), overwrite=True, window_len=400):
    ard = preprocess_auto_reg(X)
    err = process_stream(ard[:,:-2], ard[:,-1], model=model, overwrite=overwrite, window_len=window_len)
    return {"err": err, "model": str(model), "overwrite": overwrite, "window_length":window_len, "method":"AR-Err"}
def process_stream(Xs,ys, model, window_len, overwrite, min_train=50):
    assert min_train < window_len
    wXs,wYs = np.empty(shape=(window_len,Xs.shape[1])),np.empty(shape=(window_len))
    err = []
    trained_model = None
    for i,(x,y) in enumerate(zip(Xs,ys)):
        if i > min_train:
            filled = min(i-1,window_len-1)
            if overwrite or trained_model is None or i-2 < window_len:
                trained_model = model.fit(wXs[:filled],wYs[:filled])
            err.append( ((trained_model.predict([x])-y)**2).sum() ) 
        if overwrite or i < window_len+2:
            wXs[i % window_len] = x
            wYs[i % window_len] = y
    return np.array(err)
def preprocess_auto_reg(X, max_len=26):
    return np.array([X[i:i+max_len] for i in range(X.shape[0]-max_len)])

############################################################################################

def time_it(f,x):
    t0 = time.time()
    try:
        y = f(x)
        y["status"] = "success"
    except Exception as e:
        y = {"status":"failed", "error":e, "method":f.__name__, "param":x}
    t1 = time.time()
    y["comp-time"] = t1-t0
    return y


dataset_file = "Datasets.pkl.xz"

if len(sys.argv) not in [2,3] or sys.argv[1] not in ["--make","--setup","--run_exp"]:
    print("Use --make | --setup #n | --run_exp #i")
    exit(1)
if sys.argv[1] == "--make":
    if len(sys.argv) != 2:
        print("WARN: --make does not take argument")
        
    datasets = []
    n = 1000

    for s in np.linspace(0,1,10):
        for lin in [True,False]:
            datasets.append({"generator": "jump", "s": s, "lin": lin, "X": jump(n,s,lin)})

    d,s = 1,0
    for s in np.linspace(0,5,10):
        for lin in [True,False]:
            datasets.append({"generator": "multi_jump", "s": s, "d":d, "lin": lin, "X": multi_jump(n,s,d,lin)})
    d,s = 1,0
    for d in np.linspace(0,1,10):
        for lin in [True,False]:
            datasets.append({"generator": "multi_jump", "s": s, "d":d, "lin": lin, "X": multi_jump(n,s,d,lin)})

    d,s = 1,0
    for s in np.linspace(0,np.pi,10):
        for lin in [True,False]:
            datasets.append({"generator": "lemniscate", "s": s, "d":d, "lin": lin, "X": lemniscate(n,s,d,lin)})
    d,s = 1,0
    for d in np.linspace(0,1,10):
        for lin in [True,False]:
            datasets.append({"generator": "lemniscate", "s": s, "d":d, "lin": lin, "X": lemniscate(n,s,d,lin)})

    with lzma.open(dataset_file, 'wb') as file:
        pickle.dump(datasets, file)
    
elif len(sys.argv) != 3:
    print("Use --make | --setup #n | --run_exp #i")
    exit(1)
if sys.argv[1] == "--setup":
    try:
        n_setups = int(sys.argv[2])
        assert str(n_setups) == sys.argv[2]
        assert n_setups >= 1
    except:
        print("number must be an integer >= 1")
        exit(1)
    
    with lzma.open(dataset_file, 'rb') as file:
        datasets = pickle.load(file)
    
    keys=list(range(len(datasets)))
    setups = [keys[i::n_setups] for i in range(n_setups)]
    
    with open("setups.json", "w") as file:
        json.dump(setups, file)
    
elif sys.argv[1] == "--run_exp":
    try:
        setup_id = int(sys.argv[2])
        assert str(setup_id) == sys.argv[2]
    except:
        print("id must be an integer >= 1")
        exit(1)
    
    with open("setups.json", "r") as file:
        setups = json.load(file)[setup_id]
    print("running setup %i (%s)"%(setup_id,str(setups)))
    
    with lzma.open(dataset_file, 'rb') as file:
        datasets = pickle.load(file)
    datasets = [datasets[key] for key in setups]
          
    def get_data():
        for ds in datasets:
            data_key = {k:v for k,v in ds.items() if k != "X"}
            for i,data in enumerate(list(ds["X"])):
                data_key["row"] = i
                yield data_key,data
    def count_data():
        r = 0
        for ds in datasets:
            r += ds["X"].shape[0]
        return r


    start_entry = 0
    out_file = "./out/main_exp-%i.pkl.xz"%setup_id
    if start_entry == 0:
        with lzma.open(out_file, 'wb') as file:
            pickle.dump([], file)

    results = []
    t0 = time.time()
    for i,(data_key,data) in enumerate(tqdm.tqdm(get_data(),total=count_data())):
        if i >= start_entry:
            results += [{"entry":i, "key":data_key, "result": time_it(f, data)} for f in [
                mmd,
                kfdr,
                glob_mod,
                kcpd,
                lambda x:AR_err(x,Ridge(),True,400),   
                lambda x:AR_err(x,Ridge(),False,400),  
                lambda x:AR_err(x,Ridge(),True,75),   
                lambda x:AR_err(x,Ridge(),False,75),  
                lambda x:AR_err(x,KNeighborsRegressor(),True,400),  
                lambda x:AR_err(x,KNeighborsRegressor(),False,400),
                ft,
                lambda x:ft(x,False),
                period,
                lambda x:period(x, False),
                ADF,
                KPSS,
            ]]
        if time.time()-t0 > 30*60:
            with lzma.open(out_file, 'ab') as file:
                pickle.dump(results, file)
            results = []
            t0 = time.time()
    with lzma.open(out_file, 'ab') as file:
        pickle.dump(results, file)
else:
    print("Unexpected path")
    exit(1)
