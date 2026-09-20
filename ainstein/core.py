from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class R:
    x: np.ndarray
    z: np.ndarray

@dataclass(frozen=True)
class Ex:
    pair: tuple[int,int]
    rs: tuple[R,R]
    y: np.ndarray
    probe: np.ndarray

@dataclass(frozen=True)
class World:
    train: tuple[Ex,...]
    validation: tuple[Ex,...]
    test: tuple[Ex,...]
    side: int = 4

def _orth(rng,d):
    q,r=np.linalg.qr(rng.normal(size=(d,d)))
    s=np.sign(np.diag(r)); s[s==0]=1
    return q*s

def _split(i,j,seed):
    h=(3*i+5*j+seed)%7
    return "test" if h==0 else "validation" if h==1 else "train"

def make_world(seed,n=12,d=4,noise=.05,repeats=3):
    rng=np.random.default_rng(seed)
    A=rng.normal(size=(n,d)); A/=np.linalg.norm(A,axis=1,keepdims=True)
    B=rng.normal(size=(n,d)); B/=np.linalg.norm(B,axis=1,keepdims=True)
    UA,UB,Q=_orth(rng,d),_orth(rng,d),_orth(rng,d*d)
    buckets={k:[] for k in ("train","validation","test")}
    for i in range(n):
        for j in range(n):
            y=Q@np.kron(A[i],B[j])
            for _ in range(repeats):
                a=R(UA@A[i]+noise*rng.normal(size=d),np.array([1.,0.]))
                b=R(UB@B[j]+noise*rng.normal(size=d),np.array([0.,1.]))
                rs=(a,b) if rng.random()<.5 else (b,a)
                p=rng.normal(size=d); p/=np.linalg.norm(p)+1e-12
                buckets[_split(i,j,seed)].append(Ex((i,j),rs,y.copy(),p))
    return World(*(tuple(buckets[k]) for k in ("train","validation","test")))

def roles(ex,mode="intact"):
    a=np.zeros_like(ex.rs[0].x); b=a.copy()
    for r in ex.rs:
        z=r.z
        if mode=="erased": z=np.array([.5,.5])
        elif mode=="shuffled": z=z[::-1]
        a+=z[0]*r.x; b+=z[1]*r.x
    return a,b

def _sum(a,b): return a+b
def _cat(a,b): return np.r_[a,b]
def _had(a,b): return a*b
def _sym(a,b): return np.kron(a,b)+np.kron(b,a)
def _outer(a,b): return np.kron(a,b)
FEATURES={"sum":_sum,"linear_concat":_cat,"hadamard":_had,"symmetric_outer":_sym,"ordered_outer":_outer}

def matrix(xs,f,mode="intact"):
    X=[]; Y=[]
    for e in xs:
        a,b=roles(e,mode); X.append(f(a,b)); Y.append(e.y)
    return np.asarray(X),np.asarray(Y)

def fit(X,Y,ridge=1e-4):
    X=np.column_stack([X,np.ones(len(X))]); R=np.eye(X.shape[1])*ridge; R[-1,-1]=0
    return np.linalg.solve(X.T@X+R,X.T@Y)

def pred(X,W): return np.column_stack([X,np.ones(len(X))])@W

def r2(Y,P):
    sse=np.sum((Y-P)**2); sst=np.sum((Y-Y.mean(axis=0,keepdims=True))**2)
    return float(1-sse/sst)

def _branch(xs):
    A=[];B=[];Y=[]
    for e in xs:
        a,b=roles(e); A.append(a);B.append(b);Y.append(e.y)
    return np.asarray(A),np.asarray(B),np.asarray(Y)

def _best_mix(Y,A,B):
    scores=[r2(Y,w*A+(1-w)*B) for w in np.linspace(0,1,101)]
    k=int(np.argmax(scores)); return k/100.0,scores[k]

def _lookup(train,test):
    d={}
    for e in train: d.setdefault(e.pair,[]).append(e.y)
    g=np.mean([e.y for e in train],axis=0); d={k:np.mean(v,axis=0) for k,v in d.items()}
    return np.asarray([d.get(e.pair,g) for e in test])

def _postmix(xs):
    return np.asarray([np.r_[.5*roles(e)[0],.5*roles(e)[1]] for e in xs])

def _rf_train(X,seed,width=128):
    rng=np.random.default_rng(seed)
    W=rng.normal(scale=2/np.sqrt(X.shape[1]),size=(X.shape[1],width))
    b=rng.uniform(-1,1,width)
    return np.tanh(X@W+b),(W,b)

def _rf(X,p):
    W,b=p
    return np.tanh(X@W+b)

def _apply(xs,ops,side=4):
    return np.einsum("nij,nj->ni",np.asarray(ops).reshape(-1,side,side),np.asarray([e.probe for e in xs]))

def _span(J,A,B):
    out=[]
    for j,a,b in zip(J,A,B):
        M=np.column_stack([a,b])
        c,*_=np.linalg.lstsq(M,j,rcond=None)
        q=M@c
        out.append(np.linalg.norm(j-q)/max(np.linalg.norm(j),1e-20))
    return float(np.mean(out))

def run_world(seed):
    w=make_world(seed); models={}; scores={}
    for name,f in FEATURES.items():
        X,Y=matrix(w.train,f); W=fit(X,Y); Xv,Yv=matrix(w.validation,f)
        models[name]=W; scores[name]=r2(Yv,pred(Xv,W))
    chosen=max(scores,key=scores.get); f=FEATURES[chosen]; W=models[chosen]
    Xt,Yt=matrix(w.test,f); joint=pred(Xt,W)

    At,Bt,Ytr=_branch(w.train); Av,Bv,Yv=_branch(w.validation); Ae,Be,_=_branch(w.test)
    wa,wb=fit(At,Ytr),fit(Bt,Ytr)
    pa_v,pb_v=pred(Av,wa),pred(Bv,wb)
    alpha,_=_best_mix(Yv,pa_v,pb_v)
    pa,pb=pred(Ae,wa),pred(Be,wb)
    mix=alpha*pa+(1-alpha)*pb

    erased=pred(matrix(w.test,f,"erased")[0],W)
    shuffled=pred(matrix(w.test,f,"shuffled")[0],W)
    lookup=_lookup(w.train,w.test)

    Xa=_postmix(w.train); Xe=_postmix(w.test)
    H,p=_rf_train(Xa,seed+10000)
    post=pred(_rf(Xe,p),fit(H,Ytr,1e-2))

    true_app=_apply(w.test,Yt); joint_app=_apply(w.test,joint)
    mix_app=_apply(w.test,mix); post_app=_apply(w.test,post)

    tp={e.pair for e in w.train}; vp={e.pair for e in w.validation}; ep={e.pair for e in w.test}
    m={
      "joint_heldout_operator_r2":r2(Yt,joint),
      "joint_heldout_operator_nrmse":float(np.linalg.norm(Yt-joint)/np.linalg.norm(Yt)),
      "branch_a_r2":r2(Yt,pa),
      "branch_b_r2":r2(Yt,pb),
      "convex_branch_mix_r2":r2(Yt,mix),
      "pair_lookup_r2":r2(Yt,lookup),
      "provenance_erased_r2":r2(Yt,erased),
      "provenance_shuffled_r2":r2(Yt,shuffled),
      "joint_span_residual":_span(joint,pa,pb),
      "heldout_application_r2":r2(true_app,joint_app),
      "convex_application_r2":r2(true_app,mix_app),
      "postmix_mlp_control_r2":r2(Yt,post),
      "postmix_mlp_application_r2":r2(true_app,post_app),
      "convex_weight_on_a":alpha
    }
    return {
      "seed":seed,
      "selected_family":chosen,
      "validation_r2":scores,
      "pair_leakage":bool(tp&vp or tp&ep or vp&ep),
      "metrics":m
    }

def aggregate(ws):
    names=ws[0]["metrics"]
    m={k:float(np.mean([w["metrics"][k] for w in ws])) for k in names}
    frac=float(np.mean([w["selected_family"]=="ordered_outer" for w in ws]))
    weak=max(m[k] for k in ("branch_a_r2","branch_b_r2","convex_branch_mix_r2","pair_lookup_r2"))
    drop=m["joint_heldout_operator_r2"]-m["provenance_erased_r2"]
    checks={
      "ordered_outer_selected_fraction_ge_0_90":frac>=.9,
      "heldout_operator_r2_ge_0_90":m["joint_heldout_operator_r2"]>=.9,
      "heldout_application_r2_ge_0_90":m["heldout_application_r2"]>=.9,
      "branch_convex_lookup_r2_le_0_10":weak<=.1,
      "provenance_drop_ge_0_40":drop>=.4,
      "joint_span_residual_ge_0_50":m["joint_span_residual"]>=.5,
      "transformer_edge_control_r2_ge_0_80":m["postmix_mlp_control_r2"]>=.8,
      "no_pair_leakage":not any(w["pair_leakage"] for w in ws)
    }
    return {
      "worlds":len(ws),
      "selected_family_counts":{k:sum(w["selected_family"]==k for w in ws) for k in FEATURES},
      "ordered_outer_selected_fraction":frac,
      "mean_metrics":m,
      "strongest_branch_convex_or_lookup_r2":weak,
      "provenance_erased_drop":drop,
      "gate_checks":checks,
      "gate_pass":all(checks.values())
    }
