"""LOG.md Iteration 37 (run locally on results/diag_gpt2, pulled from results/last_mlp). Diagnostics: why do connectomes agree across datasets (r ~0.3-0.6) while partitions do not?
GPT-2 MLP final arm (9,984 units), 4 prose datasets x 2 halves, dense |r| on disk."""
import glob, os, sys, time, itertools
import numpy as np, h5py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari

D = '/home/andrea/Documents/postdoc/Stanford/parcelmate/results/diag_gpt2'
DS = ['wikitext', 'bookcorpus', 'agnews', 'tldr17']
H = ['halfA', 'halfB']
t0 = time.time()
R = {}
for d in DS:
    for h in H:
        with h5py.File(f'{D}/conn/connectivity_{d}_{h}.h5', 'r') as f:
            R[(d, h)] = np.abs(np.nan_to_num(f['connectivity'][()].astype(np.float32)))
N = R[(DS[0], H[0])].shape[0]
iu = np.triu_indices(N, 1)
sel = np.random.RandomState(0).choice(len(iu[0]), 3_000_000, replace=False)
I, J = iu[0][sel], iu[1][sel]
print('loaded', N, 'units, %.0fs' % (time.time() - t0)); sys.stdout.flush()

def pairs():
    """(label, (d1,h1), (d2,h2)): within = same dataset other half; across = half A of d1 vs half B of d2."""
    for d in DS:
        yield 'within', (d, 'halfA'), (d, 'halfB')
    for d1, d2 in itertools.permutations(DS, 2):
        yield 'across', (d1, 'halfA'), (d2, 'halfB')

def summarize(name, fn):
    w, a = [], []
    for lab, x, y in pairs():
        (w if lab == 'within' else a).append(fn(x, y))
    print('  %-58s within %.3f   across %.3f   (across/within %.2f)' % (name, np.mean(w), np.mean(a), np.mean(a) / np.mean(w)))
    sys.stdout.flush()

def vec(M):
    return M[I, J].astype(np.float64)

def dcenter(M):
    m = M.copy().astype(np.float64)
    np.fill_diagonal(m, np.nan)
    r = np.nanmean(m, 1)
    g = np.nanmean(m)
    out = M - r[:, None] - r[None, :] + g
    return out

# ---------------- D1 connectome similarity, raw and with unit strength removed
print('\nD1 connectome similarity (Pearson over 3M sampled unit pairs)')
V = {k: vec(v) for k, v in R.items()}
summarize('raw |r|', lambda x, y: np.corrcoef(V[x], V[y])[0, 1])
S = {k: (v.sum(1) - 1) / (N - 1) for k, v in R.items()}
summarize('unit strength vectors (mean |r| per unit)', lambda x, y: np.corrcoef(S[x], S[y])[0, 1])
# share of |r| variance explained by the additive strength model s_i + s_j
for d in DS[:2]:
    M = R[(d, 'halfA')]
    pred = (S[(d, 'halfA')][I] + S[(d, 'halfA')][J])
    print('  %s: variance of |r| explained by s_i + s_j: %.2f' % (d, np.corrcoef(pred, V[(d, 'halfA')])[0, 1] ** 2))
DC = {k: vec(dcenter(v)) for k, v in R.items()}
summarize('|r| double-centred (strength removed)', lambda x, y: np.corrcoef(DC[x], DC[y])[0, 1])
L = {k: vec(dcenter(np.log(np.maximum(v, 1e-4)))) for k, v in R.items()}
summarize('log|r| double-centred (multiplicative strength removed)', lambda x, y: np.corrcoef(L[x], L[y])[0, 1])
del DC, L

# ---------------- D2 what the pipeline clusters: profiles
print('\nD2 unit profiles as the pipeline sees them')
def profiles(M, sparsify=True):
    eps = 1 - 1e-3
    P = np.arctanh(np.clip(M * eps, 0, eps)).astype(np.float32)
    np.fill_diagonal(P, 0)
    P = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-8)
    if sparsify:
        thr = np.quantile(P, 0.9, axis=1, keepdims=True)
        P = np.where(P >= thr, P, 0).astype(np.float32)
    return P
PR = {k: profiles(v) for k, v in R.items()}
def rowcorr(A, B):
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    return np.median((A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12))
summarize('per-unit profile corr, Fisher+z+top10% (pipeline)', lambda x, y: rowcorr(PR[x], PR[y]))
PZ = {k: profiles(v, sparsify=False) for k, v in R.items()}
summarize('per-unit profile corr, Fisher+z, no sparsification', lambda x, y: rowcorr(PZ[x], PZ[y]))
def topk_jacc(x, y, q=0.1):
    k = int(q * N)
    A = np.argpartition(-R[x], k, axis=1)[:, :k]
    B = np.argpartition(-R[y], k, axis=1)[:, :k]
    js = []
    for i in range(0, N, 50):
        a, b = A[i], B[i]
        js.append(len(np.intersect1d(a, b)) / len(np.union1d(a, b)))
    return np.mean(js)
summarize('top-10% partner overlap (Jaccard, every 50th unit)', topk_jacc)
del PZ

# ---------------- D3 stored partitions
print('\nD3 stored k = 100 partitions (ARI)')
P = {}
for d in DS:
    for h in H:
        with h5py.File(f'{D}/parc/parcellation_{d}_{h}.h5', 'r') as f:
            P[(d, h)] = f['parcellation'][()].argmax(1)
summarize('stored partitions, ARI', lambda x, y: ari(P[x], P[y]))

# ---------------- D4 re-cluster with pipeline steps ablated
print('\nD4 re-clustering with steps ablated (PCA-100, k-means, 4 restarts; ARI)')
def cluster(F, k=100, seed=0):
    X = PCA(100, random_state=seed).fit_transform(F)
    return KMeans(k, n_init=4, random_state=seed).fit_predict(X)
variants = {
    'pipeline: Fisher+z+top10%, k=100': lambda M: profiles(M),
    'no sparsification: Fisher+z, k=100': lambda M: profiles(M, sparsify=False),
    'raw |r| rows (no z, no sparsify), k=100': lambda M: M,
    'double-centred |r| rows, k=100': lambda M: dcenter(M).astype(np.float32),
}
for name, f in variants.items():
    t = time.time()
    C = {k: cluster(f(v)) for k, v in R.items()}
    summarize(name + ' [%.0fs]' % (time.time() - t), lambda x, y: ari(C[x], C[y]))
C = {k: cluster(profiles(v), k=20) for k, v in R.items()}
summarize('pipeline, k=20', lambda x, y: ari(C[x], C[y]))
print('done %.0fs' % (time.time() - t0))
