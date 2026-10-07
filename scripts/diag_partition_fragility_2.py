"""LOG.md Iteration 37 (run locally on results/diag_gpt2, pulled from results/last_mlp). Is the k=100 hard partition fragile? Are cross-dataset partitions different but equally good?"""
import itertools, sys, time
import numpy as np, h5py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari

D = '/home/andrea/Documents/postdoc/Stanford/parcelmate/results/diag_gpt2'
DS = ['wikitext', 'bookcorpus', 'agnews', 'tldr17']
H = ['halfA', 'halfB']
R, P, PN = {}, {}, {}
for d in DS:
    for h in H:
        with h5py.File(f'{D}/conn/connectivity_{d}_{h}.h5', 'r') as f:
            R[(d, h)] = np.abs(np.nan_to_num(f['connectivity'][()].astype(np.float32)))
        with h5py.File(f'{D}/parc/parcellation_{d}_{h}.h5', 'r') as f:
            P[(d, h)] = f['parcellation'][()].argmax(1)
        with h5py.File(f'{D}/parc_null/parcellation_{d}_{h}.h5', 'r') as f:
            PN[(d, h)] = f['parcellation'][()].argmax(1)
N = R[(DS[0], 'halfA')].shape[0]
k = 100
rs = np.random.RandomState(0)
iu = np.triu_indices(N, 1)
sel = rs.choice(len(iu[0]), 2_000_000, replace=False)
I, J = iu[0][sel], iu[1][sel]

def block_pred(labels, M_fit, kk=None):
    kk = kk or labels.max() + 1
    O = np.zeros((N, kk), np.float32); O[np.arange(N), labels] = 1
    S = O.T @ M_fit @ O
    n = O.sum(0)
    cnt = np.outer(n, n) - np.diag(n)
    S = S - np.diag(np.einsum('ij,ij->j', O, (M_fit.diagonal()[:, None] * O)))
    B = np.where(cnt > 0, S / np.maximum(cnt, 1), 0)
    return B[labels[I], labels[J]]

def fid(labels, M_fit, M_eval):
    return np.corrcoef(block_pred(labels, M_fit), M_eval[I, J])[0, 1]

print('E1 fidelity on dataset e (block means fitted on e half A, scored on e half B), by WHOSE labels are used')
rows = []
for e in DS:
    own = fid(P[(e, 'halfA')], R[(e, 'halfA')], R[(e, 'halfB')])
    others = [fid(P[(d, 'halfA')], R[(e, 'halfA')], R[(e, 'halfB')]) for d in DS if d != e]
    nul = fid(PN[(e, 'halfA')], R[(e, 'halfA')], R[(e, 'halfB')])
    rnd = fid(rs.randint(0, k, N), R[(e, 'halfA')], R[(e, 'halfB')])
    ceil = np.corrcoef(R[(e, 'halfA')][I, J], R[(e, 'halfB')][I, J])[0, 1]
    rows.append((own, np.mean(others), nul, rnd))
    print('  %-10s own labels %.3f | other datasets\' labels %.3f | null partition %.3f | random %.3f | raw ceiling %.3f' % (e, own, np.mean(others), nul, rnd, ceil))
r = np.array(rows).mean(0)
print('  mean: own %.3f  others %.3f  null %.3f  random %.3f  -> others reach %.0f%% of own (above null)' % (*r, 100 * (r[1] - r[2]) / (r[0] - r[2])))
sys.stdout.flush()

print('\nE2 how fragile is a hard k=100 partition? re-cluster after perturbing ONE connectome')
def profiles(M):
    eps = 1 - 1e-3
    Q = np.arctanh(np.clip(M * eps, 0, eps)).astype(np.float32)
    np.fill_diagonal(Q, 0)
    Q = (Q - Q.mean(1, keepdims=True)) / (Q.std(1, keepdims=True) + 1e-8)
    thr = np.quantile(Q, 0.9, axis=1, keepdims=True)
    return np.where(Q >= thr, Q, 0).astype(np.float32)
def cluster(M, seed=0):
    X = PCA(100, random_state=seed).fit_transform(profiles(M))
    return KMeans(k, n_init=4, random_state=seed).fit_predict(X)
base = R[('wikitext', 'halfA')]
c0 = cluster(base)
c_rep = cluster(R[('wikitext', 'halfB')])
print('  reference: wikitext half A vs half B           ARI %.3f' % ari(c0, c_rep))
c_seed = cluster(base, seed=1)
print('  same matrix, different seed                    ARI %.3f' % ari(c0, c_seed))
sd = base[I, J].std()
for target in (0.9, 0.7, 0.5):
    for kind in ('iid', 'lowrank'):
        if kind == 'iid':
            E = rs.randn(N, N).astype(np.float32); E = (E + E.T) / np.sqrt(2)
        else:
            U = rs.randn(N, 50).astype(np.float32); E = U @ U.T / np.sqrt(50)
        np.fill_diagonal(E, 0)
        E *= sd / E[I, J].std()
        s = np.sqrt(1 / target ** 2 - 1)
        M2 = base + s * E
        rr = np.corrcoef(base[I, J], M2[I, J])[0, 1]
        c2 = cluster(np.abs(M2))
        print('  %-7s noise, connectome r with original %.2f   ARI %.3f   fidelity of perturbed labels on original %.3f' % (
            kind, rr, ari(c0, c2), fid(c2, base, R[('wikitext', 'halfB')])))
        sys.stdout.flush()
print('  own-label fidelity on wikitext for comparison  %.3f' % fid(c0, base, R[('wikitext', 'halfB')]))

print('\nE3 co-membership: given two units share a network in dataset d, how often in d\'?')
def comember(a, b, m=400000):
    x, y = rs.randint(0, N, m), rs.randint(0, N, m)
    s = a[x] == a[y]
    return (b[x][s] == b[y][s]).mean(), (b[x] == b[y]).mean()
w, ac, base_rate = [], [], []
for d in DS:
    p, br = comember(P[(d, 'halfA')], P[(d, 'halfB')]); w.append(p)
for d1, d2 in itertools.permutations(DS, 2):
    p, br = comember(P[(d1, 'halfA')], P[(d2, 'halfB')]); ac.append(p); base_rate.append(br)
print('  within %.3f   across %.3f   base rate %.3f   (across = %.1fx chance)' % (np.mean(w), np.mean(ac), np.mean(base_rate), np.mean(ac) / np.mean(base_rate)))
