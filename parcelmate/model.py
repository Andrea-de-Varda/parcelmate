import math
import warnings
import os
import copy
import numpy as np
import time
from scipy import optimize, linalg as scipy_linalg
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
import torch
from transformers import AutoModel, AutoTokenizer

from parcelmate.constants import *
from parcelmate.data import *
from parcelmate.util import *
from parcelmate.metrics import blockmodel_refine, block_sse, center_connectivity
from parcelmate.plot import *


class PerturbedModel(torch.nn.Module):
    def __init__(self, model, perturbation_coordinates, perturbation_values=None, *args, **kwargs):
        super(PerturbedModel, self).__init__(*args, **kwargs)
        self.model = model
        self.perturbation_coordinates = perturbation_coordinates
        if perturbation_values is None:  # Default to zero (knockout)
            perturbation_values = np.zeros(len(self.perturbation_coordinates))
        assert len(perturbation_values) == len(perturbation_coordinates), \
            'perturbation_values must match perturbation_coordinates'
        self.perturbation_values = perturbation_values

        layers_attr = 'h'
        layer_indices = np.unique(perturbation_coordinates[:, 0])  # 0th dimension is layer
        layers = getattr(self.model, layers_attr)
        perturbation_coordinate_tensors = {}
        perturbation_value_tensors = {}
        layer_selection = {}  # key -> boolean mask into perturbation_coordinates, for value updates
        n_layers = len(layers)
        for l_ix in layer_indices:
            if l_ix == 0:
                key = 'embedding'
            elif l_ix == n_layers:
                # The last hidden state is emitted by the closing layer norm (ln_f), not by the last block
                key = 'final_norm'
            else:
                key = l_ix - 1 # Shifted down bc of embedding layer
            sel = perturbation_coordinates[:, 0] == l_ix  # 0th dimension is layer
            layer_coordinates = perturbation_coordinates[sel][:, 1]  # 1st dimension is hidden unit
            layer_coordinates = torch.nn.Parameter(
                torch.as_tensor(
                    layer_coordinates
                ),
                requires_grad=False
            )
            perturbation_coordinate_tensors[key] = layer_coordinates
            layer_values = perturbation_values[sel]
            layer_values = torch.nn.Parameter(
                torch.as_tensor(
                    layer_values,
                    dtype=model.dtype
                ),
                requires_grad=False
            )
            perturbation_value_tensors[key] = layer_values
            layer_selection[key] = sel

        self.perturbation_coordinate_tensors = perturbation_coordinate_tensors
        self.perturbation_value_tensors = perturbation_value_tensors
        self._layer_selection = layer_selection

        for l_ix in layer_indices:
            if l_ix == 0:
                _l_ix = 'embedding'
                source_layer = self.model.drop
            elif l_ix == n_layers:
                _l_ix = 'final_norm'
                source_layer = self.model.ln_f
            else:
                _l_ix = l_ix - 1  # Shifted down bc of embedding layer
                source_layer = layers[_l_ix]
            layer = PerturbedLayer(
                source_layer,
                perturbation_coordinates=self.perturbation_coordinate_tensors[_l_ix],
                perturbation_values=self.perturbation_value_tensors[_l_ix]
            )
            if _l_ix == 'embedding':
                self.model.drop = layer
            elif _l_ix == 'final_norm':
                self.model.ln_f = layer
            else:
                layers[_l_ix] = layer

    def set_perturbation_values(self, perturbation_values):
        """Replace the perturbation values in place, keeping the coordinates fixed.

        Needed for mean-ablation, where the replacement value is each unit's mean
        activation *under the domain currently being processed* -- a unit's mean differs
        substantially across domains (see the Iteration 1 diagnostics), so one set of
        values cannot serve every domain. Coordinates never change, so only the value
        tensors are rewritten.
        """
        perturbation_values = np.asarray(perturbation_values)
        assert len(perturbation_values) == len(self.perturbation_coordinates), \
            'perturbation_values must match perturbation_coordinates'
        self.perturbation_values = perturbation_values
        for key, coords in self.perturbation_coordinate_tensors.items():
            sel = self._layer_selection[key]
            target = self.perturbation_value_tensors[key]
            target.data.copy_(
                torch.as_tensor(perturbation_values[sel], dtype=target.dtype, device=target.device)
            )

    def forward(self, *args, **kwargs):
        out = self.model.forward(*args, **kwargs)
        return out


class PerturbedLayer(torch.nn.Module):
    def __init__(self, layer, perturbation_coordinates=None, perturbation_values=None, *args, **kwargs):
        super(PerturbedLayer, self).__init__(*args, **kwargs)
        self.layer = layer
        self.perturbation_coordinates = perturbation_coordinates
        self.perturbation_values = perturbation_values

    def forward(self, *args, **kwargs):
        out = self.layer.forward(*args, **kwargs)
        if self.perturbation_coordinates is not None:
            if isinstance(out, torch.Tensor):
                out[..., self.perturbation_coordinates] = self.perturbation_values
            else:
                out0 = out[0]
                out0[..., self.perturbation_coordinates] = self.perturbation_values
                out = (out0,) + out[1:]

        return out


def select_network_units(parcellation, network, knockout_thresh=0.5):
    """Boolean mask of units whose membership in `network` reaches the threshold.

    One network at a time, deliberately. The previous implementation OR-ed the mask over
    every column of the parcellation, so it could only ever build a single model with the
    union of all shared subnetworks lesioned at once, and could not ask what any one
    subnetwork contributes (LOG.md S1).
    """
    assert 0 <= network < parcellation.shape[1], \
        'network %d out of range for a parcellation with %d networks' % (network, parcellation.shape[1])

    return parcellation[:, network] >= knockout_thresh


def get_model_and_tokenizer(
        model_name,
        knockout_probs=None,
        knockout_thresh=0.5,
        coordinates=None,
        network=None,
        perturbation_values=None
):
    model = AutoModel.from_pretrained(model_name)
    if knockout_probs is not None:
        assert coordinates is not None, 'coordinates must be provided if knockout_probs is not None'
        assert network is not None, 'network must be provided if knockout_probs is not None'
        sel = select_network_units(knockout_probs, network, knockout_thresh=knockout_thresh)
        model = PerturbedModel(
            model,
            perturbation_coordinates=coordinates[sel],
            perturbation_values=perturbation_values  # None -> zeros (zero-ablation)
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def get_timecourses(
        model,
        input_ids,
        attention_mask,
        batch_size=8,
        highpass=None,
        lowpass=None,
        step=0.2,
        timecourse_pca_components=None,
        timecourse_ica_components=None,
        unit_type='hidden',
        units_per_layer=None,
        unit_seed=None,
        seed=None,
        verbose=True,
        indent=0,
        **kwargs
):
    """Stream the model over the inputs and return one timecourse per unit.

    `unit_type` decides what a "unit" is (LOG.md Iteration 14, YOLO 3):

      hidden  the residual stream at every layer boundary (`output_hidden_states`), the
              inherited choice: for GPT-2, 13 x 768. The residual stream has no privileged
              basis (any rotation gives an equivalent model), and dimension d at layer l is
              dimension d at layer l+1 minus one block's update, so the strongest structure
              in this connectome is 768 chains of 13 units.
      mlp     the post-nonlinearity MLP neurons of each block, captured as the input to the
              MLP's output projection (`mlp.c_proj`), which is what every transformers
              version feeds the activation into. GELU breaks rotational symmetry, so these
              units have a privileged basis and no cross-layer chain. GPT-2: 12 x 3072.

    `units_per_layer` keeps a fixed random subset per layer, drawn once from `unit_seed`
    (so the same subset serves every sample, domain and tree). 832 per layer makes an MLP
    run 12 x 832 = 9,984 units, the same count as the residual stream, so reliability and
    fidelity at a given k are compared on equal footing. Coordinates keep the ORIGINAL
    neuron index, so a unit remains traceable to the model.
    """
    assert unit_type in ('hidden', 'mlp'), 'unit_type must be hidden or mlp, got %r' % (unit_type,)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if verbose:
        stderr('%sGetting timecourses (unit_type=%s)\n' % (' ' * indent, unit_type))
    hooks, captured = [], {}
    if unit_type == 'mlp':
        blocks = getattr(model, 'h', None)
        assert blocks is not None and len(blocks) and hasattr(blocks[0], 'mlp') \
            and hasattr(blocks[0].mlp, 'c_proj'), (
            'unit_type=mlp expects a GPT-2 style model with `h[i].mlp.c_proj` (got %s); '
            'a wrapped/perturbed model is not supported here' % type(model).__name__)

        def make_hook(layer):
            def hook(module, inputs):
                captured[layer] = inputs[0]
            return hook

        for layer, block in enumerate(blocks):
            hooks.append(block.mlp.c_proj.register_forward_pre_hook(make_hook(layer)))
    unit_index = None  # per layer: which columns are kept
    timecourses = None
    coordinates = None
    t = 0
    T = int(attention_mask.detach().numpy().sum())
    B = int(math.ceil(input_ids.size(0) / batch_size))
    indent += 2
    for i in range(0, input_ids.size(0), batch_size):
        if verbose:
            stderr('\r%sBatch %d/%d' % (' ' * indent, i // batch_size + 1, B))
        _input_ids = input_ids[i:i + batch_size].to(device)
        _attention_mask = attention_mask[i:i + batch_size].to(device)
        with torch.no_grad():  # Activations are only ever read; graphs here doubled GPU memory
            captured.clear()
            output = model(
                input_ids=_input_ids,
                attention_mask=_attention_mask,
                output_hidden_states=(unit_type == 'hidden'),
                **kwargs
            )
            if unit_type == 'hidden':
                states = output.hidden_states
            else:
                assert len(captured) == len(hooks), \
                    'captured %d MLP layers, expected %d' % (len(captured), len(hooks))
                states = [captured[l] for l in range(len(hooks))]
        mask = _attention_mask.detach().cpu().numpy().astype(bool)
        _t = int(mask.sum())
        if unit_index is None:
            unit_index = []
            for s, state in enumerate(states):
                width = int(state.size(-1))
                if units_per_layer and units_per_layer < width:
                    rng = np.random.RandomState(derive_seed(unit_seed, 'units', s) % (2 ** 32))
                    unit_index.append(np.sort(rng.choice(width, int(units_per_layer), replace=False)))
                else:
                    unit_index.append(np.arange(width))
        if timecourses is None:
            out_shape = (sum(len(ix) for ix in unit_index), T)
            timecourses = np.zeros(out_shape, dtype=np.float32)
        if coordinates is None:
            coordinates = np.zeros((sum(len(ix) for ix in unit_index), 2), dtype=np.int32)
        h = 0
        for s, state in enumerate(states):
            ix = unit_index[s]
            _h = len(ix)
            timecourses[h:h + _h, t:t + _t] = bandpass(
                state.detach().cpu().numpy()[mask][:, ix].T,
                step=step,
                lower=highpass,
                upper=lowpass
            )
            coordinates[h:h + _h, 0] = s
            coordinates[h:h + _h, 1] = ix
            h += _h
        t += _t
    if verbose:
        stderr('\n')
    for hk in hooks:
        hk.remove()

    model.to('cpu')
    torch.cuda.empty_cache()

    # Per-unit activation statistics, computed here because the PCA/ICA transforms below
    # replace the token axis with a component axis, after which a "mean activation" no
    # longer exists. Used as the replacement value for mean-ablation (LOG.md S1) and, for
    # the std, to size variance-matched random-lesion controls later.
    unit_means = timecourses.mean(axis=-1)
    unit_stds = timecourses.std(axis=-1)
    n_obs = timecourses.shape[-1]

    if timecourse_pca_components:
        t = timecourses.shape[-1]
        n_components = min(timecourse_pca_components, t)
        if verbose:
            stderr('%sPCA transforming (n components = %s)' % (' ' * indent, n_components))
        t1 = time.time()
        n_components = min(n_components, t)
        m = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, svd_solver='auto', whiten=True, random_state=seed))
        ])
        timecourses = m.fit_transform(timecourses)
        stderr(' (%0.2fs)\n' % (time.time() - t1))
    if timecourse_ica_components:
        t = timecourses.shape[-1]
        n_components = min(timecourse_ica_components, t)
        n_components = min(n_components, t)
        if verbose:
            stderr('%sICA transforming (n components = %s)' % (' ' * indent, n_components))
        t1 = time.time()
        m = Pipeline([
            ('scaler', StandardScaler()),
            ('ica', FastICA(n_components=n_components, whiten='unit-variance', random_state=seed))
        ])
        timecourses = m.fit_transform(timecourses)
        stderr(' (%0.2fs)\n' % (time.time() - t1))

    return dict(
        timecourses=timecourses,  # <n_neurons, n_tokens/n_components>
        coordinates=coordinates,  # <n_neurons>
        unit_means=unit_means,  # <n_neurons>
        unit_stds=unit_stds,  # <n_neurons>
        n_obs=n_obs  # tokens contributing to the two above, for pooling across samples
    )


def pool_unit_stats(means, stds, counts):
    """Pool per-unit means and stds over samples of possibly unequal token count.

    Means combine linearly; stds go through the second moment, since averaging stds
    directly would understate the spread whenever the sample means differ.
    """
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)[:, None]
    n = counts.sum()
    mean = (means * counts).sum(axis=0) / n
    second = ((stds ** 2 + means ** 2) * counts).sum(axis=0) / n
    var = np.maximum(second - mean ** 2, 0.0)  # clamp float error at exact-zero variance

    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def get_connectivity(timecourses, n_components=None, seed=None, use_gpu=None):
    X = timecourses
    if n_components:
        m = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, random_state=seed))
        ])
        X = m.fit_transform(X)
    R = correlate(X, rowvar=True, use_gpu=use_gpu)

    return R


def sample_parcellations(
        connectivity,
        n_networks=50,
        n_samples=100,
        binarize_connectivity=True,
        connectivity_pca_components=None,
        connectivity_ica_components=None,
        clustering_kwargs=None,
        legacy_binarize=False,
        fisher_transform=False,
        standardize_profiles=False,
        clustering='minibatch',
        pca_whiten=True,
        blockmodel_refine_labels=False,
        blockmodel_max_iter=50,
        binarize_scope='row',
        sparsify_fisher=False,
        sparsify_profiles=False,
        blockmodel_center=None,
        signed_connectivity=None,
        ica_max_iter=1000,
        ica_tol=1e-4,
        seed=None,
        verbose=True,
        indent=0
):
    """Cluster units by their connectivity profiles.

    Three arms are supported, for the reliability/fidelity comparison (see
    PARCELLATION_DESIGN.md):

      legacy    binarize_connectivity=True, legacy_binarize=True, pca=200
                Reproduces the pre-2026-09-03 behaviour, in which the quantile was
                broadcast column-wise and the binarized matrix came out transposed, so
                clustering was partly driven by hubness. Kept only as a comparison arm.
      current   binarize_connectivity=True, legacy_binarize=False, pca=200
      ablation  binarize_connectivity=False, fisher_transform=True, pca=None
                Keeps the magnitudes (Fisher-transformed to stabilize variance) and drops
                the PCA truncation and whitening entirely.
      vmf       ...plus standardize_profiles=True
                The Yeo et al. (2011) construction: each unit is described by its
                connectivity profile, standardized, and units are grouped by the similarity
                of those profiles. See `standardize_profiles` below for why this is exactly
                spherical k-means and what it removes.

    `clustering` selects the algorithm run on the prepared profiles (LOG.md Iteration 14):

      minibatch  sklearn MiniBatchKMeans, the inherited default. Stochastic and rarely at a
                 local optimum: the restart-split ceiling of the ladder run read 0.42-0.64,
                 i.e. two consensuses of the SAME matrix disagreed almost as much as two
                 halves of the data did, so reliability was tracking the optimizer.
      kmeans     full Lloyd k-means, k-means++ init, one init per restart. On 9,984 units
                 an iteration is one n x k x d product; affordable, and converged.
      ward       Ward agglomerative on the same features. Deterministic, so a single
                 sample is the parcellation and the restart-split ceiling is 1 by
                 construction; `n_samples` is forced to 1.
      ward_kmeans
                 Ward, then Lloyd k-means started from Ward's centroids (LOG.md Iteration
                 17, T2). Deterministic like Ward, so `n_samples` is forced to 1, but it
                 ends at a local optimum of the k-means objective, which Ward's greedy
                 merges do not reach. `clustering_kwargs` go to the k-means step.

      ica        MRI-style spatial ICA (LOG.md Iteration 15, YOLO 4). Needs
                 `signed_connectivity`, the signed correlation matrix: the whitened spatial
                 PCs are its top-k eigenvectors scaled to unit variance over units, which
                 is exactly the spatial PCA MELODIC would compute from the token x unit
                 matrix, so no activations are stored. FastICA over units then gives k
                 independent spatial maps per restart; each map's sign is fixed to positive
                 skew, maps are z-scored, and winner-take-all over components gives the
                 hard labels that go through the same alignment and consensus as k-means
                 (Yeo et al. 2011 compared clustering with ICA this way). The z-scored maps
                 are returned as `maps` so the consensus map and a soft reliability can be
                 scored too. The |r| profile preprocessing does not apply to this path.

    `binarize_scope='global'` thresholds the whole matrix at its 90th percentile instead of
    each row at its own, so hubs keep more partners than weak units (the row-wise version
    equalises density and thereby erases hubness from the input). `sparsify_fisher` keeps the
    Fisher-transformed magnitudes above the same per-row (or global) threshold and zeroes
    the rest: the missing rung between binarizing and keeping the dense magnitudes.

    `pca_whiten=False` keeps the PCA truncation but not the whitening, which gives the last
    retained noise direction the same weight as the first structured one. `blockmodel_refine_labels`
    follows each restart with `metrics.blockmodel_refine` on the RAW |r| the scorer targets,
    so the arm optimizes the block-model error that fidelity measures rather than the
    row-profile error k-means minimizes (YOLO 2).

    `sparsify_profiles` keeps the top 10% of each STANDARDIZED profile and re-standardizes
    (`sparsify_standardized`): the sparsity of the row-binarized arms with the row scale of
    the standardized ones (LOG.md Iteration 17, T2). `blockmodel_center` ('double' or
    'degree') centres the refinement target first (`metrics.center_connectivity`), so the
    refinement can gain only by fitting the pattern of the connectivity, not each unit's
    overall level (T5).
    """
    assert clustering in ('minibatch', 'kmeans', 'ward', 'ward_kmeans', 'ica'), \
        'clustering must be minibatch, kmeans, ward, ward_kmeans or ica, got %r' % (clustering,)
    assert binarize_scope in ('row', 'global'), \
        'binarize_scope must be row or global, got %r' % (binarize_scope,)
    assert not (sparsify_fisher and not fisher_transform), \
        'sparsify_fisher keeps Fisher magnitudes, so it needs fisher_transform: true'
    assert not (sparsify_fisher and binarize_connectivity), \
        'sparsify_fisher and binarize_connectivity are alternatives'
    assert not (sparsify_profiles and not standardize_profiles), \
        'sparsify_profiles sparsifies the standardized profiles, so it needs standardize_profiles: true'
    assert not (sparsify_profiles and (sparsify_fisher or binarize_connectivity)), \
        'sparsify_profiles, sparsify_fisher and binarize_connectivity are alternatives'
    # z-scores already share one scale, so a global threshold would keep more partners in
    # heavy-tailed rows and put a per-unit property back into the profiles.
    assert not (sparsify_profiles and binarize_scope != 'row'), \
        'sparsify_profiles is row-wise; binarize_scope must be row'
    if blockmodel_center in ('none', 'None', False):
        blockmodel_center = None
    assert blockmodel_center in (None, 'double', 'degree'), \
        'blockmodel_center must be null, double or degree, got %r' % (blockmodel_center,)
    assert not (blockmodel_center and not blockmodel_refine_labels), \
        'blockmodel_center sets the refinement target, so it needs blockmodel_refine_labels: true'
    if clustering in ('ward', 'ward_kmeans') and n_samples != 1:
        if verbose:
            stderr('%s%s is deterministic: n_samples %d -> 1\n' % (' ' * indent, clustering, n_samples))
        n_samples = 1
    if verbose:
        stderr('%sSampling (n_networks=%d, clustering=%s)\n' % (' ' * indent, n_networks, clustering))
    indent += 2

    if clustering_kwargs is None:
        clustering_kwargs = {}
    # One RNG shared across the reductions and all clustering restarts. Drawing from a
    # single stream (rather than reusing one fixed random_state) keeps the restarts
    # different from one another, which the consensus averaging depends on, while making
    # the whole set of restarts reproducible.
    rng = np.random.RandomState(seed if seed is None else int(seed) % (2 ** 32))

    if clustering == 'ica':
        assert signed_connectivity is not None, \
            'clustering=ica needs signed_connectivity (the signed correlation matrix)'
        assert not (binarize_connectivity or fisher_transform or standardize_profiles
                    or connectivity_pca_components or connectivity_ica_components
                    or blockmodel_refine_labels), (
            'clustering=ica works on the signed correlation matrix directly; set '
            'binarize_connectivity, fisher_transform, standardize_profiles and '
            'blockmodel_refine_labels to false and the *_components to null')
        Rs = np.nan_to_num(np.asarray(signed_connectivity, dtype=np.float64))
        n_units = Rs.shape[0]
        if verbose:
            stderr('%sTop-%d eigenvectors of the signed correlation matrix' % (' ' * indent, n_networks))
        t1 = time.time()
        _, evecs = scipy_linalg.eigh(Rs, subset_by_index=[n_units - n_networks, n_units - 1])
        del Rs
        Z = evecs * np.sqrt(n_units)   # whitened spatial PCs: unit variance over units
        if verbose:
            stderr(' (%0.2fs)\n%sDrawing samples\n' % (time.time() - t1, ' ' * indent))
        samples = np.zeros((n_samples, n_units))
        scores = np.zeros(n_samples)   # FastICA exposes no objective; restarts are unweighted
        maps = np.zeros((n_samples, n_units, n_networks), dtype=np.float32)
        for i in range(n_samples):
            if verbose and n_samples > 1:
                stderr('\r%s  Sample %d/%d' % (' ' * indent, i + 1, n_samples))
            ica = FastICA(n_components=n_networks, whiten='unit-variance', random_state=rng,
                          max_iter=ica_max_iter, tol=ica_tol)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')   # non-convergence is handled by the consensus
                S = ica.fit_transform(Z)           # (n_units, n_networks): the spatial maps
            S = S - S.mean(axis=0)
            skew = (S ** 3).mean(axis=0)
            S *= np.where(skew < 0, -1.0, 1.0)     # positive skew: the "active" tail is up
            S /= S.std(axis=0) + 1e-12
            maps[i] = S
            samples[i] = S.argmax(axis=1)          # winner-take-all
        if verbose and n_samples > 1:
            stderr('\n')
        return dict(samples=samples, scores=scores, maps=maps)

    X = connectivity
    assert not (binarize_connectivity and fisher_transform), \
        'fisher_transform is pointless after binarizing: arctanh of a 0/1 matrix is 0/inf'
    if binarize_connectivity:
        if legacy_binarize:
            # Deliberately reproduces the pre-2026-09-03 bug (LOG.md S2): without keepdims
            # the quantile broadcasts along the last axis, so the test is X[i,j] > q[j] and
            # the result is the transpose of a row-wise threshold. Row densities then vary
            # with hubness instead of being uniform. Comparison arm only -- never a default.
            X = (X > np.quantile(X, 0.9, axis=1)).astype(int)
        elif binarize_scope == 'global':
            # One threshold for the whole matrix: hubs keep many partners, weak units few.
            X = (X > np.quantile(X, 0.9)).astype(int)
        else:
            X = (X > np.quantile(X, 0.9, axis=1, keepdims=True)).astype(int)
    elif fisher_transform:
        # Variance-stabilizing, and it spreads out the crowded high-|r| tail that otherwise
        # dominates Euclidean distances. Copy first: fisher() writes in place, and the
        # caller's connectivity array is reused across arms. Clip before transforming --
        # arctanh is NaN outside [-1, 1], and while correlations are bounded in theory, a
        # value fractionally above 1 from accumulation error would otherwise poison a whole
        # row silently rather than raising.
        # The clip below is for accumulation error a hair above 1, NOT a licence to feed
        # this branch something that is not a correlation. A surrogate-normalized matrix
        # (|z| = |r| / sigma, LOG.md Iteration 10) runs to tens, and fisher() would map every
        # entry above 1 to the same 3.8 with no error raised -- silently binarizing the arm.
        # run_parcellation turns this branch off for normalized trees; this assert is what
        # makes any other route to it loud.
        assert X.max() <= 1.0 + 1e-6, (
            'fisher_transform on a matrix with entries up to %.3g: not a correlation matrix. '
            'If the tree is surrogate-normalized, the Fisher transform is neither needed '
            '(the normalization already stabilizes the variance) nor defined.' % X.max())
        X = fisher(np.clip(np.array(X, dtype=np.float64, copy=True), -1.0, 1.0))
        assert np.isfinite(X).all(), 'Fisher transform produced non-finite values'
        # Zero the self-connection. arctanh maps the diagonal (|r| = 1) to 3.80 while a
        # typical off-diagonal is 0.038 -- a hundredfold spike carrying a median 24.7% of
        # each unit's squared profile norm, measured on wikitext. It is pure artifact: the
        # diagonal is 1 by construction and says nothing about connectivity. Left in it
        # would handicap this arm against the binarized ones, where the diagonal is 1 of
        # 999 kept partners (0.1% of the row) and therefore harmless.
        np.fill_diagonal(X, 0.0)
        if sparsify_fisher:
            thr = np.quantile(X, 0.9) if binarize_scope == 'global' \
                else np.quantile(X, 0.9, axis=1, keepdims=True)
            X = np.where(X > thr, X, 0.0)
    if standardize_profiles:
        # z-score each unit's profile across its columns. Two things follow, and both are
        # the point of this arm.
        #
        # 1. It is exactly spherical k-means. z-scoring puts every row on a sphere of
        #    radius sqrt(n), so ||z_i - z_j||^2 = 2n(1 - r_ij) -- Euclidean distance
        #    becomes an exact monotone function of the correlation *between profiles*.
        #    L2-normalizing afterwards would divide everything by the same constant and
        #    change nothing, which is why no separate normalization step appears here.
        #    This is the Yeo et al. (2011) fMRI parcellation construction.
        #
        # 2. It removes hubness. Without it, a unit's overall connection strength enters
        #    the distance, so two units with the same connectivity *pattern* but different
        #    overall strength are far apart. Measured on the first full run, the
        #    unstandardized Fisher arm had AMI 0.354 with hubness decile -- it was
        #    substantially grouping units by how strongly connected they are rather than by
        #    what they connect to. Standardizing asks only about the pattern.
        #
        # The tradeoff is real: if hubness carries genuine functional signal, this discards
        # it. `triviality_ami_hubness` and `fidelity_within` together adjudicate -- hubness
        # was nuisance if AMI falls and fidelity holds, signal if fidelity falls with it.
        X = standardize_array(np.asarray(X, dtype=np.float64), axis=-1)
        if sparsify_profiles:
            X = sparsify_standardized(X)
    if connectivity_pca_components:
        n_components = connectivity_pca_components
        if n_components == 'auto':
            n_components = n_networks - 1
        if verbose:
            stderr('%sPCA transforming (n components = %s)' % (' ' * indent, n_components))
        t1 = time.time()
        n_components = min(n_components, X.shape[-1])
        m = PCA(n_components=n_components, svd_solver='auto', whiten=bool(pca_whiten), random_state=rng)
        X = m.fit_transform(X)
        stderr(' (%0.2fs)\n' % (time.time() - t1))
    if connectivity_ica_components:
        n_components = connectivity_ica_components
        if n_components == 'auto':
            n_components = n_networks - 1
        n_components = min(n_components, X.shape[-1])
        if verbose:
            stderr('%sICA transforming (n components = %s)' % (' ' * indent, n_components))
        t1 = time.time()
        m = FastICA(n_components=n_components, whiten='unit-variance', random_state=rng)
        X = m.fit_transform(X)
        stderr(' (%0.2fs)\n' % (time.time() - t1))

    if verbose:
        stderr('%sDrawing samples\n' % (' ' * indent))
    indent += 2
    n_units = X.shape[0]
    samples = np.zeros((n_samples, n_units))
    scores = np.zeros(n_samples)
    R_target = None
    if blockmodel_refine_labels:
        # The scorer's target is |r| itself (util.connectivity_matrix), not the Fisher or
        # standardized features k-means saw, so the refinement runs on `connectivity`.
        R_target = np.asarray(connectivity, dtype=np.float64)
        if blockmodel_center:
            # Fidelity is still scored on raw |r|; only what the refinement may fit changes.
            # Uncentred, the refinement bought within-domain fit with hubness (Iteration 16).
            R_target = center_connectivity(R_target, blockmodel_center)
    for i in range(n_samples):
        if verbose and n_samples > 1:
            stderr('\r%sSample %d/%d' % (' ' * indent, i + 1, n_samples))
        _clustering_kwargs = dict(clustering_kwargs)
        if clustering == 'ward':
            m = AgglomerativeClustering(n_clusters=n_networks, linkage='ward', **_clustering_kwargs)
            _sample = m.fit_predict(X)
            _score = 0.0  # no inertia; every restart is identical anyway
        elif clustering == 'ward_kmeans':
            # Lloyd from Ward's centroids. A tree cut at k leaves no cluster empty, so every
            # centroid is defined, and with an explicit init Lloyd is deterministic.
            ward_labels = AgglomerativeClustering(n_clusters=n_networks, linkage='ward').fit_predict(X)
            onehot = np.zeros((n_units, n_networks))
            onehot[np.arange(n_units), ward_labels] = 1.0
            centers = (onehot.T @ np.asarray(X, dtype=np.float64)) / onehot.sum(axis=0)[:, None]
            _clustering_kwargs.setdefault('n_init', 1)
            _clustering_kwargs.setdefault('random_state', rng)  # so nothing falls back to the global RNG
            m = KMeans(n_clusters=n_networks, init=centers, **_clustering_kwargs)
            _sample = m.fit_predict(X)
            _score = m.inertia_
        else:
            _clustering_kwargs.setdefault('random_state', rng)  # Explicit config still wins
            if clustering == 'kmeans':
                _clustering_kwargs.setdefault('n_init', 1)  # restarts are the outer loop
                m = KMeans(n_clusters=n_networks, **_clustering_kwargs)
            else:
                m = MiniBatchKMeans(n_clusters=n_networks, **_clustering_kwargs)
            _sample = m.fit_predict(X)
            _score = m.inertia_
        if blockmodel_refine_labels:
            _sample, _score = blockmodel_refine(R_target, _sample, n_networks,
                                                max_iter=blockmodel_max_iter)
        samples[i, :] = _sample
        scores[i] = _score

    if n_samples > 1:
        stderr('\n')

    return dict(
        samples=samples,  # <n_samples, n_units>
        scores=scores  # <n_samples>
    )


def sparsify_standardized(Z, q=0.9):
    """Keep each standardized profile's entries above its q-quantile, zero the rest, and
    re-standardize (LOG.md Iteration 17, T2).

    The ORDER is the point. The kept set is the same whether one thresholds the magnitudes
    or their z-scores, since a z-score is a monotone function of its row, but the kept
    values are not. Thresholded magnitudes keep their offset from zero, so the kept entries
    of a unit with a high baseline stand far above the zeros (a nearly binary profile) while
    those of a unit with a low baseline are graded, and the unit's level leaks back into the
    shape of its profile. Kept z-scores start from the same place in every row, so two units
    with the same pattern at different levels get identical profiles. Re-standardizing puts
    each row back on the sphere, so Euclidean distance is again a monotone function of the
    correlation between profiles.
    """
    thr = np.quantile(Z, q, axis=1, keepdims=True)
    return standardize_array(np.where(Z > thr, Z, 0.0), axis=-1)


def _align_samples(
        samples,
        w=None,
        n_alignments=None,
        shuffle=False,
        greedy=True,
        seed=None,
        verbose=True,
        indent=0
):
    rng = np.random.RandomState(seed if seed is None else int(seed) % (2 ** 32))
    if w is None:
        _w = 1
    else:
        _w = w[0]
    n_samples = samples.shape[0]
    n_units = samples.shape[1]
    if samples.ndim == 3:
        # Soft maps (n_samples, n_units, n_networks), e.g. ICA components: aligned by the
        # same Hungarian matching on standardized maps, averaged into a consensus map.
        n_networks = samples.shape[2]
        reference = samples[0].T.astype(float)
    else:
        n_networks = samples.max() + 1
        reference = (samples[0][None, ...] == np.arange(n_networks)[..., None]).astype(float)
    parcellation = None
    C = 0

    # Align subsequent samples
    if shuffle:
        s_ix = rng.permutation(n_samples)
        samples = samples[s_ix]
    n = n_alignments
    if n is None:
        n = n_samples
    i = 0
    for i_cum in range(n):
        if verbose:
            stderr('\r%sAlignment %d/%d' % (' ' * indent, i_cum + 1, n))

        if w is not None:
            _w = w[i]
        else:
            _w = 1

        if _w != 0:
            if len(samples.shape) == 2:
                s = (samples[i][None, ...] == np.arange(n_networks)[..., None])
            else:
                s = samples[i].T
            s = s.astype(float)
            _reference = standardize_array(reference)
            _s = standardize_array(s)
            scores = np.dot(
                _reference,
                _s.T,
            ) / n_units

            _, ix_r = optimize.linear_sum_assignment(scores, maximize=True)
            s = s[ix_r]
            if parcellation is None:
                parcellation = s * _w
            else:
                parcellation = parcellation + s * _w
            if greedy:
                reference = parcellation
            C += _w

        i += 1
        if i >= n_samples:
            i = 0
            if shuffle:
                s_ix = rng.permutation(n_samples)
                samples = samples[s_ix]

    if verbose and n > 0:
        stderr('\n')

    parcellation = parcellation / C

    return parcellation


def align_samples(
        samples,
        scores,
        n_alignments=None,
        weight_samples=False,
        seed=None,
        verbose=True,
        indent=0
):
    if verbose:
        stderr('%sAligning samples\n' % (' ' * indent))
    indent += 1

    s_ix = np.argsort(scores)
    samples = samples[s_ix]
    scores = scores[s_ix]
    if weight_samples:
        # Min-max normalize inertias to [0, 1] and flip, so the best (lowest-inertia) sample gets weight 1
        # and the worst gets weight 0. Raw inertias are unbounded, so 1 - scores would go negative.
        w = 1 - minmax_normalize_array(scores)
    else:
        w = None

    parcellation = _align_samples(
        samples,
        w=w,
        n_alignments=n_alignments,
        shuffle=False,
        greedy=True,
        seed=seed,
        verbose=verbose,
        indent=indent + 2
    ).T

    indent -= 1

    return parcellation


def run_connectivity(
        model_name='gpt2',
        output_dir=OUTPUT_DIR,
        n_samples=N_SAMPLES,
        domains=('wikitext', 'bookcorpus', 'agnews', 'tldr17', 'codeparrot', 'random', 'whitespace'),
        seq_len=1024,
        n_tokens=None,
        split='train',
        take=100000,
        wrap=True,
        shuffle=True,
        batch_size=8,
        highpass=None,
        lowpass=None,
        step=0.2,
        timecourse_pca_components=None,
        timecourse_ica_components=None,
        unit_type='hidden',
        units_per_layer=None,
        eps=1e-3,
        data_kwargs=None,
        model_kwargs=None,
        knockout_filepath=None,
        knockout_network=None,
        knockout_thresh=0.5,
        ablation='mean',
        ablation_stats_dir=None,
        null_model=None,
        null_output_dir=None,
        n_surrogates=0,
        seed=None,
        overwrite=False,
        verbose=True,
        indent=0
):
    """Estimate unit-by-unit connectivity, optionally alongside a null.

    `null_model='circshift'` additionally computes connectivity from independently circularly
    shifted timecourses and writes it to `null_output_dir`, using identical filenames so
    every downstream step can be pointed at either tree unchanged. The null is computed
    inside the same forward pass: the model queries dominate the cost, so this is far
    cheaper than a second full run, and it guarantees the null sees exactly the same
    tokens as the real data.

    `n_surrogates=K` additionally computes K FURTHER circular shifts per sample and stores
    the per-pair variance of the resulting correlations as `surrogate_var`. An arm that sets
    `normalize='surrogate'` then clusters effect sizes r_ij / sigma_ij, which removes the positive mean
    field that `|r|` otherwise manufactures out of noise -- see `util.connectivity_matrix`
    for why that field existed and LOG.md Iteration 9 for what it was doing to the metrics.
    The surrogates are drawn under a different seed key from the scored null, so the null
    tree is never whitened by its own noise; K=32 gives each variance a relative error of
    about 1/sqrt(2K) = 12%, and that error is independent across pairs, so it costs a little
    power but cannot manufacture a coherent field of its own.

    Cost is K correlation matmuls per sample. The forward passes, which dominate the step,
    are reused, so this is far cheaper than K extra runs.
    """
    if data_kwargs is None:
        data_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
    set_seed(seed)
    if n_tokens is None:
        n_tokens = (N_TOKENS // (seq_len * batch_size)) * seq_len * batch_size

    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_NAME)
    assert null_model in (None, 'circshift'), 'Unrecognized null_model: %s' % null_model
    if null_model is not None:
        assert null_output_dir, 'null_output_dir must be set when null_model is requested'
        assert os.path.abspath(null_output_dir) != os.path.abspath(output_dir), \
            'null_output_dir must differ from output_dir; filenames are identical in both trees'
    null_connectivity_dir = os.path.join(null_output_dir, CONNECTIVITY_NAME) if null_model else None

    assert knockout_filepath is None or unit_type == 'hidden', \
        'knockout lesions residual-stream units; unit_type=%r is not supported there' % (unit_type,)
    knockout_probs = knockout_coordinates = None
    if knockout_filepath is not None:
        data = load_h5_data(knockout_filepath, verbose=verbose, indent=indent)
        assert 'parcellation' in data, 'If provided, knockout_filepath must contain the field "parcellation"'
        knockout_probs = data['parcellation']
        knockout_coordinates = data['coordinates']

    assert ablation in ('mean', 'zero'), 'ablation must be "mean" or "zero", got %s' % ablation
    knockout_sel = None
    if knockout_probs is not None:
        knockout_sel = select_network_units(knockout_probs, knockout_network, knockout_thresh=knockout_thresh)
        if verbose:
            stderr('%sLesioning network %d: %d/%d units at thresh %s (%s-ablation)\n' % (
                ' ' * indent, knockout_network, int(knockout_sel.sum()),
                knockout_sel.size, knockout_thresh, ablation
            ))

    model, tokenizer = get_model_and_tokenizer(
        model_name,
        knockout_probs=knockout_probs,
        coordinates=knockout_coordinates,
        knockout_thresh=knockout_thresh,
        network=knockout_network
    )

    if isinstance(domains, str):
        domains = (domains,)

    for domain in domains:
        if verbose:
            stderr('%sRunning connectivity for %s\n' % (' ' * indent, domain))
        indent += 2
        _data_kwargs = copy.deepcopy(data_kwargs)
        if domain == 'wikitext':
            _data_kwargs.update(dict(
                dataset='Salesforce/wikitext',
                tokenizer= tokenizer,
                name='wikitext-103-raw-v1',
            ))
        elif domain == 'bookcorpus':
            # Parquet mirror; `bookcorpus` is script-based and unsupported since datasets 4.x.
            _data_kwargs.update(dict(
                dataset='rojagtap/bookcorpus'
            ))
        elif domain == 'agnews':
            _data_kwargs.update(dict(
                dataset='fancyzhx/ag_news'
            ))
        elif domain == 'codeparrot':
            _data_kwargs.update(dict(
                dataset='codeparrot/codeparrot-clean'
            ))
        elif domain == 'tldr17':
            # 50k-post parquet subset; `webis/tldr-17` is script-based and unsupported since datasets 4.x.
            _data_kwargs.update(dict(
                dataset='dim/tldr_17_50k'
            ))
        elif domain == 'random':
            _data_kwargs.update(dict(
                dataset='random'
            ))
        elif domain == 'whitespace':
            _data_kwargs.update(dict(
                dataset='whitespace'
            ))
        else:
            raise ValueError('Unrecognized input data name: %s' % domain)
        _data_kwargs['tokenizer'] = tokenizer

        # Mean-ablation: replace the lesioned units with their mean activation under THIS
        # domain, read from the baseline run's stats. A unit's mean differs substantially
        # across domains, so a single set of values would put the ablation off-distribution
        # for most of them -- which is the whole reason for preferring mean over zero.
        if knockout_sel is not None and ablation == 'mean':
            stats_path = os.path.join(
                ablation_stats_dir or connectivity_dir,
                '%s_%s_avg%s' % (CONNECTIVITY_NAME, domain, EXTENSION)
            )
            assert 'unit_means' in h5_keys(stats_path), (
                'mean-ablation needs baseline unit_means for domain "%s" at %s. Run the '
                'connectivity step on the unperturbed model first, or set ablation: zero.'
                % (domain, stats_path)
            )
            domain_means = load_h5_array(stats_path, 'unit_means')
            model.set_perturbation_values(domain_means[knockout_sel])
            if verbose:
                stderr('%sMean-ablation values set from %s\n' % (' ' * indent, os.path.basename(stats_path)))

        # Per-domain seed, so re-running one domain reproduces what the full run produced
        # for it (the HDF5 cache makes single-domain re-runs a normal operation).
        domain_seed = derive_seed(seed, 'data', domain)

        input_ids, attention_mask = get_dataset(
            n_tokens=n_tokens * n_samples,
            split=split,
            take=take,
            seq_len=seq_len,
            wrap=wrap,
            shuffle=shuffle,
            seed=domain_seed,
            verbose=verbose,
            indent=indent,
            **_data_kwargs
        )

        if not os.path.exists(connectivity_dir):
            os.makedirs(connectivity_dir)

        if verbose:
            stderr('%sQuerying model\n' % (' ' * indent))
        n = int(np.ceil(len(input_ids) / n_samples))
        connectivity = []
        null_connectivity = []
        coordinates = None
        sample_means, sample_stds, sample_counts = [], [], []
        surrogate_vars = []
        indent += 2
        new = False
        for i in range(0, len(input_ids), n):
            t0 = time.time()
            filepath = os.path.join(
                connectivity_dir,
                '%s_%s_%s%d%s' % (
                    CONNECTIVITY_NAME,
                    domain,
                    SAMPLE_NAME,
                    i // n + 1,
                    EXTENSION
                )
            )
            if verbose:
                stderr('%sSample %d/%d\n' % (' ' * indent, i // n + 1, n_samples))
            null_filepath = os.path.join(
                null_connectivity_dir,
                '%s_%s_%s%d%s' % (CONNECTIVITY_NAME, domain, SAMPLE_NAME, i // n + 1, EXTENSION)
            ) if null_model else None
            if os.path.exists(filepath) and not overwrite:
                out = load_h5_data(filepath, verbose=False)
            else:
                out = {}
            # A cached real matrix does not imply a cached null one, so the null's absence
            # also forces a recompute. Both come from the same timecourses, so recomputing
            # either means recomputing both -- the forward passes are the cost, not the
            # correlations.
            null_missing = bool(null_model) and not (
                os.path.exists(null_filepath) and not overwrite
                and 'connectivity' in h5_keys(null_filepath)
            )
            indent += 2
            # unit_means/unit_stds are part of this step's output as of the mean-ablation
            # work, so a cached file lacking them counts as incomplete and is recomputed.
            if null_missing or not all(k in out for k in ('connectivity', 'coordinates', 'unit_means', 'unit_stds', 'n_obs')):
                _input_ids = input_ids[i:i+n]
                _attention_mask = attention_mask[i:i+n]
                out = get_timecourses(
                    model,
                    _input_ids,
                    _attention_mask,
                    batch_size=batch_size,
                    highpass=highpass,
                    lowpass=lowpass,
                    step=step,
                    timecourse_pca_components=timecourse_pca_components,
                    timecourse_ica_components=timecourse_ica_components,
                    unit_type=unit_type,
                    units_per_layer=units_per_layer,
                    unit_seed=seed,  # master seed: one subset for every sample, domain, tree
                    seed=derive_seed(seed, 'timecourses', domain, i // n + 1),
                    verbose=verbose,
                    indent=indent,
                    **model_kwargs
                )
                timecourses = out['timecourses']
                coordinates = out['coordinates']
                if null_model == 'circshift':
                    # Before the real one: get_connectivity centers and normalizes its
                    # input in place, so it consumes whichever array it is handed.
                    if verbose:
                        stderr('%sComputing circshift null\n' % (' ' * indent))
                    shifted = circshift_timecourses(
                        timecourses,
                        rng=np.random.RandomState(
                            derive_seed(seed, 'null', domain, i // n + 1) % (2 ** 32)
                        )
                    )
                    _null_connectivity = get_connectivity(shifted)
                    del shifted
                    # Further shifts, under a DIFFERENT seed key, purely to estimate how
                    # large a correlation this pair of units produces by chance. Held out
                    # from `_null_connectivity` above so the scored null is normalized by
                    # variances it did not contribute to.
                    if n_surrogates:
                        if verbose:
                            stderr('%sEstimating null variance from %d surrogates\n'
                                   % (' ' * indent, n_surrogates))
                        _surrogate_var = None
                        for k in range(n_surrogates):
                            s = circshift_timecourses(
                                timecourses,
                                rng=np.random.RandomState(
                                    derive_seed(seed, 'surrogate', domain,
                                                i // n + 1, k) % (2 ** 32)
                                )
                            )
                            r_k = get_connectivity(s)
                            del s
                            # Second moment, not variance about the sample mean: under the
                            # null the mean of signed r is zero by construction, and
                            # subtracting an estimated mean would only add noise.
                            r_k *= r_k
                            _surrogate_var = r_k if _surrogate_var is None \
                                else _surrogate_var + r_k
                            del r_k
                        _surrogate_var /= float(n_surrogates)
                else:
                    _null_connectivity = None
                _connectivity = get_connectivity(timecourses)
                if not (null_model == 'circshift' and n_surrogates):
                    _surrogate_var = None
                save = True
                new = True
            else:
                _connectivity = out['connectivity']
                _null_connectivity = None
                _surrogate_var = out.get('surrogate_var')
                coordinates = out['coordinates']
                save = False
            if _surrogate_var is not None:
                surrogate_vars.append(_surrogate_var)
            connectivity.append(_connectivity)
            if null_model:
                if _null_connectivity is None:  # cached; read it back for the average
                    _null_connectivity = load_h5_data(null_filepath, verbose=False)['connectivity']
                null_connectivity.append(_null_connectivity)
                if n_samples > 1 and save:
                    # Same unit_means/unit_stds as the real data: a circular shift permutes
                    # each unit's timecourse, so its marginal statistics are unchanged.
                    null_data = dict(
                        connectivity=_null_connectivity,
                        coordinates=coordinates,
                        unit_means=out['unit_means'],
                        unit_stds=out['unit_stds'],
                        n_obs=np.asarray(out['n_obs'])
                    )
                    # The SAME variances go in both trees. Normalizing the two by different
                    # denominators would make every real-minus-null difference partly a
                    # difference of scalings.
                    if _surrogate_var is not None:
                        null_data['surrogate_var'] = _surrogate_var
                    save_h5_data(
                        null_data,
                        null_filepath,
                        verbose=verbose,
                        indent=indent
                    )
            sample_means.append(out['unit_means'])
            sample_stds.append(out['unit_stds'])
            sample_counts.append(int(np.asarray(out['n_obs']).item()))
            if n_samples > 1 and save:
                out_data = dict(
                    connectivity=_connectivity,
                    coordinates=coordinates,
                    unit_means=out['unit_means'],
                    unit_stds=out['unit_stds'],
                    n_obs=np.asarray(out['n_obs'])
                )
                if _surrogate_var is not None:
                    out_data['surrogate_var'] = _surrogate_var
                warn_dropped_keys(filepath, out_data, verbose=verbose, indent=indent)
                save_h5_data(
                    out_data,
                    filepath,
                    verbose=verbose,
                    indent=indent
                )
            if verbose:
                stderr('%sElapsed time: %.2f s\n' % (' ' * indent, time.time() - t0))
            indent -= 2
        indent -= 2
        if n_samples > 1:
            connectivity = fisher_average(*connectivity, eps=eps)
        else:
            connectivity = connectivity[0]
        filepath = os.path.join(
            connectivity_dir,
            '%s_%s_avg%s' % (
                CONNECTIVITY_NAME,
                domain,
                EXTENSION
            ),
        )
        pooled_means, pooled_stds = pool_unit_stats(sample_means, sample_stds, sample_counts)
        if null_model:
            null_avg = fisher_average(*null_connectivity, eps=eps) if n_samples > 1 \
                else null_connectivity[0]
            null_avg_data = dict(
                connectivity=null_avg,
                coordinates=coordinates,
                unit_means=pooled_means,
                unit_stds=pooled_stds,
                n_obs=np.asarray(sum(sample_counts))
            )
            assert not surrogate_vars or len(surrogate_vars) == n_samples, \
                '%s: %d of %d samples carry surrogate_var' % (
                    domain, len(surrogate_vars), n_samples)
            if surrogate_vars:
                null_avg_data['surrogate_var'] = average_surrogate_var(surrogate_vars)
            save_h5_data(
                null_avg_data,
                os.path.join(
                    null_connectivity_dir,
                    '%s_%s_avg%s' % (CONNECTIVITY_NAME, domain, EXTENSION)
                ),
                verbose=verbose,
                indent=indent
            )
            del null_avg, null_connectivity
        save = True
        if os.path.exists(filepath) and not overwrite:
            # Key check only -- loading the file here read the whole connectivity matrix
            # (~400 MB for GPT-2) just to test for the presence of a few keys.
            keys = h5_keys(filepath)
            if all(k in keys for k in ('connectivity', 'coordinates', 'unit_means', 'unit_stds')) and not new:
                save = False
        if save:
            out_data = dict(
                connectivity=connectivity,
                coordinates=coordinates,
                unit_means=pooled_means,
                unit_stds=pooled_stds,
                n_obs=np.asarray(sum(sample_counts))
            )
            if surrogate_vars:
                out_data['surrogate_var'] = average_surrogate_var(surrogate_vars)
            # This write truncates, dropping any parcellation previously stored here. That
            # is correct -- it was derived from the old connectivity -- but say so (M8).
            warn_dropped_keys(filepath, out_data, verbose=verbose, indent=indent)
            save_h5_data(
                out_data,
                filepath,
                verbose=verbose,
                indent=indent
            )
        indent -= 2


def run_parcellation(
        output_dir=OUTPUT_DIR,
        n_networks=50,
        n_samples=100,
        binarize_connectivity=True,
        connectivity_pca_components=200,
        connectivity_ica_components=None,
        clustering_kwargs=None,
        legacy_binarize=False,
        fisher_transform=False,
        standardize_profiles=False,
        normalize=None,
        clustering='minibatch',
        pca_whiten=True,
        blockmodel_refine_labels=False,
        blockmodel_max_iter=50,
        binarize_scope='row',
        sparsify_fisher=False,
        sparsify_profiles=False,
        blockmodel_center=None,
        ica_max_iter=1000,
        ica_tol=1e-4,
        n_alignments=None,
        weight_samples=False,
        parcellate_samples=False,
        parcellate_keys=None,
        seed=None,
        variant='default',
        overwrite=False,
        verbose=True,
        indent=0
):
    """Cluster each connectivity matrix, writing one parcellation file per source matrix.

    Output goes to `<output_dir>/<variant>/parcellation/`, NOT back into the connectivity
    file. Connectivity is expensive and shared; parcellations are cheap and there are many
    of them (one per arm of a method comparison), so a single `parcellation` slot inside
    the connectivity file meant each arm silently overwrote the last. Every file records
    the settings that produced it, the code commit, and a fingerprint of its source matrix,
    so a stale parcellation is detectable rather than silently mismatched.
    """
    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_NAME)
    assert variant not in RESERVED_VARIANT_NAMES, \
        'variant name %r collides with a shared run directory; reserved: %s' % (
            variant, ', '.join(RESERVED_VARIANT_NAMES))
    parcellation_dir = os.path.join(output_dir, variant, PARCELLATION_NAME)
    set_seed(seed)
    if verbose:
        stderr('%sParcellating (variant=%s) -> %s\n' % (' ' * indent, variant, parcellation_dir))

    for path in sorted(os.listdir(connectivity_dir)):
        t0 = time.time()
        match = INPUT_NAME_RE.match(path)
        if not match:
            continue
        # By default parcellate the sample-average and the two split halves; the halves
        # are what reliability and fidelity are computed from. Per-sample files are skipped
        # unless asked for (M4): parcellating those multiplies the cost of the longest
        # stage to produce output nothing currently reads, and the per-sample connectivity
        # is cached, so it stays recoverable without new forward passes.
        keys = tuple(parcellate_keys) if parcellate_keys else ('avg',) + HALF_NAMES
        if parcellate_samples:
            keys = keys + tuple(
                k for k in (match.group(3),) if k.startswith(SAMPLE_NAME)
            )
        if match.group(3) not in keys:
            continue
        inpath = os.path.join(connectivity_dir, path)
        outpath = os.path.join(
            parcellation_dir,
            '%s_%s_%s%s' % (PARCELLATION_NAME, match.group(2), match.group(3), EXTENSION)
        )
        if os.path.exists(outpath) and not overwrite:
            if verbose:
                stderr('%sSkipping %s (exists)\n' % (' ' * indent, os.path.basename(outpath)))
            continue
        data = load_h5_data(inpath, verbose=verbose, indent=indent)

        # |r|, or |z| for an arm that asks for it. One implementation, shared with the
        # scorer -- which reads `normalize` back from this file's provenance rather than
        # from any config -- so the two can never disagree about the matrix (LOG.md S10).
        R = connectivity_matrix(data, normalize)
        # A |z| arm receives an effect size that is already variance-stabilized and can
        # exceed 1, so arctanh is both redundant and undefined (S11). The arm keeps its
        # config but the transform is not applied; both facts go in the provenance below.
        input_normalized = normalize is not None
        fisher_applied = bool(fisher_transform) and not input_normalized

        sample = sample_parcellations(
            R,
            n_networks=n_networks,
            n_samples=n_samples,
            binarize_connectivity=binarize_connectivity,
            connectivity_pca_components=connectivity_pca_components,
            connectivity_ica_components=connectivity_ica_components,
            clustering_kwargs=clustering_kwargs,
            legacy_binarize=legacy_binarize,
            fisher_transform=fisher_applied,
            standardize_profiles=standardize_profiles,
            clustering=clustering,
            pca_whiten=pca_whiten,
            blockmodel_refine_labels=blockmodel_refine_labels,
            blockmodel_max_iter=blockmodel_max_iter,
            binarize_scope=binarize_scope,
            sparsify_fisher=sparsify_fisher,
            sparsify_profiles=sparsify_profiles,
            blockmodel_center=blockmodel_center,
            # ICA needs the sign structure; every other path sees |r| (or |z|).
            signed_connectivity=(np.nan_to_num(data['connectivity']) if clustering == 'ica' else None),
            ica_max_iter=ica_max_iter,
            ica_tol=ica_tol,
            seed=derive_seed(seed, 'parcellation', path),
            verbose=verbose,
            indent=indent + 2
        )
        parcellation = align_samples(
            sample['samples'],
            sample['scores'],
            n_alignments=n_alignments,
            weight_samples=weight_samples,
            seed=derive_seed(seed, 'alignment', path),
            verbose=verbose,
            indent=indent + 2
        )
        extra = {}
        if 'maps' in sample:
            # The soft object ICA actually produces: per-restart z-scored maps, aligned by
            # the same Hungarian matching and averaged. Scored by `map_reliability`.
            assert not weight_samples, 'ICA restarts carry no objective to weight by'
            extra['ica_maps'] = align_samples(
                sample['maps'], sample['scores'], n_alignments=n_alignments,
                weight_samples=False, seed=derive_seed(seed, 'alignment_maps', path),
                verbose=False, indent=indent + 2).astype(np.float32)
        # Two further consensuses, each from a disjoint half of the SAME restarts. This is
        # the reliability ceiling: how much of the cross-half disagreement is merely
        # k-means instability rather than the data differing. Costs two Hungarian
        # alignments, ~0.1% of the k-means time, versus a second full clustering run.
        # Stored as arrays rather than reduced to a scalar, so downstream analysis can
        # recompute any comparison and plot the raw components.
        mid = len(sample['samples']) // 2
        if mid == 0:
            # A single (deterministic) sample: both "halves" are the parcellation itself
            # and the ceiling is 1 by construction, which is the honest statement for an
            # algorithm with no restart variance.
            splits = [parcellation, parcellation]
        else:
            splits = [
                align_samples(
                    sample['samples'][sl],
                    sample['scores'][sl],
                    n_alignments=n_alignments,
                    weight_samples=weight_samples,
                    seed=derive_seed(seed, 'alignment_split', path, half_ix),
                    verbose=False,
                    indent=indent + 2
                )
                for half_ix, sl in enumerate((slice(None, mid), slice(mid, None)))
            ]
        save_h5_data(
            dict(parcellation=parcellation, coordinates=data['coordinates'],
                 parcellation_split1=splits[0], parcellation_split2=splits[1], **extra),
            outpath,
            attrs=dict(
                variant=variant,
                domain=match.group(2),
                key=match.group(3),
                n_networks=int(n_networks),
                n_samples=int(len(sample['samples'])),
                clustering=str(clustering),
                pca_whiten=bool(pca_whiten),
                blockmodel_refine_labels=bool(blockmodel_refine_labels),
                blockmodel_max_iter=int(blockmodel_max_iter),
                binarize_scope=str(binarize_scope),
                sparsify_fisher=bool(sparsify_fisher),
                sparsify_profiles=bool(sparsify_profiles),
                blockmodel_center=str(blockmodel_center),
                ica_max_iter=int(ica_max_iter),
                ica_tol=float(ica_tol),
                binarize_connectivity=bool(binarize_connectivity),
                legacy_binarize=bool(legacy_binarize),
                fisher_transform=bool(fisher_transform),
                fisher_transform_applied=bool(fisher_applied),
                normalize=str(normalize),
                input_normalized=bool(input_normalized),
                standardize_profiles=bool(standardize_profiles),
                connectivity_pca_components=str(connectivity_pca_components),
                connectivity_ica_components=str(connectivity_ica_components),
                weight_samples=bool(weight_samples),
                n_alignments=str(n_alignments),
                seed=str(seed),
                # Source identity, so a parcellation built from a connectivity matrix
                # that has since been recomputed is detectable instead of silently
                # describing a matrix that no longer exists (the M8 failure mode).
                source_path=os.path.relpath(inpath, output_dir),
                source_fingerprint=array_fingerprint(data['connectivity']),
                git_commit=git_commit(),
                created=time.strftime('%Y-%m-%dT%H:%M:%S'),
            ),
            verbose=verbose,
            indent=indent + 2
        )

        if verbose:
            stderr('%sElapsed time: %.2f s\n' % (' ' * (indent + 2), time.time() - t0))


def run_split_halves(
        output_dir=OUTPUT_DIR,
        eps=1e-3,
        overwrite=False,
        verbose=True,
        indent=0
):
    """Combine the per-sample connectivity into two independent halves per domain.

    Reliability and fidelity both need two connectivity estimates of the same domain built
    from disjoint tokens. With n_samples=4 the halves are Fisher-avg(samples 1,2) and
    Fisher-avg(samples 3,4), ~197k tokens each. Written as first-class connectivity files
    (`connectivity_<domain>_halfA.h5`) so the ordinary parcellation step consumes them,
    they are cached, and every parcellation of a half carries the usual provenance --
    rather than being recomputed in memory by whatever script happens to need them.
    """
    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_NAME)
    if verbose:
        stderr('%sBuilding split halves in %s\n' % (' ' * indent, connectivity_dir))
    indent += 2

    by_domain = {}
    for path in sorted(os.listdir(connectivity_dir)):
        match = INPUT_NAME_RE.match(path)
        if not match or match.group(1) != CONNECTIVITY_NAME:
            continue
        key = match.group(3)
        if not key.startswith(SAMPLE_NAME):
            continue
        by_domain.setdefault(match.group(2), []).append((int(key[len(SAMPLE_NAME):]), path))

    for domain in sorted(by_domain):
        samples = [p for _, p in sorted(by_domain[domain])]
        assert len(samples) >= 2, \
            'domain %s has %d sample(s); split halves need at least 2' % (domain, len(samples))
        if len(samples) % 2:
            stderr('%sNOTE: %s has an odd number of samples (%d); dropping the last so the '
                   'halves are balanced\n' % (' ' * indent, domain, len(samples)))
            samples = samples[:-1]
        mid = len(samples) // 2
        for name, group in zip(HALF_NAMES, (samples[:mid], samples[mid:])):
            outpath = os.path.join(
                connectivity_dir,
                '%s_%s_%s%s' % (CONNECTIVITY_NAME, domain, name, EXTENSION)
            )
            if os.path.exists(outpath) and not overwrite:
                if verbose:
                    stderr('%sSkipping %s (exists)\n' % (' ' * indent, os.path.basename(outpath)))
                continue
            mats, means, stds, counts, coordinates = [], [], [], [], None
            svars = []
            for path in group:
                d = load_h5_data(os.path.join(connectivity_dir, path), verbose=False)
                mats.append(d['connectivity'])
                if 'surrogate_var' in d:
                    svars.append(d['surrogate_var'])
                means.append(d['unit_means'])
                stds.append(d['unit_stds'])
                counts.append(int(np.asarray(d['n_obs']).item()))
                coordinates = d['coordinates']
            connectivity = fisher_average(*mats, eps=eps) if len(mats) > 1 else mats[0]
            pooled_means, pooled_stds = pool_unit_stats(means, stds, counts)
            half_data = dict(
                connectivity=connectivity,
                coordinates=coordinates,
                unit_means=pooled_means,
                unit_stds=pooled_stds,
                n_obs=np.asarray(sum(counts))
            )
            # A half is an average of fewer samples than the avg file, so it has a LARGER
            # null variance. Propagating it (rather than reusing the avg's) is what keeps
            # the effect sizes comparable between the halves and the averages.
            assert not svars or len(svars) == len(mats), \
                'Half %s/%s: %d of %d samples carry surrogate_var. A partial set would be ' \
                'averaged with the wrong denominator; re-run connectivity for this domain ' \
                'with a consistent n_surrogates.' % (domain, name, len(svars), len(mats))
            if svars:
                half_data['surrogate_var'] = average_surrogate_var(svars)
            save_h5_data(
                half_data,
                outpath,
                attrs=dict(
                    domain=domain,
                    key=name,
                    sources=', '.join(group),
                    n_obs=int(sum(counts)),
                    git_commit=git_commit(),
                    created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                ),
                verbose=verbose,
                indent=indent
            )


def _is_reciprocal_clique(assignment, domains, shared_subnetworks):
    """True iff every pair of domains agrees on a reciprocal best match for this assignment."""
    for i, domain1 in enumerate(domains):
        for domain2 in domains[i + 1:]:
            match = shared_subnetworks.get(domain1, {}).get(domain2, {}).get(assignment[domain1])
            if match != assignment[domain2]:
                return False

    return True


def run_subnetwork_extraction(
        output_dir=OUTPUT_DIR,
        domains=None,
        variant='default',
        verbose=True,
        indent=0
):
    parcellation_dir = os.path.join(output_dir, variant, PARCELLATION_NAME)
    subnetwork_dir = os.path.join(output_dir, variant, SUBNETWORK_NAME)
    assert os.path.isdir(parcellation_dir), \
        'No parcellations for variant %r at %s -- run the parcellation step first' % (
            variant, parcellation_dir)

    if verbose:
        stderr('Extracting subnetworks\n')
    indent += 2

    parcellations = {}
    coordinates = None
    for path in sorted(os.listdir(parcellation_dir)):
        match = INPUT_NAME_RE.match(path)
        if match and match.group(1) == PARCELLATION_NAME:
            domain = match.group(2)
        else:
            continue
        key = match.group(3)
        if key != 'avg':
            continue

        filepath = os.path.join(parcellation_dir, path)
        data = load_h5_data(filepath, verbose=verbose, indent=indent)
        if 'parcellation' not in data:
            continue
        if coordinates is None:
            coordinates = data['coordinates']
        parcellations[domain] = data['parcellation']

    shared_subnetworks = {}
    if domains is None:
        domains = sorted(list(parcellations.keys()))
    else:
        # Explicit domain list, so that e.g. the `random`/`whitespace` baselines can be
        # excluded from the definition of "domain-general" (see LOG.md M6).
        if isinstance(domains, str):
            domains = (domains,)
        missing = [d for d in domains if d not in parcellations]
        assert not missing, 'No parcellation found for requested domain(s): %s' % ', '.join(missing)
        domains = sorted(domains)
    assert domains, 'No parcellated domains found in %s' % parcellation_dir
    n_domains = len(domains)
    for d1 in range(len(domains)):
        domain1 = domains[d1]
        for d2 in range(d1 + 1, len(domains)):
            domain2 = domains[d2]
            parcellation1 = parcellations[domain1].T  # <n_networks, n_units>
            parcellation2 = parcellations[domain2].T  # <n_networks, n_units>
            n_networks = parcellation1.shape[0]
            n_units = parcellation1.shape[1]

            _parcellation1 = standardize_array(parcellation1)
            _parcellation2 = standardize_array(parcellation2)
            scores = np.dot(
                _parcellation1,
                _parcellation2.T,
            ) / n_units
            alignment1 = np.argmax(scores, axis=1)
            alignment2 = np.argmax(scores, axis=0)
            matches = np.arange(n_networks) == alignment2[alignment1]
            ix1 = np.arange(n_networks)[matches]
            ix2 = alignment1[matches]
            if domain1 not in shared_subnetworks:
                shared_subnetworks[domain1] = {}
            if domain2 not in shared_subnetworks:
                shared_subnetworks[domain2] = {}
            shared_subnetworks[domain1][domain2] = {int(x):int(y) for x, y in zip(ix1, ix2)}
            shared_subnetworks[domain2][domain1] = {int(y):int(x) for x, y in zip(ix1, ix2)}

    # A network survives only if its assignment forms a full clique of reciprocal best
    # matches: for EVERY pair of domains, the two networks assigned must be each other's
    # best match. The previous implementation chained matches along the alphabetically
    # sorted domain list, which verified only n_domains - 1 of the
    # n_domains * (n_domains - 1) / 2 pairs. That made the surviving set depend on domain
    # *names* (sort position decided which domains were interior, load-bearing links) and
    # let network identity drift across hops, since the two ends of the chain were never
    # compared. The clique criterion is order-independent and checks every pair.
    reference = domains[0]
    n_units, n_networks = parcellations[reference].shape
    networks = []
    for start in range(n_networks):
        assignment = {reference: start}
        for domain in domains[1:]:
            partner = shared_subnetworks.get(reference, {}).get(domain, {}).get(start)
            if partner is None:
                break
            assignment[domain] = partner
        if len(assignment) != n_domains:
            continue
        if not _is_reciprocal_clique(assignment, domains, shared_subnetworks):
            continue
        network = np.stack(
            [parcellations[domain][..., assignment[domain]] for domain in domains],
            axis=0
        ).mean(axis=0)
        networks.append(network)

    if verbose:
        stderr('%s%d/%d networks form a reciprocal-best-match clique across %d domains (%s)\n' % (
            ' ' * indent, len(networks), n_networks, n_domains, ', '.join(domains)
        ))
    if networks:
        networks = np.stack(networks, axis=1)
    else:
        # Possible with a strict criterion over many domains. Save an empty (n_units, 0)
        # parcellation rather than letting np.stack raise on an empty list.
        stderr('%sWARNING: no shared subnetworks found; saving an empty parcellation.\n' % (' ' * indent))
        networks = np.zeros((n_units, 0), dtype=parcellations[reference].dtype)

    if not os.path.exists(subnetwork_dir):
        os.makedirs(subnetwork_dir)

    save_h5_data(
        dict(
            parcellation=networks,
            coordinates=coordinates
        ),
        os.path.join(
            subnetwork_dir,
            '%s_%s_%s%s' % (
                PARCELLATION_NAME,
                'shared',
                'avg',
                EXTENSION
            )
        ),
        verbose=verbose,
        indent=indent
    )


def resolve_networks(networks, n_networks):
    """Turn the `networks` option into an explicit list of network indices.

    None means "do not lesion anything" -- knocking out is opt-in, so that running `-s all`
    never silently lesions a model. 'all' means every shared subnetwork; a list or a single
    int names them explicitly.
    """
    if networks is None:
        return []
    if isinstance(networks, str):
        assert networks == 'all', 'networks must be None, "all", an int, or a list of ints; got %r' % networks
        return list(range(n_networks))
    if isinstance(networks, (int, np.integer)):
        networks = [networks]
    networks = [int(k) for k in networks]
    bad = [k for k in networks if not 0 <= k < n_networks]
    assert not bad, 'network index/indices %s out of range for %d shared networks' % (bad, n_networks)

    return networks


def run_knockout(
        output_dir=OUTPUT_DIR,
        variant='default',
        model_name='gpt2',
        networks=None,
        ablation='mean',
        knockout_thresh=0.5,
        connectivity_kwargs=None,
        steps=('plot_stability',),
        seed=None,
        overwrite=False,
        verbose=True,
        indent=0
):
    """Lesion shared subnetworks one at a time and re-measure connectivity.

    `networks` is opt-in: left unset, this is a no-op. Set it to 'all' or a list of indices
    to build one perturbed model per named subnetwork (LOG.md S1). The union-lesion the
    previous implementation performed -- every unit of every shared subnetwork removed in a
    single model -- is no longer reachable; it could not answer what any one subnetwork
    contributes, which is the question the step exists to ask.
    """
    if connectivity_kwargs is None:
        connectivity_kwargs = {}
    # model_name is the single source of truth from the connectivity config, so a lesioned
    # run can never silently use a different model than the baseline it is compared with.
    connectivity_kwargs = {
        k: v for k, v in connectivity_kwargs.items()
        if k not in ('model_name', 'output_dir', 'overwrite', 'seed',
                     'knockout_filepath', 'knockout_network', 'knockout_thresh',
                     'ablation', 'ablation_stats_dir')
    }
    subnetwork_dir = os.path.join(output_dir, variant, SUBNETWORK_NAME)
    baseline_connectivity_dir = os.path.join(output_dir, CONNECTIVITY_NAME)  # shared across variants
    knockout_root = os.path.join(output_dir, variant, KNOCKOUT_NAME)

    if verbose:
        stderr('Running knockout\n')
    indent += 2

    if not os.path.exists(subnetwork_dir):
        stderr('%sNo subnetwork directory at %s; run subnetwork_extraction first. Skipping.\n' % (
            ' ' * indent, subnetwork_dir))
        return

    for path in sorted(os.listdir(subnetwork_dir)):
        match = INPUT_NAME_RE.match(path)
        if not match:
            continue
        knockout_filepath = os.path.join(subnetwork_dir, path)
        if 'parcellation' not in h5_keys(knockout_filepath):
            continue
        parcellation = load_h5_array(knockout_filepath, 'parcellation')
        n_networks = parcellation.shape[1]
        selected = resolve_networks(networks, n_networks)

        if not selected:
            stderr('%s%s: %d shared networks available, none selected. Set `networks: all` '
                   '(or a list of indices) under `subnetwork_knockout` to run lesions.\n' % (
                       ' ' * indent, path, n_networks))
            continue

        if verbose:
            stderr('%s%s: lesioning %d of %d shared networks (%s-ablation)\n' % (
                ' ' * indent, path, len(selected), n_networks, ablation))

        for network in selected:
            network_dir = os.path.join(knockout_root, '%s%d' % (SUBNETWORK_NAME, network))
            if verbose:
                stderr('%sNetwork %d -> %s\n' % (' ' * (indent + 2), network, network_dir))
            run_connectivity(
                model_name=model_name,
                output_dir=network_dir,
                knockout_filepath=knockout_filepath,
                knockout_network=network,
                knockout_thresh=knockout_thresh,
                ablation=ablation,
                ablation_stats_dir=baseline_connectivity_dir,
                seed=derive_seed(seed, 'knockout', path, network),
                overwrite=overwrite,
                verbose=verbose,
                indent=indent + 4,
                **connectivity_kwargs
            )

            for step in steps:
                if step == 'plot_stability':
                    plot_stability(
                        output_dir=network_dir,
                        verbose=verbose,
                        indent=indent + 4
                    )
                else:
                    raise ValueError('Unrecognized step: %s' % step)
