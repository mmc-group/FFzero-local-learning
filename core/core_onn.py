import os
import glob as _glob

# ── Must be set BEFORE tensorflow is imported ─────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Fix: libdevice not found at ./libdevice.10.bc
# TF PTX compiler looks for libdevice relative to CWD by default.
# Point it at the real CUDA installation. Try common paths; first match wins.
_libdevice_candidates = [
    "/usr/local/cuda/nvvm/libdevice",
    "/usr/local/cuda-*/nvvm/libdevice",
    "/usr/lib/cuda/nvvm/libdevice",
    "/opt/cuda/nvvm/libdevice",
]
_libdevice_dir = None
for _pat in _libdevice_candidates:
    _m = sorted(_glob.glob(_pat))
    if _m:
        _libdevice_dir = _m[-1]
        break

if _libdevice_dir:
    _nvvm_dir   = os.path.dirname(_libdevice_dir)   # .../nvvm
    _cuda_root  = os.path.dirname(_nvvm_dir)         # .../cuda
    _xla_flag   = f"--xla_gpu_cuda_data_dir={_cuda_root}"
    os.environ["XLA_FLAGS"]    = (os.environ.get("XLA_FLAGS", "") + " " + _xla_flag).strip()
    os.environ.pop("TF_XLA_FLAGS", None)
    print(f"[CUDA] libdevice resolved: {_libdevice_dir}")
    print(f"[CUDA] XLA_FLAGS = {os.environ['XLA_FLAGS']}")
else:
    os.environ.pop("TF_XLA_FLAGS", None)
    os.environ["XLA_FLAGS"] = ""
    print("[CUDA] libdevice not found — XLA JIT disabled, GPU ops still active.")

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from neurophox.tensorflow.layers import MeshLayer
from neurophox.meshmodel import RectangularMeshModel
    

def load_data(crop_size=5, n_classes=5):
    assert 1 <= n_classes <= 10, "MNIST has 10 classes (0-9)"
    assert 1 <= crop_size <= 28, "MNIST images are 28×28; crop_size must be ≤ 28"

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    mask_tr = y_train < n_classes
    mask_ts = y_test  < n_classes
    x_train, y_train = x_train[mask_tr], y_train[mask_tr]
    x_test,  y_test  = x_test[mask_ts],  y_test[mask_ts]

    def transform(x):
        x_f = x.astype(np.float32) / 255.0            # (N, 28, 28)

        if crop_size < 28:
            c     = 14                                 # centre of 28-px image
            half  = crop_size // 2
            extra = crop_size % 2                      # 1 if odd, 0 if even
            x_f   = x_f[:, c - half : c + half + extra,
                            c - half : c + half + extra]

        flat = x_f.reshape(-1, crop_size * crop_size)  # (N, crop_size²)

        # Per-sample peak normalisation
        amax = flat.max(axis=1, keepdims=True).clip(min=1e-8)
        flat = flat / amax

        # Complex encoding: amplitude = pixel, phase = 0
        return flat.astype(np.complex64)

    xtr = transform(x_train)
    ytr = tf.one_hot(y_train, n_classes).numpy().astype(np.float32)
    xts = transform(x_test)
    yts = tf.one_hot(y_test,  n_classes).numpy().astype(np.float32)

    return xtr, ytr, xts, yts


def forward_all_cos_sims(model, xb, refs):
    L = len(model.all_layers)
    x = xb
    cos_sims_list = []

    for i, layer in enumerate(model.all_layers):
        if i < L - 1:
            x_nl = layer(x, training=False)
            
            E_lin = layer.forward_linear(x)          # call ONCE, reuse
            ff = tf.math.real(E_lin * tf.math.conj(E_lin))
        else:
            ff   = layer.forward_ff(x)
            x_nl = x

        norm    = tf.math.l2_normalize(ff, axis=1)
        cos_sim = tf.matmul(norm, refs[i], transpose_b=True)
        cos_sims_list.append(cos_sim)

        if i < L - 1:
            x = x_nl

    cos_sims    = tf.stack(cos_sims_list)
    preds_early = tf.argmax(cos_sims[:-1], axis=2, output_type=tf.int32)
    preds_early = tf.transpose(preds_early, perm=[1, 0])
    last_scores = cos_sims[-1]
    vote_pred   = majority_vote_tiebreak(preds_early, last_scores)

    return cos_sims, vote_pred

def eval_model_cosine_simplex(model, x, y, refs, DEVICE, batch_size=256):

    ds = (tf.data.Dataset.from_tensor_slices((x, y))
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE))

    layer_acc = []
    for layer_idx in range(len(model.all_layers)):
        loss_m = tf.keras.metrics.Mean()
        acc_m  = tf.keras.metrics.SparseCategoricalAccuracy()
        ref    = refs[layer_idx]

        for xb, yb in ds:
            with tf.device(DEVICE):
                xb = tf.identity(xb)
                yb = tf.identity(yb)
            #==================================================================
            true     = tf.argmax(yb, axis=1, output_type=tf.int32)
            out      = model.forward_ff(xb, int(layer_idx), training=False)
            out      = tf.math.l2_normalize(out, axis=1)
            cos_sim  = tf.matmul(out, ref, transpose_b=True)  # [batch, num_prototypes]
            # Goodness for ground-truth prototype
            true_sim = tf.reduce_sum(cos_sim * yb, axis=1)    # [batch]
            loss     = tf.reduce_mean(1.0 - true_sim)

            loss_m.update_state(tf.reduce_mean(loss))
            acc_m.update_state(true, cos_sim)

        layer_acc.append(float(acc_m.result()))

    vote_acc = eval_vote_accuracy(model, x, y, refs, DEVICE, batch_size=batch_size)

    return layer_acc, vote_acc

def eval_vote_accuracy(model, x, y, refs_tf, DEVICE, batch_size=256):
    ds = (tf.data.Dataset.from_tensor_slices((x, y))
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE))
    correct, total = 0, 0
    for xb, yb in ds:
        with tf.device(DEVICE):
            xb = tf.identity(xb)
            yb = tf.identity(yb)
        true = tf.argmax(yb, axis=1, output_type=tf.int32)
        _, vote_pred = forward_all_cos_sims(model, xb, refs_tf)
        correct += int(tf.reduce_sum(tf.cast(tf.equal(true, vote_pred), tf.int32)).numpy())
        total   += int(tf.size(true).numpy())
    return correct / max(total, 1)

@tf.function
def majority_vote_tiebreak(preds_early, last_scores):
    C         = tf.shape(last_scores)[1]
    votes = tf.reduce_sum(tf.one_hot(preds_early, depth=C, dtype=tf.int32), axis=1)
    max_count = tf.reduce_max(votes, axis=1, keepdims=True)
    tied      = tf.equal(votes, max_count)
    n_tied    = tf.reduce_sum(tf.cast(tied, tf.int32), axis=1)
    vote_winner = tf.argmax(votes, axis=1, output_type=tf.int32)
    neg_inf      = tf.constant(-1e9, dtype=last_scores.dtype)
    masked_scores = tf.where(tied, last_scores, neg_inf)
    tie_winner   = tf.argmax(masked_scores, axis=1, output_type=tf.int32)
    return tf.where(n_tied > 1, tie_winner, vote_winner)

def haar_measure_real(n, rng=None):
    if rng is None:
        rng = np.random
    A = rng.randn(n, n)
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def get_simplex_references(dimension, classes, rng=None):
    if classes > dimension:
        raise ValueError("Number of classes must be <= number of neurons.")

    identity_matrix    = np.eye(classes)
    buffered_identity  = np.zeros((classes, dimension))
    buffered_identity[:, :classes] = identity_matrix
    centroid           = buffered_identity.mean(axis=0)
    simplex            = buffered_identity - centroid

    for i in range(classes):
        row_norm = np.linalg.norm(simplex[i])
        if row_norm > 0:
            simplex[i] /= row_norm

    orthogonal_matrix = haar_measure_real(dimension, rng=rng)
    simplex = simplex @ orthogonal_matrix

    return tf.constant(simplex.astype(np.float32), dtype=tf.float32)

class ElectroopticNonlinearity(tf.keras.layers.Layer):
    def __init__(self, alpha=0.1, g=0.05 * np.pi, phi_b=np.pi):
        super().__init__()
        self._sqrt_one_minus_alpha = float(np.sqrt(1.0 - alpha))
        self._g_half               = float(0.5 * g)
        self._phi_b_half           = float(0.5 * phi_b)

    def call(self, E):
        I       = tf.math.real(E * tf.math.conj(E))
        phase   = self._g_half * I + self._phi_b_half
        scale   = self._sqrt_one_minus_alpha * tf.cos(phase)
        scale_c = tf.cast(scale, tf.complex64)
        exp_c   = tf.dtypes.complex(tf.cos(-phase), tf.sin(-phase))

        return scale_c * exp_c * E
    

def make_train_step_ffad(model, layer_idx, ref, optimizer):
    ref        = tf.convert_to_tensor(ref, dtype=tf.float32)
    theta      = model.all_layers[layer_idx].mesh.theta
    phi        = model.all_layers[layer_idx].mesh.phi
    gamma        = model.all_layers[layer_idx].mesh.gamma
    train_vars = (theta, phi, gamma)

    @tf.function(reduce_retracing=True)
    def train_step(xb, yb):
        true = tf.argmax(yb, axis=1, output_type=tf.int32)
        with tf.GradientTape() as tape:
            out      = model.forward_ff(xb, int(layer_idx), training=True)
            out      = tf.math.l2_normalize(out, axis=1)
            cos_sim  = tf.matmul(out, ref, transpose_b=True)
            true_sim = tf.reduce_sum(cos_sim * yb, axis=1)
            loss     = tf.reduce_mean(1.0 - true_sim) # before

        grads = tape.gradient(loss, train_vars)
        grads = [tf.zeros_like(v) if g is None else g
                 for g, v in zip(grads, train_vars)]
        optimizer.apply_gradients(zip(grads, train_vars))
        return loss, cos_sim, true

    return train_step

class Photoniclayer(tf.keras.Model):
    def __init__(self, n_ports):

        super().__init__()
        self.n_ports = n_ports
        self.mesh    = MeshLayer(mesh_model=RectangularMeshModel(n_ports))
        self.nl = ElectroopticNonlinearity()

    def call(self, x, training=False):
        return self.nl(self.mesh(x))

    def forward_linear(self, x):
        return self.mesh(x)

    def forward_ff(self, x):
        E = self.mesh(x)
        E = tf.math.real(E * tf.math.conj(E))
        return E


class Photonicmodel(tf.keras.Model):
    def __init__(self, n_ports, n_classes = 5, n_layers=2):
        super().__init__()
        self.n_layers  = n_layers
        self.all_layers = []
        self.n_classes = n_classes
        self.n_ports = n_ports
        for i in range(n_layers):
            layer = Photoniclayer(n_ports)
            setattr(self, f"photonic_layer_{i}", layer)
            self.all_layers.append(layer)

    def call(self, x, training=False):
        for i, layer in enumerate(self.all_layers):
            if i < self.n_layers - 1:
                x = layer(x, training=training)
            else:
                x = layer.forward_linear(x)
        return x

    def forward_ff(self, x, i_layer, training=False):
        i_layer = int(i_layer)
        for i in range(i_layer + 1):
            layer = self.all_layers[i]
            if i < i_layer:
                x = tf.stop_gradient(layer(x, training=training))
            else:
                x = layer.forward_ff(x)
        return x
    
    def forward_bp(self, x, training=False):
        for i, layer in enumerate(self.all_layers):
            if i < self.n_layers - 1:
                x = layer(x, training=training)
            else:
                x = layer.forward_linear(x)
        # Square-law detection → real float32 logits
        x = tf.math.real(x * tf.math.conj(x))   # avoids tf.abs on GPU

        return x[:, :self.n_classes] #x[:, :self.n_classes]
    

# ============================================================
# 6) Directional-derivative training step
# ============================================================
def rand_unit_like(var):
    d = tf.random.normal(tf.shape(var), dtype=var.dtype)
    n = tf.norm(tf.reshape(d, [-1])) + tf.cast(1e-12, var.dtype)
    return d / n


@tf.function(reduce_retracing=True)
def directional_derivative_ff(eps, lr, num_directions, x, y, refs, model, i_layer):
    layer = model.all_layers[i_layer]
    theta = layer.mesh.theta
    phi   = layer.mesh.phi
    gamma   = layer.mesh.gamma

    def loss_fn():
        z        = model.forward_ff(x, i_layer)
        z        = tf.math.l2_normalize(z, axis=1)
        cos_sim  = tf.matmul(z, refs, transpose_b=True)
        true_sim = tf.reduce_sum(cos_sim * y, axis=1)
        y_pred   = tf.argmax(cos_sim, axis=1, output_type=tf.int32)
        loss     = tf.reduce_mean(1.0 - true_sim)

        return y_pred, loss

    y_pred, original_loss = loss_fn()

    g_theta = tf.zeros_like(theta)
    g_phi   = tf.zeros_like(phi)
    g_gamma   = tf.zeros_like(gamma)

    for _ in tf.range(num_directions):
        d_theta = rand_unit_like(theta)
        d_phi   = rand_unit_like(phi)
        d_gamma   = rand_unit_like(gamma)

        theta.assign_add(tf.cast(eps, theta.dtype) * d_theta)
        phi.assign_add(tf.cast(eps, phi.dtype) * d_phi)
        gamma.assign_add(tf.cast(eps, gamma.dtype) * d_gamma)
        _, loss_pos = loss_fn()

        theta.assign_add(tf.cast(-2.0 * eps, theta.dtype) * d_theta)
        phi.assign_add(tf.cast(-2.0 * eps, phi.dtype) * d_phi)
        gamma.assign_add(tf.cast(-2.0 * eps, gamma.dtype) * d_gamma)
        _, loss_neg = loss_fn()

        theta.assign_add(tf.cast(eps, theta.dtype) * d_theta)
        phi.assign_add(tf.cast(eps, phi.dtype) * d_phi)
        gamma.assign_add(tf.cast(eps, gamma.dtype) * d_gamma)

        dl       = (loss_pos - loss_neg) / tf.cast(2.0 * eps, loss_pos.dtype)
        g_theta += tf.cast(dl, theta.dtype) * d_theta
        g_phi   += tf.cast(dl, phi.dtype)   * d_phi
        g_gamma   += tf.cast(dl, gamma.dtype)   * d_gamma

    g_theta /= tf.cast(num_directions, theta.dtype)
    g_phi   /= tf.cast(num_directions, phi.dtype)
    g_gamma   /= tf.cast(num_directions, gamma.dtype)

    dim_theta = tf.cast(tf.size(theta), theta.dtype)
    dim_phi   = tf.cast(tf.size(phi),   phi.dtype)
    dim_gamma   = tf.cast(tf.size(gamma),   gamma.dtype)

    theta.assign_sub(tf.cast(lr, theta.dtype) * dim_theta * g_theta)
    phi.assign_sub(tf.cast(lr, phi.dtype)   * dim_phi   * g_phi)
    gamma.assign_sub(tf.cast(lr, gamma.dtype)   * dim_gamma   * g_gamma)

    return y_pred, original_loss

def eval_model_acc(model, x, y, DEVICE, batch_size=256):
    ds = (tf.data.Dataset.from_tensor_slices((x, y))
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE))

    acc_m = tf.keras.metrics.SparseCategoricalAccuracy()
    for xb, yb in ds:
        with tf.device(DEVICE):
            xb_d = tf.identity(xb)
            yb_d = tf.identity(yb)
        out = model.forward_bp(xb_d, training=False)
        acc_m.update_state(tf.argmax(yb_d, axis=1), out)

    return float(acc_m.result())


@tf.function(reduce_retracing=True)
def directional_derivative_bp(eps, lr, num_directions, x, y, model):

    # pick the layer + vars (your in-situ variables)
    all_vars = []
    for layer in model.all_layers:
        all_vars.append(layer.mesh.theta)
        all_vars.append(layer.mesh.phi)
        all_vars.append(layer.mesh.gamma)

    def loss_fn():
        out  = model.forward_bp(x, training=False)
        y_pred = tf.argmax(out, axis=1, output_type=tf.int32)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
        
        return y_pred, loss
    
    y_pred, original_loss = loss_fn()
    # ── gradient accumulators (one per variable) ──────────────────────────────
    accum = [tf.zeros_like(v) for v in all_vars]

    # ── loop over random directions ───────────────────────────────────────────
    for _ in tf.range(num_directions):

        dirs = [rand_unit_like(v) for v in all_vars]

        # +eps — all vars simultaneously
        for v, d in zip(all_vars, dirs):
            v.assign_add(tf.cast(eps, v.dtype) * d)
        _, loss_pos = loss_fn()

        # -2eps — from +eps to -eps
        for v, d in zip(all_vars, dirs):
            v.assign_add(tf.cast(-2.0 * eps, v.dtype) * d)
        _, loss_neg = loss_fn()

        # restore
        for v, d in zip(all_vars, dirs):
            v.assign_add(tf.cast(eps, v.dtype) * d)

        # scalar directional derivative for this direction
        dl = (loss_pos - loss_neg) / tf.cast(2.0 * eps, loss_pos.dtype)

        # accumulate
        accum = [a + tf.cast(dl, v.dtype) * d
                 for a, v, d in zip(accum, all_vars, dirs)]

    # ── average then update ───────────────────────────────────────────────────
    nd = tf.cast(num_directions, tf.float32)
    for v, g in zip(all_vars, accum):
        dim = tf.cast(tf.size(v), v.dtype)
        v.assign_sub(tf.cast(lr, v.dtype) * dim * g / tf.cast(nd, v.dtype))

    return y_pred, original_loss


def make_train_step_bpad(model, optimizer):
    """
    Returns a compiled train_step that shares the optimizer across calls.
    The optimizer must be created BEFORE this function is called and
    passed in — this avoids re-creating Adam slots on each step.
    """
    train_vars = []
    for layer in model.all_layers:
        train_vars.append(layer.mesh.theta)
        train_vars.append(layer.mesh.phi)
        train_vars.append(layer.mesh.gamma)

    # Use input_signature to prevent retracing for different batch sizes
    # (the last batch may be smaller — reduce_retracing handles this)
    @tf.function(reduce_retracing=True)
    def train_step(x_b, y_b):
        with tf.GradientTape() as tape:
            out  = model.forward_bp(x_b, training=True)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_b, logits=out))
        grads = tape.gradient(loss, train_vars)
        grads = [tf.zeros_like(v) if g is None else g
                 for g, v in zip(grads, train_vars)]
        optimizer.apply_gradients(zip(grads, train_vars))
        y_pred = tf.argmax(out, axis=1, output_type=tf.int32)
        return loss, y_pred

    return train_step


# ============================================================
# 8) Save / Load model parameters
# ============================================================
def save_model(model, simplex_refs, path="model.npz"):
    """
    Save all mesh theta/phi phases and simplex references to a .npz file.

    Parameters
    ----------
    model        : trained Photonicmodel
    simplex_refs : tf.Tensor  (L, C, D)
    path         : output file path  (e.g. "run1/model.npz")
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {}
    for li, layer in enumerate(model.all_layers):
        arrays[f"theta_{li}"] = layer.mesh.theta.numpy()
        arrays[f"phi_{li}"]   = layer.mesh.phi.numpy()
        arrays[f"gamma_{li}"]   = layer.mesh.gamma.numpy()
    if simplex_refs is not None:
        arrays["simplex_refs"] = (simplex_refs.numpy()
                                  if hasattr(simplex_refs, "numpy")
                                  else np.array(simplex_refs))
    np.savez(path, **arrays)
    print(f"  ✓  model saved → {path}  (refs={'yes' if simplex_refs is not None else 'none'})")


def load_model(model, path="model.npz"):
    """
    Load mesh theta/phi phases back into an already-constructed model.
    Also returns simplex_refs if they were saved, otherwise returns None.

    Parameters
    ----------
    model : Photonicmodel  (must have same architecture as when saved)
    path  : .npz file written by save_model()

    Returns
    -------
    simplex_refs : tf.Tensor  (L, C, D)  or  None
    """
    dummy = tf.zeros([1, model.n_ports], dtype=tf.complex64)
    data = np.load(path)

    for li, layer in enumerate(model.all_layers):
        layer.mesh.theta.assign(data[f"theta_{li}"])
        layer.mesh.phi.assign(data[f"phi_{li}"])
        layer.mesh.gamma.assign(data[f"gamma_{li}"])
    _ = model(dummy, training=False)

    if "simplex_refs" in data:
        simplex_refs = tf.constant(data["simplex_refs"], dtype=tf.float32)
    else:
        simplex_refs = None
    print(f"  ✓  model loaded ← {path}  (refs={'yes' if simplex_refs is not None else 'none'})")
    return simplex_refs