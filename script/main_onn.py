import sys
sys.path.append("../")
from core.core_onn import *
from core.onn_viz import save_all_figures, capture_phases, capture_transfer_matrices
import numpy as np
import tensorflow as tf
import argparse

def train_model(model, x_train, y_train, opt, refs):

    epochs  = opt['epochs']
    batch_size  = opt['batch_size']
    lr = opt['lr']
    DEVICE  = f"/GPU:{opt['device']}"

    ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
          .shuffle(min(len(x_train), 10000))
          .batch(batch_size, drop_remainder=False)
          .prefetch(tf.data.AUTOTUNE))

    if opt['solver'] == 'bp_ad':

        acc_history  = []
        loss_history = []

        loss_m    = tf.keras.metrics.Mean()
        train_acc = tf.keras.metrics.CategoricalAccuracy(name="train_acc")

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        train_step = make_train_step_bpad(model, optimizer)

        for ep in range(1, epochs + 1):
            loss_m.reset_state()
            train_acc.reset_state()

            for xb, yb in ds:
                with tf.device(DEVICE):
                    xb_d = tf.identity(xb)
                    yb_d = tf.identity(yb)

                loss, y_pred = train_step(xb_d, yb_d)

                true = tf.argmax(yb_d, axis=1, output_type=tf.int32)
                loss_m.update_state(loss)
                train_acc.update_state(true, y_pred)

            acc_history.append(float(train_acc.result()))
            loss_history.append(float(loss_m.result()))

            print(f"  Epoch {ep:02d}/{epochs}  "
                f"loss={loss_history[-1]:.4f}  "
                f"acc={acc_history[-1]:.4f}")
        
    elif opt['solver'] == 'bp_dd':

        acc_history  = []
        loss_history = []

        loss_m    = tf.keras.metrics.Mean()
        train_acc = tf.keras.metrics.CategoricalAccuracy(name="train_acc")

        for ep in range(1, epochs + 1):
            loss_m.reset_state()
            train_acc.reset_state()

            for xb, yb in ds:
                with tf.device(DEVICE):
                    xb_d = tf.identity(xb)
                    yb_d = tf.identity(yb)

                y_pred, loss = directional_derivative_bp(opt['eps'], lr, opt['num_directions'], xb, yb, model)

                true = tf.argmax(yb_d, axis=1, output_type=tf.int32)
                loss_m.update_state(loss)
                train_acc.update_state(true, y_pred)

            acc_history.append(float(train_acc.result()))
            loss_history.append(float(loss_m.result()))

            print(f"  Epoch {ep:02d}/{epochs}  "
                f"loss={loss_history[-1]:.4f}  "
                f"acc={acc_history[-1]:.4f}")
            
    elif opt['solver'] == 'ff_ad':

        acc_history  = {}
        loss_history = {}

        for layer_idx in range(len(model.all_layers)):
            acc_history[layer_idx]  = []
            loss_history[layer_idx] = []

            ref        = refs[layer_idx]
            optimizer  = tf.keras.optimizers.Adam(learning_rate=lr)
            train_step = make_train_step_ffad(model, layer_idx, ref, optimizer)

            loss_m = tf.keras.metrics.Mean()
            acc_m  = tf.keras.metrics.SparseCategoricalAccuracy()

            print(f"\n>>> Training Layer {layer_idx+1}/{len(model.all_layers)}")

            for ep in range(1, epochs + 1):
                loss_m.reset_state()
                acc_m.reset_state()

                for xb, yb in ds:
                    with tf.device(DEVICE):
                        xb = tf.identity(xb)
                        yb = tf.identity(yb)
                    loss, cos_sim, true = train_step(xb, yb)
                    loss_m.update_state(loss)
                    acc_m.update_state(true, cos_sim)

                acc_history[layer_idx].append(float(acc_m.result()))
                loss_history[layer_idx].append(float(loss_m.result()))

                print(f"  Epoch {ep:02d}/{epochs}  "
                    f"loss={loss_history[layer_idx][-1]:.4f}  "
                    f"acc={acc_history[layer_idx][-1]:.4f}")
                
    elif opt['solver'] == 'ff_dd':

        acc_history  = {}
        loss_history = {}

        for layer_idx in range(len(model.all_layers)):
            acc_history[layer_idx]  = []
            loss_history[layer_idx] = []

            ref        = refs[layer_idx]
            train_loss = tf.keras.metrics.Mean(name="train_loss")
            train_acc  = tf.keras.metrics.CategoricalAccuracy(name="train_acc")

            print(f"\n>>> Training Layer {layer_idx+1}/{len(model.all_layers)}")

            for ep in range(1, epochs + 1):
                train_loss.reset_state()
                train_acc.reset_state()

                for xb, yb in ds:
                    with tf.device(DEVICE):
                        xb_d = tf.identity(xb)
                        yb_d = tf.identity(yb)

                    true   = tf.argmax(yb_d, axis=1, output_type=tf.int32)
                    y_pred, loss = directional_derivative_ff(opt['eps'], lr, opt['num_directions'], xb_d, yb_d, ref, model, layer_idx)

                    train_loss.update_state(loss)
                    train_acc.update_state(true, y_pred)

                acc_history[layer_idx].append(float(train_acc.result()))
                loss_history[layer_idx].append(float(train_loss.result()))

                print(f"  Epoch {ep:02d}/{epochs}  "
                    f"loss={loss_history[layer_idx][-1]:.4f}  "
                    f"acc={acc_history[layer_idx][-1]:.4f}")

    return acc_history, loss_history


def main(opt):

    DEVICE  = f"/GPU:{opt['device']}"
    crop_size = opt['crop_size']     # n for n×n centre crop (1-28)
    n_classes = opt['n_classes']     # number of MNIST classes to use (1-10)
    n_layers  = opt['n_layers']      # number of photonic mesh layers
    n_ports = crop_size * crop_size

    print("\n" + "=" * 60)
    print(f"Loading MNIST  classes 0-{n_classes-1}  "
          f"crop {crop_size}x{crop_size} = {n_ports} ports")
    print("=" * 60)
    x_train, y_train, x_test, y_test = load_data(crop_size=crop_size, n_classes=n_classes)
    print("Train:", x_train.shape, y_train.shape)
    print("Test :", x_test.shape,  y_test.shape)

    # Build model on GPU
    with tf.device(DEVICE):
        model = Photonicmodel(n_ports = n_ports,n_classes = n_classes,  n_layers = n_layers)
        for layer in model.all_layers:
            layer.mesh.trainable = True
        dummy = tf.zeros([1, n_ports], dtype = tf.complex64)
        _ = model(dummy, training=False)

    print(f"[GPU] Model built on {DEVICE}")

    total_params = sum(int(np.prod(v.shape)) for l in model.all_layers for v in [l.mesh.theta, l.mesh.phi, l.mesh.gamma])
    print(f"[GPU] Total trainable phase params: {total_params:}")
    
    for i, layer in enumerate(model.all_layers):

        theta = layer.mesh.theta.numpy().reshape(-1)
        phi   = layer.mesh.phi.numpy().reshape(-1)
        gamma   = layer.mesh.gamma.numpy().reshape(-1)
        
        print(f"Layer {i} -- theta: {len(theta)} params, phi: {len(phi)}, gamma: {len(gamma)} params")

    # Snapshot phases before training
    phases_before   = capture_phases(model)
    matrices_before = capture_transfer_matrices(model)
    outdir = f"../results/onn_{opt['solver']}_{crop_size}_{n_layers}_layers"

    if 'ff' in opt['solver']:
        rng          = np.random.RandomState(opt['seed'])
        simplex_refs = np.zeros((n_layers, n_classes, n_ports), dtype=np.float32)
        for ell in range(n_layers):
            simplex_refs[ell] = get_simplex_references(n_ports, n_classes, rng=rng)
        simplex_refs = tf.convert_to_tensor(simplex_refs, dtype=tf.float32)
        print("Fixed simplex anchors shape:", simplex_refs.shape)
    else:
        simplex_refs = None

    acc_history, loss_history = train_model(model, x_train, y_train, opt, simplex_refs)
    
    print("\n" + "=" * 60)
    print("Final Performance")
    print("=" * 60)

    if 'ff' in opt['solver']:
        train_layer_acc, train_acc = eval_model_cosine_simplex(model, x_train, y_train, simplex_refs, DEVICE)
        test_layer_acc,  test_acc  = eval_model_cosine_simplex(model, x_test,  y_test,  simplex_refs, DEVICE)
        
        print(f"Train -- layer acc: {train_layer_acc}  voting acc: {100 * train_acc:6.2f}%")
        print(f"Test  -- layer acc: {test_layer_acc}   voting acc: {100 * test_acc:6.2f}%")

    else:
        train_acc = eval_model_acc(model, x_train, y_train, DEVICE)
        test_acc  = eval_model_acc(model, x_test,  y_test, DEVICE)

        print(f"Train acc:   {100 * train_acc:6.2f}%")
        print(f"Test  acc:   {100 * test_acc:6.2f}%")


    # ── Visualisations ────────────────────────────────────────────────────────
    save_all_figures(
                    model           = model,
                    x_test          = x_test,
                    y_test          = y_test,
                    refs            = simplex_refs,
                    n_classes       = n_classes,
                    acc_history     = acc_history,
                    loss_history    = loss_history,
                    train_acc       = train_acc,
                    test_acc        = test_acc,
                    outdir          = outdir,
                    phases_before   = phases_before,
                    matrices_before = matrices_before,
                    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default = 0, help="random seed")
    parser.add_argument('--crop_size',type=int, default = 28, help = "input dimension of the image")
    parser.add_argument('--solver', type = str, default = 'ff_dd', choices=['ff_dd', 'bp_dd', 'ff_ad', 'bp_ad'], help="solver type")
    parser.add_argument('--device', type = int, default = 0, help="cuda device")
    parser.add_argument('--epochs', type = int, default = 100, help = "epochs")
    parser.add_argument('--n_classes', type = int, default = 10, help = "number of classes")
    parser.add_argument('--batch_size', type = int, default = 128, help = "batch size")
    parser.add_argument('--n_layers', type=int, default = 2, help = "number of layers")
    parser.add_argument('--lr', default = 1e-3, help = "learning rate")
    parser.add_argument('--eps', default = 1e-3, help = "epsilon in DD")
    parser.add_argument('--num_directions', type = int, default = 1, help = "number of directions in DD")
    
    args = parser.parse_args()
    opt = vars(args)

    main(opt)

