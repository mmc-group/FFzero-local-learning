import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import logging
import shutil
import sys
from matplotlib import cm
sys.path.append("../")

def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def compute_max_angle_degrees(num_classes):
    """
    Compute the maximal angle between vertices of a regular simplex corresponding
    to a given number of classes.

    The angle is defined by placing the N classes as vertices of a regular simplex.
    """
    if num_classes == 1:
        return 0.0
    angle_rad = np.arccos(-1 / (num_classes - 1))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def haar_measure_real(n): #  this code is a modified version of the pseudocode in: 
    # "How to generate random matrices from the classical compact groups" from Francesco Mezzadri
    
    """
    Generate a random n×n orthogonal matrix distributed according to the
    Haar measure on O(n).
    """

    while True:
        
        # Create an n×n matrix with i.i.d. N(0,1) entries
        A = np.random.randn(n, n)   # Must be from NORMAL disttr. , if uniform distribution 
                                    # the resulting distribution is not rotationally invariant 
        #    Do a QR decomposition
        #    A = Q * R, where Q has orthonormal columns, R is upper triangular
        Q, R = np.linalg.qr(A)
        
        #    The diagonal elmeents of R could be negative.  Compute sign of
        #    the diagonal to fix them.
        #    diag(R) / abs(diag(R)) would be +1 or -1 for each diagonal entry.
        d = np.diag(R)
        ph = np.sign(d)
        
        #   Multiply Q by these sings so that Q * diag(ph) sets
        #    the diagonal of R effectively to be positive (or correct sign).
        Q = Q * ph  # broadcasting: each column i of Q is multiplied by ph[i]

        if np.linalg.det(Q) > 0.:
            break

    return Q

def simplex_vectors(dimension, classes):
    
    # Create a dxd identity matrix
    identity_matrix = np.eye(classes)
 
    if classes > dimension:
        raise ValueError("Number of classes must be equal or smaller than number of neurons.")

    buffered_identity = np.zeros((classes, dimension))
    buffered_identity[:, :classes] = identity_matrix

    # Compute mean of each row  and centroid --> The mean of each row gives the coordinate of the centroid 
    centroid = buffered_identity.mean(axis=0) 

    # Subtract Centroid from each basis vector that make up the identity matrix
    simplex = buffered_identity - centroid

    # Normalize each row
    for i in range(classes):
        row_norm = np.linalg.norm(simplex[i])
        if row_norm > 0:
            simplex[i] /= row_norm

    # Apply a random orthogonal transformation
    orthogonal_matrix = haar_measure_real(dimension)    # this does not chnage length --> still normalized
    simplex = simplex @ orthogonal_matrix               # @ is, what the cool kids use for dot product
    
    return simplex

def ref_vectors(opt):

    rand_vecs = {}
    if opt['model'] == 'mlp':

        for iter_j, dim_j in enumerate(opt['mlp_ref_dim']):

            if opt['task'] == 'regression':
                ref = np.random.randn(opt['num_classes'], dim_j)
                ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
            else:
                ref = simplex_vectors(dim_j, opt['num_classes']) # n_classes * dim_j
            
            ref = torch.tensor(ref).to(opt['device'])
            rand_vecs[f'layer_{iter_j}'] = ref

    elif opt['model'] == 'cnn':

        for iter_j, dim_j in enumerate(opt['cnn_ref_dim']+opt['cnn_fc_ref_dim']):
            if opt['task'] == 'regression':
                # cnn layer
                if iter_j < len(opt['cnn_ref_dim']):
                    ref_all = []
                    for _ in range(opt['cnn_channels'][iter_j]):
                        ref = np.random.randn(opt['num_classes'], dim_j)
                        ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
                        ref = torch.tensor(ref).to(opt['device'])
                        ref_all.append(ref)
                    rand_vecs[f'layer_{iter_j}'] = ref_all
                # fc layer
                else:
                    ref = np.random.randn(opt['num_classes'], dim_j)
                    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
                    ref = torch.tensor(ref).to(opt['device'])
                    rand_vecs[f'layer_{iter_j}'] = ref
            else:
                # cnn layer
                if iter_j < len(opt['cnn_ref_dim']):
                    ref_all = []
                    for _ in range(opt['cnn_channels'][iter_j]):
                        ref = simplex_vectors(dim_j, opt['num_classes']) # n_classes * dim_j
                        ref = torch.tensor(ref).to(opt['device'])
                        ref_all.append(ref)
                    rand_vecs[f'layer_{iter_j}'] = ref_all
                # fc layer
                else:
                    ref = simplex_vectors(dim_j, opt['num_classes']) # n_classes * dim_j
                    ref = torch.tensor(ref).to(opt['device'])
                    rand_vecs[f'layer_{iter_j}'] = ref

    return rand_vecs



def plot_history(opt, logger, save_log_path, train_acc_all, test_acc_all):

    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:olive', 
              'tab:cyan', 'tab:orange', 'tab:pink', 'tab:brown', 
              'tab:purple', 'tab:gray','lime']
    
    for i, layer_i in enumerate(train_acc_all[0].keys()):

        train_acc_layer = []
        test_acc_layer = []

        for run_id in range(opt['num_runs']):

            train_acc_hist = train_acc_all[run_id]
            test_acc_hist = test_acc_all[run_id]

            train_acc_layer.append(train_acc_hist[layer_i])
            test_acc_layer.append(test_acc_hist[layer_i])

        train_acc_layer = np.array(train_acc_layer)
        test_acc_layer = np.array(test_acc_layer)

        train_acc_median  = np.median(train_acc_layer, axis=0)
        train_acc_q1      = np.percentile(train_acc_layer, 25, axis=0)
        train_acc_q3      = np.percentile(train_acc_layer, 75, axis=0)

        test_acc_median   = np.median(test_acc_layer, axis=0)
        test_acc_q1       = np.percentile(test_acc_layer, 25, axis=0)
        test_acc_q3       = np.percentile(test_acc_layer, 75, axis=0)

        final_train_acc = train_acc_median[-1]
        final_test_acc  = test_acc_median[-1]

        logger.info('accuracy median: {}, {}'.format(final_train_acc, final_test_acc))

        num_epochs = len(list(train_acc_hist.values())[0])

        epochs = np.arange(1, num_epochs + 1)
        fig.suptitle(f"{opt['dataset']}-{opt['task']}-{opt['model']}-{opt['solver']}-{opt['num_directions']}",fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.plot(epochs, train_acc_median, color=colors[i], linestyle=':', label=f'L-{i}: Train', linewidth=2)
        ax1.fill_between(epochs, train_acc_q1, train_acc_q3,
                        color=colors[i], alpha=0.2)

        ax1.plot(epochs, test_acc_median, color=colors[i], label=f'L-{i}: Test', linewidth=2)
        ax1.fill_between(epochs, test_acc_q1, test_acc_q3,
                        color=colors[i], alpha=0.2)
        
        ax1.set_xlim(1, num_epochs)
        ax1.set_title("Accuracy (Median ± IQR)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend(ncol=1)

    plt.savefig(save_log_path + '/accuracy.png', dpi=300)




