import sys
sys.path.append("../")
import torch
import os
from core.data_loader import *
import argparse
import numpy as np
from core.solver import *
import logging
import time
import pprint


def main(opt):

    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    train_loader, test_loader, opt = data_loader(opt)

    # Set up experiment logging directory based on model type
    if opt['model'] == 'mlp':
        save_log_path = f"../results/log-{opt['dataset']}-{opt['task']}-{opt['model']}-{opt['solver']}-{opt['num_directions']}-{opt['input_dim']}-{opt['mlp_ref_dim']}-{time.strftime('%Y%m%d-%H%M%S')}"
    elif opt['model'] == 'cnn':
        save_log_path = f"../results/log-{opt['dataset']}-{opt['task']}-{opt['model']}-{opt['solver']}-{opt['num_directions']}-{opt['input_dim']}-{opt['cnn_kernel_size']}-{opt['cnn_channels']}-{opt['cnn_ref_dim']}-{time.strftime('%Y%m%d-%H%M%S')}"

    create_exp_dir(save_log_path, scripts_to_save=[f for f in os.listdir('./') if f.endswith('.py')])
    logger = create_logger('global_logger', save_log_path + '/log.txt')
    logger.info('args:{}'.format(pprint.pformat(opt)))
    logger = logging.getLogger('global_logger')
    logger.info('starting\n')

    # Configure model architecture and prototype vectors per dataset and model type
    if opt['dataset'] == 'MNIST' or opt['dataset'] == 'FashionMNIST':

        if opt['model'] == 'mlp':

            if opt['task'] == 'classification':
                ref = ref_vectors(opt)
                opt['ref'] = ref 
                if opt['downsample'] == False:
                    input_dim = 28*28
                else:
                    input_dim = opt['input_dim']*opt['input_dim']

                opt['mlp_layer_dim'] = [input_dim] + opt['mlp_ref_dim']

            elif opt['task'] == 'regression':
                opt['num_classes'] = 1
                ref = ref_vectors(opt)
                opt['ref'] = ref 
                if opt['downsample'] == False:
                    input_dim = 28*28
                else:
                    input_dim = opt['input_dim']*opt['input_dim']

                # FF solver needs output dim=10 for cosine similarity; BP uses dim=1
                if 'ff' in opt['solver']:
                    opt['mlp_layer_dim'] = [input_dim] + opt['mlp_ref_dim'][:-1]+[10]
                else:
                    opt['mlp_layer_dim'] = [input_dim] + opt['mlp_ref_dim'][:-1]+[1]

        elif opt['model'] == 'cnn':

            if opt['task'] == 'classification':
                opt['num_classes'] = 10
                ref = ref_vectors(opt)
                opt['ref'] = ref 
                x_train = train_loader.dataset[0][0]
                input_dim = x_train.shape[0]
                opt['cnn_channels'] = [input_dim] + opt['cnn_channels']

            elif opt['task'] == 'regression':
                opt['num_classes'] = 1
                ref = ref_vectors(opt)
                opt['ref'] = ref 
                x_train = train_loader.dataset[0][0]
                input_dim = x_train.shape[0]
                opt['cnn_channels'] = [input_dim] + opt['cnn_channels']

    # Synthetic functions: input dim varies by function, output always scalar
    elif 'function' in opt['dataset']:

        opt['num_classes'] = 1
        if opt['dataset'] == 'function1':
            input_dim = 2
        elif opt['dataset'] == 'function2':
            input_dim = 5

        # FF solver needs output dim=10 for cosine similarity; BP uses dim=1
        if 'ff' in opt['solver']:
            opt['mlp_ref_dim'] = opt['mlp_ref_dim'][:-1]+[10]
            opt['mlp_layer_dim'] = [input_dim] + opt['mlp_ref_dim'][:-1]+[10]

        else:
            opt['mlp_ref_dim'] = opt['mlp_ref_dim'][:-1]+[1]
            opt['mlp_layer_dim'] = [input_dim] + opt['mlp_ref_dim'][:-1]+[1]

        ref = ref_vectors(opt)
        opt['ref'] = ref 

    angle_deg = compute_max_angle_degrees(opt['num_classes'])
    
    logger.info("Maximal angle for {} classes: {} degrees".format(opt['num_classes'], round(angle_deg)))
    opt['logger'] = logger
    train_acc_all = []
    test_acc_all = []

    # Run training multiple times
    for run_idx in range(opt['num_runs']):

        print(f"\n=== Starting run {run_idx+1}/{opt['num_runs']} ===")
        model, train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist = solver(train_loader, test_loader, opt)

        train_acc_all.append(train_acc_hist)
        test_acc_all.append(test_acc_hist)

    plot_history(opt, logger, save_log_path, train_acc_all, test_acc_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default = 0, help="random seed")
    parser.add_argument('--task', default = 'classification', choices=['regression', 'classification'], help="task type")
    parser.add_argument('--dataset', type = str, default='MNIST', choices=['MNIST', 'FashionMNIST', 'function1', 'function2'], help="dataset")
    parser.add_argument('--model', default = 'cnn', choices=['cnn', 'mlp'], help="model type")
    parser.add_argument('--downsample', type = bool, default = False, help="downsample input image")
    parser.add_argument('--input_dim', default = 28, help = "input dimension of the image")
    parser.add_argument('--solver', type = str, default = 'ff_dd', choices=['ff_dd', 'bp_dd', 'ff_ad', 'bp_ad'], help="solver type")
    parser.add_argument('--device', default = 0, help="cuda device")
    parser.add_argument('--num_runs', default = 1, help = "num of runs")
    parser.add_argument('--epochs', default = 100, help = "epochs")
    parser.add_argument('--kwargs', default = {'num_workers': 32, 'pin_memory': True}, help = "kwargs")
    parser.add_argument('--batch_size', default = 256, help = "batch size")
    parser.add_argument('--num_classes', default = 10, help="num of classes")
    parser.add_argument('--classes', default = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, help="classes, should match number of classes")

    # DD (directional derivative) optimizer parameters
    parser.add_argument('--num_directions', default = 1, help = "num of directions in DD optimization")
    parser.add_argument('--eps', default = 1e-3, help = "epsilon in DD optimization")
    parser.add_argument('--lr', default = 1e-3, help = "learning rate")  
    
    # MLP architecture parameters
    parser.add_argument('--mlp_ref_dim', default = [100]*3+[10], help="dimension of prototype vectors")
    parser.add_argument('--mlp_dropout', default = [0.]*4, help="dropout") 

    # cnn parameters for MNIST
    parser.add_argument('--cnn_padding', default = [2]*2, help="")
    parser.add_argument('--cnn_dropout', default = [0.0]*2, help="")
    parser.add_argument('--cnn_kernel_size', default = [6]*2, help="dimension of prototype vectors")
    parser.add_argument('--cnn_channels', default = [4]*2, help="dimension of prototype vectors")
    parser.add_argument('--cnn_kernel_stride', default = [1]*2, help="dimension of prototype vectors")
    parser.add_argument('--cnn_pooling_size', default = [2]*2, help="dimension of prototype vectors")
    parser.add_argument('--cnn_pooling_stride', default = [2]*2, help="dimension of prototype vectors")
    parser.add_argument('--cnn_ref_dim', default = [10]*2, help="dimension of prototype vectors in covolution layers")
    parser.add_argument('--cnn_fc_ref_dim', default = [10], help="dimension of prototype vectors in FC layer")
    parser.add_argument('--cnn_fc_dim', default = [10], help="dimension of FC layer")
    parser.add_argument('--cnn_fc_dropout', default = [0], help="dropout ratio in FC layer")

    # cnn parameters for FMNIST
    # parser.add_argument('--cnn_padding', default = [2]*2, help="")
    # parser.add_argument('--cnn_dropout', default = [0.1]*2, help="")
    # parser.add_argument('--cnn_kernel_size', default = [6]*2, help="dimension of prototype vectors")
    # parser.add_argument('--cnn_channels', default = [16]*2, help="dimension of prototype vectors")
    # parser.add_argument('--cnn_kernel_stride', default = [1]*2, help="dimension of prototype vectors")
    # parser.add_argument('--cnn_pooling_size', default = [2]*2, help="dimension of prototype vectors")
    # parser.add_argument('--cnn_pooling_stride', default = [2]*2, help="dimension of prototype vectors")
    # parser.add_argument('--cnn_ref_dim', default = [10]*2, help="dimension of prototype vectors")
    # parser.add_argument('--cnn_fc_ref_dim', default = [100, 10], help="")
    # parser.add_argument('--cnn_fc_dim', default = [100, 10], help="")
    # parser.add_argument('--cnn_fc_dropout', default = [0.1]*2, help="")

    args = parser.parse_args()
    opt = vars(args)

    main(opt)