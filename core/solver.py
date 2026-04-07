
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from core.core import *
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader

class cnn_layer(nn.Module):
    def __init__(self, 
                 padding, 
                 dropout,
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 kernel_stride, 
                 batch_norm, 
                 pooling_size, 
                 pooling_stride, 
                 ref, 
                 opt):
        
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channels, 
                              out_channels = out_channels, 
                              kernel_size = kernel_size,
                              stride = kernel_stride, 
                              padding = padding, bias=False)
        
        self.paras_to_optimize = list(self.conv.parameters())
        self.batch_norm = nn.BatchNorm2d(batch_norm)
        self.paras_to_optimize += list(self.batch_norm.parameters())
        if pooling_size > 0:
            self.pool = nn.MaxPool2d(kernel_size = pooling_size, stride = pooling_stride)
        else:
            self.pool = None
        self.ref = ref  # shape: (num_classes, output_dim)
        self.opt = opt
        self.dropout = dropout
        
    # channel-wise reference
    def mapping_layer(self, x): # create untrainable mapping layer for dimension reduction

        x = self.conv(x)
        x = self.batch_norm(x)
        if self.pool is not None:
            z = self.pool(x)
        else:
            z = x
        self.opt['logger'].info('layer output before pooing: {}'.format(x.shape))
        self.opt['logger'].info('layer output after pooing: {}'.format(z.shape))
        # create random dimensionality reduction layer
        x = x.view(x.shape[0], x.shape[1], -1)
        in_dim = x.shape[2]
        self.mapping = []

        for iter_i in range(x.shape[1]):
            out_dim = self.ref[iter_i].shape[1]
            self.mapping.append(torch.nn.Linear(in_dim, out_dim, bias=False).to(self.opt['device']))
        self.opt['logger'].info('--------------------\n {}'.format(self.mapping))

        return z # for size control only, z has no effect on the results
    
    def forward_ff(self, x): # linear layer
        
        x = self.conv(x)
        x = self.batch_norm(x)

        return x
    
    def forward_bp(self, x, training): # linear layer + nonlinear layer
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        x = F.dropout(x, p=self.dropout, training=training)
        return x
    
    def forward_loader(self, data_loader, training):

        z_list, y_list = [], []
        for x_data, y_data in data_loader:
            x_data = x_data.to(self.opt['device'])
            z = self.forward_bp(x_data, training=training)
            z_list.append(z.detach().cpu())
            y_list.append(y_data.cpu())

        z_all = torch.cat(z_list)
        y_all = torch.cat(y_list)
        new_dataset = TensorDataset(z_all, y_all)

        # Create new DataLoader using same configuration
        new_loader = DataLoader(
            new_dataset,
            batch_size=data_loader.batch_size,
            shuffle = False,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
            drop_last=data_loader.drop_last
        )

        del data_loader
        return new_loader
    
    # # channel-wise ref
    def z_mapping(self, x):
        z = self.forward_ff(x)
        z = z.view(z.shape[0], z.shape[1], -1)
        
        # for each channel
        z_all = []
        for iter_i, layer in enumerate(self.mapping):
            z_in = z[:, iter_i,:].view(z.shape[0], -1)
            z_out = layer(z_in)
            z_all.append(z_out)

        return z_all

    # channel-wise prediction
    def prediction(self, x): # linear layer prediction with ref.
        z_all = self.z_mapping(x)
        sim_all = torch.zeros(z_all[0].shape[0], self.ref[0].shape[0]).to(self.opt['device'])
        for iter_i, z in enumerate(z_all):
            z = z / z.norm(p=2, dim=1, keepdim = True)
            sim = torch.matmul(z, self.ref[iter_i].T.float())
            sim_all += sim
        if self.opt['task'] == 'regression':
            y_pred = sim/len(z_all)
        elif self.opt['task'] == 'classification': 
            y_pred = torch.argmax(sim_all, dim=1)

        return y_pred

    def accuracy(self, x, y): # linear layer prediction with ref.
        y_pred = self.prediction(x)
        accuracy = (y_pred == y).float().mean().item()
        return accuracy
    
    def regression_loss(self, z_all, y_train):

        loss_func = torch.nn.MSELoss()
        sim_all = torch.zeros(z_all[0].shape[0], self.ref[0].shape[0]).to(self.opt['device'])
        
        for iter_i, z in enumerate(z_all):
            z = z / z.norm(p=2, dim=1, keepdim = True)
            sim = torch.matmul(z, self.ref[iter_i].T.float()) # similarity with all refs -> prediction
            sim_all += sim

        loss  = loss_func(y_train, sim_all/len(z_all))

        return loss
    

    # channel-wise
    def multi_class_margin_loss(self, z_all, y_train, margin=0.3): 

        loss_all = torch.zeros(len(z_all), 1).to(self.opt['device'])
        for iter_i, z in enumerate(z_all):
            z = z / z.norm(p=2, dim=1, keepdim = True)
            sim = torch.matmul(z, self.ref[iter_i].T.float()) # similarity with all refs
            true_sim = sim[torch.arange(len(y_train)), y_train] # similarity with true ref
            margin_loss = F.relu(margin + sim - true_sim.unsqueeze(1)) # max with true ref and min with other refs -> minimize margin loss
            loss_all[iter_i] = margin_loss.sum()

        return loss_all

    # channel-wise
    def forward_pass(self, x, y):
        z_all = self.z_mapping(x)
        sim_all = torch.zeros(z_all[0].shape[0], self.ref[0].shape[0]).to(self.opt['device'])
        
        for iter_i, z in enumerate(z_all):
            z = z / z.norm(p=2, dim=1, keepdim = True)
            sim = torch.matmul(z, self.ref[iter_i].T.float()) # similarity with all refs -> prediction
            sim_all += sim # sum all similarity for all channels

        if self.opt['task'] == 'regression':

            y_pred = sim_all/len(z_all)
            loss = self.regression_loss(z_all, y)
            accuracy = torch.tensor([0.]).to(self.opt['device'])

        elif self.opt['task'] == 'classification':
            y_pred = torch.argmax(sim_all, dim=1)
            loss = self.multi_class_margin_loss(z_all, y)
            accuracy = self.accuracy(x, y)

        return y_pred, loss.sum(), accuracy

    def directional_derivative(self, x, y, loss_func):

        z = self.z_mapping(x)
        original_loss = loss_func(z, y)

        for iter_i in range(len(z)):
            para = torch.nn.utils.parameters_to_vector([self.paras_to_optimize[0][iter_i], self.paras_to_optimize[1][iter_i]])
            original_params = torch.nn.utils.parameters_to_vector(self.paras_to_optimize)
            grad_acc = torch.zeros_like(para).to(self.opt['device'])

            for iter_j in range(self.opt['num_directions']):

                direction = torch.randn_like(para)
                direction /= direction.norm()
                offset = self.opt['eps'] * direction

                new_para = original_params.clone()
                new_para[iter_i*(para.shape[0]-1):(iter_i+1)*(para.shape[0]-1)] = para[:-1] + offset[:-1]
                new_para[-(len(z)-iter_i)] = para[-1] + offset[-1]

                torch.nn.utils.vector_to_parameters(new_para, self.paras_to_optimize)
                z_pos = self.z_mapping(x)
                loss_pos = loss_func(z_pos, y)

                new_para = original_params.clone()
                new_para[iter_i*(para.shape[0]-1):(iter_i+1)*(para.shape[0]-1)] = para[:-1] - offset[:-1]
                new_para[-(len(z)-iter_i)] = para[-1] - offset[-1]

                torch.nn.utils.vector_to_parameters(new_para, self.paras_to_optimize)
                z_neg = self.z_mapping(x)
                loss_neg = loss_func(z_neg, y)

                if self.opt['task'] == 'classification':
                    grad = (loss_pos[iter_i] - loss_neg[iter_i])/self.opt['eps']/2 * direction # directional derivative
                else:
                    grad = (loss_pos - loss_neg)/self.opt['eps']/2 * direction # directional derivative
                grad_acc += grad

                torch.nn.utils.vector_to_parameters(original_params, self.paras_to_optimize)

            move = grad_acc/self.opt['num_directions'] * self.opt['lr'] * grad_acc.shape[0]

            new_para = original_params.clone()
            new_para[iter_i*(para.shape[0]-1):(iter_i+1)*(para.shape[0]-1)] = para[:-1] - move[:-1]
            new_para[-(len(z)-iter_i)] = para[-1] - move[-1]
            torch.nn.utils.vector_to_parameters(new_para, self.paras_to_optimize)

        return original_loss.sum()

    def unnormalization(self, y):

        if self.opt['dataset'] == 'MNIST':

            y = (np.array(y)/2+0.5)*9

        elif self.opt['dataset'] == 'function':

            y = (np.array(y)/2+0.5)*(self.opt['y_max']-self.opt['y_min'])+self.opt['y_min']

        return y
    
    def train_layer(self, train_loader, test_loader):
        
        layer_loss_train = []
        layer_accuracy_train = []
        layer_loss_test = []
        layer_accuracy_test = []
        
        if self.opt['solver'] == 'ff_ad':
            optimizer = torch.optim.Adam(self.paras_to_optimize, lr=self.opt['lr'])
            for iter_i in range(self.opt['epochs']):

                if self.opt['task'] == 'classification':
                    correct = 0
                    total = 0
                    y_train_all = torch.tensor([]).to(self.opt['device'])
                    y_train_pred_all = torch.tensor([]).to(self.opt['device'])

                    for iter_j, (x_train, y_train) in enumerate(train_loader):

                        x_train = x_train.to(self.opt['device'])
                        y_train = y_train.to(self.opt['device'])
                        y_train_all = torch.cat([y_train_all,y_train])
                        y_train_pred, loss_train, accuracy_train = self.forward_pass(x_train, y_train)
                        y_train_pred_all = torch.cat([y_train_pred_all,y_train_pred])
                        correct += accuracy_train * x_train.shape[0]
                        total += x_train.shape[0]
                        optimizer.zero_grad()
                        loss_train.backward()
                        optimizer.step()

                    train_accuracy = 100 * correct / total
                    layer_accuracy_train.append(train_accuracy)

                    with torch.no_grad():
                        correct = 0
                        total = 0
                        y_test_all = torch.tensor([]).to(self.opt['device'])
                        y_test_pred_all = torch.tensor([]).to(self.opt['device'])
                        
                        for iter_j, (x_test, y_test) in enumerate(test_loader):
                            x_test = x_test.to(self.opt['device'])
                            y_test = y_test.to(self.opt['device'])
                            y_test_all = torch.cat([y_test_all, y_test])
                            y_test_pred, loss_test, accuracy_test = self.forward_pass(x_test, y_test)
                            y_test_pred_all = torch.cat([y_test_pred_all,y_test_pred])
                            correct += accuracy_test * x_test.shape[0]
                            total += x_test.shape[0]

                        test_accuracy = 100 * correct / total
                        layer_accuracy_test.append(test_accuracy)

                elif self.opt['task'] == 'regression':
                
                    all_true = []
                    all_prediction = []
                    y_train_all = torch.tensor([]).to(self.opt['device'])
                    y_train_pred_all = torch.tensor([]).to(self.opt['device'])

                    for i, (x_train, y_train) in enumerate(train_loader):
                        x_train = x_train.to(self.opt['device'])
                        y_train = y_train.to(self.opt['device']).view(-1,1)
                        y_train_all = torch.cat([y_train_all,y_train])
                        y_train_pred, loss_train, accuracy_train = self.forward_pass(x_train, y_train)
                        y_train_pred_all = torch.cat([y_train_pred_all,y_train_pred])
                        optimizer.zero_grad()
                        loss_train.backward()
                        optimizer.step()

                        all_true += y_train.tolist()
                        all_prediction += y_train_pred.tolist()
                
                    y_train = self.unnormalization(all_true)
                    y_train_pred = self.unnormalization(all_prediction)
                    r2_train = r2_score(y_train, y_train_pred)
                    layer_accuracy_train.append(r2_train)

                    all_true = []
                    all_prediction = []
                    y_test_all = torch.tensor([]).to(self.opt['device'])
                    y_test_pred_all = torch.tensor([]).to(self.opt['device'])

                    for i, (x_test, y_test) in enumerate(test_loader):

                        x_test = x_test.to(self.opt['device'])
                        y_test = y_test.to(self.opt['device']).view(-1,1)
                        y_test_all = torch.cat([y_test_all, y_test])
                        y_test_pred, loss_test, accuracy_test = self.forward_pass(x_test, y_test)
                        y_test_pred_all = torch.cat([y_test_pred_all,y_test_pred])
                        all_true += y_test.tolist()
                        all_prediction += y_test_pred.tolist()
                
                    y_test = self.unnormalization(all_true)
                    y_test_pred = self.unnormalization(all_prediction)
                    r2_test = r2_score(y_test, y_test_pred)

                    layer_accuracy_test.append(r2_test)

                if self.opt['task'] == 'classification':
                    self.opt['logger'].info("Epoch [{}/{}], train acc: {}, test acc: {}".format(iter_i, self.opt['epochs'], train_accuracy, test_accuracy))

                elif self.opt['task'] == 'regression':
                    self.opt['logger'].info("Epoch [{}/{}], train R2: {}, test R2: {}".format(iter_i, self.opt['epochs'], r2_train, r2_test))

        elif self.opt['solver'] == 'ff_dd':

            if self.opt['task'] == 'regression':
                loss_func = self.regression_loss

            elif self.opt['task'] == 'classification':
                loss_func = self.multi_class_margin_loss

            for iter_i in range(self.opt['epochs']):

                with torch.no_grad():

                    if self.opt['task'] == 'classification':
                        correct = 0
                        total = 0
                        y_train_all = torch.tensor([]).to(self.opt['device'])
                        y_train_pred_all = torch.tensor([]).to(self.opt['device'])

                        for iter_j, (x_train, y_train) in enumerate(train_loader):
                            x_train = x_train.to(self.opt['device'])
                            y_train = y_train.to(self.opt['device'])
                            y_train_all = torch.cat([y_train_all,y_train])
                            loss_train = self.directional_derivative(x_train, y_train, loss_func)
                            y_train_pred, loss_train, accuracy_train = self.forward_pass(x_train, y_train)
                            y_train_pred_all = torch.cat([y_train_pred_all,y_train_pred])
                            correct += accuracy_train * x_train.shape[0]
                            total += x_train.shape[0]

                        train_accuracy = 100 * correct / total
                        layer_accuracy_train.append(train_accuracy)

                        correct = 0
                        total = 0
                        y_test_all = torch.tensor([]).to(self.opt['device'])
                        y_test_pred_all = torch.tensor([]).to(self.opt['device'])

                        for iter_j, (x_test, y_test) in enumerate(test_loader):
                            x_test = x_test.to(self.opt['device'])
                            y_test = y_test.to(self.opt['device'])
                            y_test_all = torch.cat([y_test_all, y_test])
                            y_test_pred, loss_test, accuracy_test = self.forward_pass(x_test, y_test)
                            y_test_pred_all = torch.cat([y_test_pred_all, y_test_pred])
                            correct += accuracy_test * x_test.shape[0]
                            total += x_test.shape[0]

                        test_accuracy = 100 * correct / total
                        layer_accuracy_test.append(test_accuracy)

                    elif self.opt['task'] == 'regression':
                        all_true = []
                        all_prediction = []
                        y_train_all = torch.tensor([]).to(self.opt['device'])
                        y_train_pred_all = torch.tensor([]).to(self.opt['device'])

                        for i, (x_train, y_train) in enumerate(train_loader):
                            x_train = x_train.to(self.opt['device'])
                            y_train = y_train.to(self.opt['device']).view(-1,1)
                            y_train_all = torch.cat([y_train_all,y_train])
                            loss_train = self.directional_derivative(x_train, y_train, loss_func)
                            y_train_pred, loss_train, accuracy_train = self.forward_pass(x_train, y_train)
                            y_train_pred_all = torch.cat([y_train_pred_all,y_train_pred])
                            all_true += y_train.tolist()
                            all_prediction += y_train_pred.tolist()
                    
                        y_train = self.unnormalization(all_true)
                        y_train_pred = self.unnormalization(all_prediction)
                        r2_train = r2_score(y_train, y_train_pred)
                        layer_accuracy_train.append(r2_train)

                        all_true = []
                        all_prediction = []
                        y_test_all = torch.tensor([]).to(self.opt['device'])
                        y_test_pred_all = torch.tensor([]).to(self.opt['device'])

                        for i, (x_test, y_test) in enumerate(test_loader):
                            x_test = x_test.to(self.opt['device'])
                            y_test = y_test.to(self.opt['device']).view(-1,1)
                            y_test_all = torch.cat([y_test_all, y_test])
                            y_test_pred, loss_test, accuracy_test = self.forward_pass(x_test, y_test)
                            y_test_pred_all = torch.cat([y_test_pred_all, y_test_pred])
                            all_true += y_test.tolist()
                            all_prediction += y_test_pred.tolist()
                    
                        y_test = self.unnormalization(all_true)
                        y_test_pred = self.unnormalization(all_prediction)
                        r2_test = r2_score(y_test, y_test_pred)
                        layer_accuracy_test.append(r2_test)
                
                if self.opt['task'] == 'classification':
                    self.opt['logger'].info("Epoch [{}/{}], train acc: {}, test acc: {}".format(iter_i, self.opt['epochs'], train_accuracy, test_accuracy))

                elif self.opt['task'] == 'regression':
                    self.opt['logger'].info("Epoch [{}/{}],  train R2: {}, test R2: {}".format(iter_i, self.opt['epochs'], r2_train, r2_test))
    
        return layer_loss_train, layer_loss_test, layer_accuracy_train, layer_accuracy_test, y_train_all, y_train_pred_all, y_test_all, y_test_pred_all



class cnn_model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        all_layers = []
        for iter_i in range(len(opt['cnn_channels'])-1):
            layer = cnn_layer(
                    padding = opt['cnn_padding'][iter_i], 
                    dropout = opt['cnn_dropout'][iter_i], 
                    in_channels = opt['cnn_channels'][iter_i], 
                    out_channels = opt['cnn_channels'][iter_i+1], 
                    kernel_size = opt['cnn_kernel_size'][iter_i], 
                    kernel_stride = opt['cnn_kernel_stride'][iter_i], 
                    batch_norm =  opt['cnn_channels'][iter_i+1], 
                    pooling_size = opt['cnn_pooling_size'][iter_i], 
                    pooling_stride = opt['cnn_pooling_stride'][iter_i], 
                    ref = opt['ref'][f'layer_{iter_i}'], 
                    opt=opt).to(opt['device']) 
            
            all_layers.append(layer)

        self.all_layers = all_layers

    # channel-wise mapping
    def mapping_layer(self, x): # predefined untrainable mapping layer for dimensionality reduction

        params_to_optimize = []
        for iter_i, layer in enumerate(self.all_layers):
            params_to_optimize += list(layer.conv.parameters())
            x = layer.mapping_layer(x) # size control only

        x = x.view(x.shape[0], -1)
        self.num_cnn_layers = len(self.all_layers)
        # add fc layers
        fc_layer = mlp_layer(input_dim = x.shape[1], 
                                output_dim = self.opt['cnn_fc_dim'][0], 
                                dropout = self.opt['cnn_fc_dropout'][0], 
                                ref = list(self.opt['ref'].values())[len(self.opt['cnn_kernel_size'])], 
                                opt = self.opt).to(self.opt['device'])
        
        self.all_layers.append(fc_layer)
        params_to_optimize += list(fc_layer.parameters())
        for iter_i in range(len(self.opt['cnn_fc_dim'])-1):
            fc_layer = mlp_layer(input_dim = self.opt['cnn_fc_dim'][iter_i], 
                                 output_dim = self.opt['cnn_fc_dim'][iter_i+1], 
                                 dropout = self.opt['cnn_fc_dropout'][iter_i+1],
                                 ref = list(self.opt['ref'].values())[iter_i+1+len(self.opt['cnn_kernel_size'])], 
                                 opt = self.opt).to(self.opt['device'])
            
            self.all_layers.append(fc_layer)
            params_to_optimize += list(fc_layer.parameters())

        self.num_fc_layers = len(self.all_layers) - self.num_cnn_layers
        self.layers = nn.ModuleList(self.all_layers)
        self.params_to_optimize = params_to_optimize
        total_paras = 0
        for para in self.params_to_optimize:
            num_para = para.view(-1).shape[0]
            total_paras += num_para
            self.opt['logger'].info('parameters to optimize: {}'.format(num_para))

        self.opt['logger'].info('total parameters to optimize: {}'.format(total_paras))
        self.opt['logger'].info(self.layers)

    def forward(self, x, training):

        for iter_i in range(self.num_cnn_layers):
            layer = self.layers[iter_i]
            x = layer.forward_bp(x, training=training)
        x = x.view(x.shape[0], -1)
        # mlp layer
        for iter_i in range(self.num_cnn_layers, len(self.layers)):
            layer = self.layers[iter_i]
            if iter_i < len(self.layers) - 1:
                x = layer.forward(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.opt['cnn_fc_dropout'][iter_i-self.num_cnn_layers], training=training)
            else:
                x = layer.forward(x)

        return x
    
    # forward DataLoader
    def flatten(self, data_loader):

        z_list, y_list = [], []
        for x_data, y_data in data_loader:
            z = x_data.view(x_data.shape[0], -1)
            z_list.append(z.detach().cpu())
            y_list.append(y_data.cpu())

        z_all = torch.cat(z_list)
        y_all = torch.cat(y_list)
        new_dataset = TensorDataset(z_all, y_all)

        # Create new DataLoader using same configuration
        new_loader = DataLoader(
            new_dataset,
            batch_size=data_loader.batch_size,
            shuffle = False,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
            drop_last=data_loader.drop_last
        )

        del data_loader
        return new_loader

    def directional_derivative(self, x, y, loss_function = torch.nn.CrossEntropyLoss()):

        z = self.forward(x, training = True)
        original_loss = loss_function(z, y)
        original_params = torch.nn.utils.parameters_to_vector(self.params_to_optimize)
        grad_acc = torch.zeros_like(original_params).to(self.opt['device'])

        for _ in range(self.opt['num_directions']):

            direction = torch.randn_like(original_params)
            direction /= direction.norm()
            offset = self.opt['eps'] * direction
            torch.nn.utils.vector_to_parameters(original_params + offset, self.params_to_optimize) # positive perturbation
            z_pos = self.forward(x, training = True)
            loss_pos = loss_function(z_pos, y)
            torch.nn.utils.vector_to_parameters(original_params - offset, self.params_to_optimize) # negative  perturbation
            z_neg = self.forward(x, training = True)
            loss_neg = loss_function(z_neg, y)
            grad = (loss_pos - loss_neg)/self.opt['eps']/2 * direction # directional derivative
            grad_acc += grad

            torch.nn.utils.vector_to_parameters(original_params, self.params_to_optimize)

        move = grad_acc/self.opt['num_directions'] * self.opt['lr'] * grad_acc.shape[0]
        new_weight = original_params - move
        torch.nn.utils.vector_to_parameters(new_weight, self.params_to_optimize)

        return original_loss

    def voting(self, all_predictions_train_flatten):

        num_samples = all_predictions_train_flatten.size(1)
        y_pred = torch.empty(num_samples, dtype=all_predictions_train_flatten.dtype, device=all_predictions_train_flatten.device)

        for j in range(num_samples):
            col = all_predictions_train_flatten[:, j].long() 
            counts = torch.bincount(col)
            max_count = counts.max()
            modes = torch.where(counts == max_count)[0]
            idx_max = 0
            select_mode = modes[0]

            for iid, mode in enumerate(modes):
                idx = torch.where(col == mode)[0].sum()
                if idx > idx_max:
                    idx_max = idx
                    select_mode = modes[iid]

            y_pred[j] = select_mode

        return y_pred
    
    def unnormalization(self, y):

        if self.opt['dataset'] == 'MNIST':

            y = (np.array(y)/2+0.5)*9

        elif self.opt['dataset'] == 'function':

            y = (np.array(y)/2+0.5)*(self.opt['y_max']-self.opt['y_min'])+self.opt['y_min']

        return y
    
    def train_all_layers(self, train_loader, test_loader):

        train_loss_history = {}
        test_loss_history = {}
        train_accuracy_history = {}
        test_accuracy_history = {}

        if self.opt['task'] == 'classification':
            loss_function = torch.nn.CrossEntropyLoss()
        elif self.opt['task'] == 'regression':
            loss_function = torch.nn.MSELoss()

        if self.opt['solver'] == 'bp_ad':

            train_accuracy_layer = []
            test_accuracy_layer = []
            optimizer = torch.optim.Adam(self.params_to_optimize, lr=self.opt['lr'], weight_decay=1e-5)
            
            for epoch in range(self.opt['epochs']):
                if self.opt['task'] == 'classification':

                    correct = 0
                    total = 0
                    y_train_all = torch.tensor([]).to(self.opt['device'])
                    y_train_pred_all = torch.tensor([]).to(self.opt['device'])

                    for i, (x_train, y_train) in enumerate(train_loader):
                        x_train = x_train.to(self.opt['device'])
                        y_train = y_train.to(self.opt['device'])
                        y_train_all = torch.cat([y_train_all,y_train])
                        y_train_pred = self.forward(x_train, training = True)
                        y_train_pred_all = torch.cat([y_train_pred_all,y_train_pred])
                        train_loss = loss_function(y_train_pred, y_train)
                        total += y_train.size(0)
                        correct += (torch.argmax(y_train_pred, dim=1) == y_train).sum().item()
                        train_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    train_accuracy = 100 * correct / total
                    train_accuracy_layer.append(train_accuracy)

                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for images, labels in test_loader:
                            images = images.to(self.opt['device'])
                            labels = labels.to(self.opt['device'])
                            outputs = self.forward(images, training = False)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            del images, labels, outputs

                        test_accuracy = 100 * correct / total
                        test_accuracy_layer.append(test_accuracy)

                elif self.opt['task'] == 'regression':

                    all_true = []
                    all_prediction = []
                    for i, (x_train, y_train) in enumerate(train_loader):
                        x_train = x_train.to(self.opt['device'])
                        y_train = y_train.to(self.opt['device']).view(-1,1)
                        y_train_pred = self.forward(x_train, True)
                        train_loss = loss_function(y_train_pred, y_train)  
                        
                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()

                        all_true += y_train.tolist()
                        all_prediction += y_train_pred.tolist()
                
                    y_train = self.unnormalization(all_true)
                    y_train_pred = self.unnormalization(all_prediction)
                    r2_train = r2_score(y_train, y_train_pred)
                    train_accuracy_layer.append(r2_train)

                    with torch.no_grad():
                        all_true = []
                        all_prediction = []
                        for i, (x_test, y_test) in enumerate(test_loader):
                            x_test = x_test.to(self.opt['device'])
                            y_test = y_test.to(self.opt['device']).view(-1,1)
                            y_test_pred = self.forward(x_test, False)
                            all_true += y_test.tolist()
                            all_prediction += y_test_pred.tolist()
                    
                        y_test = self.unnormalization(all_true)
                        y_test_pred = self.unnormalization(all_prediction)
                        r2_test = r2_score(y_test, y_test_pred)

                        test_accuracy_layer.append(r2_test)
                    
                if self.opt['task'] == 'classification':
                    self.opt['logger'].info("Epoch [{}/{}], train acc: {}, test acc: {}".format(epoch, self.opt['epochs'], train_accuracy, test_accuracy))

                elif self.opt['task'] == 'regression':
                    self.opt['logger'].info("Epoch [{}/{}], train R2: {}, test R2: {}".format(epoch, self.opt['epochs'], r2_train, r2_test))

            train_accuracy_history['layer_all'] = train_accuracy_layer
            test_accuracy_history['layer_all'] = test_accuracy_layer

        elif self.opt['solver'] == 'bp_dd':

            train_accuracy_layer = []
            test_accuracy_layer = []

            for epoch in range(self.opt['epochs']):

                with torch.no_grad():

                    if self.opt['task'] == 'classification':

                        correct = 0
                        total = 0

                        for i, (x_train, y_train) in enumerate(train_loader):

                            x_train = x_train.to(self.opt['device'])
                            y_train = y_train.to(self.opt['device'])
                            train_loss = self.directional_derivative(x_train, y_train, loss_function)
                            y_train_pred = self.forward(x_train, training = True)
                            total += y_train.size(0)
                            correct += (torch.argmax(y_train_pred, dim=1) == y_train).sum().item()

                        train_accuracy = 100 * correct / total
                        train_accuracy_layer.append(train_accuracy)

                        # test dataset
                        correct = 0
                        total = 0
                        for images, labels in test_loader:
                            images = images.to(self.opt['device'])
                            labels = labels.to(self.opt['device'])
                            outputs = self.forward(images, training = False)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            del images, labels, outputs

                        test_accuracy = 100 * correct / total
                        test_accuracy_layer.append(test_accuracy)

                    elif self.opt['task'] == 'regression':

                        all_true = []
                        all_prediction = []
                        for i, (x_train, y_train) in enumerate(train_loader):
                            x_train = x_train.to(self.opt['device'])
                            y_train = y_train.to(self.opt['device']).view(-1,1)
                            train_loss = self.directional_derivative(x_train, y_train, loss_function)
                            y_train_pred = self.forward(x_train, True)

                            all_true += y_train.tolist()
                            all_prediction += y_train_pred.tolist()
                    
                        y_train = self.unnormalization(all_true)
                        y_train_pred = self.unnormalization(all_prediction)

                        r2_train = r2_score(y_train, y_train_pred)
                        train_accuracy_layer.append(r2_train)

                        with torch.no_grad():
                            all_true = []
                            all_prediction = []
                            for i, (x_test, y_test) in enumerate(test_loader):
                                x_test = x_test.to(self.opt['device'])
                                y_test = y_test.to(self.opt['device']).view(-1,1)
                                y_test_pred = self.forward(x_test, False)
                                all_true += y_test.tolist()
                                all_prediction += y_test_pred.tolist()
                        
                            y_test = self.unnormalization(all_true)
                            y_test_pred = self.unnormalization(all_prediction)
                            r2_test = r2_score(y_test, y_test_pred)

                            test_accuracy_layer.append(r2_test)

                if self.opt['task'] == 'classification':
                    self.opt['logger'].info("Epoch [{}/{}], train acc: {}, test acc: {}".format(epoch, self.opt['epochs'], train_accuracy, test_accuracy))

                elif self.opt['task'] == 'regression':
                    self.opt['logger'].info("Epoch [{}/{}], train R2: {}, test R2: {}".format(epoch, self.opt['epochs'], r2_train, r2_test))

            train_accuracy_history['layer_all'] = train_accuracy_layer
            test_accuracy_history['layer_all'] = test_accuracy_layer

        else: # forward-forward

            all_predictions_train = torch.tensor([]).view(1,-1).to(self.opt['device'])
            all_predictions_test = torch.tensor([]).view(1,-1).to(self.opt['device'])
            
            for i, layer in enumerate(self.layers):
                self.opt['logger'].info('----------- Training layer {} -----------'.format(i))
                
                if i < len(self.layers) - len(self.opt['cnn_fc_dim']): # cnn layer
                    layer_loss_train, layer_loss_test, layer_accuracy_train, layer_accuracy_test, y_train_all, y_train_pred_all, y_test_all, y_test_pred_all = layer.train_layer(train_loader, test_loader)
                    
                    all_predictions_train = torch.cat([all_predictions_train, y_train_pred_all.view(1, -1)], 1)
                    all_predictions_train_flatten = all_predictions_train.view(i+1, -1)
                    
                    if self.opt['task'] == 'classification':
                        y_pred_train = self.voting(all_predictions_train_flatten)
                    else:
                        y_pred_train = y_train_all

                    accuracy_train = (y_pred_train == y_train_all).float().mean().item()

                    all_predictions_test = torch.cat([all_predictions_test, y_test_pred_all.view(1, -1)], 1)
                    all_predictions_test_flatten = all_predictions_test.view(i+1, -1)
                    if self.opt['task'] == 'classification':
                        y_pred_test = self.voting(all_predictions_test_flatten)
                    else:
                        y_pred_test = y_test_all

                    accuracy_test = (y_pred_test == y_test_all).float().mean().item()

                    self.opt['logger'].info('majority voting -- train acc: {}, test acc: {}'.format(accuracy_train*100, accuracy_test*100))

                    with torch.no_grad():
                        train_loader = layer.forward_loader(train_loader, training = True)
                        test_loader = layer.forward_loader(test_loader, training = False)

                else: # fc layer
                    train_loader = self.flatten(train_loader)
                    test_loader = self.flatten(test_loader)
                    layer_loss_train, layer_loss_test, layer_accuracy_train, layer_accuracy_test, y_train_all, y_train_pred_all, y_test_all, y_test_pred_all = layer.train_layer(train_loader, test_loader)
                    
                    all_predictions_train = torch.cat([all_predictions_train, y_train_pred_all.view(1, -1)], 1)
                    all_predictions_train_flatten = all_predictions_train.view(i+1, -1)

                    if self.opt['task'] == 'classification':
                        y_pred_train = self.voting(all_predictions_train_flatten)
                    else:
                        y_pred_train = y_train_all

                    accuracy_train = (y_pred_train == y_train_all).float().mean().item()

                    all_predictions_test = torch.cat([all_predictions_test, y_test_pred_all.view(1, -1)], 1)
                    all_predictions_test_flatten = all_predictions_test.view(i+1, -1)

                    if self.opt['task'] == 'classification':
                        y_pred_test = self.voting(all_predictions_test_flatten)
                    else:
                        y_pred_test = y_test_all

                    accuracy_test = (y_pred_test == y_test_all).float().mean().item()

                    self.opt['logger'].info('majority voting -- train acc: {}, test acc: {}'.format(accuracy_train*100, accuracy_test*100))
                    
                    with torch.no_grad():
                        train_loader = layer.forward_loader(train_loader, training = True)
                        test_loader = layer.forward_loader(test_loader, training = False)

                train_loss_history[f'layer_{i}'] = layer_loss_train
                train_accuracy_history[f'layer_{i}'] = layer_accuracy_train
                test_loss_history[f'layer_{i}'] = layer_loss_test
                test_accuracy_history[f'layer_{i}'] = layer_accuracy_test

        return train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history

    def predict_ff(self, x):

        with torch.no_grad():
            z = self.forward(x)
            z = z / z.norm(p=2, dim=1, keepdim = True)
            ref = list(list(self.opt['ref'].items()))[-1][-1]
            sim = torch.matmul(z, ref.T.float())
            y_pred = torch.argmax(sim, dim=1)

        return y_pred
    
    def predict_bp(self, x):

        with torch.no_grad():
            z = self.forward(x)
            y_pred = torch.argmax(z, dim=1)

        return y_pred


class mlp_layer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, ref, opt):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.ref = ref  # shape: (num_classes, output_dim)
        self.opt = opt
        self.dropout = dropout
        self.paras_to_optimize = list(self.linear.parameters())

    def forward(self, x):
        x = self.linear(x)
        return x
    
    def forward_loader(self, data_loader, training):

        z_list, y_list = [], []
        for x_data, y_data in data_loader:
            z = x_data.to(self.opt['device'])
            z = self.linear(z)
            z = F.dropout(z, p = self.dropout, training = training)
            z_list.append(z.detach().cpu())
            y_list.append(y_data.cpu())

        z_all = torch.cat(z_list)
        y_all = torch.cat(y_list)

        new_dataset = TensorDataset(z_all, y_all)
        # Create new DataLoader using same configuration
        new_loader = DataLoader(
            new_dataset,
            batch_size=data_loader.batch_size,
            shuffle = False,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
            drop_last=data_loader.drop_last
        )

        del data_loader
        return new_loader
    
    def prediction(self, x):

        z = self.forward(x)
        z = z / z.norm(p=2, dim=1, keepdim = True)
        sim = torch.matmul(z, self.ref.T.float())

        if self.opt['task'] == 'regression':
            y_pred = sim
        elif self.opt['task'] == 'classification': 
            y_pred = torch.argmax(sim, dim=1)

        return y_pred

    def accuracy(self, x, y):

        y_pred = self.prediction(x)
        accuracy = (y_pred == y).float().mean().item()

        return accuracy

    def multi_class_margin_loss(self, z, y_train, margin=0.3):

        z = z / z.norm(p=2, dim=1, keepdim = True)
        sim = torch.matmul(z, self.ref.T.float()) # similarity with all refs
        true_sim = sim[torch.arange(len(y_train)), y_train] # similarity with true ref
        margin_loss = F.relu(margin + sim - true_sim.unsqueeze(1)) # max with true ref and min with other refs -> minimize margin loss

        return margin_loss.sum()
    
    def regression_loss(self, z, y_train):

        z = z / z.norm(p=2, dim=1, keepdim = True)
        sim = torch.matmul(z, self.ref.T.float()) # similarity with all refs -> prediction
        loss_func = torch.nn.MSELoss()
        loss  = loss_func(y_train, sim)

        return loss

    def forward_pass(self, x, y):

        z = self.forward(x) # linear layer
        z = z / z.norm(p=2, dim=1, keepdim = True)
        sim = torch.matmul(z, self.ref.T.float()) # similarity with all refs -> prediction

        if self.opt['task'] == 'regression':
            y_pred = sim
            loss = self.regression_loss(z, y)
            accuracy = torch.tensor([0.]).to(self.opt['device'])
        elif self.opt['task'] == 'classification':
            y_pred = torch.argmax(sim, dim=1)
            loss = self.multi_class_margin_loss(z, y)
            accuracy = self.accuracy(x, y)

        return y_pred, loss, accuracy

    
    def directional_derivative(self, x, y, loss_func):

        z = self.forward(x) # linear
        original_loss = loss_func(z, y)
        original_params = torch.nn.utils.parameters_to_vector(self.paras_to_optimize)
        grad_acc = torch.zeros_like(original_params).to(self.opt['device'])

        for _ in range(self.opt['num_directions']):

            direction = torch.randn_like(original_params)
            direction /= direction.norm()
            offset = self.opt['eps'] * direction
            torch.nn.utils.vector_to_parameters(original_params + offset, self.paras_to_optimize)
            z_pos = self.forward(x)
            loss_pos = loss_func(z_pos, y)
            torch.nn.utils.vector_to_parameters(original_params - offset, self.paras_to_optimize)
            
            z_neg = self.forward(x)
            loss_neg = loss_func(z_neg, y)
            grad = (loss_pos - loss_neg)/self.opt['eps']/2 * direction # directional derivative
            grad_acc += grad
            torch.nn.utils.vector_to_parameters(original_params, self.paras_to_optimize)

        move = grad_acc/self.opt['num_directions'] * self.opt['lr'] * grad_acc.shape[0]
        new_weight = original_params - move
        torch.nn.utils.vector_to_parameters(new_weight, self.paras_to_optimize)

        return original_loss

    def unnormalization(self, y):

        if self.opt['dataset'] == 'MNIST':
            y = (np.array(y)/2+0.5)*9

        elif self.opt['dataset'] == 'function':
            y = (np.array(y)/2+0.5)*(self.opt['y_max']-self.opt['y_min'])+self.opt['y_min']

        return y
    
    def train_layer(self, train_loader, test_loader):
        
        layer_loss_train = []
        layer_accuracy_train = []
        layer_loss_test = []
        layer_accuracy_test = []
        
        if self.opt['solver'] == 'ff_ad':
            optimizer = torch.optim.Adam(self.paras_to_optimize, lr = self.opt['lr'])

            for iter_i in range(self.opt['epochs']):

                if self.opt['task'] == 'classification':
                    correct = 0
                    total = 0
                    y_train_all = torch.tensor([]).to(self.opt['device'])
                    y_train_pred_all = torch.tensor([]).to(self.opt['device'])

                    for iter_j, (x_train, y_train) in enumerate(train_loader):
                        x_train = x_train.to(self.opt['device'])
                        y_train = y_train.to(self.opt['device'])
                        y_train_all = torch.cat([y_train_all,y_train])
                        y_train_pred, loss_train, accuracy_train = self.forward_pass(x_train, y_train)
                        y_train_pred_all = torch.cat([y_train_pred_all,y_train_pred])
                        correct += accuracy_train * x_train.shape[0]
                        total += x_train.shape[0]

                        optimizer.zero_grad()
                        loss_train.backward()
                        optimizer.step()

                    train_accuracy = 100 * correct / total
                    layer_accuracy_train.append(train_accuracy)

                    with torch.no_grad():
                        correct = 0
                        total = 0
                        y_test_all = torch.tensor([]).to(self.opt['device'])
                        y_test_pred_all = torch.tensor([]).to(self.opt['device'])
                        for i, (x_test, y_test) in enumerate(test_loader):
                            x_test = x_test.to(self.opt['device'])
                            y_test = y_test.to(self.opt['device'])
                            y_test_all = torch.cat([y_test_all, y_test])
                            y_test_pred, loss_test, accuracy_test = self.forward_pass(x_test, y_test)
                            y_test_pred_all = torch.cat([y_test_pred_all,y_test_pred])
                            correct += accuracy_test*x_test.shape[0]
                            total += x_test.shape[0]

                        test_accuracy = 100 * correct / total # last layer
                        layer_accuracy_test.append(test_accuracy)

                elif self.opt['task'] == 'regression':

                    all_true = []
                    all_prediction = []
                    y_train_all = torch.tensor([]).to(self.opt['device'])
                    y_train_pred_all = torch.tensor([]).to(self.opt['device'])
                    for i, (x_train, y_train) in enumerate(train_loader):
                        x_train = x_train.to(self.opt['device'])
                        y_train = y_train.to(self.opt['device']).view(-1,1)
                        y_train_all = torch.cat([y_train_all,y_train])
                        y_train_pred, loss_train, accuracy_train = self.forward_pass(x_train, y_train)
                        y_train_pred_all = torch.cat([y_train_pred_all,y_train_pred])
                        optimizer.zero_grad()
                        loss_train.backward()
                        optimizer.step()

                        all_true += y_train.tolist()
                        all_prediction += y_train_pred.tolist()
                
                    y_train = self.unnormalization(all_true)
                    y_train_pred = self.unnormalization(all_prediction)
                    r2_train = r2_score(y_train, y_train_pred)
                    layer_accuracy_train.append(r2_train)

                    all_true = []
                    all_prediction = []
                    y_test_all = torch.tensor([]).to(self.opt['device'])
                    y_test_pred_all = torch.tensor([]).to(self.opt['device'])

                    for i, (x_test, y_test) in enumerate(test_loader):
                        x_test = x_test.to(self.opt['device'])
                        y_test = y_test.to(self.opt['device']).view(-1,1)
                        y_test_all = torch.cat([y_test_all, y_test])
                        y_test_pred, loss_test, accuracy_test = self.forward_pass(x_test, y_test)
                        y_test_pred_all = torch.cat([y_test_pred_all,y_test_pred])
                        
                        all_true += y_test.tolist()
                        all_prediction += y_test_pred.tolist()
                
                    y_test = self.unnormalization(all_true)
                    y_test_pred = self.unnormalization(all_prediction)
                    r2_test = r2_score(y_test, y_test_pred)
                    layer_accuracy_test.append(r2_test)

                if self.opt['task'] == 'classification':
                    self.opt['logger'].info("Epoch [{}/{}], train acc: {}, test acc: {}".format(iter_i, self.opt['epochs'], train_accuracy, test_accuracy))

                elif self.opt['task'] == 'regression':
                    self.opt['logger'].info("Epoch [{}/{}], train R2: {}, test R2: {}".format(iter_i, self.opt['epochs'], r2_train, r2_test))

        elif self.opt['solver'] == 'ff_dd':

            if self.opt['task'] == 'regression':
                loss_func = self.regression_loss
            elif self.opt['task'] == 'classification':
                loss_func = self.multi_class_margin_loss

            for iter_i in range(self.opt['epochs']):

                with torch.no_grad():
                    if self.opt['task'] == 'classification':

                        correct = 0
                        total = 0
                        y_train_all = torch.tensor([]).to(self.opt['device'])
                        y_train_pred_all = torch.tensor([]).to(self.opt['device'])
                        for iter_j, (x_train, y_train) in enumerate(train_loader):
                            x_train = x_train.to(self.opt['device'])
                            y_train = y_train.to(self.opt['device'])
                            y_train_all = torch.cat([y_train_all,y_train])
                            loss_train = self.directional_derivative(x_train, y_train, loss_func)
                            y_train_pred = self.prediction(x_train)
                            y_train_pred_all = torch.cat([y_train_pred_all,y_train_pred])
                            accuracy_train = self.accuracy(x_train, y_train)
                            correct += accuracy_train * x_train.shape[0]
                            total += x_train.shape[0]

                        train_accuracy = 100 * correct / total
                        layer_accuracy_train.append(train_accuracy)

                        correct = 0
                        total = 0
                        y_test_all = torch.tensor([]).to(self.opt['device'])
                        y_test_pred_all = torch.tensor([]).to(self.opt['device'])

                        for i, (x_test, y_test) in enumerate(test_loader):
                            x_test = x_test.to(self.opt['device'])
                            y_test = y_test.to(self.opt['device'])
                            y_test_all = torch.cat([y_test_all, y_test])
                            y_test_pred, loss_test, accuracy_test = self.forward_pass(x_test, y_test)
                            y_test_pred_all = torch.cat([y_test_pred_all,y_test_pred])
                            correct += accuracy_test*x_test.shape[0]
                            total += x_test.shape[0]

                        test_accuracy = 100 * correct / total # last layer
                        layer_accuracy_test.append(test_accuracy)

                    elif self.opt['task'] == 'regression':

                        all_true = []
                        all_prediction = []
                        y_train_all = torch.tensor([]).to(self.opt['device'])
                        y_train_pred_all = torch.tensor([]).to(self.opt['device'])

                        for i, (x_train, y_train) in enumerate(train_loader):
                            x_train = x_train.to(self.opt['device'])
                            y_train = y_train.to(self.opt['device']).view(-1,1)
                            y_train_all = torch.cat([y_train_all,y_train])
                            loss_train = self.directional_derivative(x_train, y_train, loss_func)
                            y_train_pred = self.prediction(x_train)
                            y_train_pred_all = torch.cat([y_train_pred_all,y_train_pred])

                            all_true += y_train.tolist()
                            all_prediction += y_train_pred.tolist()
                    
                        y_train = self.unnormalization(all_true)
                        y_train_pred = self.unnormalization(all_prediction)
                        r2_train = r2_score(y_train, y_train_pred)
                        layer_accuracy_train.append(r2_train)

                        all_true = []
                        all_prediction = []
                        y_test_all = torch.tensor([]).to(self.opt['device'])
                        y_test_pred_all = torch.tensor([]).to(self.opt['device'])
                        for i, (x_test, y_test) in enumerate(test_loader):

                            x_test = x_test.to(self.opt['device'])
                            y_test = y_test.to(self.opt['device']).view(-1,1)
                            y_test_all = torch.cat([y_test_all, y_test])
                            y_test_pred = self.prediction(x_test)
                            y_test_pred_all = torch.cat([y_test_pred_all,y_test_pred])
                            all_true += y_test.tolist()
                            all_prediction += y_test_pred.tolist()
                    
                        y_test = self.unnormalization(all_true)
                        y_test_pred = self.unnormalization(all_prediction)
                        r2_test = r2_score(y_test, y_test_pred)
                        layer_accuracy_test.append(r2_test)

                if self.opt['task'] == 'classification':
                    self.opt['logger'].info("Epoch [{}/{}], train acc: {}, test acc: {}".format(iter_i, self.opt['epochs'], train_accuracy, test_accuracy))

                elif self.opt['task'] == 'regression':
                    self.opt['logger'].info("Epoch [{}/{}], train R2: {}, test R2: {}".format(iter_i, self.opt['epochs'], r2_train, r2_test))

        return layer_loss_train, layer_loss_test, layer_accuracy_train, layer_accuracy_test, y_train_all, y_train_pred_all, y_test_all, y_test_pred_all

class mlp_model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.layers = nn.ModuleList([mlp_layer(opt['mlp_layer_dim'][i], 
                                               opt['mlp_layer_dim'][i+1], 
                                               self.opt['mlp_dropout'][i], 
                                               opt['ref'][f'layer_{i}'], self.opt) 
                                     for i in range(len(opt['mlp_layer_dim'])-1)]).to(opt['device'])
        
        self.paras_to_optimize = list(self.parameters())
        total = 0
        for para in self.paras_to_optimize:
            self.opt['logger'].info('trainable parameters: {}'.format(para.view(-1).shape))
            total += para.view(-1).shape[0]
        self.opt['logger'].info('total trainable parameters: {}'.format(total))

    def forward(self, x, training): # no layer normalization
        for iter_i, layer in enumerate(self.layers):
            if iter_i < len(self.layers)-1:
                x = layer.forward(x)
                x = F.dropout(x, p=layer.dropout, training=training)

            else:
                x = layer.forward(x) 
        return x

    def directional_derivative(self, x, y, loss_function = torch.nn.CrossEntropyLoss()):
        
        z = self.forward(x, training = True)
        original_loss = loss_function(z, y)
        original_params = torch.nn.utils.parameters_to_vector(self.paras_to_optimize)
        grad_acc = torch.zeros_like(original_params).to(self.opt['device'])

        for _ in range(self.opt['num_directions']):

            direction = torch.randn_like(original_params)
            direction /= direction.norm()

            offset = self.opt['eps'] * direction
            torch.nn.utils.vector_to_parameters(original_params + offset, self.paras_to_optimize)
            z_pos = self.forward(x, training = True)
            loss_pos = loss_function(z_pos, y)

            torch.nn.utils.vector_to_parameters(original_params - offset, self.paras_to_optimize)
            z_neg = self.forward(x, training = True)
            loss_neg = loss_function(z_neg, y)
            grad = (loss_pos - loss_neg)/self.opt['eps']/2 * direction # directional derivative
            grad_acc += grad

            torch.nn.utils.vector_to_parameters(original_params, self.paras_to_optimize)

        move = grad_acc/self.opt['num_directions'] * self.opt['lr'] * grad_acc.shape[0]
        new_weight = original_params - move
        torch.nn.utils.vector_to_parameters(new_weight, self.paras_to_optimize)

        return original_loss
    
    def voting(self, all_predictions_train_flatten):

        num_samples = all_predictions_train_flatten.size(1)
        y_pred = torch.empty(num_samples, dtype=all_predictions_train_flatten.dtype, device=all_predictions_train_flatten.device)

        for j in range(num_samples):
            col = all_predictions_train_flatten[:, j].long()   # votes from all models
            counts = torch.bincount(col)
            max_count = counts.max()
            modes = torch.where(counts == max_count)[0]
            idx_max = 0
            select_mode = modes[0]
            for iid, mode in enumerate(modes):
                idx = torch.where(col == mode)[0].sum()
                if idx > idx_max:
                    idx_max = idx
                    select_mode = modes[iid]

            y_pred[j] = select_mode

        return y_pred
    
    def unnormalization(self, y):

        if self.opt['dataset'] == 'MNIST':
            y = (np.array(y)/2+0.5)*9

        elif self.opt['dataset'] == 'function':
            y = (np.array(y)/2+0.5)*(self.opt['y_max']-self.opt['y_min'])+self.opt['y_min']

        return y

    def train_all_layers(self, train_loader, test_loader):

        train_loss_history = {}
        test_loss_history = {}
        train_accuracy_history = {}
        test_accuracy_history = {}

        if self.opt['solver'] == 'bp_ad':
            train_accuracy_layer = []
            test_accuracy_layer = []
            optimizer = torch.optim.Adam(self.paras_to_optimize, lr = self.opt['lr'])

            if self.opt['task'] == 'classification':
                loss_function = torch.nn.CrossEntropyLoss()
            elif self.opt['task'] == 'regression':
                loss_function = torch.nn.MSELoss()

            for epoch in range(self.opt['epochs']):

                if self.opt['task'] == 'classification':
                    correct = 0
                    total = 0

                    for i, (x_train, y_train) in enumerate(train_loader):
                        
                        x_train = x_train.to(self.opt['device'])
                        y_train = y_train.to(self.opt['device'])
                        y_train_pred = self.forward(x_train, training = True)
                        _, predicted = torch.max(y_train_pred.data, 1)
                        total += y_train.size(0)
                        correct += (predicted == y_train).sum().item()
                        train_loss = loss_function(y_train_pred, y_train)  
                        train_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    train_accuracy = 100 * correct / total
                    train_accuracy_layer.append(train_accuracy)

                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for images, labels in test_loader:
                            images = images.to(self.opt['device'])
                            labels = labels.to(self.opt['device'])
                            outputs = self.forward(images, training = False)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            del images, labels, outputs

                    test_accuracy = 100 * correct / total
                    test_accuracy_layer.append(test_accuracy)

                elif self.opt['task'] == 'regression':

                    all_true = []
                    all_prediction = []
                    for i, (x_train, y_train) in enumerate(train_loader):

                        x_train = x_train.to(self.opt['device'])
                        y_train = y_train.to(self.opt['device']).view(-1,1)
                        y_train_pred = self.forward(x_train, True)
                        train_loss = loss_function(y_train_pred, y_train)  
                        train_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        all_true += y_train.tolist()
                        all_prediction += y_train_pred.tolist()
                
                    y_train = self.unnormalization(all_true)
                    y_train_pred = self.unnormalization(all_prediction)
                    r2_train = r2_score(y_train, y_train_pred)
                    train_accuracy_layer.append(r2_train)

                    all_true = []
                    all_prediction = []

                    with torch.no_grad():

                        for i, (x_test, y_test) in enumerate(test_loader):
                            x_test = x_test.to(self.opt['device'])
                            y_test = y_test.to(self.opt['device']).view(-1,1)
                            y_test_pred = self.forward(x_test, False)
                            all_true += y_test.tolist()
                            all_prediction += y_test_pred.tolist()
                    
                        y_test = self.unnormalization(all_true)
                        y_test_pred = self.unnormalization(all_prediction)
                        r2_test = r2_score(y_test, y_test_pred)

                    test_accuracy_layer.append(r2_test)

                if self.opt['task'] == 'classification':
                    self.opt['logger'].info("Epoch [{}/{}], train acc: {}, test acc: {}".format(epoch, self.opt['epochs'], train_accuracy, test_accuracy))

                elif self.opt['task'] == 'regression':
                    self.opt['logger'].info("Epoch [{}/{}], train R2: {}, test R2: {}".format(epoch, self.opt['epochs'], r2_train, r2_test))

            train_accuracy_history['layer_all'] = train_accuracy_layer
            test_accuracy_history['layer_all'] = test_accuracy_layer

        elif self.opt['solver'] == 'bp_dd':

            train_loss_layer = []
            test_loss_layer = []
            train_accuracy_layer = []
            test_accuracy_layer = []
            if self.opt['task'] == 'classification':
                loss_function = torch.nn.CrossEntropyLoss()
            elif self.opt['task'] == 'regression':
                loss_function = torch.nn.MSELoss()

            for epoch in range(self.opt['epochs']):

                with torch.no_grad():
                    if self.opt['task'] == 'classification':
                        correct = 0
                        total = 0
                        for i, (x_train, y_train) in enumerate(train_loader):

                            x_train = x_train.to(self.opt['device'])
                            y_train = y_train.to(self.opt['device'])
                            train_loss = self.directional_derivative(x_train, y_train, loss_function)
                            y_train_pred = self.forward(x_train, training = True)
                            _, predicted = torch.max(y_train_pred.data, 1)
                            total += y_train.size(0)
                            correct += (predicted == y_train).sum().item()

                        train_accuracy = 100 * correct / total
                        train_accuracy_layer.append(train_accuracy)

                        correct = 0
                        total = 0
                        for images, labels in test_loader:
                            images = images.to(self.opt['device'])
                            labels = labels.to(self.opt['device'])
                            outputs = self.forward(images, training = False)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            del images, labels, outputs

                        test_accuracy = 100 * correct / total
                        test_accuracy_layer.append(test_accuracy)

                    elif self.opt['task'] == 'regression':

                        all_true = []
                        all_prediction = []

                        for i, (x_train, y_train) in enumerate(train_loader):
                            x_train = x_train.to(self.opt['device'])
                            y_train = y_train.to(self.opt['device']).view(-1,1)
                            train_loss = self.directional_derivative(x_train, y_train, loss_function)
                            y_train_pred = self.forward(x_train, True)

                            all_true += y_train.tolist()
                            all_prediction += y_train_pred.tolist()

                        y_train = self.unnormalization(all_true)
                        y_train_pred = self.unnormalization(all_prediction)
                        r2_train = r2_score(y_train, y_train_pred)
                        train_accuracy_layer.append(r2_train)

                        all_true = []
                        all_prediction = []

                        for i, (x_test, y_test) in enumerate(test_loader):
                            x_test = x_test.to(self.opt['device'])
                            y_test = y_test.to(self.opt['device']).view(-1,1)
                            y_test_pred = self.forward(x_test, False)
                            all_true += y_test.tolist()
                            all_prediction += y_test_pred.tolist()
                    
                        y_test = self.unnormalization(all_true)
                        y_test_pred = self.unnormalization(all_prediction)
                        r2_test = r2_score(y_test, y_test_pred)

                        test_accuracy_layer.append(r2_test)

                if self.opt['task'] == 'classification':
                    self.opt['logger'].info("Epoch [{}/{}], train acc: {}, test acc: {}".format(epoch, self.opt['epochs'], train_accuracy, test_accuracy))

                elif self.opt['task'] == 'regression':
                    self.opt['logger'].info("Epoch [{}/{}], train R2: {}, test R2: {}".format(epoch, self.opt['epochs'], r2_train, r2_test))

            train_loss_history['layer_all'] = train_loss_layer
            train_accuracy_history['layer_all'] = train_accuracy_layer

            test_loss_history['layer_all'] = test_loss_layer
            test_accuracy_history['layer_all'] = test_accuracy_layer

        else:
            # training
            all_predictions_train = torch.tensor([]).view(1,-1).to(self.opt['device'])
            all_predictions_test = torch.tensor([]).view(1,-1).to(self.opt['device'])
            
            for i, layer in enumerate(self.layers):

                self.opt['logger'].info('----------- Training layer {} -----------'.format(i))
                layer_loss_train, layer_loss_test, layer_accuracy_train, layer_accuracy_test, y_train_all, y_train_pred_all, y_test_all, y_test_pred_all  = layer.train_layer(train_loader, test_loader)
                all_predictions_train = torch.cat([all_predictions_train, y_train_pred_all.view(1, -1)], 1)
                all_predictions_train_flatten = all_predictions_train.view(i+1, -1)

                if self.opt['task'] == 'classification':
                    y_pred_train = self.voting(all_predictions_train_flatten)
                else:
                    y_pred_train = y_train_all

                accuracy_train = (y_pred_train == y_train_all).float().mean().item()
                all_predictions_test = torch.cat([all_predictions_test, y_test_pred_all.view(1, -1)], 1)
                all_predictions_test_flatten = all_predictions_test.view(i+1, -1)

                if self.opt['task'] == 'classification':
                    y_pred_test = self.voting(all_predictions_test_flatten)
                else:
                    y_pred_test = y_test_all
                accuracy_test = (y_pred_test == y_test_all).float().mean().item()

                self.opt['logger'].info('majority voting -- train acc: {}, test acc: {}'.format(accuracy_train*100, accuracy_test*100))
                
                with torch.no_grad():
                    train_loader = layer.forward_loader(train_loader, training = True)
                    test_loader = layer.forward_loader(test_loader, training = False)

                train_loss_history[f'layer_{i}'] = layer_loss_train
                train_accuracy_history[f'layer_{i}'] = layer_accuracy_train

                test_loss_history[f'layer_{i}'] = layer_loss_test
                test_accuracy_history[f'layer_{i}'] = layer_accuracy_test

        return train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history
   


def solver(train_loader, test_loader, opt):

    if opt['model'] == 'mlp':
        model = mlp_model(opt)

    elif opt['model'] == 'cnn':

        model = cnn_model(opt)
        # initialization of model and dimensionality reduction layers
        x_train = train_loader.dataset[0][0]
        x_train = x_train.view(1, x_train.shape[0], x_train.shape[1],x_train.shape[2]).to(opt['device'])
        model.mapping_layer(x_train)

    train_loss_history, test_loss_history, train_accuracy_history,test_accuracy_history = model.train_all_layers(train_loader, test_loader)

    return model, train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history


