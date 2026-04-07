import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
import torch.nn.functional as F

class Downsample(object):
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def __call__(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)

        x_small = F.adaptive_avg_pool2d(x, (self.input_dim, self.input_dim))

        return x_small 
    

class FloatLabelDataset(Dataset):
    """Wraps a dataset and converts its labels to float tensors."""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]

        return x, 2*(torch.tensor(float(y), dtype=torch.float32)/9-0.5)

def filter_by_class(dataset, classes):
    indices = [i for i, (_, y) in enumerate(dataset) if y in classes]
    return Subset(dataset, indices)


def data_loader(opt):

    if opt['task'] == 'classification':

        if opt['dataset'] == 'MNIST':
            if opt['model'] == 'mlp':
                if opt['downsample']:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # per-image min–max
                        Downsample(input_dim=opt['input_dim']),
                        transforms.Lambda(lambda x: x.view(-1))  # flatten
                        ])
                else:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # per-image min–max
                        transforms.Lambda(lambda x: x.view(-1))  # flatten
                        ])

            elif opt['model'] == 'cnn':
                if opt['downsample']:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # per-image min–max
                        Downsample(opt),
                        ])
                else:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # per-image min–max
                        ])
                    
            train = datasets.MNIST(root="../dataset", train=True, download=True, transform=transform)
            test  = datasets.MNIST(root="../dataset", train=False, download=True, transform=transform)

            if opt['num_classes'] < 10:
                selected_classes = opt['classes']
                train = filter_by_class(train, selected_classes)
                test  = filter_by_class(test, selected_classes)

        elif opt['dataset'] == 'FashionMNIST':
            if opt['model'] == 'mlp':
                if opt['downsample']:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # per-image min–max
                        Downsample(opt),
                        transforms.Lambda(lambda x: x.view(-1))  # flatten
                        ])
                else:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # per-image min–max
                        transforms.Lambda(lambda x: x.view(-1))  # flatten
                        ])

            elif opt['model'] == 'cnn':
                if opt['downsample']:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # per-image min–max
                        Downsample(opt),
                        ])
                else:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # per-image min–max
                        ])
                    
            train = datasets.FashionMNIST(root="../dataset", train=True, download=True, transform=transform)
            test  = datasets.FashionMNIST(root="../dataset", train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train, batch_size=opt['batch_size'], shuffle=True, **opt['kwargs'])
        test_loader = torch.utils.data.DataLoader(test, batch_size=opt['batch_size'], shuffle=False, **opt['kwargs'])

    elif opt['task'] == 'regression':

        if 'function' in opt['dataset']:
            # ==================
            def f1(X):
                return (
                    np.sin(X[:, 0]) + np.cos(X[:, 1])
                )
            
            def f2(X):
                return (
                    np.exp(X[:, 0]) * np.sin(X[:, 1])
                    + X[:, 2] * np.cos(X[:, 3])
                    - X[:, 4] * X[:, 0]
                )
            
            def normalize(y):
                return 2 * (y - y_min) / (y_max - y_min) - 1
            # -------------------
            # Generate full dataset
            n_total = 12000
            
            if opt['dataset'] == 'function1':
                X = np.random.uniform(-1, 1, (n_total, 2))
                y = f1(X)
            elif opt['dataset'] == 'function2':
                X = np.random.uniform(-1, 1, (n_total, 5))
                y = f2(X)

            # Train / test split
            n_train = 10000
            perm = np.random.permutation(n_total)

            x_train = X[perm[:n_train]]
            y_train = y[perm[:n_train]]

            x_test  = X[perm[n_train:]]
            y_test  = y[perm[n_train:]]
            
            noise_std = 0.05 * np.std(y_train)
            y_train_noisy = y_train + noise_std * np.random.randn(len(y_train))

            y_min = y_train_noisy.min()
            y_max = y_train_noisy.max()

            y_train = normalize(y_train_noisy)
            y_test  = normalize(y_test)

            x_train = torch.tensor(x_train, dtype=torch.float32).to(opt['device'])
            x_test = torch.tensor(x_test, dtype=torch.float32).to(opt['device'])
            y_train = torch.tensor(y_train, dtype=torch.float32).to(opt['device'])
            y_test = torch.tensor(y_test, dtype=torch.float32).to(opt['device'])

            train_dataset = TensorDataset(x_train, y_train)
            test_dataset  = TensorDataset(x_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size = opt['batch_size'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size = opt['batch_size'], shuffle=False)
            

        elif opt['dataset'] == 'MNIST' and opt['model'] == 'mlp':

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # per-image min–max
                transforms.Lambda(lambda x: x.view(-1))  # flatten
                ])
            train = datasets.MNIST(root="../dataset", train=True, download=True, transform=transform)
            test  = datasets.MNIST(root="../dataset", train=False, download=True, transform=transform)
                
            train_reg_dataset = FloatLabelDataset(train)
            test_reg_dataset  = FloatLabelDataset(test)

            train_loader = DataLoader(train_reg_dataset, batch_size=opt['batch_size'], shuffle=True)
            test_loader  = DataLoader(test_reg_dataset, batch_size=opt['batch_size'], shuffle=False)

        
        elif opt['dataset'] == 'MNIST' and opt['model'] == 'cnn':

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # per-image min–max
                ])

            train = datasets.MNIST(root="../dataset", train=True, download=True, transform=transform)
            test  = datasets.MNIST(root="../dataset", train=False, download=True, transform=transform)
                
            train_reg_dataset = FloatLabelDataset(train)
            test_reg_dataset  = FloatLabelDataset(test)

            train_loader = DataLoader(train_reg_dataset, batch_size = opt['batch_size'], shuffle=True)
            test_loader = DataLoader(test_reg_dataset, batch_size = opt['batch_size'], shuffle=False)

    return train_loader, test_loader, opt
