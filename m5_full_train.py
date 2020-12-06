## This Python 3 environment comes with many helpful analytics libraries installed
## It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
## For example, here's several helpful packages to load
import os
import os.path as osp
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc
import tqdm
import glob
import yaml
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset 
from torch.autograd import Variable

import datetime
import pytz

## Input data files are available in the read-only "../input/" directory
## For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
## import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
## You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
## You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

## Reduce memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

class M5Dataset(Dataset):
    def __init__(self, train_dir, test_dir, product_dir, split='train', transform=True):
        super(M5Dataset, self).__init__()
        self.split = split
        self._transform = transform

        if self.split == 'train':
            self.df_data = pd.read_csv(train_dir)
        else:
            self.df_data =  pd.read_csv(test_dir)
        self.df_product = pd.read_csv(product_dir)
        self.df_data = reduce_mem_usage(self.df_data)
        gc.collect()

    def __len__(self):
        return len(self.df_product)
    
    def __getitem__(self, idx):
        item_id = self.df_product['item_id'][idx]
        store_id = self.df_product['store_id'][idx]
        mean = self.df_product['mean'][idx]
        std = self.df_product['std'][idx]
        state = store_id[-4:-2]
        data = self.df_data[self.df_data['item_store'] == item_id + '_' + store_id]
        embedding_data = data[['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_name_2', 'event_type_1', 'event_type_2']]
        feat_data = data[['wday', 'month', 'year', 'snap_'+state, 'sell_price', 'available']]
        target = data[['demand_norm']] 
        embedding_data, feat_data, target = embedding_data.to_numpy(), feat_data.to_numpy(), target.to_numpy() 
        if self._transform:
            embedding_data, feat_data, target = self.transform(embedding_data, feat_data, target)
        return item_id, store_id, mean, std, embedding_data, feat_data, target
        
    def transform(self, embedding_data, feat_data, target):
        embedding_data = embedding_data.astype(np.long)
        embedding_data = torch.from_numpy(embedding_data).long()
        feat_data = feat_data.astype(np.float64)
        feat_data = torch.from_numpy(feat_data).float()
        target = target.astype(np.float64)
        target = torch.from_numpy(target).float()
        return embedding_data, feat_data, target

class M5Net(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_lstm_layer, n_classifier_layer, output_dim, batch_first=True):
        super(M5Net, self).__init__()
        self.embedding_dim =embedding_dim
        self.hidden_dim = hidden_dim
        self.n_lstm_layer = n_lstm_layer
        self.n_classifier_layer = n_classifier_layer
        self.output_dim = output_dim
        self.batch_first = batch_first
        
        self.item_embedding = nn.Embedding(3049, embedding_dim)
        self.dept_embedding = nn.Embedding(7, embedding_dim)
        self.cat_embedding = nn.Embedding(3, embedding_dim)
        self.store_embedding = nn.Embedding(10, embedding_dim)
        self.state_embedding = nn.Embedding(3, embedding_dim)
        self.event_name_embedding = nn.Embedding(31, embedding_dim)
        self.event_type_embedding = nn.Embedding(5, embedding_dim)
        self.calendar_prices_feat = nn.Linear(6, embedding_dim)
        self.sigmoid = nn.Sigmoid()
        self.encode = nn.Linear(4*embedding_dim, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_lstm_layer, batch_first=batch_first)
        
        conv28_layer = [nn.Conv1d(hidden_dim, hidden_dim, 19, dilation=3, padding=27), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
        self.conv28 = nn.Sequential(*conv28_layer)
        conv7_layer = [nn.Conv1d(hidden_dim, hidden_dim, 7, dilation=2, padding=6), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
        self.conv7 = nn.Sequential(*conv7_layer)
        conv3_layer = [nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
        self.conv3 = nn.Sequential(*conv3_layer)
        
        classifier_layers = []
        for i in range(n_classifier_layer - 1):
            classifier_layers.append(nn.Dropout(0.2))
            classifier_layers.append(nn.Linear(hidden_dim, hidden_dim))
            classifier_layers.append(nn.Sigmoid())
        classifier_layers.append(nn.Dropout(0.2))
        classifier_layers.append(nn.Linear(hidden_dim, output_dim))
        classifier_layers.append(nn.Tanh())
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x1, x2):
        item = self.item_embedding(x1[:,:,0])
        dept = self.dept_embedding(x1[:,:,1])
        cat = self.cat_embedding(x1[:,:,2])
        store = self.store_embedding(x1[:,:,3])
        state = self.state_embedding(x1[:,:,4])
        en1 = self.event_name_embedding(x1[:,:,5])
        en2 = self.event_name_embedding(x1[:,:,6])
        et1 = self.event_type_embedding(x1[:,:,7])
        et2 = self.event_type_embedding(x1[:,:,8])
        x2[:,:,4] = self.sigmoid(x2[:,:,4])
        i = item + dept + cat
        s = store + state
        e = en1 + en2 + et1 + et2
        c = self.calendar_prices_feat(x2)
        x = self.encode(torch.cat((i, s, e, c), 2))
        y, (_, _) = self.lstm(x)
        y = y.transpose(1,2)
        c3 = self.conv3(y)
        c7 = self.conv7(y)
        c28 = self.conv28(y)
        y = y + c3 + c7 + c28
        y = y.transpose(1,2)
        out = self.classifier(y)
        return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ## Define hyper parameters
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--embedding-dim', type=int, default=10, help='embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=48, help='hidden dimension')
    parser.add_argument('--output-dim', type=int, default=1, help='hidden dimension')
    parser.add_argument('--n-lstm-layer', type=int, default=2, help='number of lstm layers')
    parser.add_argument('--n-classifier-layer', type=int, default=2, help='number of classifier layers')

    args = parser.parse_args()
    args.model = 'm5'

    here = osp.dirname(osp.abspath(__file__))
    log_dir = osp.join(here, 'log', args.model)
    runs = sorted(glob.glob(os.path.join(log_dir, 'experiment_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    output_dir = os.path.join(log_dir, 'experiment_{}'.format(str(run_id)))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    product_dir = osp.join(here, 'product.csv')
    train_dir = osp.join(here, 'train.csv')
    test_dir = osp.join(here, 'test.csv')

    with open(osp.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    print('Argument Parser: ')
    print(args.__dict__)

    cuda = torch.cuda.is_available()
    print('Cuda available: ', cuda)
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    ## Define data loader
    # kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(M5Dataset(train_dir, test_dir, product_dir, split='train', transform=True), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(M5Dataset(train_dir, test_dir, product_dir, split='valid', transform=True), batch_size=args.batch_size, shuffle=False)

    ## Define neural network model
    model = M5Net(args.embedding_dim, args.hidden_dim, args.n_lstm_layer, args.n_classifier_layer, args.output_dim)
    checkpoint = torch.load(osp.join(here, 'checkpoint.pth.tar'))
    model.load_state_dict(checkpoint['model_state_dict'])

    if cuda:
        model = model.cuda()

    # Define optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Define loss function
    loss_fn = nn.MSELoss()

    if not osp.exists(osp.join(output_dir, 'log.csv')):
        with open(osp.join(output_dir,'log.csv'), 'w') as f:
            header = ['epoch','iteration', 'MSE']
            header = map(str, header)
            f.write(','.join(header) + '\n')


    for epoch in range(args.epochs):
        iteration = 0
        if not osp.exists(osp.join(output_dir, 'valid_%d.csv'%epoch)):
            with open(osp.join(output_dir,'valid_%d.csv'%epoch), 'w') as f:
                header = ['id'] + ['F%d'%i for i in range(1,29)]
                header = map(str, header)
                f.write(','.join(header) + '\n')
        if not osp.exists(osp.join(output_dir, 'eval_%d.csv'%epoch)):
            with open(osp.join(output_dir,'eval_%d.csv'%epoch), 'w') as f:
                header = ['id'] + ['F%d'%i for i in range(1,29)]
                header = map(str, header)
                f.write(','.join(header) + '\n')
                
        model.train()
        for batch_idx, (item_id, store_id,  mean, std, embedding_data, feat_data, target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), ncols=80, desc='Train epoch=%d'%epoch, leave=False):
            assert model.training
            if cuda:
                embedding_data, feat_data, target = embedding_data.cuda(), feat_data.cuda(), target.cuda()
            embedding_data, feat_data, target = Variable(embedding_data), Variable(feat_data), Variable(target)
            for j in range(100):
                rand = np.random.randint(low=0, high=300)
                sub_embedding_data, sub_feat_data, sub_target = embedding_data[:,rand:rand+100,:], feat_data[:,rand:rand+100,:], target[:,rand:rand+100,:]
                optim.zero_grad()
                predict = model(sub_embedding_data, sub_feat_data)
                loss = loss_fn(predict, sub_target)
                loss.backward()
                train_loss = loss.item()
                optim.step()
                if j == 99:
                    with open(osp.join(output_dir, 'log.csv'), 'a') as f:
                        log = [epoch, iteration, train_loss]
                        log = map(str, log)
                        f.write(','.join(log) + '\n')
            iteration = iteration + 1

        checkpoint = {
            'model': model,
            'optim_state_dict': optim.state_dict(),
            'model_state_dict': model.state_dict()
        }
        
        torch.save(checkpoint, 'checkpoint%d.pth.tar'%epoch)
        model.eval()
        for batch_idx, (item_id, store_id,  mean, std, embedding_data, feat_data, target) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), ncols=80, desc='Validation epoch=%d'%epoch, leave=False):
            if cuda:
                embedding_data, feat_data = embedding_data.cuda(), feat_data.cuda()
            embedding_data, feat_data = Variable(embedding_data), Variable(feat_data)
            with torch.no_grad():
                predicts = model(embedding_data, feat_data)
                predicts = predicts.data.cpu().numpy().squeeze(axis=2)
                target = target.numpy().squeeze(axis=2)
                mean, std =  mean.numpy(), std.numpy()
                with open(osp.join(output_dir, 'valid_%d.csv'%epoch), 'a') as f:
                    for (item, store, m, s, pred, tgt) in zip(item_id, store_id, mean, std, predicts, target):
                        pred = (pred * s + m)
                        tgt = (tgt * s + m)
                        log = [item+'_'+store+'_validation'] + [p for p in pred[-56:-28]]+[t for t in tgt[-56:-28]]
                        log = map(str, log)
                        f.write(','.join(log) + '\n')
                with open(osp.join(output_dir, 'eval_%d.csv'%epoch), 'a') as f:
                    for (item, store, pred) in zip(item_id, store_id, predicts):
                        pred = (pred * s + m)
                        log = [item+'_'+store+'_evaluation'] + [p for p in pred[-28:]]
                        log = map(str, log)
                        f.write(','.join(log) + '\n')
        