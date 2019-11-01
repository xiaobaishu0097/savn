import torch
import h5py
import datetime
import os
import argparse

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight-decay', type=float, default=0.00001)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--clip-norm', type=float, default=1)
parser.add_argument('--epochs', type=int, default=10)


class OptimalPath(nn.Module):
    def __init__(self):
        super(OptimalPath, self).__init__()
        self.Linear_1 = nn.Linear(25088, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.Linear_2 = nn.Linear(260, 6)

    def forward(self, obs, det):
        x = obs.view(obs.size(0), -1)
        x = self.Linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = torch.cat((x, det), dim=1)
        x = self.Linear_2(x)

        return x


class FeatureLoader(Dataset):
    def __init__(self, data_path):
        self.data_reader = h5py.File(data_path, 'r')
        self.positions = list(self.data_reader.keys())

    def __getitem__(self, index):
        pos = self.positions[index]
        image_feature = self.data_reader[pos]['feature'][()]
        detection = self.data_reader[pos]['detection'][()]
        optimal_action = self.data_reader[pos]['optimal_action'][()]

        return {'feature': image_feature, 'detection': detection, 'optimal_action': optimal_action}

    def __len__(self):
        return len(self.positions)


def train(data_loader, model, criterion, optimizer, clip_norm, epoch, print_freq=100):
    model.train()

    for i, sample in enumerate(data_loader):
        print('Fuck')
        feature, detection, optimal_action = sample['feature'], sample['detection'], sample['optimal_action']
        detection = detection.float()
        feature, detection, optimal_action = feature.cuda(), detection.cuda(), optimal_action.cuda()

        action_pro = model(feature, detection)
        loss = criterion(action_pro, torch.max(optimal_action.long(), 1)[1])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm, norm_type=2)
        optimizer.step()

        _, preds = torch.max(action_pro, dim=1)
        correct = float((optimal_action.int() == preds.int()).sum())
        accuracy = correct / len(optimal_action)

        if i % print_freq == 0:
            print(
                'Train:\t'
                'Epoch:[{0}][{1}/{2}]   \t'
                'Acc: {acc:.3f}\t'
                'Loss: {loss:.4f}\t'.format(
                    epoch, i + 1, len(data_loader), acc=accuracy, loss=loss)
            )

        if i > (len(data_loader) / (len(optimal_action) * 4)):
            break


def main():
    print(datetime.datetime.now())

    args = parser.parse_args()
    lr = args.lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    num_workers = args.num_workers
    clip_norm = args.clip_norm
    epochs = args.epochs

    train_data_path = './data/Meta_train_Data.hdf5'
    val_data_path = './data/Meta_val_Data.hdf5'
    test_data_path = './data/Meta_test_Data.hdf5'

    train_dataset = FeatureLoader(train_data_path)
    val_dataset = FeatureLoader(val_data_path)
    test_dataset = FeatureLoader(test_data_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                             num_workers=num_workers)

    model = OptimalPath()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        train(val_loader, model, criterion, optimizer, clip_norm, epoch)


if __name__ == '__main__':
    main()
