import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import logging
import torchvision.transforms as transforms

from config import cfg
from DogNet import DogNet
from utils.model import *
from utils.data import DogsDataset
from tensorboardX import SummaryWriter


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(filename=cfg.routes['log']+'/log_'+str(cfg.train_params['arch'])+'.log', level=logging.INFO)


def fit_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train(True)

    running_loss = 0.0
    processed_data = 0

    for batch, labels in data_loader:
        batch = batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        prediction_x, prediction_aux1, prediction_aux2 = [model(batch)[j] for j in range(3)]
        loss_x = loss_fn(prediction_x, labels)
        loss_aux1 = loss_fn(prediction_aux1, labels)
        loss_aux2 = loss_fn(prediction_aux2, labels)

        loss_x.backward()
        loss_aux1.backward()
        loss_aux2.backward()
        optimizer.step()

        running_loss += loss_x.item() * batch.size(0)
        processed_data += batch.size(0)

    train_loss = running_loss / processed_data

    return train_loss


def eval_epoch(model, data_loader, loss_fn, device, epoch_num, tensorboard_writer):
    model.eval()

    running_loss = 0.0
    processed_size = 0

    for batch, labels in data_loader:
        batch = batch.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            prediction_x = model(batch)[0]
            loss_x = loss_fn(prediction_x, labels)

        running_loss += loss_x.item() * batch.size(0)
        processed_size += batch.size(0)

    loss = running_loss / processed_size
    message = f"epoch {epoch_num}: loss value={loss}"
    logging.info(message)
    print(message)

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar('Loss', loss, epoch_num)

    return loss


def train(data_loaders, model):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    if cfg.routes['tensorboard'] is not None:
        path = cfg.routes['tensorboard']
        tensorboard_writer = SummaryWriter('{}/{}'.format(path, cfg.train_params['arch']))
    else:
        tensorboard_writer = None

    for epoch in range(cfg.train_params['start_epoch'], cfg.train_params['epochs']):
        fit_epoch(model, data_loaders['train'], criterion, optimizer, DEVICE)
        eval_epoch(model, data_loaders['val'], criterion, DEVICE, epoch, tensorboard_writer)

        torch.save(model.state_dict(), cfg.routes['weights'] + '/' + cfg.train_params['arch'] + '_epoch' +
                   repr(epoch) + '.pth')


if __name__ == '__main__':
    # Prepare environment
    if not os.path.exists(cfg.routes['weights']):
        os.mkdir(cfg.routes['weights'])
    if not os.path.exists(cfg.routes['log']):
        os.mkdir(cfg.routes['log'])
    if cfg.routes['tensorboard'] is not None and not os.path.exists(cfg.routes['tensorboard']):
        os.mkdir(cfg.routes['tensorboard'])

    if cfg.env_params['random_seed'] is not None:
        torch.manual_seed(cfg.env_params['random_seed'])
        np.random.seed(cfg.env_params['random_seed'])
        random.seed(cfg.env_params['random_seed'])

    cudnn.benchmark = True

    # Prepare model
    if cfg.train_params['arch'] == 'DogNet':
        model = DogNet(out_classes=120)
    else:
        logging.info(f"{cfg.train_params['arch']} is not correct; correct architectures: "
                     f"{['FeatherNetA', 'FeatherNetB']}")
        raise NameError

    if cfg.train_params['resume_net'] is not None:
        logging.info('Loading resume network...')
        model = load_weights(model, cfg.train_params['resume_net'], DEVICE)

    model.to(DEVICE)

    # Load data
    transform = {
        'train': transforms.Compose([
            transforms.Resize((cfg.transform_params['rescale_size'], cfg.transform_params['rescale_size'])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.transform_params['mean'], std=cfg.transform_params['std'])]),
        'val': transforms.Compose([
            transforms.Resize((cfg.transform_params['rescale_size'], cfg.transform_params['rescale_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.transform_params['mean'], std=cfg.transform_params['std'])])
    }

    datasets = {
        'train': DogsDataset(mode='train', transforms=transform['train']),
        'val': DogsDataset(mode='val', transforms=transform['val'])
    }
    loaders = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=cfg.train_params['batch_size'],
                                       shuffle=True, num_workers=cfg.train_params['num_workers'])
        for x in ['train', 'val']
    }

    # Train model
    train(loaders, model)
