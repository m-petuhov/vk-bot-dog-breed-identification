import torchvision.transforms as transforms
import torch
import pandas as pd

from DogNet import DogNet
from utils.data import DogsDataset, labels
from utils.model import load_weights
from config import cfg


if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    transform = transforms.Compose([
            transforms.Resize((cfg.transform_params['rescale_size'], cfg.transform_params['rescale_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.transform_params['mean'], std=cfg.transform_params['std'])])
    dataset = DogsDataset(mode='test', transforms=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.train_params['batch_size'],
                                shuffle=False, num_workers=cfg.train_params['num_workers'])

    # Prepare model
    model = DogNet(out_classes=120)
    model.to(DEVICE)

    load_weights(model, './weights/DogNet_epoch7.pth', DEVICE)
    model.eval()

    print(len(dataset))
    i = 0
    with torch.no_grad():
        logits = []

        for batch in loader:
            if i % 5 == 0:
                print(i * 32)
            batch = batch.to(DEVICE)
            outputs = model(batch)[0].cpu()
            logits.append(outputs)
            i += 1

    probs = torch.softmax(torch.cat(logits), dim=-1).numpy()
    test_filenames = [name[len('dataset/test/'):-4] for name in dataset.images]

    ids = []
    expected = []
    for i in range(len(probs)):
        ids += [test_filenames[i]]
        expected += [probs[i]]

    my_submit = pd.DataFrame(expected, index=ids, columns=labels)

    my_submit.index.name = 'id'
    my_submit.to_csv('./my_submission.csv')


