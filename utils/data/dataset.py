import pandas as pd

from pathlib import Path
from config import cfg
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from PIL import Image

labels = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier',
'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog',
'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 'border_terrier',
'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel',
'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel',
'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter',
'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees',
'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel',
'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor',
'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
'norwegian_elkhound', 'norwich_terrier','old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 'pembroke',
'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke',
'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky',
'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer',
'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound',
'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier',
'yorkshire_terrier']

encoder = preprocessing.LabelEncoder()
encoder.fit(labels)

class DogsDataset(Dataset):

    __DATA_MODES__ = ['train', 'val', 'test']

    def __init__(self, mode='train', dataset_root=cfg.routes['data'], transforms=None):
        self.mode = mode
        self.dataset_root = dataset_root
        self.transform = transforms
        self.labels = pd.read_csv(self.dataset_root + '/labels.csv')
        self.label_encoder = encoder

        if self.mode not in self.__DATA_MODES__:
            print(f"{self.mode} is not correct; correct modes: {self.__DATA_MODES__}")
            raise NameError

        self.images = self._prepare_files()

    def __getitem__(self, item):
        img = self._load_sample(self.images[item])

        if self.transform is not None:
            img = self.transform(img)

        if self.mode != 'test':
            label = self.labels[self.labels['id'] == str(self.images[item])[len('dataset/train/'):-4]]['breed'].values[0]
            return img, self.label_encoder.transform([label])[0]
        else:
            return img

    def __len__(self):
        return len(self.images)

    def _prepare_files(self):
        train_val_files = sorted(list(Path(self.dataset_root + '/train').rglob('*.jpg')))
        train_val_labels = [self.labels[self.labels['id'] == str(file)[len('dataset/train/'):-4]]['breed'].values[0]
                            for file in train_val_files]

        test_files = sorted(list(Path(self.dataset_root + '/test').rglob('*.jpg')))
        train_files, val_files = train_test_split(train_val_files, test_size=0.2, stratify=train_val_labels,
                                                  random_state=cfg.env_params['random_seed'])

        if self.mode == 'train':
            return train_files
        elif self.mode == 'test':
            return test_files
        else:
            return val_files

    @staticmethod
    def _load_sample(file):
        image = Image.open(file)
        image.load()
        return image
