import shutil
import urllib

import requests
import torch
import vk_api
import random
import urllib.request

from torch.autograd import Variable
from torchvision.transforms import transforms
from vk_api.longpoll import VkLongPoll, VkEventType
from PIL import Image

from DogNet import DogNet
from config import cfg
from utils.model import load_weights
from vk_bot import VkBot
from utils.data.dataset import encoder


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def write_msg(user_id, message):
    vk.method('messages.send', {'user_id': user_id, 'message': message, 'random_id': random.randint(0, 2048)})

# API-ключ созданный ранее
token = "d945835225cf3f6afb9ed24c0e720d8d83c71831fe06bd0352401bc9a191b1b801231a3f8c515db02e991"

# Авторизуемся как сообщество
vk = vk_api.VkApi(token=token)
vk_sess = vk.get_api()

# Работа с сообщениями
longpoll = VkLongPoll(vk)


print("Server started")

transform = transforms.Compose([
            transforms.Resize((cfg.transform_params['rescale_size'], cfg.transform_params['rescale_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.transform_params['mean'], std=cfg.transform_params['std'])])

model = DogNet(out_classes=120)
model.to(DEVICE)

load_weights(model, './weights/DogNet_epoch7.pth', DEVICE)
model.eval()

for event in longpoll.listen():
    if event.type == VkEventType.MESSAGE_NEW:
        if event.to_me:
            print(f'New message from {event.user_id}')

            if len(event.attachments) != 0 and event.attachments['attach1_type'] == 'photo':
                bot = VkBot(event.user_id, model)
                items = vk_sess.messages.getById(message_ids=event.message_id)
                item_url = items["items"][0]["attachments"][0]["photo"]["sizes"][4]["url"]
                response = requests.get(item_url, stream=True)
                local_file = open('./local.jpg', 'wb')
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, local_file)
                image = Image.open('./local.jpg')
                image.load()
                output = model(torch.Tensor(transform(image)).unsqueeze_(0))
                label = encoder.inverse_transform([torch.argmax(torch.softmax(output[0], dim=-1)).item()]).item()
                write_msg(event.user_id, f'{label}')
            else:
                write_msg(event.user_id, 'Oooops. Некорректный формат данных')