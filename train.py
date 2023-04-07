import random
import pandas as pd
import numpy as np
import os
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import albumentations as A

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings(action='ignore') 

import argparse
import logging
from pathlib import Path

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_dict, transforms, cfg, train_mode = 'train'):
        self.video_path_list = video_path_list
        self.label_list = label_dict
        self.mode = train_mode
        self.transforms = transforms
        self.cfg = cfg
        
    def __getitem__(self, index):        
        images = self.get_frames(self.video_path_list[index])
                        
        if self.transforms is not None:
            res = self.transforms(**images)
            images = np.zeros((len(images), self.cfg.img_size, self.cfg.img_size, 3))
            images[0, :, :, :] = res["image"]
            for i in range(1, len(images)):
                images[i, :, :, :] = res[f"image{i}"]

        images = torch.FloatTensor(images).permute(3, 0, 1, 2)
        # images = torch.FloatTensor(images)
        if self.label_list is not None:
            label = self.label_list[index]
            return images, label
        else:
            return images

    def __len__(self):
        return len(self.video_path_list) 
    
    def get_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        imgs = []        
        for fidx in range(frames):
            _, img = cap.read()            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        
        ret = {f"image{i}":imgs[i] for i in range(1, len(imgs))}
        ret['image'] = imgs[0]

        return ret

class WTmodel(nn.Module):
    def __init__(self, num_classes=3):
        super(WTmodel, self).__init__()
        self.backbone = torchvision.models.convnext_large(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        i = np.random.randint(50)
        x = x[:,:,i,:]
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
class BaseModel(nn.Module):
    def __init__(self, hidden_size=400):
        super(BaseModel, self).__init__()
        self.ego = torchvision.models.video.r3d_18(pretrained=True)   
        self.weather = WTmodel()
        self.timing = WTmodel(num_classes=2)
        
        self.ego_involve_fc = self.create_linear_layer(hidden_size, 3)
        self.dropout = nn.Dropout(0.5)
        
    def create_linear_layer(self, in_features, out_features, bias=True):
        layer = nn.Linear(in_features, out_features, bias = bias)
        layer.weight.data.normal_(mean=0.0, std=1.0)
        if bias:
            layer.bias.data.zero_()
        return layer
        
    def forward(self, x):
        ego_x = self.dropout(F.relu(self.ego(x)))
        weather_output = self.weather(x)
        timing_output = self.timing(x)
        
        ego_output = self.ego_involve_fc(ego_x)
        
        return ego_output, weather_output, timing_output

def train_model(model, optimizer, cfg, train_loader, val_loader, scheduler, device):
    
    save_modelpath = os.path.join(cfg.save_dir, cfg.save_name)
    criterion = {
        "ego": nn.CrossEntropyLoss(label_smoothing=0.1).to(device),
        "weather": nn.CrossEntropyLoss(label_smoothing=0.1).to(device),
        "timing": nn.CrossEntropyLoss().to(device)
    }
    
    best_score = 0
    best_model = None
    
    for epoch in range(1, cfg.epochs+1):
        model.train()
        train_loss = []
        logger.info('-'*100)
        logger.info(f'[{epoch}] Epoch Training................')
        for videos, labels in tqdm(iter(train_loader)):
            videos = videos.to(device)

            elabels = labels[1].to(device)
            wlabels = labels[2].to(device)
            tlabels = labels[3].to(device)

            optimizer.zero_grad()
            
            ego_output, weather_output, timing_output = model(videos)
            loss = criterion['ego'](ego_output, elabels) + criterion['weather'](weather_output, wlabels) + criterion['timing'](timing_output, tlabels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_ego_score, _val_weather_score, _val_timing_score = validation(model, criterion, val_loader, device)
        _val_score = _val_ego_score * _val_weather_score * _val_timing_score
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val score : [{_val_score:.5f}] Val ego F1 : [{_val_ego_score:.4f}] Val weather F1 : [{_val_weather_score:.4f}] Val timing F1 : [{_val_timing_score:.4f}]')
        logger.info(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val score : [{_val_score:.5f}] Val ego F1 : [{_val_ego_score:.4f}] Val weather F1 : [{_val_weather_score:.4f}] Val timing F1 : [{_val_timing_score:.4f}]')
        
        if scheduler is not None:
            scheduler.step(_val_score)
            
        if best_score <= _val_score:
            best_score = _val_score
            best_model = model
            torch.save(best_model.state_dict(),  save_modelpath + '/best_model.pt')
            print('Model Saved.')   
            logger.info("saved new best score model")
    
    logger.info(f"train completed, best_metric: {best_score:.4f}")
    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    ego_preds, ego_trues = [], []
    weather_preds, weather_trues = [], []
    timing_preds, timing_trues = [], []
    
    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)

            elabels = labels[1].to(device)
            wlabels = labels[2].to(device)
            tlabels = labels[3].to(device)
            
            ego_output, weather_output, timing_output = model(videos)
            
            loss = criterion['ego'](ego_output, elabels) + criterion['weather'](weather_output, wlabels) + criterion['timing'](timing_output, tlabels)
            
            val_loss.append(loss.item())
            
            ego_preds += ego_output.argmax(1).detach().cpu().numpy().tolist()
            ego_trues += elabels.detach().cpu().numpy().tolist()

            weather_preds += weather_output.argmax(1).detach().cpu().numpy().tolist()
            weather_trues += wlabels.detach().cpu().numpy().tolist()
            
            timing_preds += timing_output.argmax(1).detach().cpu().numpy().tolist()
            timing_trues += tlabels.detach().cpu().numpy().tolist()
            
        _val_loss = np.mean(val_loss)
    
    _val_ego_score = f1_score(ego_trues, ego_preds, average='macro')
    _val_weather_score = f1_score(weather_trues, weather_preds, average='macro')
    _val_timing_score = f1_score(timing_trues, timing_preds, average='macro')
    
    return _val_loss, _val_ego_score, _val_weather_score, _val_timing_score

def make_logger(cfg, name='train'):
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    console = logging.StreamHandler()
    save_logpath = os.path.join(cfg.save_dir, cfg.save_name)
    Path(save_logpath).mkdir(parents = True, exist_ok = True)
    file_handler = logging.FileHandler(save_logpath + '/train.log')
    
    console.setLevel(logging.WARNING)
    file_handler.setLevel(logging.DEBUG)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--vide_length", type=int, default=50)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--num_class", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=str, default=3e-4)
    
    parser.add_argument('--gpu', type=str, default='1,2')
    parser.add_argument("--save_dir", type=str, default='./save_log')
    parser.add_argument("--save_name", type=str, default='0308conv')
    
    cfg = parser.parse_args()
    
    seed_everything(cfg.seed) # Seed 고정

    os.environ["CUDA_VISIBLE_DEVICES"]= cfg.gpu  # Set the GPU 1 to use
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    logger = make_logger(cfg)
    logger.info(cfg)
    
    label_dict = {
            -1:[-1,-1,-1,-1],
            0:[0,0,0,0],
            1:[1,1,0,0],
            2:[1,1,0,1],
            3:[1,1,1,0],
            4:[1,1,1,1],
            5:[1,1,2,0],
            6:[1,1,2,1],
            7:[1,2,0,0],
            8:[1,2,0,1],
            9:[1,2,1,0],
            10:[1,2,1,1],
            11:[1,2,2,0],
            12:[1,2,2,1]
        }
    
    pseudo = pd.read_csv('train_persudo2.csv')
    pseudo['video_path'] = pseudo['video_path'].apply(lambda x : '/data/jhjang_datasets/car_crash' + x[1:])
    pseudo['ego-involve'] = 0
    pseudo['crash'] = 0

    pseudo['label_split'] = pseudo['label'].apply(label_dict.get)
    pseudo['tmp'] = pseudo[['crash', 'ego-involve', 'pseudo_weather', 'pseudo_timing']].values.tolist()
    pseudo.loc[(pseudo['label']==0), 'label_split'] = pseudo[pseudo['label'] == 0]['tmp']

    train, val = train_test_split(pseudo, test_size=0.1, random_state=cfg.seed)

    train_transforms = A.Compose([
        A.Resize(height=cfg.img_size, width=cfg.img_size),
        # A.HorizontalFlip(p=0.5),
        # A.OneOf([
        #     A.RandomBrightnessContrast(p=1),
        #     A.Blur(p=1),
        #     A.CLAHE(p=1)], p=0.5),
        A.Normalize()
    ], p=1, additional_targets={f"image{i}":"image" for i in range(1, 50)})

    test_transforms = A.Compose([
        A.Resize(height=cfg.img_size, width=cfg.img_size),
        A.Normalize()
    ], p=1, additional_targets={f"image{i}":"image" for i in range(1, 50)})
    
    train_dataset = CustomDataset(train['video_path'].values, train['label_split'].values, train_transforms, cfg, train_mode='train')
    train_loader = DataLoader(train_dataset, batch_size = cfg.batch_size, shuffle=True, num_workers=0)

    val_dataset = CustomDataset(val['video_path'].values, val['label_split'].values, test_transforms, cfg, train_mode='valid')
    val_loader = DataLoader(val_dataset, batch_size = cfg.batch_size, shuffle=True, num_workers=0)

    _model = BaseModel().cuda()
    model = nn.DataParallel(_model).to(device)
    model.eval()
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

    infer_model = train_model(model, optimizer, cfg, train_loader, val_loader, scheduler, device)