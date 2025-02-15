from IntelImageClassification.constants import *
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from IntelImageClassification.entity.config_entity import TrainingConfig



class Training:
    def __init__(self,config:TrainingConfig):
        self.config=config
    
    def get_resnet_model(self):
        self.model=torch.load(self.config.resnet_model_path).to(torch.device('cuda'))
    
    def train_valid_generator(self):
        self.training_data=datasets.ImageFolder(root=self.config.training_data,transform=transforms.Compose([
        transforms.Resize(size=(150 , 150)) ,
        transforms.RandomCrop(size=(150,150)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])]))
        self.test_data=datasets.ImageFolder(root=self.config.test_data , transform = transforms.Compose([
        transforms.Resize((150, 150)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])]))
    
    def class_finder(self):
        self.classes = sorted(i.name for i in os.scandir(self.config.training_data) if i.is_dir())
        if not self.classes:
            raise FileNotFoundError(f'This directory dose not have any classes : {self.training_data}')
        self.class_to_inx = {name : value for name , value in enumerate(self.classes) }
    

    def dataloaders(self):
        self.number_train=len(self.training_data)
        self.indx = list(range(self.number_train))
        np.random.shuffle(self.indx)
        self.split = int(0.10 * self.number_train)
        train_idx, valid_idx = self.indx[self.split:], self.indx[:self.split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        self.train_dataloader=DataLoader(dataset= self.training_data , 
                              batch_size= 32 ,  
                              num_workers=0,
                              sampler=train_sampler)
        self.valid_dataloader = DataLoader(dataset=self.training_data , 
                             batch_size=32 , 
                             num_workers=0,
                             sampler=valid_sampler                            
                             )
        self.test_dataloader = DataLoader(dataset=self.test_data,
                            batch_size=32,
                            num_workers=0,
                            shuffle=False)
        
    def train_model(self):
        self.criterion=nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.epochs=1
        self.train_samples_num = 12630
        self.val_samples_num = 1404
        self.train_costs=[]
        self.val_costs=[]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'



        for epoch in range(self.epochs):

            train_running_loss = 0
            correct_train = 0
        
            self.model.train().cuda()
        
            for inputs, labels in self.train_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
            
                self.optimizer.zero_grad()
                prediction = self.model(inputs)
                        
                loss = self.criterion(prediction, labels)
          
                loss.backward()         
                self.optimizer.step()
                _, predicted_outputs = torch.max(prediction.data, 1)
                correct_train += (predicted_outputs == labels).float().sum().item()
                train_running_loss += (loss.data.item() * inputs.shape[0])


            train_epoch_loss = train_running_loss / self.train_samples_num
        
            self.train_costs.append(train_epoch_loss)
        
            train_acc =  correct_train / self.train_samples_num
            val_running_loss = 0
            correct_val = 0
      
            self.model.eval().cuda()
    
            with torch.no_grad():
                for inputs, labels in self.valid_dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    prediction = self.model(inputs)
                    loss = self.criterion(prediction, labels)
                    _, predicted_outputs = torch.max(prediction.data, 1)
                    correct_val += (predicted_outputs == labels).float().sum().item()

                val_running_loss += (loss.data.item() * inputs.shape[0])

                val_epoch_loss = val_running_loss / self.val_samples_num
                self.val_costs.append(val_epoch_loss)
                val_acc =  correct_val / self.val_samples_num
        
            info = "[Epoch {}/{}]: train-loss = {:0.6f} | train-acc = {:0.3f} | val-loss = {:0.6f} | val-acc = {:0.3f}"
        
            print(info.format(epoch+1, self.epochs, train_epoch_loss, train_acc, val_epoch_loss, val_acc))
        
            torch.save(self.model.state_dict(), 'checkpoint_gpu_{}'.format(epoch + 1)) 
                                                                
        torch.save(self.model.state_dict(), Path('artifacts/training/resnet-50_weights_gpu'))  
        
        return self.train_costs, self.val_costs
