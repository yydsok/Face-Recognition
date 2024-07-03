from cho_models.resnet import resnet50
from cho_models.senet import senet50
from cho_models.vit import ViT
import matplotlib.pyplot as plt

from torchvision.models import vit_b_16
from cho_models.ceit import CustomViTModel
import os
import torch
from transformers import AutoModel
from huggingface_hub import hf_hub_download
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import v2
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])



from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import timm
import fnmatch
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
data_dir = 'D:/face/lfw'
name_list = os.listdir(data_dir)
face_names = [name for name in os.listdir(data_dir) if len(glob.glob(os.path.join(data_dir, name, '*.jpg'))) >= 60]
face_names_mapping = {}
js = 0
for name in face_names:
    face_names_mapping[name] = js
    js+=1
print(len(face_names))
def find_images(directory, pattern='*.jpg'):
    for root, dirs, files in os.walk(directory):
        for file in fnmatch.filter(files, pattern):
            if '_'.join(file.split('_')[:-1]) in face_names:
                yield file

# 指定你要搜索的文件夹路径
directory_path = data_dir

# 保存所有找到的图片路径的列表
imgs_list = list(find_images(directory_path))
print(face_names_mapping)
class LFWDataset(Dataset):
    def __init__(self, imgs, transform=None):
        self.transform = transform
        self.labels = [face_names_mapping['_'.join(img.split('_')[:-1])] for img in imgs]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img_path = os.path.join(data_dir, '_'.join(imgs_list[idx].split('_')[:-1]), imgs_list[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


dataset = LFWDataset(imgs_list, transform=transform)
random.seed(42)
rand_list = [i for i in range(len(dataset))]
random.shuffle(rand_list)
length = int(len(dataset) * 0.85)
train_idx, val_idx = rand_list[:length], rand_list[length:]
train_dataset = LFWDataset([imgs_list[i] for i in train_idx], transform=transform)
val_dataset = LFWDataset([imgs_list[i] for i in val_idx], transform=transform)
print(len(train_dataset),len(val_dataset))
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size = batch_size,shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = batch_size,shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = senet50(include_top=True, num_classes=len(face_names))

#model = vit_b_16(weights='IMAGENET1K_V1')
#model.heads = nn.Linear(model.heads[0].in_features,len(face_names)) 

model = timm.create_model('vgg19_bn.tv_in1k', pretrained=True, num_classes=12,).eval()

#processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
#model = CustomViTModel(num_classes=len(face_names))




model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08)
import torch.optim.lr_scheduler as lr_scheduler

# 初始化学习率调度器
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
from tqdm import tqdm

num_epochs = 30
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)  # 通过模型获取输出
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_accuracies.append(train_accuracy)
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

    scheduler.step()

# 绘制训练准确率图表
print(train_accuracies)
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_accuracies, label='Val Accuracy', marker='o')
plt.title(' demo Accuracy ')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()