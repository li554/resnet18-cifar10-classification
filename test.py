from torchvision import transforms
from dataset import CifarDataset
from models.resnet import Resnet18
import torch.utils.data as Data
import pandas as pd
import torch
from tqdm import tqdm

model = Resnet18(num_classes=10, improved=True)

model.load_state_dict(torch.load('runs/exp6/weights/best.pth'))
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
dataset = CifarDataset(img_dir='dataset/test/', img_label=None, transform=transform)
sampler = Data.SequentialSampler(dataset)
test_iter = Data.DataLoader(dataset, batch_size=128, shuffle=False, sampler=sampler)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
predictions = []
for X, _ in tqdm(test_iter):
    X = X.to(device)
    y_hat = model(X)
    predictions += list(y_hat.argmax(dim=1).cpu().detach().numpy())
df = pd.DataFrame(enumerate(predictions), columns=['id', 'label'])
df['id'] += 1
df['label'] = df['label'].apply(lambda x: classes[int(x)])
df.to_csv('submission.csv', index=False)
