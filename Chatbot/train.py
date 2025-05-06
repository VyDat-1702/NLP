import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_untils import bag_of_words, tokenize, stem
from model import Neuron_NW


#đọc file json
with open('/home/vydat/Code/Chatbot/Data/intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = [] 

for intent in intents['intents']:
    tag = intent['tag']
    
    tags.append(tag)
    for pattern in intent['patterns']:
        #tách từ trong câu
        w = tokenize(pattern)
        # thêm từ vào list chứa tất cả từ
        all_words.extend(w)
        # thêm tất cả từ vừa token kèm tag vào xy
        xy.append((w, tag))

#định dạng và loại bỏ ký tự không cần thiết
ignore_words = ['?', '.', '!', '"\"']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# loại bỏ trùng lặp và sắp xếp
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)#[1,1,0,0,1,0,]
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

num_epochs = 200
batch_size =16 
learning_rate = 0.0012 
input_size = len(X_train[0])
hidden_size = int(input_size*1.5)
output_size = len(tags)

print(f'input_size: {input_size}, output_size: {output_size}')

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # truy cập mẫu dữ liệu bằng index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # trả về tổng số mẫu dữ liệu
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Neuron_NW(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# mô hình train
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        outputs = model(words)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 10 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "/home/vydat/Code/Chatbot/Data/data.pth" #luu trang thai cua mo hinh, khong can train lai khi chay
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')