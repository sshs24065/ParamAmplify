from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, LeakyReLU, Dropout, CrossEntropyLoss, Module, Softmax
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassFilter(Dataset):
    def __init__(self, examples, filter_function):
        self.data = [item for item in examples if filter_function(item[1])]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


train_dataset = ClassFilter(CIFAR100(r"\data", train=True, transform=ToTensor(), download = True), lambda x: x < 80)
test_dataset = ClassFilter(CIFAR100(r"\data", train=False, transform=ToTensor(), download = True), lambda x: x < 80)
next_train_dataset = ClassFilter(CIFAR100(r"\data", train=True, transform=ToTensor(), download = True), lambda x:x > 79)
next_test_dataset = ClassFilter(CIFAR100(r"\data", train=False, transform=ToTensor(), download = True), lambda x:x > 79)
whole_train_dataset = CIFAR100(r"\data", train=True, transform=ToTensor(), download = True)
whole_test_dataset = CIFAR100(r"\data", train=False, transform=ToTensor(), download = True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
next_train_loader = DataLoader(next_train_dataset, batch_size=64, shuffle=True)
next_test_loader = DataLoader(next_test_dataset, batch_size=64, shuffle=True)
whole_train_loader = DataLoader(whole_train_dataset, batch_size=64, shuffle=True)
whole_test_loader = DataLoader(whole_test_dataset, batch_size=64, shuffle=True)

FIRST_TRAIN = 1
SECOND_TRAIN = 2
WHOLE_TRAIN = 3

class MyNet(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.l2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.l3 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.l4 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.x1 = Linear(8*8*128, 512)
        self.x2 = Linear(512, 256)
        self.x3 = Linear(256, 80)
        self.relu = LeakyReLU()
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()
        self.x1_add = Linear(8*8*128, 512)
        self.x3_add = Linear(256, 20)
        self.mode = FIRST_TRAIN
        self.softmax = Softmax()

    def forward(self, inp):
        s1 = self.relu(self.l1(inp))
        s2 = self.pool(self.relu(self.l2(s1)))
        s3 = self.relu(self.l3(s2))
        s4 = self.pool(self.relu(self.l4(s3)))
        p1 = self.x1(self.flatten(s4))
        if self.mode == SECOND_TRAIN or self.mode == WHOLE_TRAIN:
            p1 += self.x1_add(self.flatten(s4))
        p2 = self.relu(self.x2(self.dropout(self.relu(p1))))
        p3 = self.x3(self.dropout(p2))
        if self.mode == SECOND_TRAIN or self.mode == WHOLE_TRAIN:
            p3 = torch.cat([p3, self.x3_add(self.dropout(p2))], dim=1)
        return p3

    def setup_layer(self):
        if self.mode == SECOND_TRAIN:
            for param in self.parameter():
                param.requires_grad = False
            self.x1_add.requires_grad = True
            self.x3_add.requires_grad = True
        else:
            for param in self.parameter():
                param.requires_grad = True

model = MyNet().to(device)

loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)
st_time = time.time()

loss_lst = []
t_lst = []
for epoch in range(30):
    print(epoch)
    model.train()
    for batch in tqdm(train_loader, unit="batch", total=len(train_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        loss_lst.append(loss.item())
        t_lst.append(time.time()-st_time)
    torch.save(model.state_dict(), f"./results/first_train/params{epoch}.pt")


model.mode = SECOND_TRAIN

for epoch in range(30, 40):
    print(epoch)
    model.train()
    for batch in tqdm(next_train_loader, unit="batch", total=len(next_train_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        loss_lst.append(loss.item())
        t_lst.append(time.time()-st_time)
    torch.save(model.state_dict(), f"./results/second_train/params{epoch}.pt")

model.mode = WHOLE_TRAIN
plt.plot(t_lst, loss_lst)
for epoch in range(40, 50):
    print(epoch)
    model.train()
    for batch in tqdm(whole_train_loader, unit="batch", total=len(next_train_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        loss_lst.append(loss.item())
        t_lst.append(time.time()-st_time)
    torch.save(model.state_dict(), f"./results/second_train/params{epoch}.pt")

plt.plot(t_lst, loss_lst)





model = MyNet().to(device)

loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)
st_time = time.time()

model.mode = WHOLE_TRAIN

loss_lst = []
t_lst = []
for epoch in range(50):
    print(epoch)
    model.train()
    for batch in tqdm(whole_train_loader, unit="batch", total=len(train_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        loss_lst.append(loss.item())
        t_lst.append(time.time()-st_time)
    torch.save(model.state_dict(), f"./results/whole_train/params{epoch}.pt")

plt.plot(t_lst, loss_lst)

plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()