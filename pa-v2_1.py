from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, LeakyReLU, Dropout, CrossEntropyLoss, Module, Softmax, Parameter
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
WHOLE_TRAIN = 2

class MyNet(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.l2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.l3 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.l4 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.x1 = Linear(8*8*128, 512 - 128)
        self.x1_ext = Linear(8*8*128, 512)
        self.x2 = Linear(512 - 128, 256 - 64)
        self.x2_ext = Linear(512, 256)
        self.x3 = Linear(256 - 64, 80)
        self.x3_ext = Linear(256, 100)
        self.relu = LeakyReLU()
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()
        self.mode = FIRST_TRAIN
        self.softmax = Softmax()

    def forward(self, inp):
        s1 = self.relu(self.l1(inp))
        s2 = self.pool(self.relu(self.l2(s1)))
        s3 = self.relu(self.l3(s2))
        s4 = self.pool(self.relu(self.l4(s3)))
        p1 = self.ssum_x1(self.flatten(s4))
        p2 = self.relu(self.ssum_x2(self.dropout(self.relu(p1))))
        p3 = self.ssum_x3(self.dropout(p2))
        return p3

    def setup_layer(self):
        if self.mode == WHOLE_TRAIN:
            self.x1_ext.weight = Parameter(torch.cat((self.x1.weight, torch.zeros(128, 8*8*128).to(device)), dim=0))
            self.x1_ext.bias = Parameter(torch.cat((self.x1.bias, torch.zeros(128).to(device))))
            self.ssum_x1 = self.x1_ext
            self.x2_ext.weight = Parameter(torch.cat((torch.cat((self.x2.weight, torch.zeros(256 - 64, 128).to(device)), dim=1), torch.zeros(64, 512).to(device)), dim=0))
            self.x2_ext.bias = Parameter(torch.cat((self.x2.bias, torch.zeros(64).to(device))))
            self.ssum_x2 = self.x2_ext
            self.x3_ext.weight = Parameter(torch.cat((torch.cat((self.x3.weight, torch.zeros(80, 64).to(device)), dim=1), torch.zeros(20, 256).to(device)), dim=0))
            self.x3_ext.bias = Parameter(torch.cat((self.x3.bias, torch.zeros(20).to(device))))
            self.ssum_x3 = self.x3_ext
        else:
            self.ssum_x1 = self.x1
            self.ssum_x2 = self.x2
            self.ssum_x3 = self.x3


model = MyNet().to(device)

loss_fn = CrossEntropyLoss()

st_time = time.time()

model.mode = FIRST_TRAIN
model.setup_layer()
optimizer = Adam(model.parameters(), lr=1e-4)

loss_lst = []
t_lst = []
loss_mean_lst = []
loss_mean_t_lst = []
for epoch in range(40):
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
    model.eval()
    ss = 0
    cnt = 0
    for batch in tqdm(test_loader, unit="batch", total=len(test_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        output = model(inputs)
        _, pred = torch.max(output, 1)
        ss += torch.sum((pred == labels).sqeeze())
        cnt += torch.numel(pred)
    loss_mean_lst.append(ss/cnt)
    loss_mean_t_lst.append(time.time()-st_time)
    #torch.save(model.state_dict(), f"./results/first_train/params{epoch}.pt")

model.mode = WHOLE_TRAIN
model.setup_layer()
optimizer = Adam(model.parameters(), lr=1e-4)

for epoch in range(40, 70):
    ss = 0
    print(epoch)
    model.train()
    for batch in tqdm(whole_train_loader, unit="batch", total=len(whole_train_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        loss_lst.append(loss.item())
        t_lst.append(time.time()-st_time)
    model.eval()
    ss = 0
    cnt = 0
    for batch in tqdm(whole_test_loader, unit="batch", total=len(whole_test_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        output = model(inputs)
        _, pred = torch.max(output, 1)
        ss += torch.sum((pred == labels).sqeeze())
        cnt += torch.numel(pred)
    loss_mean_lst.append(ss/cnt)
    loss_mean_t_lst.append(time.time()-st_time)
    #torch.save(model.state_dict(), f"./results/second_train/params{epoch}.pt")

plt.plot(t_lst, loss_lst)
plt.plot(loss_mean_t_lst, loss_mean_lst)


model = MyNet().to(device)

loss_fn = CrossEntropyLoss()
st_time = time.time()

model.mode = WHOLE_TRAIN
model.setup_layer()
optimizer = Adam(model.parameters(), lr=1e-4)

loss_lst = []
loss_mean_lst = []
loss_mean_t_lst = []
t_lst = []
for epoch in range(55):
    ss = 0
    print(epoch)
    model.train()
    for batch in tqdm(whole_train_loader, unit="batch", total=len(whole_train_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        loss_lst.append(loss.item())
        t_lst.append(time.time()-st_time)
    model.eval()
    ss = 0
    cnt = 0
    for batch in tqdm(whole_test_loader, unit="batch", total=len(whole_test_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        output = model(inputs)
        _, pred = torch.max(output, 1)
        ss += torch.sum((pred == labels).sqeeze())
        cnt += torch.numel(pred)
    loss_mean_lst.append(ss/cnt)
    loss_mean_t_lst.append(time.time()-st_time)
    #torch.save(model.state_dict(), f"./results/whole_train/params{epoch}.pt")

plt.plot(t_lst, loss_lst)
plt.plot(loss_mean_t_lst, loss_mean_lst)

plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()