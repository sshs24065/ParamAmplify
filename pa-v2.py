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


first_train_dataset = ClassFilter(CIFAR100(r"\data", train=True, transform=ToTensor(), download = True), lambda x: x < 80)
first_test_dataset = ClassFilter(CIFAR100(r"\data", train=False, transform=ToTensor(), download = True), lambda x: x < 80)
whole_train_dataset = CIFAR100(r"\data", train=True, transform=ToTensor(), download = True)
whole_test_dataset = CIFAR100(r"\data", train=False, transform=ToTensor(), download = True)

first_train_loader = DataLoader(first_train_dataset, batch_size=64, shuffle=True)
first_test_loader = DataLoader(first_test_dataset, batch_size=64, shuffle=True)
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
        self.x1 = Linear(8*8*128, 512)
        self.x1_ext = Linear(8*8*128, 512 + 128)
        self.x2 = Linear(512, 256)
        self.x2_ext = Linear(512 + 128, 256 + 64)
        self.x3 = Linear(256, 80)
        self.x3_ext = Linear(256 + 64, 100)
        self.relu = LeakyReLU()
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.dropout = Dropout(0.3)
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
            self.x2_ext.weight = Parameter(torch.cat((torch.cat((self.x2.weight, torch.zeros(256, 128).to(device)), dim=1), torch.zeros(64, 512+128).to(device)), dim=0))
            self.x2_ext.bias = Parameter(torch.cat((self.x2.bias, torch.zeros(64).to(device))))
            self.ssum_x2 = self.x2_ext
            self.x3_ext.weight = Parameter(torch.cat((torch.cat((self.x3.weight, torch.zeros(80, 64).to(device)), dim=1), torch.zeros(20, 256+64).to(device)), dim=0))
            self.x3_ext.bias = Parameter(torch.cat((self.x3.bias, torch.zeros(20).to(device))))
            self.ssum_x3 = self.x3_ext
        else:
            self.ssum_x1 = self.x1
            self.ssum_x2 = self.x2
            self.ssum_x3 = self.x3

loss_fn = CrossEntropyLoss()
def calc_loss_and_accuracy(model, output, labels):
    inputs, labels = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad()
    loss = loss_fn(output, labels)
    output = model(inputs)
    _, pred = torch.max(output, 1)
    return (loss, torch.sum((pred == labels).squeeze()), torch.numel(pred))



model = MyNet().to(device)


st_time = time.time()

model.mode = FIRST_TRAIN
model.setup_layer()
optimizer = Adam(model.parameters(), lr=1e-4)

train_loss_pa = []
train_acc_pa = []
train_t_pa = []
train_acc_t_pa = []
test_loss_pa = []
test_acc_pa = []
test_t_pa = []
test_acc_t_pa = []

for epoch in range(40):
    ss, cnt = 0, 0
    print(epoch)
    model.train()
    for batch in tqdm(first_train_loader, unit="batch", total=len(first_train_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss, acc, cor = calc_loss_and_accuracy(model, output, labels)
        loss.backward()
        optimizer.step()
        ss += acc.item()
        cnt += cor
        train_loss_pa.append(loss.item())
        train_t_pa.append(time.time()-st_time)
    train_acc_pa.append(ss/cnt*100)
    train_acc_t_pa.append(time.time()-st_time)
    ss, cnt, ls = 0, 0, 0.0
    model.eval()
    for batch in tqdm(first_test_loader, unit="batch", total=len(first_test_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        output = model(inputs)
        loss, acc, cor = calc_loss_and_accuracy(model, output, labels)
        ss += acc.item()
        cnt += cor
        ls += loss.item() * cor
    test_loss_pa.append(ls/cnt)
    test_t_pa.append(time.time()-st_time)
    test_acc_pa.append(ss/cnt*100)
    test_acc_t_pa.append(time.time()-st_time)
    torch.save(model.state_dict(), f"./results/first_train/params{epoch}.pt")


model.mode = WHOLE_TRAIN
model.setup_layer()
optimizer = Adam(model.parameters(), lr=1e-4)

for epoch in range(40, 70):
    ss, cnt = 0, 0
    print(epoch)
    model.train()
    for batch in tqdm(whole_train_loader, unit="batch", total=len(whole_train_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss, acc, cor = calc_loss_and_accuracy(model, output, labels)
        loss.backward()
        optimizer.step()
        ss += acc.item()
        cnt += cor
        train_loss_pa.append(loss.item())
        train_t_pa.append(time.time()-st_time)
    train_acc_pa.append(ss/cnt*100)
    train_acc_t_pa.append(time.time()-st_time)
    ss, cnt, ls = 0, 0, 0.0
    model.eval()
    for batch in tqdm(whole_test_loader, unit="batch", total=len(whole_test_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        output = model(inputs)
        loss, acc, cor = calc_loss_and_accuracy(model, output, labels)
        ss += acc.item()
        cnt += cor
        ls += loss.item() * cor
    test_loss_pa.append(ls/cnt)
    test_t_pa.append(time.time()-st_time)
    test_acc_pa.append(ss/cnt*100)
    test_acc_t_pa.append(time.time()-st_time)
    torch.save(model.state_dict(), f"./results/second_train/params{epoch}.pt")

plt.figure(1)
plt.plot(train_t_pa, train_loss_pa, label='train loss: PA')
plt.plot(test_t_pa, test_loss_pa, label='test loss: PA')
plt.figure(2)
plt.plot(train_acc_t_pa, train_acc_pa, label='train accuracy: PA')
plt.plot(test_acc_t_pa, test_acc_pa, label='test accuracy: PA')


model = MyNet().to(device)

loss_fn = CrossEntropyLoss()
st_time = time.time()

model.mode = WHOLE_TRAIN
model.setup_layer()
optimizer = Adam(model.parameters(), lr=1e-4)

train_loss_cnn = []
train_acc_cnn = []
train_t_cnn = []
train_acc_t_cnn = []
test_loss_cnn = []
test_acc_cnn = []
test_t_cnn = []
test_acc_t_cnn = []
for epoch in range(50):
    ss, cnt = 0, 0
    print(epoch)
    model.train()
    for batch in tqdm(whole_train_loader, unit="batch", total=len(whole_train_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss, acc, cor = calc_loss_and_accuracy(model, output, labels)
        loss.backward()
        optimizer.step()
        ss += acc.item()
        cnt += cor
        train_loss_cnn.append(loss.item())
        train_t_cnn.append(time.time()-st_time)
    train_acc_cnn.append(ss/cnt*100)
    train_acc_t_cnn.append(time.time()-st_time)
    ss, cnt, ls = 0, 0, 0.0
    model.eval()
    for batch in tqdm(whole_test_loader, unit="batch", total=len(whole_test_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        output = model(inputs)
        loss, acc, cor = calc_loss_and_accuracy(model, output, labels)
        ss += acc.item()
        cnt += cor
        ls += loss.item() * cor
    test_loss_cnn.append(ls/cnt)
    test_t_cnn.append(time.time()-st_time)
    test_acc_cnn.append(ss/cnt*100)
    test_acc_t_cnn.append(time.time()-st_time)
    torch.save(model.state_dict(), f"./results/whole_train/params{epoch}.pt")

plt.figure(1)
plt.plot(train_t_cnn, train_loss_cnn, label='train loss: CNN')
plt.plot(test_t_cnn, test_loss_cnn, label='test loss: CNN')
plt.figure(2)
plt.plot(train_acc_t_cnn, train_acc_cnn, label='train accuracy: CNN')
plt.plot(test_acc_t_cnn, test_acc_cnn, label='test accuracy: CNN')

plt.figure(1)
plt.xlabel('Time (s)')
plt.ylabel('Loss')
plt.title('Training Loss / Test Loss')
plt.legend()
plt.figure(2)
plt.xlabel('Time (s)')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy / Test Accuracy')
plt.legend()
plt.show()