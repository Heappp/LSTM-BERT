import pandas as pd
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


class MyDataSet(Dataset):
    def __init__(self):
        df1 = pd.read_json('archive/Sarcasm_Headlines_Dataset.json', lines=True)
        df2 = pd.read_json('archive/Sarcasm_Headlines_Dataset_v2.json', lines=True)
        df = pd.concat([df1, df2], axis=0)

        self.data = df.headline.values
        self.label = df.is_sarcastic.values
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __getitem__(self, idx):
        text = self.data[idx]
        text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=200, padding='max_length', return_token_type_ids=False)

        input_ids = torch.tensor(text['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(text['attention_mask'], dtype=torch.long)
        label = torch.tensor(self.label[idx], dtype=torch.long)

        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.data)


class BERT(torch.nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, 2)

    def forward(self, text, mask):
        _, x = self.bert(text, attention_mask=mask, return_dict=False)
        x = self.dropout(x)
        x = self.linear(x)
        return x


def Train():
    data = MyDataSet()
    train_data, test_data = random_split(data, [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)])

    net = BERT().to(device)
    # net.load_state_dict(torch.load('save\BERT9.pt', map_location='cpu'))
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    my_writer = SummaryWriter("tf-logs")

    epoch = 10
    batch_size = 64

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)

    for step in range(epoch):
        # 训练
        net.train()
        train_loss, train_acc = 0, 0
        for input_ids, attention_mask, label in train_dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            y = net.forward(input_ids, attention_mask)
            loss = loss_function(y, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += torch.sum(torch.eq(torch.max(y, dim=1)[1], label)).item()
        scheduler.step()

        # 测试T
        net.eval()
        test_loss, test_acc = 0, 0
        for input_ids, attention_mask, label in test_dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)

            y = net.forward(input_ids, attention_mask)
            loss = loss_function(y, label)

            test_loss += loss.item()
            test_acc += torch.sum(torch.eq(torch.max(y, dim=1)[1], label)).item()

        # 统计
        my_writer.add_scalars("Loss", {"train": train_loss / len(train_dataloader), "test": test_loss / len(test_dataloader)}, step)
        my_writer.add_scalars("Acc", {"train": train_acc / len(train_data), "test": test_acc / len(test_data)}, step)
        torch.save(net.state_dict(), "save/BERT" + str(step) + ".pt")


Train()
