import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

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
        text = self.tokenizer.encode(text)

        input_ids = torch.tensor(text, dtype=torch.long)
        label = torch.tensor(self.label[idx], dtype=torch.long)

        return input_ids, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch_list):
        batch_list.sort(key=lambda x: len(x[0]), reverse=True)
        length = [len(item[0]) for item in batch_list]

        input_ids = pad_sequence([item[0] for item in batch_list], batch_first=True, padding_value=0)
        label = torch.tensor([item[1] for item in batch_list])
        length = torch.tensor(length)

        return input_ids, label, length


class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=30522, embedding_dim=768)
        self.lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.linear = nn.Linear(768 * 2, 2)

    def forward(self, text, length):
        text = self.embedding(text)
        pack = pack_padded_sequence(text, length, batch_first=True, enforce_sorted=True)
        _, (h, _) = self.lstm(pack)
        x = torch.concat([h[-1, :, :], h[-2, :, :]], dim=1)
        x = self.linear(x)
        return x


def Train():
    data = MyDataSet()
    train_data, test_data = random_split(data, [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)])

    net = LSTM().to(device)
    # net.load_state_dict(torch.load('save\lstm.pt', map_location='cpu'))
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    my_writer = SummaryWriter("tf-logs")

    epoch = 10
    batch_size = 64

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=MyDataSet.collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=MyDataSet.collate_fn)

    for step in range(epoch):
        # 训练
        net.train()
        train_loss, train_acc = 0, 0
        for input_ids, label, length, in train_dataloader:
            input_ids = input_ids.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            y = net.forward(input_ids, length)
            loss = loss_function(y, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += torch.sum(torch.eq(torch.max(y, dim=1)[1], label)).item()
        scheduler.step()

        # 测试
        net.eval()
        test_loss, test_acc = 0, 0
        for input_ids, label, length in test_dataloader:
            input_ids = input_ids.to(device)
            label = label.to(device)

            y = net.forward(input_ids, length)
            loss = loss_function(y, label)

            test_loss += loss.item()
            test_acc += torch.sum(torch.eq(torch.max(y, dim=1)[1], label)).item()

        # 统计
        my_writer.add_scalars("Loss", {"train": train_loss / len(train_dataloader), "test": test_loss / len(test_dataloader)}, step)
        my_writer.add_scalars("Acc", {"train": train_acc / len(train_data), "test": test_acc / len(test_data)}, step)
        torch.save(net.state_dict(), "save/LSTM" + str(step) + ".pt")


Train()
