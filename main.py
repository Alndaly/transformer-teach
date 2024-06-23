import torch
from rich import print
from torch.utils.data import Dataset, DataLoader
from day04.main import MyData, Transformer

if __name__ == '__main__':
    model = Transformer()
    loss_fn = torch.nn.CrossEntropyLoss()
    train_dataloader = DataLoader(MyData(), batch_size=12, shuffle=True)
    data = next(iter(train_dataloader))
    for batch in train_dataloader:
        # [batch_size, max_seq_len]
        # TODO 将所有的训练过程中的模型层增加第一个batch维度
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = loss_fn(outputs, batch.label)
        loss.backward()