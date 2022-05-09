import torchvision
from torch import nn
import torch
from torch.utils.data import DataLoader
from model import SwinTransformer

train_dataloader = DataLoader(torchvision.datasets.MNIST('./data/mnist/', train=True, download=True,
                                                         transform=torchvision.transforms.Compose([
                                                             torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize(
                                                                 (0.1307,), (0.3081,))
                                                         ])), shuffle=True, batch_size=64)

test_dataloader = DataLoader(torchvision.datasets.MNIST('./data/mnist/', train=False, download=True,
                                                        transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize(
                                                                (0.1307,), (0.3081,))
                                                        ])), shuffle=True, batch_size=64)
# [64, 1, 28, 28] => 48, 14, 14 => 96, 7, 7 =>
net = nn.Sequential(
    SwinTransformer(img_size=28, patch_size=2, n_channels=1,
                    embed_dim=48, window_size=7, mlp_ratio=4,
                    qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                    patch_norm=True, n_swin_blocks=(2, 2), n_attn_heads=(2, 4)),
    nn.Linear(96, 10)
)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 10001):
    net.train()
    sum_loss = 0.
    for d, l in train_dataloader:
        pl = net(d)
        loss = criterion(pl, l)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    print(f'[epoch {epoch}, train loss: {sum_loss}')

    if epoch % 100 == 0:
        net.eval()
        with torch.no_grad():
            acc_num = 0
            sum_num = 0
            for d, l in train_dataloader:
                pl = net(d)
                acc_num += torch.sum(torch.argmax(pl, dim=-1) == l)
                sum_num += len(l)
            print(f'acc: {acc_num / sum_num :.4f}')

