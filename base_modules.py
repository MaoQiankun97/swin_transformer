import torch
from torch import nn
from timm.models.layers import trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """
    将输入图片进行分区，将每4*4个区域视为一个patch, 将其元素投射到channel维度上
    使用 conv 操作同时进行分区和线性映射
    该操作对应论文架构中的PatchPartition和stage1的LinearEmbedding
    """

    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, patch_norm=True):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        if patch_norm:
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)

        return x


class PatchMerging(nn.Module):
    """
    首先将相邻的2*2个patch在channel上合并
    然后将得到的channel维度 4C 映射为 2C
    input shape: [batch_size, n_channels, h, w]
    output shape: [batch_size, 2*n_channels, h/2, w/2]
    """

    def __init__(self, in_channels):
        super(PatchMerging, self).__init__()
        self.norm = nn.LayerNorm(4 * in_channels)
        # 使用1*1 conv来进行reduction
        self.reduction = nn.Conv2d(in_channels=4 * in_channels, out_channels=2 * in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # b, c, h, w = x.size()  # 分别为batch_size, n_channels, height_size, width_size
        x_0 = x[:, :, 0::2, 0::2]  # 宽高从0开始每两个元素取一个，等价于取每一个2*2块的左上角
        x_1 = x[:, :, 1::2, 0::2]  # 高从1开始每两个元素取一个，等价于取每一个2*2块的左下角
        x_2 = x[:, :, 0::2, 1::2]  # 宽从1开始每两个元素取一个，等价于取每一个2*2块的右上角
        x_3 = x[:, :, 1::2, 1::2]  # 宽高从1开始每两个元素取一个，等价于取每一个2*2块的右下角
        x = torch.cat([x_0, x_1, x_2, x_3], dim=1)  # 在channel维度上将所有的块cat起来
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.reduction(x)

        return x


def window_partition(x, window_size):
    """
    [batch_size, n_c, n_h, n_w] => [n_windows * batch_size, n_c, window_size, window_size]
    """
    b, c, h, w = x.shape
    x = x.view(b, c, h // window_size, window_size, w // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, c, window_size, window_size)

    return windows


def window_reverse(windows, window_size, img_size):
    """
    [n_windows * batch_size, n_c, window_size, window_size] => [batch_size, n_c, n_h, n_w]
    """
    b_, n_c, _, _ = windows.shape
    h_windows = img_size // window_size
    w_windows = h_windows
    n_windows = h_windows * w_windows
    windows = windows.reshape(b_ // n_windows, h_windows, w_windows, n_c, window_size, window_size).permute(0, 3, 4, 1, 5, 2)
    x = windows.reshape(b_ // n_windows, n_c, h_windows * window_size, w_windows * window_size)

    return x


def get_relative_position_index(window_size):
    # get pair-wise relative position index for each token inside the window
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

    return relative_position_index


def get_mask(img_size, window_size, shift_size):
    """
    根据total_size, window_size和shift_size获取mask矩阵
    """
    img_mask = torch.zeros((1, 1, img_size, img_size))  # 1 H W 1
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0  # cnt代表当前窗口的序号，即把在原图中属不同窗口的各个区域通过序号给区分开来
    for h in h_slices:
        for w in w_slices:
            img_mask[:, :, h, w] = cnt
            cnt += 1
    # img_mask[0, 0, h, w] = [
    # tensor([[0., 0., 0., 0., 1., 1., 2., 2.],
    #         [0., 0., 0., 0., 1., 1., 2., 2.],
    #         [0., 0., 0., 0., 1., 1., 2., 2.],
    #         [0., 0., 0., 0., 1., 1., 2., 2.],
    #         [3., 3., 3., 3., 4., 4., 5., 5.],
    #         [3., 3., 3., 3., 4., 4., 5., 5.],
    #         [6., 6., 6., 6., 7., 7., 8., 8.],
    #         [6., 6., 6., 6., 7., 7., 8., 8.]])

    mask_windows = window_partition(img_mask, window_size)  # 把整个img_mask划分为各个窗口, nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)  # 把每个窗口的值flatten成向量 nW, window_size*window_size
    # mask_windows[0, :] tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    # mask_windows[1, :] tensor([1., 1., 2., 2., 1., 1., 2., 2., 1., 1., 2., 2., 1., 1., 2., 2.])
    # mask_windows[2, :] tensor([3., 3., 3., 3., 3., 3., 3., 3., 6., 6., 6., 6., 6., 6., 6., 6.])
    # mask_windows[3, :] tensor([4., 4., 5., 5., 4., 4., 5., 5., 7., 7., 8., 8., 7., 7., 8., 8.])
    # shape: [n_window, window_size**2, window_size**2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # 计算每个窗口的mask矩阵, 来自不同窗口的元素的值相减以后非0
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))  # 把非0元素改为-100.

    return attn_mask


class WindowAttention(nn.Module):
    """
    在一个window内做attention
    首先根据给定的window_size获取相对位置编码矩阵
    然后将给定的window所有元素flatten之后做multi head attention
    input shape: [num_windows*batch_size,n_channels, window_size, window_size]
    output shape: [num_windows*batch_size,n_channels, window_size, window_size]
    """

    def __init__(self, window_size, n_heads, n_channels, qkv_bias=True, attention_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        self.scale = (n_channels // n_heads) ** -0.5

        # 为每个attention head初始化相位位置编码表, 每个head的表大小为 (2*window_size-1) * (2*window_size-1)
        # 由于相位位置编码index计算过程中，最大的index为2*(window_size-1)*(2*window_size-1)+2*(window_size-1)
        # = 2*(window_size-1)*2*window_size = (2 * window_size -1) * (2 * window_size - 1) - 1
        # 然后只要求相对位置编码index矩阵，然后根据index从对应head的相对位置编码表中取相对位置编码即可
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), n_heads)
        )
        # 使用正态分布值初始化相对位置编码
        trunc_normal_(self.relative_position_bias_table, std=.02)
        # 获取相对位置编码index矩阵 [window_size**2, window_size**2]
        relative_position_index = get_relative_position_index(window_size)
        self.register_buffer('relative_position_index', relative_position_index)
        # 同时求Q、K、V, 在channel维度上做attention, 对应文章中的
        # all query patches within a window share the same key set
        self.qkv = nn.Linear(n_channels, 3 * n_channels)
        self.attention_drop = nn.Dropout(attention_drop)
        self.proj = nn.Linear(n_channels, n_channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b_, n_c, window_size, _ = x.size()
        # 首先进行维度转换
        # [n_w*batch_sz, n_c, window_sz, window_sz] => [n_w*batch_sz, window_sz, window_sz, n_c]
        x = x.permute(0, 2, 3, 1)
        # 求q k v
        qkv = self.qkv(x)  # [n_w*batch_sz, window_sz, window_sz, 3 * n_c]
        qkv = qkv.reshape(b_, window_size * window_size, n_c // self.n_heads, self.n_heads, 3)
        q = qkv[:, :, :, :, 0].permute(0, 3, 1, 2)  # [b_, n_heads, window_sz * window_sz, n_c // n_heads]
        k = qkv[:, :, :, :, 1].permute(0, 3, 1, 2)  # [b_, n_heads, window_sz * window_sz, n_c // n_heads]
        v = qkv[:, :, :, :, 2].permute(0, 3, 1, 2)  # [b_, n_heads, window_sz * window_sz, n_c // n_heads]
        q *= self.scale  # 对q进行缩放
        att_matrix = q @ k.transpose(-2, -1)  # [b_, n_heads, window_sz * window_sz, window_sz * window_sz]
        # 根据relative_position_index取出relative_position_encode
        # [window_size**2 * window_size**2, n_heads]
        relative_position_encode = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_encode = relative_position_encode.view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_encode = relative_position_encode.permute(2, 0, 1)  # [n_heads, window_size**2, window_size**2]
        # 把相对位置编码加到attention矩阵中
        att_matrix = att_matrix + relative_position_encode.unsqueeze(0)
        # 对attention矩阵进行mask
        if mask is not None:
            # mask shape: [n_window, window_size**2, window_size**2] => [n_window, 1, window_size**2, window_size**2]
            n_w = mask.size()[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            att_matrix = att_matrix.view(b_ // n_w, n_w, self.n_heads, window_size * window_size, window_size * window_size)
            att_matrix = att_matrix + mask
            att_matrix = att_matrix.view(-1, self.n_heads, window_size * window_size, window_size * window_size)
        att_matrix = self.softmax(att_matrix)
        x = att_matrix @ v  # [b_, n_heads, window_sz * window_sz, n_c // n_heads]
        x = x.transpose(1, 2).reshape(b_, window_size, window_size, n_c)  # [b_, window_sz, window_sz, n_c]
        x = self.proj(x)  # [b_, window_sz, window_sz, n_c]
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)  # [b_, n_c, window_sz, window_sz]

        return x


class SwinTransformerBlock(nn.Module):
    """
    swin transformer块，支持shift和非shift
    input shape: [batch_size, n_c, n_h, n_w]
    """

    def __init__(self, n_channels, n_heads, img_size,
                 window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0.):
        super(SwinTransformerBlock, self).__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size

        self.norm_0 = nn.LayerNorm(n_channels)
        self.attn = WindowAttention(window_size, n_heads, n_channels, qkv_bias, attn_drop, drop)
        self.norm_1 = nn.LayerNorm(n_channels)
        self.mlp = Mlp(in_features=n_channels, hidden_features=int(mlp_ratio * n_channels))

        # 计算mask矩阵
        if self.shift_size > 0:
            # shape: [n_window, window_size**2, window_size**2]
            attn_mask = get_mask(img_size, window_size, shift_size)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        short_cut = x
        x = x.permute(0, 2, 3, 1)  # [batch_size, n_h, n_w, n_c]
        x = self.norm_0(x)
        x = x.permute(0, 3, 1, 2)  # [batch_size, n_c, n_h, n_w]
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        # [batch_size, n_c, n_h, n_w] =>  [n_windows * batch_size, n_c, window_size, window_size]
        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows, self.attn_mask)
        x = window_reverse(attn_windows, self.window_size, self.img_size)  # [batch_size, n_c, n_h, n_w]
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        x = short_cut + x  # 残差
        x = x.permute(0, 2, 3, 1)  # [batch_size, n_h, n_w, n_c]
        x = x + self.mlp(self.norm_1(x))  # 第二个残差
        x = x.permute(0, 3, 1, 2)  # [batch_size, n_c, n_h, n_w]

        return x


class SwinTransformerBlockStack(nn.Module):
    """
    将一个非shift的SwinTransformerBlock和一个shift的SwinTransformerBlock堆叠在一起
    """

    def __init__(self, n_channels, n_heads, img_size,
                 window_size=7, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0.):
        super(SwinTransformerBlockStack, self).__init__()
        self.no_shift_block = SwinTransformerBlock(
            n_channels, n_heads, img_size, window_size, 0, mlp_ratio, qkv_bias, drop, attn_drop)
        self.shift_block = SwinTransformerBlock(
            n_channels, n_heads, img_size, window_size, window_size // 2, mlp_ratio, qkv_bias, drop, attn_drop)

    def forward(self, x):
        x = self.no_shift_block(x)
        x = self.shift_block(x)

        return x
