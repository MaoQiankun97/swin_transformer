import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn
from base_modules import *


class SwinTransformer(nn.Module):
    def __init__(self, img_size, patch_size=4, n_channels=3,
                 embed_dim=96, window_size=7, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 patch_norm=True, n_swin_blocks=(2, 2, 6, 2), n_attn_heads=(3, 6, 12, 24)):
        super(SwinTransformer, self).__init__()
        self.n_layers = len(n_swin_blocks)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.n_layers - 1))  # 输出的n_channels, 除第一层外, 每层网络将n_channels翻一倍
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(patch_size, n_channels, embed_dim, patch_norm)
        self.pos_drop = nn.Dropout(p=drop_rate)

        img_size //= patch_size
        n_channels = embed_dim
        self.layers = nn.ModuleList()
        for i_layer in range(self.n_layers):
            cur_layer = nn.ModuleList()
            if i_layer > 0:
                cur_layer.append(PatchMerging(in_channels=n_channels))
                img_size //= 2
                n_channels *= 2
            for _ in range(n_swin_blocks[i_layer] // 2):
                cur_layer.append(
                    SwinTransformerBlockStack(
                        n_channels=n_channels,
                        n_heads=n_attn_heads[i_layer],
                        img_size=img_size,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate
                    )
                )
            cur_layer = nn.Sequential(*cur_layer)
            self.layers.append(cur_layer)

        self.norm = nn.LayerNorm(self.num_features)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        参照源码进行初始化
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)  # [_, 3, h, w] => [_, 96, h//patch_size, w//patch_size]
        x = self.pos_drop(x)

        #   [_, embed_dim, h//patch_size, w//patch_size]
        # =>[_, embed_dim*2, h//(patch_size*2), w//(patch_size*2)]
        # =>[_, embed_dim*4, h//(patch_size*4), w//(patch_size*4)]
        # =>[_, embed_dim*8, h//(patch_size*8), w//(patch_size*8)]
        for layer in self.layers:
            x = layer(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.avg_pool(x)  # [_, n_c, 1]
        x = torch.flatten(x, 1)  # [_, n_c]

        return x


if __name__ == '__main__':
    net = SwinTransformer(img_size=224)
    fake_pics = torch.randn((2, 3, 224, 224))
    ops = net(fake_pics)
    print(ops.size())
