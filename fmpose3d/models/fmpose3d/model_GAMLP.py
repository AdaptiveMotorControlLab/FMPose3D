"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import torch
import torch.nn as nn
import math
from einops import rearrange
from fmpose3d.models.fmpose3d.graph_frames import Graph
from fmpose3d.models.base_model import BaseModel, register_model
from functools import partial
from einops import rearrange
from timm.layers import DropPath

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        assert self.dim % 2 == 0, "TimeEmbedding.dim must be even"
        self.gaussian_std = 1.0
        self.register_buffer('B', torch.randn(self.dim // 2) * self.gaussian_std, persistent=True)
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        b, f = t.shape[0], t.shape[1]
        half_dim = self.dim // 2
     
        angles = (2 * math.pi) * t.to(torch.float32).unsqueeze(-1) * self.B.view(1, 1, 1, 1, half_dim)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        emb = torch.cat([sin, cos], dim=-1).reshape(b, f, self.dim).to(t.dtype)  # (B,F,dim)
        emb = self.proj(emb)  # (B,F,dim)
        return emb

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

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()

        # Register adj as buffer to follow module device automatically
        self.register_buffer('adj', adj)
        self.kernel_size = adj.size(0)
        self.conv1d = nn.Conv1d(in_channels, out_channels * self.kernel_size, kernel_size=1)
    def forward(self, x): # b, j, c
        # conv1d
        x = rearrange(x,"b j c -> b c j") 
        x = self.conv1d(x)   # b,c*kernel_size,j = b,c*4,j
        x = rearrange(x,"b ck j -> b ck 1 j")
        b, kc, t, v = x.size()
        x = x.view(b, self.kernel_size, kc//self.kernel_size, t, v) # b,k, kc/k, 1, j 
        x = torch.einsum('bkctv, kvw->bctw', (x, self.adj))   # bctw    b,c,1,j 
        x = x.contiguous()
        x = rearrange(x, 'b c 1 j -> b j c')
        return x.contiguous()

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) 
        self.attn_drop = nn.Dropout(attn_drop) 
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) 

    def forward(self, x):
        B, N, C = x.shape # b,j,c
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2]  

        attn = (q @ k.transpose(-2, -1)) * self.scale # b,heads,J,4 @ b,heads,4,J = b,heads,J,J
        attn = attn.softmax(dim=-1) 
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C).contiguous() 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module): # drop=0.1
    def __init__(self, length, dim, tokens_dim, channels_dim, adj, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # GCN
        self.norm1 = norm_layer(length)
        self.gcn1 = GCN(dim, dim, adj)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # attention
        self.norm_att1=norm_layer(dim)
        self.num_heads = 8
        qkv_bias =  True
        qk_scale = None
        attn_drop = 0.2
        proj_drop = 0.25
        self.attn = Attention(
            dim, num_heads=self.num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop) 
        
        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=1024, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # B,J,dim 
        res1 = x # b,j,c
        # gcn
        x_gcn_1 = rearrange(x,"b j c -> b c j").contiguous() 
        x_gcn_1 = self.norm1(x_gcn_1) # b,c,j
        x_gcn_1 = rearrange(x_gcn_1,"b c j -> b j c").contiguous()
        x_gcn_1 = self.gcn1(x_gcn_1)  # b,j,c
        # attention
        x_atten = self.norm_att1(x)
        x_atten = self.attn(x_atten)
        
        x = res1 + self.drop_path(x_gcn_1 + x_atten)
        
        # MLP residual
        res2 = x  # b,j,c
        x2 = self.norm_mlp(x)
        x = self.mlp(x2)
        x = res2 + self.drop_path(x)
        return x

class FMPose3D(nn.Module):
    def __init__(self, depth, embed_dim, channels_dim, tokens_dim, adj, drop_rate=0.10, length=27):
        super().__init__()
        
        drop_path_rate = 0.2
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                length, embed_dim, tokens_dim, channels_dim, adj,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm_mu = norm_layer(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_mu(x)
        return x

class encoder(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class decoder(nn.Module):
    """Output decoder: predicts velocity field for Flow Matching (no dropout needed)"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        dim_in = in_features
        dim_hid = hidden_features
        dim_out = out_features
        
        self.fc1 = nn.Linear(dim_in, dim_hid)
        self.act = act_layer()
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.fc2= nn.Linear(dim_hid, dim_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

@register_model("fmpose3d_humans")
class Model(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        ## GCN
        self.graph = Graph('hm36_gt', 'spatial', pad=1)
        # Register as buffer (not parameter) to follow module device automatically
        self.register_buffer('A', torch.tensor(self.graph.A, dtype=torch.float32))

        self.t_embed_dim = 32
        self.time_embed = TimeEmbedding(self.t_embed_dim, hidden_dim=64)
        
        self.encoder = encoder(2 + 3 + self.t_embed_dim, args.channel//2, args.channel)
        
        self.FMPose3D = FMPose3D(args.layers, args.channel, args.d_hid, args.token_dim, self.A, length=args.n_joints) # 256
        self.pred_mu = decoder(args.channel, args.channel//2, 3)
        
    def forward(self, pose_2d, y_t, t):
        # pose_2d: (B,F,J,2)  y_t: (B,F,J,3)  t: (B,F,1,1)
        b, f, j, _ = pose_2d.shape
        
        # build time embedding
        t_emb = self.time_embed(t) # (B,F,t_dim)
        t_emb = t_emb.unsqueeze(2).expand(b, f, j, self.t_embed_dim).contiguous()  # (B,F,J,t_dim)

        x_in = torch.cat([pose_2d, y_t, t_emb], dim=-1)           # (B,F,J,2+3+t_dim)
        x_in = rearrange(x_in, 'b f j c -> (b f) j c').contiguous() # (B*F,J,in)
        
        in_emb = self.encoder(x_in)
        features = self.FMPose3D(in_emb)
        v = self.pred_mu(features)                                  # (B*F,J,3)
        v = rearrange(v, '(b f) j c -> b f j c', b=b, f=f).contiguous() # (B,F,J,3)
        return v
    
if __name__ == "__main__":
    class Args:
        pass
    args = Args()
    args.channel = 512
    args.d_hid = 1024
    args.token_dim = 256
    args.layers = 5
    args.n_joints = 17
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(args).to(device)
    # Print model architecture and parameter counts
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")
    x = torch.randn(1, 17, 17, 2, device=device)
    y_t = torch.randn(1, 17, 17, 3, device=device)
    t = torch.randn(1, 1, 1, 1, device=device)
    v = model(x, y_t, t)
    print(v.shape) 

