import sys
sys.path.append("..")
import torch
import torch.nn as nn
import math
from einops import rearrange
from model.graph_frames import Graph
from functools import partial
from einops import rearrange, repeat
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()

        self.adj = adj
        self.kernel_size = adj.size(0)
        #
        self.conv1d = nn.Conv1d(in_channels, out_channels * self.kernel_size, kernel_size=1)
    def forward(self, x):
        # x.shape: b,c,1,j
        # conv1d
        x = self.conv1d(x)   # b,c*kernel_size,j=b,c*4,j
        x = rearrange(x,"b ck j -> b ck 1 j")
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v) # b,k, kc/k, 1, j 
        x = torch.einsum('nkctv, kvw->nctw', (x, self.adj)) # 
        # x.shape b,c,1,j   [128,512,17,1]
        return x.contiguous()

class linear_block(nn.Module):
    def __init__(self, ch_in, ch_out, drop=0.1):
        super(linear_block,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(ch_in, ch_out),
            nn.GELU(),
            nn.Dropout(drop)
        )
    def forward(self,x):
        x = self.linear(x)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,F,1,1) in [0,1)
        b, f, _, _ = t.shape
        half_dim = max(self.dim // 2, 1)
        inv_freq = torch.exp(-math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / max(half_dim - 1, 1))
        angles = t * inv_freq.view(1, 1, 1, 1, half_dim)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        emb = torch.cat([sin, cos], dim=-1)  # (B,F,1,1,dim)
        emb = emb.view(b, f, self.dim)
        emb = self.proj(emb)  # (B,F,dim)
        return emb

class uMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.): # zheli
        super().__init__()

        self.linear_down = linear_block(in_features,hidden_features,drop)
        self.linear_mid = linear_block(hidden_features,hidden_features,drop) 
        self.linear_up = linear_block(hidden_features,in_features,drop)

    def forward(self, x):
        # res_512 = x
        # down  
        x = self.linear_down(x)
        res_mid = x 
        # mid
        x = self.linear_mid(x)
        x = x + res_mid
        # up
        x = self.linear_up(x) 
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features) # 32 64
        self.act = act_layer() # GELU
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x) # 64,32
        x = self.drop(x)
        return x


class GCN_V2(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()

        self.adj = adj # 4,17,17
        self.kernel_size = adj.size(0)
        #
        self.conv1d = nn.Conv1d(in_channels, out_channels * self.kernel_size, kernel_size=1)
    def forward(self, x): # b, j, c
        # conv1d
        x = rearrange(x,"b j c -> b c j") 
        x = self.conv1d(x)   # b,c*kernel_size,j = b,c*4,j
        x = rearrange(x,"b ck j -> b ck 1 j")
        b, kc, t, v = x.size()
        x = x.view(b, self.kernel_size, kc//self.kernel_size, t, v) # b,k, kc/k, 1, j 
        x = torch.einsum('bkctv, kvw->bctw', (x, self.adj))   # bctw    b,c,1,j 
        # x.shape b,c,1,j   [128,512,17,1]
        x = x.contiguous()
        x = rearrange(x, 'b c 1 j -> b j c')
        # 激活函数  x.contiguous() + relu 
        return x.contiguous()

class spatial_uMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.): # zheli
        super().__init__()

        self.linear_down = linear_block(in_features, hidden_features, drop)
        self.linear_mid = linear_block(hidden_features, hidden_features, drop) 
        self.linear_up = linear_block(hidden_features, in_features, drop)
    def forward(self, x):
        # down   
        x_down = self.linear_down(x) # 17*4
        # mid
        x_mid = self.linear_mid(x_down) + x_down # 17*2
        # up1
        x = self.linear_up(x_mid) 
        return x

class spatial_uMLP_Multi(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.): # zheli
        super().__init__()

        self.linear_up1 = linear_block(in_features, hidden_features//2, drop)
        self.linear_up2 = linear_block(hidden_features//2, hidden_features, drop)
        self.linear_mid = linear_block(hidden_features, hidden_features, drop) 
        self.linear_down1 = linear_block(hidden_features, hidden_features//2, drop)
        self.linear_down2 = linear_block(hidden_features//2, out_features, drop) 
        
    def forward(self, x):
        # up1
        x = self.linear_up1(x) 
        # up2
        x_up2 =  self.linear_up2(x)
        # mid
        x_mid = self.linear_mid(x_up2) + x_up2 # 17*N
        x_down1 = self.linear_down1(x_mid) # 17*N//2
        # down   
        x_down2 = self.linear_down2(x_down1) # 17    
        return x_down2
    
class channel_uMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.): # zheli
        super().__init__()

        self.linear_down = linear_block(in_features,hidden_features,drop)
        self.linear_mid = linear_block(hidden_features,hidden_features,drop) 
        self.linear_up = linear_block(hidden_features,in_features,drop)

    def forward(self, x):
        # res_512 = x
        # down  
        x = self.linear_down(x)
        res_mid = x 
        # mid
        x = self.linear_mid(x)
        x = x + res_mid
        # up
        x = self.linear_up(x) 
        return x
    

class spatial_channel_uMLP(nn.Module):
    def __init__(self, dim = 512, length=17, act_layer=nn.GELU,norm_layer = nn.LayerNorm, drop=0.): # zheli
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.channel_umlp = channel_uMLP(in_features=dim, hidden_features=128, act_layer=act_layer, drop=0.2)

        self.norm2 = norm_layer(length)
        self.spatial_umlp = spatial_uMLP(in_features=17, hidden_features=17*4, act_layer=act_layer, drop=0.15)

    def forward(self, x):
        # uMLP channel + spatial 
        res = x  # b,j,c
        x2 = self.norm1(x)
        x_s = rearrange(x, "b j c -> b c j").contiguous() 
        x_s = self.norm2(x_s)
        x_s = self.spatial_umlp(x_s)
        x_s = rearrange(x_s, "b c j -> b j c").contiguous() 
        x =  res + self.channel_umlp(x2)+ x_s  
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 
        self.attn_drop = nn.Dropout(attn_drop) # p=0
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)  # 0

    def forward(self, x):
        B, N, C = x.shape # b,j,c
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2]  

        attn = (q @ k.transpose(-2, -1)) * self.scale # b,heads,17,4 @ b,heads,4,17 = b,heads,17,17
        attn = attn.softmax(dim=-1) 
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C).contiguous() 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, cond_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., qk_norm=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.cond_dim = cond_dim if cond_dim is not None else dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(self.cond_dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

    def forward(self, x, cond):
        # x: (B, Nq, C=dim), cond: (B, Nk, Cc)
        B, J, C = x.shape
        Bc, Jc, Cc = cond.shape
        assert B == Bc, "Batch size of x and cond must match"

        q = self.q_proj(x).reshape(B, J, self.num_heads, C // self.num_heads)
        kv = self.kv_proj(cond).reshape(B, Jc, self.num_heads, (C // self.num_heads) * 2)

        k, v = torch.split(kv, C // self.num_heads, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rearrange(q, 'b J h c -> b h J c')
        k = rearrange(k, 'b Jc h c -> b h Jc c')
        v = rearrange(v, 'b Jc h c -> b h Jc c')

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, J, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module): # drop=0.1
    def __init__(self, length, dim, tokens_dim, channels_dim, adj, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # length =17, dim = args.channel = 512, tokens_dim = args.token_dim=256, channels_dim = args.d_hid = 1024
        super().__init__()
        # GCN
        self.norm1 = norm_layer(length)
        self.gcn1 = GCN_V2(dim, dim, adj)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # cross attention (Q from hidden, K/V from condition)
        self.norm_att1=norm_layer(dim)
        self.num_heads = 8
        qkv_bias =  True
        qk_scale = None
        attn_drop = 0.2
        proj_drop = 0.25
        self.attn = CrossAttention(
            dim, cond_dim=dim, num_heads=self.num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, qk_norm=False, norm_layer=norm_layer)
        
        # channel MLP
        self.norm2 = norm_layer(dim)
        # spatial MLP
        
        self.norm3 = norm_layer(length)
        # self.spatial_mlp = spatial_uMLP_Multi(in_features=17, hidden_features=17*6, out_features=17, act_layer=act_layer, drop=0.05) # 0.15
        self.mlp = Mlp(in_features=dim, hidden_features=1024, act_layer=act_layer, drop=drop)

    def forward(self, x, cond):
        # B,J,dim 
        res1 = x # b,j,c
        # GCN
        x_gcn_1 = rearrange(x, "b j c -> b c j").contiguous() 
        x_gcn_1 = self.norm1(x_gcn_1) # b,c,j
        x_gcn_1 = rearrange(x_gcn_1,"b j c -> b c j").contiguous()
        x_gcn_1 = self.gcn1(x_gcn_1)  # b,j,c
        x = res1 + self.drop_path(x_gcn_1)

        ## attention channel 
        x_atten = x
        x_atten = self.norm_att1(x_atten)
        x_atten = self.attn(x_atten, cond)
        x = x + self.drop_path(x_atten)

        # MLP
        res3 = x.clone()  # b,j,c
        x2 = self.norm2(x)
        x = self.mlp(x2)
        x =  res3 + self.drop_path(x) 
        
        return x


class FMPose(nn.Module):
    def __init__(self, depth, embed_dim, channels_dim, tokens_dim, adj, drop_rate=0.10, length=27):
        super().__init__()
        # depth = args.layers=3, embed_dim=args.channel=512, channels_dim=args.d_hid=1024, tokens_dim=args.token_dim=256(set by myself)
        
        drop_path_rate = 0.2
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # depth_part = 2
        self.blocks = nn.ModuleList([
            Block(
                length, embed_dim, tokens_dim, channels_dim, adj,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm_mu = norm_layer(embed_dim)

    def forward(self, x, cond):
        for blk in self.blocks:
            x = blk(x, cond)
        mu = x
        mu = self.norm_mu(mu)
        return mu

class encoder(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x

class decoder(nn.Module): # 2,256,512
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        dim_in = in_features
        dim_hid = hidden_features
        dim_out = out_features
        
        self.fc1 = nn.Linear(dim_in, dim_hid)
        self.fc5 = nn.Linear(dim_hid, dim_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc5(x)
        return x

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        ## GCN
        self.graph = Graph('hm36_gt', 'spatial', pad=1)
        # follow module device; store as buffer to avoid hard binding to cuda(0)
        self.A = nn.Parameter(torch.tensor(self.graph.A, dtype=torch.float32), requires_grad=False).cuda(0)
        # self.register_buffer('A', torch.tensor(self.graph.A, dtype=torch.float32), persistent=False)

        # time embedding for t (CTM style conditioning)
        self.t_embed_dim = 32
        self.time_embed = TimeEmbedding(self.t_embed_dim, hidden_dim=64)

        # encoder maps concatenated [x2d(2), y_t(3), t_emb(self.t_embed_dim)] to args.channel
        self.encoder = encoder(2 + 3 + self.t_embed_dim, args.channel//2, args.channel)
        # project raw x2d to embedding dim for cross-attention condition
        self.cond_proj = nn.Linear(2, args.channel)
        #  
        self.FMPose = FMPose(args.layers, args.channel, args.d_hid, args.token_dim, self.A, length=args.n_joints) # 256
        self.pred_mu = decoder(args.channel, args.channel//2, 3)

    def forward(self, x, y_t, t):
        # x: (B,F,J,2)  y_t: (B,F,J,3)  t: (B,1,1,1) or (B,F,1,1)
        b, f, j, _ = x.shape
        # build time embedding
        t_emb = self.time_embed(t) # (B,F,t_dim)
        t_emb = t_emb.unsqueeze(2).expand(b, f, j, self.t_embed_dim).contiguous()  # (B,F,J,t_dim)

        x_in = torch.cat([x, y_t, t_emb], dim=-1)           # (B,F,J,2+3+t_dim)
        x_in = rearrange(x_in, 'b f j c -> (b f) j c').contiguous() # (B*F,J,in)

        # condition from raw x (2D) projected to embed dim
        cond_in = rearrange(x, 'b f j c -> (b f) j c').contiguous()  # (B*F,J,2)
        cond = self.cond_proj(cond_in)                                 # (B*F,J,channel)

        # encoder -> GCN_MLP -> regression head
        h = self.encoder(x_in)
        h = self.FMPose(h, cond)
        v = self.pred_mu(h)                                  # (B*F,J,3)
        v = rearrange(v, '(b f) j c -> b f j c', b=b, f=f).contiguous() # (B,F,J,3)
        return v 