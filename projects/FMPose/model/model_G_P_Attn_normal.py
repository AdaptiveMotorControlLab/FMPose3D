import sys
sys.path.append("..")
import torch
import torch.nn as nn
from einops import rearrange
from model.graph_frames import Graph
from functools import partial
from einops import rearrange
from timm.models.layers import DropPath

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
        # x = self.act(x)
        x = self.drop(x)
        return x

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
    
class Block(nn.Module): # drop=0.1
    def __init__(self, length, dim, tokens_dim, channels_dim, adj, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # length =17, dim = args.channel = 512, tokens_dim = args.token_dim=256, channels_dim = args.d_hid = 1024
        super().__init__()
        # GCN
        self.norm1 = norm_layer(length)
        self.gcn1 = GCN_V2(dim, dim, adj)
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

        # channel MLP
        self.norm_mlp = norm_layer(dim)
        # self.channel_mlp = uMLP(in_features=dim, hidden_features=256, act_layer=act_layer, drop=0.1)
        # spatial MLP
        self.mlp = Mlp(in_features=dim, hidden_features=1024, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # B,J,dim 
        res1 = x # b,j,c
        x_gcn_1 = x.clone()
        # gcn
        x_gcn_1 = rearrange(x_gcn_1,"b j c -> b c j").contiguous() 
        x_gcn_1 = self.norm1(x_gcn_1) # b,c,j
        x_gcn_1 = rearrange(x_gcn_1,"b c j -> b j c").contiguous()
        x_gcn_1 = self.gcn1(x_gcn_1)  # b,j,c
        # attention
        x_atten = x.clone()
        x_atten = self.norm_att1(x_atten)
        x_atten = self.attn(x_atten)
        
        x = res1 + self.drop_path(x_gcn_1 + x_atten)
        
        # MLP residual
        res2 = x  # b,j,c
        x2 = self.norm_mlp(x.clone())
        x = self.mlp(x2)
        x = res2 + self.drop_path(x)
        return x

class GCN_MLP(nn.Module):
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

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
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

        # encoder maps x2d(2) to args.channel
        self.encoder = encoder(2, args.channel//2, args.channel)
        #  
        self.GCN_MLP = GCN_MLP(args.layers, args.channel, args.d_hid, args.token_dim, self.A, length=args.n_joints) # 256
        self.pred_mu = decoder(args.channel, args.channel//2, 3)

    def forward(self, x):
        # x: (B,F,J,2) - 2D pose input
        b, f, j, _ = x.shape
        
        # reshape for processing
        x_in = rearrange(x, 'b f j c -> (b f) j c').contiguous() # (B*F,J,2)
        
        # encoder -> GCN_MLP -> regression head
        h = self.encoder(x_in)
        h = self.GCN_MLP(h)
        out = self.pred_mu(h) # (B*F,J,3)
        
        out = rearrange(out, '(b f) j c -> b f j c', b=b, f=f).contiguous() # (B,F,J,3)
        return out 