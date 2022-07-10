import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


class Embedding(nn.Module):
    def __init__(self,
                 user_num : int = 100,
                 item_num : int = 100, 
                 emb_size : int = 256, 
                 factor_num : int = 32,
                 patch_size : int = 4):
        super(Embedding, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embed_user = nn.ModuleList([nn.Embedding(user_num, factor_num).to(device) for _ in range(emb_size)])
        self.embed_item = nn.ModuleList([nn.Embedding(item_num, factor_num).to(device) for _ in range(emb_size)])
        self.projection = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, kernel_size = patch_size, stride = patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((factor_num // patch_size) ** 2 + 1, emb_size))
    
    def forward(self, user, item):
        b, _ = user.size()
 
        embed_user = [layer(user) for layer in self.embed_user]
        embed_user_ = [emb.permute(0, 2, 1) for emb in embed_user]

        embed_item = [layer(item) for layer in self.embed_item]

        embed_outer = [torch.unsqueeze(torch.bmm(emb_u, emb_i), 1) for emb_u, emb_i in zip(embed_user_, embed_item)]
        embed_outer = torch.cat(embed_outer, dim = 1)

        x = self.projection(embed_outer)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions

        embed_user = torch.cat(embed_user, dim = 1)
        embed_item = torch.cat(embed_item, dim = 1)

        return x, embed_user, embed_item, embed_outer


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 emb_size : int = 256,
                 num_heads : int = 8, 
                 dropout : float = 0.):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads if emb_size >= 8 else emb_size

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)

        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x , mask = None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h = self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h = self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h = self.num_heads)

        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, 
                 emb_size : int, 
                 expansion : int = 4, 
                 drop_p : float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, 
                 emb_size : int = 256, 
                 drop_p : float = 0., 
                 forward_expansion : int = 4,
                 forward_drop_p : float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p)
                )
             ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p)
                )
            )
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth : int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size : int = 256, out_size : int = 1):
        super().__init__(
            Reduce('b n e -> b e', reduction = 'mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, out_size)
        )

class AuxClassifier(nn.Module):
    def __init__(self, aux_depth : int = 3, emb_size : int = 256, factor_num : int= 16, out_size : int = 1):
        super().__init__()
        self.emb_size = emb_size*factor_num
        self.out_size = out_size
        cur_size = self.emb_size
        MLP_modules = []
        for i in range(aux_depth-1):
            input_size = self.emb_size // (2 ** (i))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
            cur_size = input_size//2
        MLP_modules.append(nn.Linear(cur_size, self.out_size))
        self.MLP_layers = nn.Sequential(*MLP_modules)

    def forward(self, x):
        res = self.MLP_layers(x)
        return res

class ViT(nn.Module):
    model_name = "ViT"
    def __init__(self,
                 user_num : int = 100,
                 item_num : int = 100,
                 emb_size : int = 256,
                 factor_num : int = 128,
                 patch_size : int = 4,
                 depth : int = 12,
                 aux_depth: int = 3,
                 user_out : int = 100,
                 item_out : int = 100,
                 dropout : float = 0.1,
                 **kwargs):
        """ NCF Framework Using Transformer Structure

        Args:
            user_num (int, optional): number of users. Defaults to 100.
            item_num (int, optional): number of items. Defaults to 100.
            emb_size (int, optional): number of embedding layers. Defaults to 256.
            factor_num (int, optional): number of predictive factors. Defaults to 128.
            depth (int, optional): number of EncoderBlock. Defaults to 12.
            depth_user (int, optional): number of user axiliary classifier encoderBlock. Defaults to 6.
            depth_item (int, optional): number of item axiliary classifier encoderBlock. Defaults to 6.
            user_out (int, optional) : output size of user auxiliary classifier. Defaults to 100.
            item_out (int, optional) : output size of item auxiliary classifier. Defaults to 100.
            dropout (float, optional) : dropout rate
        """
        super().__init__()

        self.user_out = user_out
        self.item_out = item_out

        self.emb = Embedding(user_num = user_num,
                             item_num = item_num, 
                             emb_size = emb_size, 
                             factor_num = factor_num,
                             patch_size = patch_size)
        self.enc = TransformerEncoder(depth = depth, emb_size = emb_size, forward_drop_p=dropout, drop_p=dropout, **kwargs)
        self.cls = ClassificationHead(emb_size = emb_size, out_size = 1)

        if self.user_out:
            self.aux_user = AuxClassifier(aux_depth = aux_depth, emb_size = emb_size, factor_num = factor_num, out_size = user_out)
            # self.enc_user = TransformerEncoder(depth_user, emb_size = emb_size, **kwargs)
            # self.cls_user = ClassificationHead(emb_size = emb_size, out_size = user_out)
        if self.item_out:
            self.aux_item = AuxClassifier(aux_depth = aux_depth, emb_size = emb_size, factor_num = factor_num, out_size = item_out)
            # self.enc_item = TransformerEncoder(depth_item, emb_size = emb_size, **kwargs)
            # self.cls_item = ClassificationHead(emb_size = emb_size, out_size = item_out)
        
    def forward(self, user, item):
        b, _ = user.size()
        x, embed_user, embed_item, embed_outer = self.emb(user, item)
        x = self.enc(x)
        x = self.cls(x)

        result = {
            'main' : x,
            'user' : None,
            'item' : None
        }

        if self.user_out:
            x_user = embed_user.view(b,-1)
            x_user = self.aux_user(x_user)
            #x_user = self.enc_user(embed_user)
            #x_user = self.cls_user(x_user)
            result['user'] = x_user
        
        if self.item_out:
            x_item = embed_item.view(b,-1)
            x_item = self.aux_item(x_item)
            #x_item = self.enc_item(embed_item)
            #x_item = self.cls_item(x_item)
            result['item'] = x_item

        return result

class ONCF(nn.Module):
    model_name='ONCF'
    def __init__(self,
                 user_num : int = 100,
                 item_num : int = 100,
                 emb_size : int = 256,
                 factor_num : int = 128,
                 patch_size : int = 4,
                 depth : int = 12,
                 aux_depth: int = 3,
                 user_out : int = 100,
                 item_out : int = 100,
                 dropout : float = 0.1,
                 **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.emb = Embedding(user_num = user_num,
                             item_num = item_num, 
                             emb_size = emb_size, 
                             factor_num = factor_num,
                             patch_size = patch_size)
        self.dropout = dropout
        conv = [nn.Conv2d(emb_size, 32, kernel_size=2, stride=2), nn.ReLU()]
        output_dim = factor_num//2
        while  output_dim != 1:
            conv.append(nn.Conv2d(32, 32, kernel_size=2, stride=2))
            conv.append(nn.ReLU())
            output_dim = output_dim // 2
        self.conv = nn.Sequential(*conv)
        self.cls = nn.Linear(32, 1)

        self.affine = nn.Linear(emb_size*factor_num, 1)

    def forward(self, user, item, is_pretrain):
        b, _ = user.size()
        x, embed_user, embed_item, embed_outer = self.emb(user, item)
        
        if is_pretrain:
            x = torch.mul(embed_user, embed_item)
            x = x.view(b, -1)
            output = self.affine(x)
        else:
            x = self.conv(embed_outer)
            x = x.view(b, -1)
            x = nn.Dropout(self.dropout)(x)
            output = self.cls(x)
        return output


                 