import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import ops
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.wtconv import WTConv2d
print("ARCH FILE:", __file__)

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class NonLocalSparseAttention(nn.Module):
    def __init__(self, n_hashes=4, channels=64, k_size=3, reduction=4, chunk_size=144, conv=default_conv, res_scale=1):
        super(NonLocalSparseAttention, self).__init__()
        self.chunk_size = chunk_size
        self.n_hashes = n_hashes
        self.reduction = reduction
        self.res_scale = res_scale
        self.conv_match = BasicBlock(conv, channels, channels // reduction, k_size, bn=False, act=None)
        self.conv_assembly = BasicBlock(conv, channels, channels, 1, bn=False, act=None)

    def LSH(self, hash_buckets, x):
        # x: [N,H*W,C]
        N = x.shape[0]
        device = x.device

        # generate random rotation matrix
        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets // 2)  # [1,C,n_hashes,hash_buckets//2]
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(N, -1, -1,
                                                                                             -1)  # [N, C, n_hashes, hash_buckets//2]

        # locality sensitive hashing
        rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations)  # [N, n_hashes, H*W, hash_buckets//2]
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)  # [N, n_hashes, H*W, hash_buckets]

        # get hash codes
        hash_codes = torch.argmax(rotated_vecs, dim=-1)  # [N,n_hashes,H*W]

        # add offsets to avoid hash codes overlapping between hash rounds
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
        hash_codes = torch.reshape(hash_codes + offsets, (N, -1,))  # [N,n_hashes*H*W]

        return hash_codes

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)

    def forward(self, input):

        N, _, H, W = input.shape
        x_embed = self.conv_match(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)
        y_embed = self.conv_assembly(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)
        L, C = x_embed.shape[-2:]

        # number of hash buckets/hash bits
        hash_buckets = min(L // self.chunk_size + (L // self.chunk_size) % 2, 128)

        # get assigned hash codes/bucket number
        hash_codes = self.LSH(hash_buckets, x_embed)  # [N,n_hashes*H*W]
        hash_codes = hash_codes.detach()

        # group elements with same hash code by sorting
        _, indices = hash_codes.sort(dim=-1)  # [N,n_hashes*H*W]
        _, undo_sort = indices.sort(dim=-1)  # undo_sort to recover original order
        mod_indices = (indices % L)  # now range from (0->H*W)
        x_embed_sorted = batched_index_select(x_embed, mod_indices)  # [N,n_hashes*H*W,C]
        y_embed_sorted = batched_index_select(y_embed, mod_indices)  # [N,n_hashes*H*W,C]

        # pad the embedding if it cannot be divided by chunk_size
        padding = self.chunk_size - L % self.chunk_size if L % self.chunk_size != 0 else 0
        x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_hashes, -1, C))  # [N, n_hashes, H*W,C]
        y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_hashes, -1, C * self.reduction))
        if padding:
            pad_x = x_att_buckets[:, :, -padding:, :].clone()
            pad_y = y_att_buckets[:, :, -padding:, :].clone()
            x_att_buckets = torch.cat([x_att_buckets, pad_x], dim=2)
            y_att_buckets = torch.cat([y_att_buckets, pad_y], dim=2)

        x_att_buckets = torch.reshape(x_att_buckets, (
            N, self.n_hashes, -1, self.chunk_size, C))  # [N, n_hashes, num_chunks, chunk_size, C]
        y_att_buckets = torch.reshape(y_att_buckets, (N, self.n_hashes, -1, self.chunk_size, C * self.reduction))

        x_match = F.normalize(x_att_buckets, p=2, dim=-1, eps=5e-5)

        # allow attend to adjacent buckets
        x_match = self.add_adjacent_buckets(x_match)
        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

        # unormalized attention score
        raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets,
                                 x_match)  # [N, n_hashes, num_chunks, chunk_size, chunk_size*3]

        # softmax
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score)  # (after softmax)
        bucket_score = torch.reshape(bucket_score, [N, self.n_hashes, -1])

        # attention
        ret = torch.einsum('bukij,bukje->bukie', score, y_att_buckets)  # [N, n_hashes, num_chunks, chunk_size, C]
        ret = torch.reshape(ret, (N, self.n_hashes, -1, C * self.reduction))

        # if padded, then remove extra elements
        if padding:
            ret = ret[:, :, :-padding, :].clone()
            bucket_score = bucket_score[:, :, :-padding].clone()

        # recover the original order
        ret = torch.reshape(ret, (N, -1, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, -1,))  # [N,n_hashes*H*W]
        ret = batched_index_select(ret, undo_sort)  # [N, n_hashes*H*W,C]
        bucket_score = bucket_score.gather(1, undo_sort)  # [N,n_hashes*H*W]

        # weighted sum multi-round attention
        ret = torch.reshape(ret, (N, self.n_hashes, L, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, L, 1))
        probs = nn.functional.softmax(bucket_score, dim=1)
        ret = torch.sum(ret * probs, dim=1)

        ret = ret.permute(0, 2, 1).view(N, -1, H, W).contiguous() * self.res_scale + input
        return ret


def patch_divide(x, step, ps):
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw


def patch_reverse(crop_x, x, step, ps):
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads, qk_dim):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)  # scaled_dot_product_attention 需要PyTorch2.0之后版本
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class LRSA(nn.Module):
    def __init__(self, dim, qk_dim=36, mlp_dim=96, heads=4, ps=12):
        super().__init__()
        self.layer = nn.ModuleList([
            PreNorm(dim, Attention(dim, heads, qk_dim)),
            PreNorm(dim, ConvFFN(dim, mlp_dim))])

    def forward(self, x, ps):  # 试试8 再后12 最后再8+12
        step = ps - 2
        crop_x, nh, nw = patch_divide(x, step, ps)  # (b, n, c, ps, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')
        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c->b c h w', h=h)
        return x


class MultiWindowLRSA(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, heads=1):
        super().__init__()
        self.lrsa_8 = LRSA(dim, qk_dim, mlp_dim, heads)
        self.lrsa_16 = LRSA(dim, qk_dim, mlp_dim, heads)
        self.fusion = nn.Conv2d(dim * 2, dim, 1)  # optional

    def forward(self, x):
        out_8 = self.lrsa_8(x, ps=8)

        out_16 = self.lrsa_16(x, ps=16)
        # out_12 = self.lrsa_12(x, ps=12)
        out = out_8 + out_16  # 偏向于＋

        return out


class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x


class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)

        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1, x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
            x = self.conv_2(x)
        return x


class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lde = MultiWindowLRSA(dim, qk_dim=36, mlp_dim=96, heads=4)  #########################对PS进行试验
        # self.lde = DMlp(dim, 2)

        # self.dw_conv = nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.dw_conv2 = WTConv2d(dim, dim, kernel_size=5, wt_levels=1, padding=2)  # 修改这里

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        # x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_s = self.dw_conv2(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))  # 修改这里
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w),
                                mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)


class FMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.smfa = SMFA(dim)
        # self.lrsa = LRSA(dim=dim, qk_dim=36, mlp_dim=96, heads=4, ps=16)
        self.pcfn = PCFN(dim, ffn_scale)

    def forward(self, x):
        x = self.smfa(F.normalize(x)) + x
        # x = self.lrsa(F.normalize(x)) + x
        x = self.pcfn(F.normalize(x)) + x
        return x


@ARCH_REGISTRY.register()
class MAAN(nn.Module):
    def __init__(self, dim=36, n_blocks=8, ffn_scale=2, upscaling_factor=4):
        super().__init__()
        self.nlsa = NonLocalSparseAttention(n_hashes=4, channels=36, k_size=3, reduction=4, chunk_size=144, conv=default_conv, res_scale=1)
        self.scale = upscaling_factor
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.feats = nn.Sequential(*[FMB(dim, ffn_scale) for _ in range(n_blocks)])
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor ** 2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        x = self.to_feat(x)
        # print("x1.size", x.size())
        x = self.nlsa(x)
        # print("x2.size", x.size())
        x = self.feats(x) + x
        x = self.nlsa(x)
        x = self.to_img(x)
        return x
