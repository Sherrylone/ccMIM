from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed
from basic_module import Block
from util.pos_embed import get_2d_sincos_pos_embed

class AttnMaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, args=None):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.contra_dim = 256

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(embed_dim, self.contra_dim, bias=False),
            nn.BatchNorm1d(self.contra_dim, affine=False)
        )

        self.args = args
        self.temperature = 0.2

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def standard_norm(self, target):
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5
        return target

    def get_mask_idx(self, x, mask_ratio):
        N, L, D = x.size()

        # apply Transformer blocks
        avg_cls_attn = torch.zeros((N, L-1)).to(x.device)
        _ = x.detach()
        with torch.no_grad():
            for blk in self.blocks:
                # calculate the attention value with CLS token.
                attn, _ = blk(_)
                # attention value of each index
                avg_cls_attn = avg_cls_attn + attn
        del _

        avg_cls_attn = F.softmax(avg_cls_attn, dim=1).detach()
        avg_cls_attn = avg_cls_attn / F.softmax(torch.rand(N, L-1, device=x.device), dim=-1)
        # avg_cls_attn = torch.rand(N, L-1, device=x.device)
        x = x[:, 1:, :]     # remove cls token
        vis_nums = int((L-1) * (1-mask_ratio))

        mask_sort = torch.argsort(avg_cls_attn, dim=1)
        ids_restore = torch.argsort(mask_sort, dim=1)

        # visible and mask patches index
        vis_idx = mask_sort[:, :vis_nums]
        x_vis = torch.gather(x, dim=1, index=vis_idx.unsqueeze(-1).repeat(1, 1, D))
        mask_idx = mask_sort[:, vis_nums:]
        x_mask = torch.gather(x, dim=1, index=mask_idx.unsqueeze(-1).repeat(1, 1, D))

        mask_binary = torch.ones([N, L-1], device=x.device)
        mask_binary[:, :vis_nums] = 0
        # unshuffle to get the binary mask
        mask_binary = torch.gather(mask_binary, dim=1, index=ids_restore)

        return x_vis, x_mask, mask_binary, ids_restore

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        x = x[:, 1:, :]
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_vis = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(x, dim=1, index=ids_mask.unsqueeze(-1).repeat(1, 1, D))
        x_vis = torch.gather(x, dim=1, index=ids_vis.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_vis, x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls_token.repeat(x.size(0), 1, 1), x], dim=1)

        if self.args.random:
            x_vis, x_mask, mask_binary, ids_restore = self.random_masking(x, mask_ratio)
        else:
            x_vis, x_mask, mask_binary, ids_restore = self.get_mask_idx(x, mask_ratio)

        x_vis = torch.cat((cls_token.expand(x_vis.shape[0], -1, -1), x_vis), dim=1)
        x_mask = torch.cat((cls_token.expand(x_mask.shape[0], -1, -1), x_mask), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            _, x_vis = blk(x_vis)
            _, x_mask = blk(x_mask)
        x_vis = self.norm(x_vis)
        x_mask = self.norm(x_mask)

        return x_vis, x_mask, mask_binary, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            _, x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_reconstruct_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_alignment_loss(self, cls_vis, cls_mask, T=0.5):
        cls_vis = self.projector(cls_vis)
        cls_mask = self.projector(cls_mask)
        cls_vis = nn.functional.normalize(cls_vis, dim=1)
        cls_mask = nn.functional.normalize(cls_mask, dim=1)
        # gather all targets
        cls_mask = concat_all_gather(cls_mask)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [cls_vis, cls_mask]) / T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * T)

    def forward(self, imgs, mask_ratio=0.75):
        embed_vis, embed_mask, mask_binary, ids_restore = self.forward_encoder(imgs, mask_ratio)
        if self.args.contrastive:
            l_align = self.forward_alignment_loss(embed_vis[:, 0, :], embed_mask[:, 0, :])
        else:
            l_align = torch.FloatTensor([0]).to(imgs.device)

        pred = self.forward_decoder(embed_vis, ids_restore)  # [N, L, p*p*3]
        l_construct = self.forward_reconstruct_loss(imgs, pred, mask_binary)

        return l_align, l_construct, pred, mask_binary


def ccmim_vit_base_patch16_dec512d8b(**kwargs):
    model = AttnMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def ccmim_vit_large_patch16_dec512d8b(**kwargs):
    model = AttnMaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def ccmim_vit_huge_patch14_dec512d8b(**kwargs):
    model = AttnMaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def ccmim_vit_small_patch16_dec512d8b(**kwargs):
    model = AttnMaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

# set recommended archs
ccmim_vit_base_patch16 = ccmim_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
ccmim_vit_large_patch16 = ccmim_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
ccmim_vit_huge_patch14 = ccmim_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
ccmim_vit_small_patch16 = ccmim_vit_small_patch16_dec512d8b
