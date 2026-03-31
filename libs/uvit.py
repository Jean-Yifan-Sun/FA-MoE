import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .timm import trunc_normal_, Mlp
import einops
import torch.utils.checkpoint
# from normalization import *

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
# ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


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
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Expert(nn.Module):
    """ A single expert in a Mixture of Experts layer. Replace the FFN in the Transformer block with this."""
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

class ECDiTRoutingLayer(nn.Module):
    """
    EC-DiT Routing Layer: Expert-Choice Routing for Diffusion Transformers
    """

    def __init__(self, dim, num_experts, expert_capacity_factor=2.0, mlp_hidden_dim=2048, act_layer=nn.GELU, num_tokens=1, counting=False):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = expert_capacity_factor
        
        # Router parameters (expert embeddings)
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        # Expert networks (FeedForward networks)
        self.experts = nn.ModuleList([
            Expert(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer) for _ in range(num_experts)
        ])
        # 添加 buffer 来记录 token 分配情况
        # Shape: [num_tokens] - 每个 token 被多少专家选中
        self.counting = True
        self.register_buffer('token_selection_count', torch.zeros(num_experts, num_tokens, dtype=torch.float))
        self.decay = 0.99  # 指数移动平均的衰减率
        
        
    def forward(self, x_prime):
        """
        x_prime: [batch_size, seq_len, dim] - input after cross-attention
        Returns: [batch_size, seq_len, dim] - routed output
        """
        batch_size, seq_len, dim = x_prime.shape
        
        # Step 1: Compute token-expert affinity scores
        # A = softmax(x' @ W_r) - equation (5) in paper
        affinity_scores = self.router(x_prime)  # [B, S, E]
        affinity_scores = F.softmax(affinity_scores, dim=-1)  # [B, S, E]
        
        # Step 2: Calculate expert capacity
        # C = S × f_c / E - capacity per expert
        expert_capacity = int(seq_len * self.capacity_factor / self.num_experts) + 1
        expert_capacity = max(1, expert_capacity)  # Ensure at least 1
        
        # Step 3: Create gating tensor G using expert-choice routing
        gating_tensor, selection_indices, selection_mask = self._expert_choice_routing(affinity_scores, expert_capacity)

        # 更新 token 分配统计
        if self.training and self.counting:
            with torch.no_grad():
                # 计算每个 token 被选中的次数
                token_counts = torch.zeros(self.num_experts, seq_len, device=x_prime.device)
                token_score = torch.ones(self.num_experts, seq_len, device=x_prime.device, dtype=torch.float)
                for _ in range(batch_size):
                    for i in range(self.num_experts):
                    # selection_indices: [B, E, C] -> [B*E*C]
                        expert_selections = selection_indices[_, i, :]  # [C]
                        
                    # 累加每个 token 被选中的次数
                        token_counts[i, expert_selections] += token_score[i, expert_selections]
                
                # 归一化: 除以专家总数
                token_counts = token_counts / (batch_size)
                
                # 使用指数移动平均更新统计
                # if self.token_selection_count.shape != token_counts.shape:
                #     self.token_selection_count = token_counts
                # else:
                self.token_selection_count = (
                        self.token_selection_count * self.decay + 
                        token_counts * (1 - self.decay)
                    )
        
        # Step 4: Process tokens through experts and combine outputs
        output = self._apply_experts(x_prime, gating_tensor, selection_indices, selection_mask, expert_capacity)
        
        return output
    
    def _expert_choice_routing(self, affinity_scores, expert_capacity):
        """
        Expert-choice routing: Each expert selects top-C tokens
        Corresponds to equation (6) in paper
        """
        batch_size, seq_len, num_experts = affinity_scores.shape
        
        # Reshape to [B, E, S] for expert-wise operations
        affinity_scores_t = affinity_scores.transpose(1, 2)  # [B, E, S]
        
        # Find top-C tokens for each expert
        topk_values, topk_indices = torch.topk(
            affinity_scores_t, 
            k=expert_capacity, 
            dim=-1
        )  # [B, E, C]
        
        # Create gating tensor initialized to zeros
        # Create selection mask and indices for gathering
        selection_indices = topk_indices  # [B, E, C]
        selection_mask = torch.ones_like(topk_values, dtype=torch.bool)  # [B, E, C]
        gating_tensor = torch.zeros_like(affinity_scores_t)  # [B, E, S]
        
        # Scatter top-k values to appropriate positions
        batch_indices = torch.arange(batch_size, device=affinity_scores.device)
        batch_indices = batch_indices.view(-1, 1, 1).expand(-1, num_experts, expert_capacity)
        
        expert_indices = torch.arange(num_experts, device=affinity_scores.device)
        expert_indices = expert_indices.view(1, -1, 1).expand(batch_size, -1, expert_capacity)
        
        # Use scatter_ to assign values
        gating_tensor.scatter_(
            dim=-1,
            index=topk_indices,
            src=topk_values
        )
        
        # Transpose back to [B, S, E]
        gating_tensor = gating_tensor.transpose(1, 2)  # [B, S, E]
        
        return gating_tensor, selection_indices, selection_mask
    
    def _apply_experts(self, x_prime, gating_tensor, selection_indices, selection_mask, expert_capacity):
        """
        Apply experts to selected tokens and combine results
        Corresponds to equation (8) in paper
        """
        batch_size, seq_len, dim = x_prime.shape
        num_experts = len(self.experts)

        # Initialize output
        output = torch.zeros_like(x_prime)  # [B, S, D]
        
        # Process each expert separately (clearer and more debuggable)
        for expert_idx in range(num_experts):
            expert_fn = self.experts[expert_idx]
            
            # Get the tokens selected by this expert
            # selection_indices: [B, E, C] -> for this expert: [B, C]
            expert_token_indices = selection_indices[:, expert_idx, :]  # [B, C]
            
            # Get gating values for these tokens
            expert_gating_values = torch.gather(
                gating_tensor[:, :, expert_idx],  # [B, S]
                dim=1,
                index=expert_token_indices  # [B, C]
            ).unsqueeze(-1)  # [B, C, 1]
            
            # Gather the actual tokens to process
            # Create batch indices
            batch_indices = torch.arange(batch_size, device=x_prime.device)
            batch_indices = batch_indices.view(-1, 1).expand(batch_size, expert_capacity)  # [B, C]
            
            # Gather tokens: [B, C, D]
            selected_tokens = x_prime[batch_indices, expert_token_indices, :]
            
            # Process through expert
            # Flatten batch and capacity for processing
            selected_tokens_flat = selected_tokens.reshape(-1, dim)  # [B*C, D]
            expert_output_flat = expert_fn(selected_tokens_flat)  # [B*C, D]
            expert_output = expert_output_flat.reshape(batch_size, expert_capacity, dim)  # [B, C, D]
            
            # Apply gating
            gated_output = expert_output * expert_gating_values  # [B, C, D]
            
            # Scatter back to output - FIXED DIMENSION HANDLING
            # We need to scatter along dimension 1 (sequence dimension)
            # output: [B, S, D]
            # expert_token_indices: [B, C] 
            # gated_output: [B, C, D]
            
            # Use scatter_add_ with proper dimensions
            output.scatter_add_(
                dim=1,  # scatter along sequence dimension
                index=expert_token_indices.unsqueeze(-1).expand(-1, -1, dim),  # [B, C, D]
                src=gated_output  # [B, C, D]
            )
        
        return output
    
    def get_expert_distribution(self):
        """获取专家选择的分布统计"""
        return self.token_selection_count.cpu().numpy()

class Block_ECDiT(nn.Module):
    """ Transformer block with EC-DiT Routing Layer. """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False, num_experts=2, expert_capacity_factor=1.0, num_tokens=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        
        # --- Recommended Change: Use a single LayerNorm before the MoE layer ---
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # --- EC-DiT components ---
        self.routing_layer = ECDiTRoutingLayer(dim=dim, num_experts=num_experts, expert_capacity_factor=expert_capacity_factor, mlp_hidden_dim=mlp_hidden_dim, act_layer=act_layer, num_tokens=num_tokens)
            
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(lambda inp, skp: self._forward(inp, skp), x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        
        # 1. Attention Block
        x = x + self.attn(self.norm1(x))
        residual = x
        
        # 2. EC-DiT Routing Layer
        x = self.norm2(x)
        x = self.routing_layer(x)
        
        # 3. Residual Connection
        x = residual + x
        
        return x
    
    def get_expert_distribution(self):
        """获取专家选择的分布统计"""
        return self.routing_layer.get_expert_distribution()

class UViT_greyscale(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True, tokens=0, low_freqs=0, use_moe=False, MoE=None):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.tokens = tokens
        self.DCT_coes = low_freqs

        if in_chans == 1:
            self.proj = nn.Linear(self.DCT_coes * 4, embed_dim, bias=True)
            self.decoder_pred = nn.Linear(embed_dim, self.DCT_coes * 4, bias=True)
        else:
            self.proj = nn.Linear(in_chans, embed_dim, bias=True)
            self.decoder_pred = nn.Linear(embed_dim, in_chans, bias=True)

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + self.tokens, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, y=None):
        # x: (b, tokens, num_low_freq*4)
        x = self.proj(x)  # (b, tokens, num_low_freq*4) --> (b, tokens, hidden_dim)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)  # (b, 1, dim)
        x = torch.cat((time_token, x), dim=1)
        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)

        image_tokens = x[:, self.extras:, :] # Select the image tokens first
        x = self.decoder_pred(image_tokens) # Then apply the prediction head ONLY to them

        # x = self.decoder_pred(x)  # (b, tokens, dim) --> (b, tokens, num_low_freq*4)
        # assert x.size(1) == self.extras + L
        # x = x[:, self.extras:, :]  # (b, tokens, num_low_freq*4)

        return x

class UViT_greyscale_MoE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True, tokens=0, low_freqs=0, use_moe=True, MoE={"typr":'normal', "depth": 1, "num_experts": 2, "router":"topk", "top_k": 2},
                 pos_normalize="minmax"):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.tokens = tokens
        self.DCT_coes = low_freqs
        assert use_moe==True, "UViT_greyscale_MoE is designed to use MoE. Set use_moe=True."
        # self.pos_normalize = False
        
        # --- MoE Configuration ---
        self.num_experts = MoE.get("num_experts", 2)
        self.router_type = MoE.get("router", "topk")
        self.top_k = MoE.get("top_k", 2) # How many experts to use per token
        self.moe_noise_eps = MoE.get("noise_eps", 1e-2)
        self.moe_layer_index = MoE.get("depth", 1) # Interpreted as placing MoE at first and last blocks

        # --- Input and Embedding Layers ---
        if in_chans != 1:
            self.proj = nn.Linear(in_chans, embed_dim, bias=True)
        else:
            self.proj = nn.Linear(self.DCT_coes * 4, embed_dim, bias=True) # For greyscale images, only use Y channel
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), 
            nn.SiLU(), 
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1
            
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + self.tokens, embed_dim))
        
        # --- Build Transformer Blocks with MoE ---
        in_blocks_list = []
        for i in range(depth // 2):
            if self.moe_layer_index == i + 1 or self.moe_layer_index == -1: # First block or all blocks
                block = Block_ECDiT(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        norm_layer=norm_layer, use_checkpoint=use_checkpoint, num_experts=self.num_experts, expert_capacity_factor=self.top_k, num_tokens=self.tokens + self.extras)
            else:
                 block = Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            in_blocks_list.append(block)
        self.in_blocks = nn.ModuleList(in_blocks_list)

        if self.moe_layer_index == -1:
            self.mid_block = Block_ECDiT(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint, num_experts=self.num_experts, expert_capacity_factor=self.top_k, num_tokens=self.tokens + self.extras)
        else:
            self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        out_blocks_list = []
        for i in range(depth // 2):
            if self.moe_layer_index + i == (depth // 2) or self.moe_layer_index == -1: # Last block
                block = Block_ECDiT(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint, num_experts=self.num_experts, expert_capacity_factor=self.top_k, num_tokens=self.tokens + self.extras)
            else:
                block = Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            out_blocks_list.append(block)
        self.out_blocks = nn.ModuleList(out_blocks_list)

        # --- Output Layers ---
        self.norm = norm_layer(embed_dim)
        if in_chans != 1:
            self.decoder_pred = nn.Linear(embed_dim, in_chans, bias=True)
        else:
            self.decoder_pred = nn.Linear(embed_dim, self.DCT_coes * 4, bias=True)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, y=None):
        # 1. Initial Projection and Embedding
        # if self.pos_normalize:
        #     x = self.input_normalize(x)

        x = self.proj(x)
        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None:
            label_emb = self.label_emb(y).unsqueeze(1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        # 2. Forward pass through the network
        skips = []

        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        # 3. Final Prediction Head
        x = self.norm(x)
        image_tokens = x[:, self.extras:, :]
        x = self.decoder_pred(image_tokens)

        return x
    
    def get_expert_distribution(self):
        """获取所有MoE层的专家选择的分布统计"""
        distributions = {}
        for i, blk in enumerate(self.in_blocks):
            if isinstance(blk, Block_ECDiT):
                distributions[f'in_block_{i}'] = blk.get_expert_distribution()
        for i, blk in enumerate(self.out_blocks):
            if isinstance(blk, Block_ECDiT):
                distributions[f'out_block_{i}'] = blk.get_expert_distribution()
        return distributions




