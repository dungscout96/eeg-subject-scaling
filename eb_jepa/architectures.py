import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel

from eb_jepa.nn_utils import init_module_weights


# ===========================================================================
# REVE Transformer Components
# Adapted from braindecode (BSD-3-Clause License)
# Original: El Ouahidi et al. (2025), "REVE: A Foundation Model for EEG"
# https://github.com/braindecode/braindecode/blob/master/braindecode/models/reve.py
# ===========================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more stable than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class GEGLU(nn.Module):
    """Gated GELU activation: splits input, uses GELU-gated half to modulate other."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class REVEFeedForward(nn.Module):
    """Transformer feedforward sublayer: RMSNorm → Linear → GEGLU → Linear."""

    def __init__(self, dim: int, hidden_dim: int, geglu: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, hidden_dim * 2 if geglu else hidden_dim, bias=False),
            GEGLU() if geglu else nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class REVEClassicalAttention(nn.Module):
    """Multi-head attention using PyTorch SDPA (flash attention when available)."""

    def __init__(self, heads: int, use_sdpa: bool = True):
        super().__init__()
        self.use_sdpa = use_sdpa
        self.heads = heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = (
            rearrange(t, "batch seq (heads dim) -> batch heads seq dim", heads=self.heads)
            for t in (q, k, v)
        )
        if self.use_sdpa:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = F.scaled_dot_product_attention(q, k, v)
        else:
            scale = q.shape[-1] ** -0.5
            dots = torch.matmul(q, k.transpose(-1, -2)) * scale
            attn = torch.softmax(dots, dim=-1)
            out = torch.matmul(attn, v)
        out = rearrange(out, "batch heads seq dim -> batch seq (heads dim)")
        return out


class REVEAttention(nn.Module):
    """Self-attention sublayer: RMSNorm → QKV projection → SDPA → output projection."""

    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.attend = REVEClassicalAttention(self.heads, use_sdpa=True)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x)
        out = self.attend(qkv)
        return self.to_out(out)


class REVETransformerBackbone(nn.Module):
    """Transformer backbone: stacks of [Attention + FeedForward] with residual connections."""

    def __init__(self, dim, depth, heads, head_dim, mlp_dim, geglu=True):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    REVEAttention(self.dim, heads=heads, head_dim=head_dim),
                    REVEFeedForward(self.dim, mlp_dim, geglu),
                ])
            )

    def forward(self, x) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class FourierEmb4D(nn.Module):
    """4D Fourier positional encoding for (x, y, z, t) coordinates.

    Uses sinusoidal features at multiple frequencies for smooth interpolation
    to unseen positions. The time dimension is scaled by increment_time.
    """

    def __init__(self, dimension: int, freqs: int, increment_time=0.1, margin: float = 0.4):
        super().__init__()
        self.dimension = dimension
        self.freqs = freqs
        self.increment_time = increment_time
        self.margin = margin

    def forward(self, positions_: torch.Tensor) -> torch.Tensor:
        positions = positions_.clone()
        positions[:, :, -1] *= self.increment_time
        input_shape = positions.shape
        batch_dims = list(input_shape[:-1])

        freqs_w = torch.arange(self.freqs).to(positions)
        freqs_z = freqs_w[:, None]
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        p_z = 2 * math.pi * freqs_z / width
        p_w = 2 * math.pi * freqs_w / width
        positions = positions[..., None, None, None, None, :]
        loc = (
            positions[..., 0] * p_x
            + positions[..., 1] * p_y
            + positions[..., 2] * p_z
            + positions[..., 3] * p_w
        )
        batch_dims.append(-1)
        loc = loc.view(batch_dims)

        half_dim = self.dimension // 2
        current_dim = loc.shape[-1]
        if current_dim != half_dim:
            if current_dim > half_dim:
                loc = loc[..., :half_dim]
            else:
                raise ValueError(
                    f"Input dimension ({current_dim}) is too small for target "
                    f"embedding dimension ({self.dimension}). Expected at least {half_dim}."
                )

        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        return emb

    @classmethod
    def add_time_patch(cls, pos: torch.Tensor, num_patches: int) -> torch.Tensor:
        """Expand position tensor [B, C, 3] by adding time dimension → [B, C*num_patches, 4].

        Each channel position is repeated for each time patch index.
        """
        batch, nchans, _ = pos.shape
        pos_repeated = pos.unsqueeze(2).repeat(1, 1, num_patches, 1)
        time_values = torch.arange(0, num_patches, 1, device=pos.device).float()
        time_values = time_values.view(1, 1, num_patches, 1).expand(batch, nchans, num_patches, 1)
        pos_with_time = torch.cat((pos_repeated, time_values), dim=-1)
        pos_with_time = pos_with_time.view(batch, nchans * num_patches, 4)
        return pos_with_time


class EEGEncoderTokens(nn.Module):
    """REVE-based EEG encoder exposing token-level representations.

    Processes all T windows jointly as a single token sequence.
    Token grid is 3D: [C, T, P] where P = patches per window per channel.
    Flattened token order: (c, t, p) → index = c * (T*P) + t * P + p.

    Input:  [B, T, C, W] raw EEG
    Output: [B, C*T*P, embed_dim] token representations
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        embed_dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        head_dim: int = 16,
        n_windows: int = 16,
        patch_size: int = 200,
        patch_overlap: int = 20,
        freqs: int = 4,
        chs_info=None,
        mlp_dim_ratio: float = 2.66,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.embed_dim = embed_dim
        self.n_windows = n_windows
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.freqs = freqs

        # Compute patches per window (matches torch.unfold output count)
        step = patch_size - patch_overlap
        assert n_times >= patch_size, f"n_times ({n_times}) must be >= patch_size ({patch_size})"
        self.n_patches_per_window = (n_times - patch_size) // step + 1

        self.n_tokens_per_window = n_chans * self.n_patches_per_window
        self.total_patches_per_channel = n_windows * self.n_patches_per_window

        # Patch embedding
        self.to_patch_embedding = nn.Linear(patch_size, embed_dim)

        # 4D positional encoding
        self.fourier4d = FourierEmb4D(embed_dim, freqs=freqs)
        self.mlp4d = nn.Sequential(
            nn.Linear(4, embed_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.ln = nn.LayerNorm(embed_dim)

        # Transformer backbone
        mlp_dim = int(embed_dim * mlp_dim_ratio)
        self.transformer = REVETransformerBackbone(
            dim=embed_dim, depth=depth, heads=heads,
            head_dim=head_dim, mlp_dim=mlp_dim, geglu=True,
        )

        # Channel positions from REVE position bank
        self.default_pos = None
        if chs_info is not None:
            from braindecode.models.reve import RevePositionBank
            position_bank = RevePositionBank()
            self.default_pos = position_bank.forward(
                [ch["ch_name"] for ch in chs_info]
            )

    def _compute_pos_embed(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Compute 4D positional embeddings for the full [C, T, P] token grid.

        Returns: [B, C*T*P, embed_dim]
        """
        if self.default_pos is None:
            raise ValueError("No channel positions available. Provide chs_info at init.")

        pos = self.default_pos.to(device)  # [C, 3]
        C = pos.shape[0]
        T = self.n_windows
        P = self.n_patches_per_window

        # Build 4D positions: (x, y, z, t) for each token in (c, t, p) order
        # token_idx = c * (T*P) + t * P + p
        positions_4d = []
        for c in range(C):
            for t in range(T):
                for p in range(P):
                    xyz = pos[c]  # [3]
                    time_idx = t * P + p
                    pos_4d = torch.cat([xyz, torch.tensor([time_idx], device=device, dtype=xyz.dtype)])
                    positions_4d.append(pos_4d)

        positions_4d = torch.stack(positions_4d, dim=0)  # [C*T*P, 4]
        positions_4d = positions_4d.unsqueeze(0).expand(batch_size, -1, -1)  # [B, C*T*P, 4]

        pos_embed = self.ln(self.fourier4d(positions_4d) + self.mlp4d(positions_4d))
        return pos_embed

    def tokenize(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract patch tokens and compute positional embeddings.

        Args:
            eeg: [B, T, C, W] raw EEG

        Returns:
            tokens: [B, C*T*P, embed_dim] patch embeddings + positional encoding
            pos_embed: [B, C*T*P, embed_dim] positional embeddings (for predictor)
        """
        B, T, C, W = eeg.shape
        step = self.patch_size - self.patch_overlap
        P = self.n_patches_per_window

        # Extract patches from each window: [B, T, C, P, patch_size]
        patches = eeg.unfold(dimension=3, size=self.patch_size, step=step)
        # patches shape: [B, T, C, P, patch_size]

        # Linear projection: [B, T, C, P, embed_dim]
        patch_embeds = self.to_patch_embedding(patches)

        # Reorder to (c, t, p) flattening: [B, C*T*P, embed_dim]
        # Current: [B, T, C, P, embed_dim] → need [B, C, T, P, embed_dim] → [B, C*T*P, embed_dim]
        patch_embeds = patch_embeds.permute(0, 2, 1, 3, 4)  # [B, C, T, P, embed_dim]
        patch_embeds = patch_embeds.reshape(B, C * T * P, self.embed_dim)

        # Compute positional embeddings
        pos_embed = self._compute_pos_embed(B, eeg.device)

        # Add positional encoding
        tokens = patch_embeds + pos_embed

        return tokens, pos_embed

    def encode_tokens(
        self, eeg: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Tokenize, optionally mask, and encode through transformer.

        Args:
            eeg: [B, T, C, W] raw EEG
            mask: [C*T*P] bool tensor — True = keep (context), False = mask out.
                  If None, all tokens are kept.

        Returns:
            [B, n_visible, embed_dim] encoded token representations
        """
        tokens, _ = self.tokenize(eeg)

        if mask is not None:
            tokens = tokens[:, mask]  # [B, n_visible, embed_dim]

        tokens = self.transformer(tokens)
        return tokens

    def pool_to_windows(self, tokens: torch.Tensor,
                        keep_channels: bool = False) -> torch.Tensor:
        """Pool full (unmasked) token sequence to per-window representations.

        Reshapes [B, C*T*P, D] → [B, C, T, P, D] then mean-pools over patches
        and (by default) channels. With ``keep_channels=True`` the channel
        axis is concatenated into the feature dim instead of averaged, giving
        probes per-CorrCA-channel resolution. With 5-component CorrCA that's
        the difference between a 64-D and a 320-D probe input.

        Args:
            tokens: [B, C*T*P, embed_dim] — must be full (unmasked) token sequence
            keep_channels: if True, return [B, C*embed_dim, T, 1, 1].

        Returns:
            [B, embed_dim, T, 1, 1] or [B, C*embed_dim, T, 1, 1].
        """
        B = tokens.shape[0]
        C = self.n_chans
        T = self.n_windows
        P = self.n_patches_per_window
        D = self.embed_dim

        x = tokens.view(B, C, T, P, D)
        if keep_channels:
            x = x.mean(dim=3)                              # [B, C, T, D]
            x = x.permute(0, 2, 1, 3).reshape(B, T, C * D) # [B, T, C*D]
        else:
            x = x.mean(dim=(1, 3))                         # [B, T, D]
        return x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)


class _ResidualAdd(nn.Module):
    """Pre-norm residual wrapper: ``out = fn(x) + x`` (not in-place)."""

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class MovieCLIPHead(nn.Module):
    """Two-tower projection head for CLIP-style EEG ↔ V-JEPA-2 InfoNCE.

    EEG side: NICE-EEG-style MLP — ``Linear(D, P) → Residual(GELU + Linear(P, P)
    + Dropout) → LayerNorm(P)``. Higher capacity than a single Linear so the
    EEG encoder doesn't have to do all the alignment work itself.

    Vision side: asymmetric (NICE-style) by default — the V-JEPA-2 space is
    already well-organized, so we treat it as a fixed anchor and let only the
    EEG side learn into it. Set ``vision_passthrough=False`` for the symmetric
    variant with a learnable Linear vision projector.

    When ``vision_passthrough=True``, the EEG projector output dim is forced to
    ``vision_in_dim`` so EEG and vision embeddings live in the same space.
    The ``proj_dim`` argument is ignored in that mode.

    A learnable scalar log-temperature controls softmax sharpness, initialised
    to ``log(1 / 0.07)`` as in OpenAI CLIP.

    Caveat: per-window vision vectors can duplicate across recordings in a batch
    when two subjects are sampled at the same window position. With B=64, T=8
    this is rare in practice and treated as a hard-negative collision (no dedup).
    """

    def __init__(
        self,
        eeg_in_dim: int,
        vision_in_dim: int,
        proj_dim: int = 256,
        temperature: float = 0.07,
        drop_proj: float = 0.5,
        vision_passthrough: bool = True,
        n_residual_blocks: int = 1,
    ):
        super().__init__()
        self.vision_passthrough = vision_passthrough
        self.n_residual_blocks = n_residual_blocks
        out_dim = vision_in_dim if vision_passthrough else proj_dim
        self.proj_dim = out_dim

        # EEG side: stacked NICE-style residual MLP head.
        # n_residual_blocks=1 (default) reproduces the original 1-block head.
        # Bumping >1 gives the projector more capacity (useful when the encoder
        # is frozen so the head has to do all the alignment work itself).
        layers: list[nn.Module] = [nn.Linear(eeg_in_dim, out_dim)]
        for _ in range(n_residual_blocks):
            layers.append(_ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
                nn.Dropout(drop_proj),
            )))
        layers.append(nn.LayerNorm(out_dim))
        self.eeg_proj = nn.Sequential(*layers)

        # Vision side: identity (asymmetric/NICE) or learnable Linear (symmetric).
        if vision_passthrough:
            self.vision_proj = nn.Identity()
        else:
            self.vision_proj = nn.Linear(vision_in_dim, proj_dim)

        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temperature).log())
        self.apply(init_module_weights)

    def project_eeg(self, x):
        # x: [B, D, T, 1, 1] → [B*T, out_dim] (L2-normalized)
        B, D, T = x.shape[:3]
        x = x.view(B, D, T).permute(0, 2, 1).reshape(B * T, D)
        z = self.eeg_proj(x)
        return torch.nn.functional.normalize(z, dim=-1)

    def project_vision(self, v):
        # v: [N, vision_in_dim] → [N, out_dim] (L2-normalized)
        z = self.vision_proj(v)
        return torch.nn.functional.normalize(z, dim=-1)


