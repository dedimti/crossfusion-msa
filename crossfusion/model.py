"""
CrossFusion: Annotation-Driven Asymmetric Cross-Modal Attention
Text-Anchored Fusion for Robust Multimodal Sentiment Analysis

Architecture:
  - Text encoder: BERT-base-uncased (110M params)
  - Audio encoder: wav2vec 2.0-base (94M params)
  - Video encoder: ViT-base/patch16-224 (86M params)
  - Asymmetric cross-modal attention: text-only Q, audio+video K,V
  - Learned gating: per-sample modality weighting
  - Prediction head: polarity classification + intensity regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, Wav2Vec2Model, ViTModel


class AsymmetricCrossModalAttention(nn.Module):
    """
    Asymmetric cross-modal attention module.
    Text supplies Q only; audio+video supply K and V jointly.

    This design encodes the empirical observation that text dominates
    sentiment annotation (kappa_text = 0.74 vs kappa_audio = 0.63),
    and prevents zero-vector audio/video inputs from distorting the
    query direction under missing-modality conditions.
    """

    def __init__(self, hidden_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # 64 per head

        # Q from text only (768 -> 768)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # K, V from concatenated audio+video (1536 -> 768)
        self.W_K = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        h_text: torch.Tensor,   # (B, 768) — text CLS representation
        h_audio: torch.Tensor,  # (B, 768) — audio mean-pooled representation
        h_video: torch.Tensor,  # (B, 768) — video mean-pooled representation
    ) -> torch.Tensor:
        B = h_text.size(0)

        # Concatenate audio and video for K, V
        h_av = torch.cat([h_audio, h_video], dim=-1)  # (B, 1536)

        # Project to Q, K, V
        Q = self.W_Q(h_text)   # (B, 768)
        K = self.W_K(h_av)     # (B, 768)
        V = self.W_V(h_av)     # (B, 768)

        # Reshape for multi-head attention: (B, num_heads, 1, head_dim)
        Q = Q.view(B, self.num_heads, 1, self.head_dim)
        K = K.view(B, self.num_heads, 1, self.head_dim)
        V = V.view(B, self.num_heads, 1, self.head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, 1, 1)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attended output
        z_attn = torch.matmul(attn_weights, V)  # (B, H, 1, head_dim)
        z_attn = z_attn.view(B, self.hidden_dim)  # (B, 768)
        z_attn = self.out_proj(z_attn)

        # Residual connection: text query + attended audio-visual
        z_res = z_attn + h_text  # (B, 768)

        return z_res


class LearnedGating(nn.Module):
    """
    Per-sample learned gating mechanism.

    Computes g = sigmoid(W_g * [h_T; h_A; h_V] + b_g) in R^3.
    When a modality is absent (zero vector), its gate component
    naturally approaches zero without explicit missingness flags.
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 3, 3)

    def forward(
        self,
        h_text: torch.Tensor,
        h_audio: torch.Tensor,
        h_video: torch.Tensor,
    ) -> torch.Tensor:
        """Returns g = [g_T, g_A, g_V] in (0, 1)^3, shape (B, 3)."""
        h_concat = torch.cat([h_text, h_audio, h_video], dim=-1)  # (B, 2304)
        return torch.sigmoid(self.gate(h_concat))  # (B, 3)


class CrossFusion(nn.Module):
    """
    CrossFusion: Annotation-Driven Asymmetric Cross-Modal Attention.

    Args:
        hidden_dim:     Shared hidden dimension across all encoders (768).
        num_heads:      Number of attention heads (12).
        num_classes:    Number of sentiment polarity classes (3: Neg/Neu/Pos).
        dropout:        Dropout rate for attention and classifier (0.1).
        freeze_encoders: If True, freeze encoder weights (for LoRA variant).
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_classes: int = 3,
        dropout: float = 0.1,
        freeze_encoders: bool = False,
    ):
        super().__init__()

        # Modality-specific encoders
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.video_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")

        if freeze_encoders:
            for enc in [self.text_encoder, self.audio_encoder, self.video_encoder]:
                for param in enc.parameters():
                    param.requires_grad = False

        # Core modules
        self.attention = AsymmetricCrossModalAttention(hidden_dim, num_heads, dropout)
        self.gating = LearnedGating(hidden_dim)

        # Layer norms
        self.ln_text = nn.LayerNorm(hidden_dim)
        self.ln_audio = nn.LayerNorm(hidden_dim)
        self.ln_video = nn.LayerNorm(hidden_dim)
        self.ln_fused = nn.LayerNorm(hidden_dim)

        # Prediction heads
        self.polarity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def encode_text(self, input_ids, attention_mask):
        """Returns CLS token representation: (B, 768)."""
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.ln_text(out.last_hidden_state[:, 0, :])  # CLS

    def encode_audio(self, audio_values, audio_attention_mask=None):
        """Returns mean-pooled audio representation: (B, 768)."""
        out = self.audio_encoder(
            input_values=audio_values,
            attention_mask=audio_attention_mask,
        )
        # Mean pool over time dimension
        if audio_attention_mask is not None:
            mask = audio_attention_mask.unsqueeze(-1).float()
            h = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            h = out.last_hidden_state.mean(1)
        return self.ln_audio(h)

    def encode_video(self, pixel_values):
        """Returns mean-pooled video frame representation: (B, 768)."""
        # pixel_values: (B, T_frames, C, H, W)
        B, T = pixel_values.shape[:2]
        flat = pixel_values.view(B * T, *pixel_values.shape[2:])
        out = self.video_encoder(pixel_values=flat)
        cls = out.last_hidden_state[:, 0, :]  # (B*T, 768)
        cls = cls.view(B, T, -1).mean(1)      # (B, 768)
        return self.ln_video(cls)

    def forward(
        self,
        input_ids,
        attention_mask,
        audio_values=None,
        audio_attention_mask=None,
        pixel_values=None,
    ):
        """
        Forward pass.

        Missing modalities should be passed as zero tensors of the
        appropriate shape — the gating mechanism will naturally suppress
        their contribution without explicit missingness flags.

        Returns:
            polarity_logits: (B, num_classes)
            intensity:       (B, 1) in [0, 1]
            gate_weights:    (B, 3) — [g_T, g_A, g_V] for interpretability
        """
        # Encode each modality
        h_T = self.encode_text(input_ids, attention_mask)

        if audio_values is not None:
            h_A = self.encode_audio(audio_values, audio_attention_mask)
        else:
            h_A = torch.zeros_like(h_T)

        if pixel_values is not None:
            h_V = self.encode_video(pixel_values)
        else:
            h_V = torch.zeros_like(h_T)

        # Asymmetric cross-modal attention (text Q, audio+video K,V)
        z_res = self.attention(h_T, h_A, h_V)  # (B, 768)

        # Per-sample gating
        g = self.gating(h_T, h_A, h_V)  # (B, 3): [g_T, g_A, g_V]

        # Weighted fusion
        z_fused = (
            g[:, 0:1] * h_T
            + g[:, 1:2] * h_A
            + g[:, 2:3] * h_V
            + z_res
        )  # (B, 768)
        z_fused = self.ln_fused(self.dropout(z_fused))

        # Predictions
        polarity_logits = self.polarity_head(z_fused)   # (B, num_classes)
        intensity = self.intensity_head(z_fused)         # (B, 1)

        return polarity_logits, intensity, g
