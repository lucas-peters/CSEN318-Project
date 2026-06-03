"""
Target workloads as GEMM dimensions.

Convolution layers are expressed as their im2col GEMM equivalents:
  - M = output_height × output_width (spatial output pixels)
  - N = num_filters (output channels)
  - K = filter_h × filter_w × input_channels (window size)

ResNet-50 layers from the standard architecture specification.
BERT-Base attention from the original transformer paper (d_model=768, 12 heads).
"""

from dataclasses import dataclass


@dataclass
class GEMMWorkload:
    name: str
    m: int
    n: int
    k: int
    description: str

    @property
    def macs(self) -> int:
        return self.m * self.n * self.k

    def __repr__(self):
        return f"{self.name}: GEMM({self.m}×{self.n}×{self.k}) = {self.macs:,} MACs"


# ── ResNet-50 convolution layers as GEMM ──────────────────────────────
# Format: (name, output_h, output_w, num_filters, filter_h, filter_w, in_channels)

_RESNET50_CONV_SPECS = [
    ("conv1",       112, 112, 64,  7, 7, 3),
    ("conv2_1_1",   56,  56,  64,  1, 1, 64),
    ("conv2_1_2",   56,  56,  64,  3, 3, 64),
    ("conv2_1_3",   56,  56,  256, 1, 1, 64),
    ("conv2_2_1",   56,  56,  64,  1, 1, 256),
    ("conv2_2_2",   56,  56,  64,  3, 3, 64),
    ("conv2_2_3",   56,  56,  256, 1, 1, 64),
    ("conv3_1_1",   28,  28,  128, 1, 1, 256),
    ("conv3_1_2",   28,  28,  128, 3, 3, 128),
    ("conv3_1_3",   28,  28,  512, 1, 1, 128),
    ("conv3_ds",    28,  28,  512, 1, 1, 256),
    ("conv4_1_1",   14,  14,  256, 1, 1, 512),
    ("conv4_1_2",   14,  14,  256, 3, 3, 256),
    ("conv4_1_3",   14,  14,  1024,1, 1, 256),
    ("conv4_ds",    14,  14,  1024,1, 1, 512),
    ("conv5_1_1",   7,   7,   512, 1, 1, 1024),
    ("conv5_1_2",   7,   7,   512, 3, 3, 512),
    ("conv5_1_3",   7,   7,   2048,1, 1, 512),
    ("conv5_ds",    7,   7,   2048,1, 1, 1024),
]

RESNET50_LAYERS = []
for name, oh, ow, nf, fh, fw, ic in _RESNET50_CONV_SPECS:
    m = oh * ow
    n = nf
    k = fh * fw * ic
    RESNET50_LAYERS.append(GEMMWorkload(
        name=f"resnet50_{name}",
        m=m, n=n, k=k,
        description=f"ResNet-50 {name}: {oh}×{ow} out, {fh}×{fw} filter, {ic}→{nf}ch"
    ))

# Representative subset for sweeps (one from each stage)
RESNET50_REPRESENTATIVE = [
    RESNET50_LAYERS[0],   # conv1: large spatial, small channels
    RESNET50_LAYERS[2],   # conv2_1_2: 56×56, 3×3, 64→64
    RESNET50_LAYERS[8],   # conv3_1_2: 28×28, 3×3, 128→128
    RESNET50_LAYERS[12],  # conv4_1_2: 14×14, 3×3, 256→256
    RESNET50_LAYERS[16],  # conv5_1_2: 7×7, 3×3, 512→512
]


# ── BERT-Base attention layers ────────────────────────────────────────
# d_model=768, num_heads=12, d_head=64, seq_len parameterized

def bert_attention_workloads(seq_len: int = 512) -> list[GEMMWorkload]:
    d_head = 64
    return [
        GEMMWorkload(
            name=f"bert_qk_seq{seq_len}",
            m=seq_len, n=seq_len, k=d_head,
            description=f"BERT Q×K^T: seq={seq_len}, d_head={d_head}"
        ),
        GEMMWorkload(
            name=f"bert_av_seq{seq_len}",
            m=seq_len, n=d_head, k=seq_len,
            description=f"BERT Attn×V: seq={seq_len}, d_head={d_head}"
        ),
        GEMMWorkload(
            name=f"bert_qkv_proj_seq{seq_len}",
            m=seq_len, n=768, k=768,
            description=f"BERT QKV projection: seq={seq_len}, 768→768"
        ),
    ]


BERT_LAYERS = bert_attention_workloads(512)


# ── BERT at seq_len = 2048 (matches the report table) ───────────────

BERT_LAYERS_2048 = bert_attention_workloads(2048)


# ── Exact workloads from the mid-project report table ────────────────
# These five workloads are the primary targets for Experiments 1–4.
# "ResNet conv3" (M=3136, N=256, K=576) is a custom variant: 56×56 spatial,
# 3×3 filter with 64 input channels producing 256 output channels.

REPORT_WORKLOADS = [
    # ResNet-50 stages 1, 3, 5 (representative compute-bound ↔ comm-bound range)
    GEMMWorkload("resnet50_conv1",         12544, 64,  147,
                 "ResNet-50 conv1: 112×112, 7×7, 3→64ch"),
    GEMMWorkload("resnet50_conv3",          3136, 256, 576,
                 "ResNet-50 conv3: 56×56, 3×3, 64→256ch (custom)"),
    GEMMWorkload("resnet50_conv5",            49, 512, 4608,
                 "ResNet-50 conv5: 7×7, 3×3, 512→512ch"),
    # BERT-Base attention (seq_len = 2048 as in the report table)
    GEMMWorkload("bert_qk_seq2048",         2048, 2048, 64,
                 "BERT Q×Kᵀ: seq=2048, d_head=64"),
    GEMMWorkload("bert_qkv_proj_seq2048",   2048, 768, 768,
                 "BERT QKV projection: seq=2048, 768→768"),
]


# ── All eight validation workloads (Experiment 5) ────────────────────
# Five ResNet stages + three BERT attention ops at seq=2048.

VALIDATION_WORKLOADS = RESNET50_REPRESENTATIVE + BERT_LAYERS_2048


# ── All workloads for sweeps ──────────────────────────────────────────

ALL_WORKLOADS = RESNET50_REPRESENTATIVE + BERT_LAYERS
