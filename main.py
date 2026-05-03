import torch
from transformer import pipeline

pipeline = pipeline(
    task="image-classification",
    model="microsoft/swinv2-tiny-patch4-window8-256",
    dtype=torch.float16,
    device=0
)