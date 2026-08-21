# PyTorch Katas

Small exercises to sharpen my PyTorch Skills in the age of agents. No aim for completeness, purely based off of what I find interesting.
I will use agents for downloading datasets, visualizations and other stuff that does not matter too much

# Topics

## Basics

[x] ResNet
[] AdamW
[] AMP training
[] Transformer from Scratch
[] VAE for Cifar10 - not sure whether it is interesting for such small resolutions
[] VAE for videos
[] Flow matching
[] Simple MoE with deepseek style load balancing

## Self-supervised Learning

[] MAE
[] LeJepa from Scratch

## Generative + SSL

[x] SIGReg on Self-Flow token embeddings (drop the EMA teacher)

Self-Flow's teacher/EMA pair is a JEPA mechanism. This kata puts LeJEPA's SIGReg on a tiny SiT's own token embeddings and compares four recipes: vanilla flow matching, Self-Flow (EMA), SIGReg-only (no teacher), and LeJEPA-style prediction + SIGReg (no EMA).

Verdict: SIGReg can replace the anti-collapse hack, not the dual-timestep predictive task. Bare Gaussianization of SiT tokens will not invent DINO-like semantics; keep Dual-Timestep Scheduling and predict the cleaner view with the same network.

Notebook: `notebooks/self_flow/self_flow_sigreg.ipynb`

## LLMs

[] GPT-2
[] n-dim RoPE

## Computer Vision

[] ViT
[] UNet
[] DiT
[] ArcFace loss

## Distributed Training & Efficiency

[] LoRA
[] DDP on 2 nodes
[] FSDP

## RL

[] PPO from Scratch
[] PPO for the lux challenges
