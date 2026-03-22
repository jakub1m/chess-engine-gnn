# SAGE: Chess Engine with GNN + Mixture of Experts

A neural network chess engine combining **Graph Neural Networks** with **Sparse Mixture of Experts** routing.

## 🚀 What's Inside

**PyTorch + PyG** — Chess board as a heterogeneous graph (65 nodes, semantic edge features)

**GATeau Layers** — Edge-aware multi-head attention for position understanding

**DS-MoE** — Differentiable sparse routing: all experts train, top-1 routes at inference

**Dual Heads** — Simultaneous position evaluation + move prediction

## 🔧 Tech Stack

- **PyTorch 2.10** + CUDA 13.0
- **PyTorch Geometric 2.7** (heterogeneous graphs)
- **python-chess** + **Stockfish**

## 📖 Implementation

See `chess_engine_implementation.ipynb` for the complete pipeline:
- Graph encoding (FEN → HeteroData)
- Model architecture (GATeau + DS-MoE blocks)
- Training loop with dual-task learning
