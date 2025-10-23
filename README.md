# FlashAttention CUDA Implementation

A minimal, educational implementation of the [FlashAttention](https://arxiv.org/abs/2205.14135) algorithm in CUDA with PyTorch integration. This implementation demonstrates the core concepts of memory-efficient attention through tiling and online softmax normalization.

## Features

- ✅ Memory-efficient attention using block-wise computation
- ✅ Online softmax with running statistics (avoids storing full attention matrix)
- ✅ PyTorch C++ extension for seamless integration
- ✅ Numerically verified against PyTorch's scaled_dot_product_attention
- ⚠️ Backward pass (in progress - not yet functional)
- 📚 Educational codebase with detailed comments

## Algorithm Overview

FlashAttention computes attention without materializing the full N×N attention matrix by:
1. **Tiling**: Breaking Q, K, V into blocks that fit in SRAM
2. **Online Softmax**: Maintaining running max and sum statistics
3. **Incremental Output**: Computing output incrementally as blocks are processed

**Memory Complexity**: O(N) instead of O(N²)

## Requirements

- CUDA Toolkit (10.0+)
- PyTorch (1.12.0+)
- Python 3.7+
- g++ compiler
- NVIDIA GPU with compute capability 6.0+

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/dino65-dev/flash_attention.git
cd flash_attention
```

### 2. Create virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install PyTorch
```bash
# CPU-only (for testing interface)
pip install torch

# Or with CUDA support (for GPU execution)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 4. Build the CUDA extension

First, set environment variables:
```bash
export CUDA_HOME=/usr/local/cuda  # Adjust if your CUDA is elsewhere
export CXX=g++
```

Then install:
```bash
pip install -e .
```

**Note**: The build process will compile the CUDA kernel for your GPU architecture. Edit `setup.py` to change the target architecture:
```python
'-arch=sm_75',  # T4, RTX 2080
# Common values: sm_61 (GTX 1050), sm_75 (T4), sm_80 (A100), sm_86 (RTX 3090)
```

## Usage

### Python API

```python
import torch
from flash_attention import FlashAttention

# Create attention module
attn = FlashAttention(head_dim=64)

# Prepare inputs [batch, heads, seq_len, head_dim]
batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64
Q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
K = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
V = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()

# Compute attention
output = attn(Q, K, V)
```

### Functional Interface

```python
from flash_attention import flash_attention

output = flash_attention(Q, K, V)
```

### Test the Implementation

```bash
python flash_attention.py
```

Expected output:
```
Testing FlashAttention...
Using device: cuda
Input shapes: Q=torch.Size([2, 4, 128, 64]), ...
Output shape: torch.Size([2, 4, 128, 64])
Max difference from PyTorch: 0.000001
✓ Results match PyTorch!
```

## GPU Architecture Support

| GPU Model      | Compute Capability | setup.py flag |
|----------------|-------------------|---------------|
| GTX 1050 Ti    | 6.1               | `-arch=sm_61` |
| T4             | 7.5               | `-arch=sm_75` |
| V100           | 7.0               | `-arch=sm_70` |
| A100           | 8.0               | `-arch=sm_80` |
| RTX 3090       | 8.6               | `-arch=sm_86` |
| RTX 4090       | 8.9               | `-arch=sm_89` |

Check your GPU's compute capability:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## Limitations

This is an educational implementation with the following limitations:

- ⚠️ **Head dimension**: Only supports `head_dim=64`
- ⚠️ **Attention masks**: Causal masking not implemented
- ⚠️ **Backward pass**: Gradient computation in progress (DO NOT USE FOR TRAINING)
- ⚠️ **Dropout**: Not supported
- ⚠️ **Block sizes**: Fixed at compile time (Br=32, Bc=32)
- ⚠️ **Data type**: Only supports FP32

For production use, consider the official [FlashAttention repository](https://github.com/Dao-AILab/flash-attention).

## Troubleshooting

### "CUDA error: no kernel image is available"
Your GPU architecture doesn't match the compiled kernel. Update `setup.py` with your GPU's compute capability and rebuild:
```bash
pip install --force-reinstall -e .
```

### "module 'flash_attention' has no attribute 'forward'"
The extension wasn't compiled with PyTorch bindings. Make sure you have the latest code and rebuild.

### Numerical differences
Small differences (< 1e-3) are expected due to floating-point precision. Larger differences indicate a bug.

## File Structure

```
flash-attention-cuda/
├── flash_attention.cu      # CUDA kernel implementation
├── flash_attention.py      # Python wrapper and testing
├── setup.py               # Build configuration
├── README.md              # This file
├── .gitignore            # Git ignore rules
└── LICENSE               # MIT License
```

## How It Works

The implementation follows the FlashAttention algorithm:

1. **Tiling**: Q is divided into blocks of size Br×d, K and V into Bc×d
2. **For each Q block**:
   - Initialize output accumulator and statistics (max, sum)
   - **For each K, V block**:
     - Load blocks into shared memory
     - Compute attention scores: S = Q @ K^T
     - Apply online softmax with running statistics
     - Accumulate output: O += softmax(S) @ V
3. **Final normalization**: Divide by final sum

## Performance Notes

This implementation prioritizes clarity over performance. Optimizations in the official FlashAttention include:
- Warp-level primitives for faster memory access
- FP16/BF16 computation
- Better memory coalescing
- Optimized block sizes per GPU architecture

## References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135) - Dao et al., 2022
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Dao, 2023
- [Official Implementation](https://github.com/Dao-AILab/flash-attention)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If you use this code for educational purposes, please cite the original FlashAttention paper:

```bibtex
@inproceedings{dao2022flashattention,
  title={Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```
