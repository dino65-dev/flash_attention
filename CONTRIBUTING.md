# Contributing to FlashAttention CUDA

Thank you for your interest in contributing! This is an educational project aimed at helping people understand the FlashAttention algorithm.

## How to Contribute

### Reporting Bugs

- Check if the issue already exists in GitHub Issues
- Include your GPU model, CUDA version, and PyTorch version
- Provide a minimal reproducible example
- Include the error message and stack trace

### Suggesting Enhancements

- Open an issue with the `enhancement` label
- Clearly describe the feature and its benefits
- Consider whether it fits the educational focus of the project

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make your changes**:
   - Add comments explaining complex CUDA code
   - Update README.md if adding new features
   - Ensure code compiles without warnings
4. **Test your changes**: Run `python flash_attention.py` and verify results match PyTorch
5. **Commit**: Use clear, descriptive commit messages
6. **Push**: `git push origin feature/your-feature`
7. **Open a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/flash-attention-cuda.git
cd flash-attention-cuda

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in editable mode
pip install -e .

# Run tests
python flash_attention.py
```

## Code Style

- **CUDA**: Follow CUDA best practices, add comments explaining block/thread logic
- **Python**: Follow PEP 8, use type hints where helpful
- **Comments**: Prioritize clarity - this is an educational project

## Ideas for Contributions

### Easy
- [ ] Add support for different block sizes (Br, Bc)
- [ ] Add benchmarking script comparing with PyTorch
- [ ] Improve error messages
- [ ] Add more examples

### Medium
- [ ] Support for different head dimensions (32, 128)
- [ ] Add causal masking support
- [ ] FP16/BF16 support
- [ ] Better shared memory utilization

### Advanced
- [ ] Backward pass implementation
- [ ] Multi-GPU support
- [ ] Integration with torch.compile
- [ ] Performance optimizations (warp-level primitives)

## Testing

When making changes to the CUDA kernel:

1. **Correctness**: Verify output matches PyTorch (max diff < 1e-3)
2. **Edge cases**: Test with different sequence lengths and batch sizes
3. **Memory**: Ensure no CUDA memory leaks

```python
# Basic test
python flash_attention.py

# Test different sizes
python -c "
import torch
from flash_attention import flash_attention
for seq in [64, 128, 256, 512]:
    Q = torch.randn(1, 1, seq, 64).cuda()
    K = torch.randn(1, 1, seq, 64).cuda()
    V = torch.randn(1, 1, seq, 64).cuda()
    out = flash_attention(Q, K, V)
    print(f'seq_len={seq}: ✓')
"
```

## Questions?

Open an issue with the `question` label or start a discussion.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
