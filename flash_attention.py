"""
FlashAttention CUDA Implementation

A PyTorch wrapper for a CUDA-based FlashAttention implementation.
This provides memory-efficient attention computation using the tiling
and online softmax techniques from the FlashAttention paper.

Example:
    >>> import torch
    >>> from flash_attention import FlashAttention
    >>> 
    >>> attn = FlashAttention(head_dim=64)
    >>> Q = torch.randn(2, 8, 512, 64).cuda()
    >>> K = torch.randn(2, 8, 512, 64).cuda()
    >>> V = torch.randn(2, 8, 512, 64).cuda()
    >>> output = attn(Q, K, V)

References:
    FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
    Dao et al., 2022 - https://arxiv.org/abs/2205.14135
"""

import torch
import torch.nn as nn

# Try to import the compiled CUDA extension
try:
    import flash_attention as flash_attention_cuda
except ImportError:
    flash_attention_cuda = None
    print("Warning: flash_attention CUDA extension not found. Did you compile it?")


class FlashAttentionFunction(torch.autograd.Function):
    """
    Custom autograd function for FlashAttention.
    """
    
    @staticmethod
    def forward(ctx, Q, K, V):
        """
        Forward pass for FlashAttention.
        
        Args:
            ctx: Context object to save tensors for backward pass
            Q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            K: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            V: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
            
        Returns:
            O: Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        # Forward returns [O, l, m]
        O, l, m = flash_attention_cuda.forward(Q, K, V)
        ctx.save_for_backward(Q, K, V, O, l, m)
        return O
    
    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass for FlashAttention.
        
        WARNING: Backward pass is currently not functional.
        DO NOT USE THIS FOR TRAINING.
        
        Args:
            ctx: Context object with saved tensors
            dO: Gradient of output
            
        Returns:
            Gradients for Q, K, V
        """
        raise NotImplementedError(
            "Backward pass is in progress and not yet functional. "
            "The CUDA kernel encounters launch errors on current GPUs. "
            "Use PyTorch's native attention for training."
        )


class FlashAttention(nn.Module):
    """
    Flash Attention module with PyTorch-like interface.
    
    Args:
        head_dim: Dimension of each attention head (must be 64)
        dropout: Dropout probability (not implemented)
    """
    
    def __init__(self, head_dim=64, dropout=0.0):
        super().__init__()
        assert head_dim == 64, "Only head_dim=64 is supported"
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
    def forward(self, Q, K, V, attn_mask=None):
        """
        Args:
            Q: [batch, heads, seq_len, head_dim]
            K: [batch, heads, seq_len, head_dim]
            V: [batch, heads, seq_len, head_dim]
            attn_mask: Not supported yet
            
        Returns:
            output: [batch, heads, seq_len, head_dim]
        """
        if attn_mask is not None:
            raise NotImplementedError("Attention masks not supported yet")
        
        # Ensure correct dtype
        Q = Q.contiguous().float()
        K = K.contiguous().float()
        V = V.contiguous().float()
        
        # Call custom CUDA kernel
        output = FlashAttentionFunction.apply(Q, K, V)
        
        return output


def flash_attention(Q, K, V):
    """
    Functional interface for FlashAttention.
    
    Args:
        Q, K, V: Tensors of shape [batch, heads, seq_len, head_dim]
    
    Returns:
        Output tensor of shape [batch, heads, seq_len, head_dim]
    
    Example:
        >>> Q = torch.randn(2, 8, 512, 64).cuda()
        >>> K = torch.randn(2, 8, 512, 64).cuda()
        >>> V = torch.randn(2, 8, 512, 64).cuda()
        >>> output = flash_attention(Q, K, V)
    """
    module = FlashAttention()
    return module(Q, K, V)


if __name__ == "__main__":
    # Test the implementation
    print("Testing FlashAttention...")
    
    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64
    
    # Create random tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("Warning: CUDA not available, using CPU (PyTorch native attention only)")
    
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    
    print(f"Input shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
    
    # Test FlashAttention only if CUDA extension is available
    if flash_attention_cuda is not None and device == 'cuda':
        try:
            flash_attn = FlashAttention(head_dim=64).to(device)
            output = flash_attn(Q, K, V)
            
            print(f"Output shape: {output.shape}")
            print(f"Output mean: {output.mean():.6f}, std: {output.std():.6f}")
            
            # Compare with PyTorch's scaled_dot_product_attention
            with torch.no_grad():
                pytorch_output = torch.nn.functional.scaled_dot_product_attention(
                    Q, K, V, dropout_p=0.0, is_causal=False
                )
            
            # Check if results are close
            max_diff = (output - pytorch_output).abs().max()
            print(f"Max difference from PyTorch: {max_diff:.6f}")
            
            if max_diff < 1e-3:
                print("✓ Results match PyTorch!")
            else:
                print("✗ Results differ significantly")
        except AttributeError as e:
            print(f"Error: {e}")
            print("\nThe flash_attention module exists but doesn't have the 'forward' function.")
            print("This means the extension needs to be rebuilt on this machine.")
            print("\nTo fix this:")
            print("1. Make sure setup.py has the correct GPU architecture (e.g., -arch=sm_75 for T4)")
            print("2. Rebuild the extension:")
            print("   cd /path/to/flash_attention")
            print("   source venv/bin/activate")
            print("   export CUDA_HOME=/usr/local/cuda")
            print("   export CXX=g++")
            print("   pip install --no-build-isolation --force-reinstall -e .")
            print("\nSee T4_INSTALLATION.md for detailed instructions.")
            
            # Fallback to PyTorch native
            print("\nRunning PyTorch native attention as fallback:")
            with torch.no_grad():
                pytorch_output = torch.nn.functional.scaled_dot_product_attention(
                    Q, K, V, dropout_p=0.0, is_causal=False
                )
            print(f"PyTorch output shape: {pytorch_output.shape}")
            print(f"PyTorch output mean: {pytorch_output.mean():.6f}, std: {pytorch_output.std():.6f}")
        except Exception as e:
            print(f"Error running FlashAttention CUDA kernel: {e}")
            print("\nThis could be due to:")
            print("1. GPU compute capability incompatibility (wrong -arch in setup.py)")
            print("2. Extension not rebuilt on this machine")
            print("3. CUDA/PyTorch version mismatch")
            print("\nCheck T4_INSTALLATION.md for troubleshooting steps.")
            
            # Fallback to PyTorch native
            print("\nRunning PyTorch native attention as fallback:")
            with torch.no_grad():
                pytorch_output = torch.nn.functional.scaled_dot_product_attention(
                    Q, K, V, dropout_p=0.0, is_causal=False
                )
            print(f"PyTorch output shape: {pytorch_output.shape}")
            print(f"PyTorch output mean: {pytorch_output.mean():.6f}, std: {pytorch_output.std():.6f}")
    else:
        print("\nFlashAttention CUDA extension not available.")
        print("Running PyTorch native attention instead:")
        with torch.no_grad():
            pytorch_output = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, dropout_p=0.0, is_causal=False
            )
        print(f"PyTorch output shape: {pytorch_output.shape}")
        print(f"PyTorch output mean: {pytorch_output.mean():.6f}, std: {pytorch_output.std():.6f}")
    
    # Note: Backward pass test removed as backward is not functional yet
    print("\nNote: Backward pass is in progress and not available for use.")
