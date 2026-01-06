"""
Tests for IntSeqDecoder module.
Tests Heteroscedastic Regression heads, Mod Spectrum heads, and Beam Search CRT solver.
"""

import pytest
import torch
import math

from intseq_bert.decoder_model import (
    IntSeqDecoder,
    extended_gcd,
    solve_congruence,
    MOD_RANGE
)


# ==========================================
# Helper Functions
# ==========================================

def create_mock_latent(batch_size: int = 2, d_model: int = 128):
    """Create mock latent vector input."""
    return torch.randn(batch_size, d_model)


@pytest.fixture
def sample_decoder():
    """Create a small decoder for testing."""
    return IntSeqDecoder(d_model=128, hidden_dim=256, dropout=0.1)


# ==========================================
# 1. Math Utility Tests
# ==========================================

class TestExtendedGcd:
    """Tests for extended_gcd function."""
    
    def test_gcd_coprime(self):
        """Test GCD of coprime numbers."""
        g, x, y = extended_gcd(3, 7)
        assert g == 1
        assert 3 * x + 7 * y == g
    
    def test_gcd_with_common_factor(self):
        """Test GCD of numbers with common factor."""
        g, x, y = extended_gcd(12, 18)
        assert g == 6
        assert 12 * x + 18 * y == g
    
    def test_gcd_with_zero(self):
        """Test GCD when one number is zero."""
        g, x, y = extended_gcd(0, 5)
        assert g == 5
        assert 0 * x + 5 * y == g


class TestSolveCongruence:
    """Tests for solve_congruence function (CRT solver)."""
    
    def test_simple_congruence(self):
        """Test simple CRT problem."""
        # x = 1 (mod 3), x = 2 (mod 5)
        # Solution: x = 7 (mod 15)
        x, lcm = solve_congruence(1, 3, 2, 5)
        assert lcm == 15
        assert x == 7
    
    def test_coprime_moduli(self):
        """Test with coprime moduli."""
        # x = 2 (mod 3), x = 3 (mod 7)
        x, lcm = solve_congruence(2, 3, 3, 7)
        assert lcm == 21
        assert x % 3 == 2
        assert x % 7 == 3
    
    def test_inconsistent_system(self):
        """Test with inconsistent system."""
        # x = 1 (mod 2), x = 0 (mod 2) -> impossible
        x, lcm = solve_congruence(1, 2, 0, 2)
        assert x is None
        assert lcm == 2
    
    def test_non_coprime_moduli_consistent(self):
        """Test with non-coprime moduli but consistent."""
        # x = 2 (mod 4), x = 6 (mod 6)
        # gcd(4,6)=2, (6-2)=4 divisible by 2 -> solvable
        x, lcm = solve_congruence(2, 4, 6, 6)
        assert x is not None
        assert x % 4 == 2
        assert x % 6 == 6 % 6  # 0


# ==========================================
# 2. Decoder Initialization Tests
# ==========================================

class TestDecoderInitialization:
    """Tests for IntSeqDecoder initialization."""
    
    def test_default_initialization(self):
        """Test decoder initializes with default parameters."""
        decoder = IntSeqDecoder()
        assert decoder.d_model == 128
    
    def test_custom_initialization(self):
        """Test decoder initializes with custom parameters."""
        decoder = IntSeqDecoder(d_model=256, hidden_dim=512, dropout=0.2)
        assert decoder.d_model == 256
    
    def test_has_required_components(self, sample_decoder):
        """Test decoder has all required components."""
        assert hasattr(sample_decoder, 'trunk')
        assert hasattr(sample_decoder, 'mag_head')
        assert hasattr(sample_decoder, 'sign_head')
        assert hasattr(sample_decoder, 'mod_heads')
    
    def test_mod_heads_count(self, sample_decoder):
        """Test decoder has correct number of mod heads."""
        # MOD_RANGE is 2..101, so 100 heads
        assert len(sample_decoder.mod_heads) == 100
        assert 'mod2' in sample_decoder.mod_heads
        assert 'mod101' in sample_decoder.mod_heads
    
    def test_mod_head_output_dims(self, sample_decoder):
        """Test each mod head has correct output dimension."""
        for m in MOD_RANGE:
            head = sample_decoder.mod_heads[f"mod{m}"]
            assert head.out_features == m


# ==========================================
# 3. Forward Pass Tests
# ==========================================

class TestForwardPass:
    """Tests for forward pass behavior."""
    
    def test_output_keys(self, sample_decoder):
        """Test forward returns correct keys."""
        x = create_mock_latent()
        output = sample_decoder(x)
        
        assert 'mag_mu' in output
        assert 'mag_logvar' in output
        assert 'sign_logits' in output
        
        # Check mod heads
        for m in MOD_RANGE:
            assert f"mod{m}" in output
    
    def test_output_shapes(self, sample_decoder):
        """Test output shapes are correct."""
        batch_size = 4
        x = create_mock_latent(batch_size=batch_size)
        output = sample_decoder(x)
        
        assert output['mag_mu'].shape == (batch_size, 1)
        assert output['mag_logvar'].shape == (batch_size, 1)
        assert output['sign_logits'].shape == (batch_size, 3)
        
        # Check some mod head shapes
        assert output['mod2'].shape == (batch_size, 2)
        assert output['mod5'].shape == (batch_size, 5)
        assert output['mod100'].shape == (batch_size, 100)
        assert output['mod101'].shape == (batch_size, 101)


# ==========================================
# 4. Loss Computation Tests
# ==========================================

class TestLossComputation:
    """Tests for compute_loss method."""
    
    def test_loss_computation(self, sample_decoder):
        """Test basic loss computation."""
        x = create_mock_latent(batch_size=4)
        predictions = sample_decoder(x)
        
        # Create mock targets
        targets = {
            'mag': torch.randn(4),  # log10 magnitude
        }
        # Add some mod targets
        for m in [2, 3, 5, 10]:
            targets[f"mod{m}"] = torch.randint(0, m, (4,))
        
        loss = sample_decoder.compute_loss(predictions, targets)
        
        assert loss.shape == ()  # Scalar
        assert not torch.isnan(loss)
        assert loss.item() >= 0
    
    def test_loss_with_ignore_index(self, sample_decoder):
        """Test loss ignores padding tokens (-100)."""
        x = create_mock_latent(batch_size=2)
        predictions = sample_decoder(x)
        
        targets = {
            'mag': torch.tensor([1.0, 2.0]),
            'mod3': torch.tensor([1, -100]),  # Second sample is padding
        }
        
        loss = sample_decoder.compute_loss(predictions, targets)
        
        # Should not crash and should return valid loss
        assert not torch.isnan(loss)


# ==========================================
# 5. Beam Search Solver Tests
# ==========================================

class TestBeamSearchSolver:
    """Tests for beam_search_solve method."""
    
    def test_solver_returns_list(self, sample_decoder):
        """Test solver returns list of tuples."""
        sample_decoder.eval()
        x = create_mock_latent(batch_size=1)
        
        with torch.no_grad():
            predictions = sample_decoder(x)
            results = sample_decoder.beam_search_solve(predictions, beam_width=5)
        
        assert isinstance(results, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    
    def test_solver_handles_zero_sign(self, sample_decoder):
        """Test solver handles zero prediction correctly."""
        sample_decoder.eval()
        x = create_mock_latent(batch_size=1)
        
        with torch.no_grad():
            predictions = sample_decoder(x)
            # Force sign to be zero
            predictions['sign_logits'] = torch.tensor([[0.0, 100.0, 0.0]])  # Strong zero
            
            results = sample_decoder.beam_search_solve(predictions)
        
        # Should return [(0, score)]
        assert len(results) == 1
        assert results[0][0] == 0
    
    def test_solver_returns_integers(self, sample_decoder):
        """Test solver returns integer values."""
        sample_decoder.eval()
        x = create_mock_latent(batch_size=1)
        
        with torch.no_grad():
            predictions = sample_decoder(x)
            # Force positive sign
            predictions['sign_logits'] = torch.tensor([[0.0, 0.0, 100.0]])
            results = sample_decoder.beam_search_solve(predictions, beam_width=10)
        
        for val, score in results:
            assert isinstance(val, int)
            assert isinstance(score, float)


# ==========================================
# 6. Gradient Flow Tests
# ==========================================

class TestGradientFlow:
    """Tests for gradient flow through the decoder."""
    
    def test_gradients_flow_through_loss(self, sample_decoder):
        """Test that gradients flow through loss computation."""
        x = create_mock_latent(batch_size=2)
        predictions = sample_decoder(x)
        
        targets = {
            'mag': torch.randn(2),
            'mod3': torch.randint(0, 3, (2,)),
            'mod5': torch.randint(0, 5, (2,)),
        }
        
        loss = sample_decoder.compute_loss(predictions, targets)
        loss.backward()
        
        # Check gradients exist
        assert sample_decoder.trunk[0].weight.grad is not None
        assert sample_decoder.mag_head.weight.grad is not None


# ==========================================
# 7. Edge Cases
# ==========================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_item_batch(self, sample_decoder):
        """Test with batch size of 1."""
        x = create_mock_latent(batch_size=1)
        output = sample_decoder(x)
        
        assert output['mag_mu'].shape == (1, 1)
    
    def test_large_batch(self, sample_decoder):
        """Test with large batch."""
        x = create_mock_latent(batch_size=64)
        output = sample_decoder(x)
        
        assert output['mag_mu'].shape == (64, 1)
    
    def test_eval_mode(self, sample_decoder):
        """Test decoder works in eval mode."""
        sample_decoder.eval()
        x = create_mock_latent()
        
        with torch.no_grad():
            output = sample_decoder(x)
        
        assert 'mag_mu' in output


# ==========================================
# 8. MOD_RANGE Configuration Tests
# ==========================================

class TestModRangeConfig:
    """Tests for MOD_RANGE configuration."""
    
    def test_mod_range_coverage(self):
        """Test that MOD_RANGE covers 2 to 101."""
        mod_list = list(MOD_RANGE)
        
        assert mod_list[0] == 2
        assert mod_list[-1] == 101
        assert len(mod_list) == 100