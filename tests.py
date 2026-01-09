import unittest
import torch
import torch.nn as nn
import math
from unittest.mock import MagicMock, patch

# Import the classes from the provided code snippet
# Assuming the code is in a module named 'solution' or pasted here.
# For the test runner, we will redefine the necessary classes if they aren't imported.

class SNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, depth, dropout_rate=0.05):
        super(SNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(depth):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SELU()
        self.dropout = nn.AlphaDropout(p=dropout_rate)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=math.sqrt(1.0 / m.in_features))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
            x = self.dropout(x)
        return self.output(x)

class TestSNNImplementation(unittest.TestCase):
    def setUp(self):
        self.input_dim = 100
        self.output_dim = 10
        self.hidden_dim = 64
        self.depth = 3
        self.batch_size = 16
        self.model = SNN(self.input_dim, self.output_dim, self.hidden_dim, self.depth)

    def test_architecture_depth(self):
        """Verify the number of layers matches configuration."""
        # 1 input layer + 'depth' hidden layers = 1 + 3 = 4 linear layers in self.layers
        # plus 1 output layer separate.
        self.assertEqual(len(self.model.layers), 1 + self.depth)
        self.assertIsInstance(self.model.act, nn.SELU)
        self.assertIsInstance(self.model.dropout, nn.AlphaDropout)

    def test_weight_initialization(self):
        """Check if weights are initialized roughly according to LeCun Normal."""
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                # Expected std = sqrt(1 / fan_in)
                fan_in = layer.in_features
                expected_std = math.sqrt(1.0 / fan_in)
                
                # Statistical check (loose bounds due to randomness)
                weight_std = layer.weight.data.std().item()
                weight_mean = layer.weight.data.mean().item()
                
                self.assertTrue(abs(weight_mean) < 0.05, f"Mean should be ~0, got {weight_mean}")
                # Allow 20% margin of error for small tensor stats
                self.assertTrue(abs(weight_std - expected_std) / expected_std < 0.2, 
                                f"Std deviation mismatch. Expected {expected_std}, got {weight_std}")
                
                if layer.bias is not None:
                    self.assertEqual(layer.bias.data.sum().item(), 0)

    def test_forward_pass_shape(self):
        """Ensure forward pass works and returns correct shape."""
        dummy_input = torch.randn(self.batch_size, self.input_dim)
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_forward_pass_image_shape(self):
        """Ensure model handles unflattened image input (Batch, C, H, W)."""
        # SNN forward method has .view(x.size(0), -1) to flatten
        dummy_image = torch.randn(self.batch_size, 1, 10, 10) # 10x10 = 100 dim
        output = self.model(dummy_image)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_backward_pass(self):
        """Ensure gradients flow through the network."""
        dummy_input = torch.randn(self.batch_size, self.input_dim)
        dummy_target = torch.randint(0, self.output_dim, (self.batch_size,))
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        output = self.model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        
        # Check if weights changed
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

if __name__ == '__main__':
    unittest.main()