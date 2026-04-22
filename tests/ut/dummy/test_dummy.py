from tests.ut.conftest import npu_test  # noqa E402


import unittest
from unittest.mock import patch

import torch


class TestWeightLoader(unittest.TestCase):
    """Test cases for weight_loader function in kv_c8.py"""

    def setUp(self):
        """Set up test environment before each test"""
        # Import the module under test
        from vllm_ascend.quantization.methods.kv_c8 import _fa_quant_weight_loader as weight_loader

        self.weight_loader = weight_loader

        # Mock distributed functions
        self.tp_rank_patch = patch("vllm_ascend.quantization.methods.kv_c8.get_tensor_model_parallel_rank")
        self.tp_size_patch = patch("vllm_ascend.quantization.methods.kv_c8.get_tensor_model_parallel_world_size")
        self.mock_tp_rank = self.tp_rank_patch.start()
        self.mock_tp_size = self.tp_size_patch.start()

    def tearDown(self):
        """Clean up after each test"""
        self.tp_rank_patch.stop()
        self.tp_size_patch.stop()

    def test_weight_loader_single_element(self):
        """Test weight_loader when both tensors contain a single element"""
        # Create tensors with single element
        param = torch.tensor([0.0])
        loaded_weight = torch.tensor([5.0])

        # Call weight_loader
        self.weight_loader(param, loaded_weight)

        # Verify the value was filled correctly
        self.assertEqual(param.item(), 5.0)
        self.assertEqual(param.dtype, torch.float32)

    def test_weight_loader_single_element_int(self):
        """Test weight_loader with integer tensors"""
        param = torch.tensor([0], dtype=torch.int32)
        loaded_weight = torch.tensor([10], dtype=torch.int32)

        self.weight_loader(param, loaded_weight)

        self.assertEqual(param.item(), 10)

    def test_weight_loader_tp_sharding_first_rank(self):
        """Test weight_loader with tensor parallelism sharding for first rank"""
        # Configure mocks for rank 0 of 4
        self.mock_tp_rank.return_value = 0
        self.mock_tp_size.return_value = 4

        # Create test tensors
        param = torch.zeros(2, 5)  # Target param shape (2,5)
        loaded_weight = torch.ones(8, 5)  # Full weight (8,5)

        # Mock narrow to track the call
        with patch.object(loaded_weight, "narrow", wraps=loaded_weight.narrow) as mock_narrow:
            self.weight_loader(param, loaded_weight)

            # Verify narrow was called correctly: narrow(dim=0, start=0, length=2)
            mock_narrow.assert_called_once_with(0, 0, 2)

            # Verify data was copied
            self.assertTrue(torch.all(param == 1))


@npu_test(num_npus=1, npu_type="a2")
def test_dummy():
    assert True


@npu_test(num_npus=2, npu_type="a3")
def test_dummy_with_a3():
    assert True


def test_dummy_without_npu():
    assert True


@npu_test(num_npus=1, npu_type="310p")
def test_dummy_with_310():
    assert True
