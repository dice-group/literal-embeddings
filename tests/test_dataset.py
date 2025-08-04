import unittest
import tempfile
import os
import shutil
import pandas as pd
import torch
import numpy as np
from src.dataset import LiteralDataset


class TestLiteralDataset(unittest.TestCase):
    """Regression tests for LiteralDataset to ensure consistent behavior"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.literals_dir = os.path.join(self.test_dir, "literals")
        os.makedirs(self.literals_dir)
        
        # Create sample entity to index mapping
        self.entity_to_idx = {
            "entity_1": 0,
            "entity_2": 1,
            "entity_3": 2,
            "entity_4": 3,
            "entity_5": 4
        }
        
        # Create sample training data
        self.train_data = [
            ("entity_1", "height", 1.75),
            ("entity_1", "weight", 70.5),
            ("entity_2", "height", 1.80),
            ("entity_2", "weight", 75.0),
            ("entity_3", "height", 1.65),
            ("entity_3", "weight", 60.0),
            ("entity_4", "height", 1.90),
            ("entity_4", "weight", 85.0),
            ("entity_5", "height", 1.70),
            ("entity_5", "weight", 65.5)
        ]
        
        # Create sample test data
        self.test_data = [
            ("entity_1", "height", 1.75),
            ("entity_2", "weight", 75.0),
            ("entity_3", "height", 1.65)
        ]
        
        # Write training data
        train_df = pd.DataFrame(self.train_data, columns=["head", "relation", "tail"])
        train_df.to_csv(os.path.join(self.literals_dir, "train.txt"), 
                       sep="\t", header=False, index=False)
        
        # Write test data
        test_df = pd.DataFrame(self.test_data, columns=["head", "relation", "tail"])
        test_df.to_csv(os.path.join(self.literals_dir, "test.txt"), 
                      sep="\t", header=False, index=False)
        
        # Create empty validation file
        val_df = pd.DataFrame([], columns=["head", "relation", "tail"])
        val_df.to_csv(os.path.join(self.literals_dir, "val.txt"), 
                     sep="\t", header=False, index=False)
    
    def tearDown(self):
        """Clean up after each test"""
        shutil.rmtree(self.test_dir)
    
    def test_dataset_initialization(self):
        """Test that dataset initializes correctly with expected properties"""
        dataset = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            normalization="z-norm"
        )
        
        # Check basic properties
        self.assertEqual(dataset.num_entities, 5)
        self.assertEqual(dataset.num_data_properties, 2)  # height, weight
        self.assertEqual(len(dataset), 10)  # 10 training samples
        
        # Check data property mapping
        expected_properties = {"height", "weight"}
        actual_properties = set(dataset.data_property_to_idx.keys())
        self.assertEqual(actual_properties, expected_properties)
    
    def test_normalization_consistency(self):
        """Test that normalization produces consistent results"""
        # Test z-norm
        dataset_znorm = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            normalization="z-norm"
        )
        
        # Test min-max norm
        dataset_minmax = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            normalization="min-max"
        )
        
        # Test no normalization
        dataset_no_norm = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            normalization=None
        )
        
        # Check that normalization parameters are stored correctly
        self.assertEqual(dataset_znorm.normalization_params["type"], "z-norm")
        self.assertEqual(dataset_minmax.normalization_params["type"], "min-max")
        self.assertEqual(dataset_no_norm.normalization_params["type"], None)
        
        # For z-norm, check that normalized data has approximately mean=0, std=1 per relation
        height_indices = (dataset_znorm.triples[:, 1] == 
                         dataset_znorm.data_property_to_idx["height"])
        height_values = dataset_znorm.tails_norm[height_indices]
        
        # Check that height values are approximately normalized
        self.assertAlmostEqual(height_values.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(height_values.std().item(), 1.0, places=5)
    
    def test_log_normalization(self):
        """Test log-based normalization methods"""
        # Test log normalization
        dataset_log = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            normalization="log"
        )
        
        # Test log-z-norm
        dataset_log_znorm = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            normalization="log-z-norm"
        )
        
        # Test log-min-max
        dataset_log_minmax = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            normalization="log-min-max"
        )
        
        # Check that normalization parameters are stored correctly
        self.assertEqual(dataset_log.normalization_params["type"], "log")
        self.assertEqual(dataset_log_znorm.normalization_params["type"], "log-z-norm")
        self.assertEqual(dataset_log_minmax.normalization_params["type"], "log-min-max")
        
        # Check that epsilon is stored for log-based methods
        self.assertIn("epsilon", dataset_log.normalization_params)
        self.assertIn("epsilon", dataset_log_znorm.normalization_params)
        self.assertIn("epsilon", dataset_log_minmax.normalization_params)
    
    def test_sampling_ratio(self):
        """Test that sampling ratio works correctly"""
        # Test with 50% sampling
        dataset_half = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            sampling_ratio=0.5
        )
        
        # Should have roughly half the data
        self.assertLessEqual(len(dataset_half), 7)  # Allow some variance due to groupby sampling
        self.assertGreaterEqual(len(dataset_half), 3)
        
        # Test with full data
        dataset_full = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            sampling_ratio=1.0
        )
        
        self.assertEqual(len(dataset_full), 10)
    
    def test_selected_attributes(self):
        """Test attribute filtering functionality"""
        # Test with only height attribute
        dataset_height_only = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            selected_attributes=["height"]
        )
        
        self.assertEqual(dataset_height_only.num_data_properties, 1)
        self.assertEqual(len(dataset_height_only), 5)  # 5 height records
        self.assertIn("height", dataset_height_only.data_property_to_idx)
        self.assertNotIn("weight", dataset_height_only.data_property_to_idx)
        
        # Test with both attributes
        dataset_both = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            selected_attributes=["height", "weight"]
        )
        
        self.assertEqual(dataset_both.num_data_properties, 2)
        self.assertEqual(len(dataset_both), 10)
    
    def test_label_perturbation(self):
        """Test label perturbation functionality"""
        # Test Gaussian perturbation
        dataset_gaussian = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            label_perturbation="gaussian",
            perturbation_ratio=0.5,
            perturbation_noise_std=0.1,
            random_seed=42
        )
        
        # Check that perturbation was applied
        self.assertTrue(hasattr(dataset_gaussian, 'perturbation_stats'))
        self.assertEqual(dataset_gaussian.perturbation_stats['perturbation_type'], 'gaussian')
        self.assertEqual(dataset_gaussian.perturbation_stats['num_perturbed'], 5)  # 50% of 10
        
        # Test uniform perturbation
        dataset_uniform = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            label_perturbation="uniform",
            perturbation_ratio=0.3,
            perturbation_noise_std=0.05,
            random_seed=42
        )
        
        self.assertEqual(dataset_uniform.perturbation_stats['perturbation_type'], 'uniform')
        self.assertEqual(dataset_uniform.perturbation_stats['num_perturbed'], 3)  # 30% of 10
        
        # Test label flip perturbation
        dataset_flip = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            label_perturbation="label_flip",
            perturbation_ratio=0.2,
            random_seed=42
        )
        
        self.assertEqual(dataset_flip.perturbation_stats['perturbation_type'], 'label_flip')
        self.assertEqual(dataset_flip.perturbation_stats['num_perturbed'], 2)  # 20% of 10
    
    def test_data_consistency_across_runs(self):
        """Test that same configuration produces identical results"""
        # Create two identical datasets
        dataset1 = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            normalization="z-norm",
            random_seed=42
        )
        
        dataset2 = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            normalization="z-norm",
            random_seed=42
        )
        
        # Check that data is identical
        self.assertTrue(torch.equal(dataset1.triples, dataset2.triples))
        self.assertTrue(torch.equal(dataset1.tails, dataset2.tails))
        self.assertTrue(torch.equal(dataset1.tails_norm, dataset2.tails_norm))
        
        # Check that normalization parameters are identical
        self.assertEqual(dataset1.normalization_params, dataset2.normalization_params)
    
    def test_perturbation_reproducibility(self):
        """Test that perturbations are reproducible with same random seed"""
        dataset1 = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            label_perturbation="gaussian",
            perturbation_ratio=0.5,
            perturbation_noise_std=0.1,
            random_seed=42
        )
        
        dataset2 = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            label_perturbation="gaussian",
            perturbation_ratio=0.5,
            perturbation_noise_std=0.1,
            random_seed=42
        )
        
        # Check that perturbations are identical
        self.assertTrue(torch.equal(dataset1.tails_norm, dataset2.tails_norm))
        self.assertEqual(dataset1.perturbation_stats['perturbed_indices'], 
                        dataset2.perturbation_stats['perturbed_indices'])
    
    def test_getitem_functionality(self):
        """Test __getitem__ method returns correct format"""
        dataset = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx
        )
        
        # Test single item access
        triple, target = dataset[0]
        
        # Check types and shapes
        self.assertIsInstance(triple, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)
        self.assertEqual(triple.shape, (2,))  # [entity_idx, relation_idx]
        self.assertEqual(target.shape, ())    # scalar
        
        # Check that indices are valid
        self.assertGreaterEqual(triple[0].item(), 0)
        self.assertLess(triple[0].item(), dataset.num_entities)
        self.assertGreaterEqual(triple[1].item(), 0)
        self.assertLess(triple[1].item(), dataset.num_data_properties)
    
    def test_get_df_functionality(self):
        """Test get_df method for different splits"""
        dataset = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx
        )
        
        # Test train split
        train_df = dataset.get_df("train")
        self.assertEqual(len(train_df), 10)
        self.assertIn("head_idx", train_df.columns)
        self.assertIn("rel_idx", train_df.columns)
        
        # Test test split
        test_df = dataset.get_df("test")
        self.assertEqual(len(test_df), 3)
        
        # Test validation split (should be empty)
        val_df = dataset.get_df("val")
        self.assertEqual(len(val_df), 0)
    
    def test_get_batch_functionality(self):
        """Test get_batch method with different configurations"""
        dataset = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx
        )
        
        # Test batch for specific entities
        entity_indices = torch.tensor([0, 1])  # entity_1, entity_2
        ent_ids, rel_ids, labels = dataset.get_batch(entity_indices)
        
        # Check that all returned entities are in the requested set
        self.assertTrue(torch.all(torch.isin(ent_ids, entity_indices)))
        
        # Test multi-regression format
        ent_ids_multi, rel_ids_multi, labels_multi = dataset.get_batch(
            entity_indices, multi_regression=True
        )
        
        # Check multi-regression output shape
        self.assertEqual(labels_multi.shape[1], dataset.num_data_properties)
    
    def test_denormalization(self):
        """Test denormalization static method for different normalization types"""
        # Test z-norm denormalization
        dataset_znorm = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            normalization="z-norm"
        )
        
        # Get some normalized predictions
        height_idx = dataset_znorm.data_property_to_idx["height"]
        height_mask = dataset_znorm.triples[:, 1] == height_idx
        height_norm_values = dataset_znorm.tails_norm[height_mask]
        
        # Denormalize
        denorm_values = LiteralDataset.denormalize(
            height_norm_values.numpy(),
            ["height"] * len(height_norm_values),
            dataset_znorm.normalization_params
        )
        
        # Check that denormalized values match original
        original_height_values = dataset_znorm.tails[height_mask].numpy()
        np.testing.assert_array_almost_equal(denorm_values, original_height_values, decimal=5)
        
        # Test log-based denormalization
        dataset_log = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            normalization="log"
        )
        
        # Test that log denormalization works
        log_norm_values = dataset_log.tails_norm[:3].numpy()
        denorm_log_values = LiteralDataset.denormalize(
            log_norm_values,
            ["height", "weight", "height"],  # Mix of attributes
            dataset_log.normalization_params
        )
        
        # Denormalized values should be positive (since they came from exp)
        self.assertTrue(np.all(denorm_log_values > 0))
    
    def test_utility_methods(self):
        """Test utility methods for dataset information"""
        dataset = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            selected_attributes=["height"]
        )
        
        # Test get_available_attributes
        available_attrs = dataset.get_available_attributes()
        self.assertIn("height", available_attrs)
        self.assertIn("weight", available_attrs)
        
        # Test get_attribute_stats
        stats = dataset.get_attribute_stats()
        self.assertIn("filtered", stats)  # Should have filtered stats since we used selected_attributes
        
        # Test get_perturbation_info (no perturbation applied)
        perturbation_info = dataset.get_perturbation_info()
        self.assertIn("message", perturbation_info)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test invalid normalization type (should fall back to no normalization)
        dataset = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            normalization="invalid_norm"
        )
        # Should fall back to no normalization (but type key won't be set for invalid values)
        # Check that tail_norm equals tail (no normalization applied)
        self.assertTrue(torch.equal(dataset.tails_norm, dataset.tails))
        
        # Test invalid sampling ratio
        with self.assertRaises(ValueError):
            dataset = LiteralDataset(
                dataset_dir=self.test_dir,
                ent_idx=self.entity_to_idx,
                sampling_ratio=1.5  # > 1.0
            )
        
        # Test non-existent selected attributes
        with self.assertRaises(ValueError):
            dataset = LiteralDataset(
                dataset_dir=self.test_dir,
                ent_idx=self.entity_to_idx,
                selected_attributes=["non_existent_attribute"]
            )
        
        # Test invalid perturbation type
        with self.assertRaises(ValueError):
            dataset = LiteralDataset(
                dataset_dir=self.test_dir,
                ent_idx=self.entity_to_idx,
                label_perturbation="invalid_perturbation"
            )
        
        # Test missing data file
        empty_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(empty_dir, "literals"))
        
        with self.assertRaises(FileNotFoundError):
            dataset = LiteralDataset(
                dataset_dir=empty_dir,
                ent_idx=self.entity_to_idx
            )
        
        shutil.rmtree(empty_dir)
    
    def test_advanced_perturbations(self):
        """Test advanced perturbation methods"""
        # Test scaled noise perturbation
        dataset_scaled = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            label_perturbation="scaled_noise",
            perturbation_ratio=0.4,
            perturbation_noise_std=0.1,
            random_seed=42
        )
        
        self.assertEqual(dataset_scaled.perturbation_stats['perturbation_type'], 'scaled_noise')
        
        # Test dropout perturbation
        dataset_dropout = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            label_perturbation="dropout",
            perturbation_ratio=0.3,
            perturbation_noise_std=0.5,  # 50% dropout rate
            random_seed=42
        )
        
        self.assertEqual(dataset_dropout.perturbation_stats['perturbation_type'], 'dropout')
        
        # Test quantization perturbation
        dataset_quant = LiteralDataset(
            dataset_dir=self.test_dir,
            ent_idx=self.entity_to_idx,
            label_perturbation="quantization",
            perturbation_ratio=0.5,
            perturbation_noise_std=0.1,  # 10 quantization levels
            random_seed=42
        )
        
        self.assertEqual(dataset_quant.perturbation_stats['perturbation_type'], 'quantization')


def run_regression_tests():
    """Function to run all regression tests"""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)
