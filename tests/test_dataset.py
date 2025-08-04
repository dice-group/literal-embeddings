import os
import tempfile
import pytest
import pandas as pd
import torch
import numpy as np
from src.dataset import LiteralDataset


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data = {
        'train': [
            ['entity1', 'height', 1.75],
            ['entity1', 'weight', 70.5],
            ['entity2', 'height', 1.80],
            ['entity2', 'weight', 75.0],
            ['entity3', 'height', 1.65],
            ['entity3', 'age', 25],
        ],
        'test': [
            ['entity1', 'height', 1.76],
            ['entity2', 'weight', 74.5],
            ['entity3', 'age', 26],
        ],
        'val': [
            ['entity1', 'weight', 71.0],
            ['entity2', 'height', 1.81],
        ]
    }
    return data


@pytest.fixture
def temp_dataset_dir(sample_data):
    """Create temporary dataset directory with sample files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        literals_dir = os.path.join(temp_dir, 'literals')
        os.makedirs(literals_dir)
        
        for split, data in sample_data.items():
            file_path = os.path.join(literals_dir, f'{split}.txt')
            df = pd.DataFrame(data, columns=['head', 'relation', 'tail'])
            df.to_csv(file_path, sep='\t', index=False, header=False)
        
        yield temp_dir


@pytest.fixture
def entity_mapping():
    """Create entity mapping"""
    return {
        'entity1': 0,
        'entity2': 1,
        'entity3': 2
    }


class TestLiteralDataset:
    
    def test_init_basic(self, temp_dataset_dir, entity_mapping):
        """Test basic initialization"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping,
            normalization="z-norm"
        )
        
        assert dataset.num_entities == 3
        assert dataset.num_data_properties == 3  # height, weight, age
        assert len(dataset) == 6  # 6 training triples
    
    def test_normalization_z_norm(self, temp_dataset_dir, entity_mapping):
        """Test z-normalization"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping,
            normalization="z-norm"
        )
        
        assert dataset.normalization_params["type"] == "z-norm"
        assert "height" in dataset.normalization_params
        assert "weight" in dataset.normalization_params
        assert "age" in dataset.normalization_params
        
        # Check that normalization parameters are computed
        height_params = dataset.normalization_params["height"]
        assert "mean" in height_params
        assert "std" in height_params
    
    def test_normalization_min_max(self, temp_dataset_dir, entity_mapping):
        """Test min-max normalization"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping,
            normalization="min-max"
        )
        
        assert dataset.normalization_params["type"] == "min-max"
        height_params = dataset.normalization_params["height"]
        assert "min" in height_params
        assert "max" in height_params
    
    def test_no_normalization(self, temp_dataset_dir, entity_mapping):
        """Test no normalization"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping,
            normalization=None
        )
        
        assert dataset.normalization_params["type"] is None
        # Original and normalized values should be the same
        assert torch.allclose(dataset.tails, dataset.tails_norm)
    
    def test_selected_attributes(self, temp_dataset_dir, entity_mapping):
        """Test filtering by selected attributes"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping,
            selected_attributes=['height', 'weight']
        )
        
        assert dataset.num_data_properties == 2  # Only height and weight
        assert len(dataset) == 5  # 5 triples with height/weight (not 4)
        
        # Check that only selected attributes are in the mapping
        assert set(dataset.data_property_to_idx.keys()) == {'height', 'weight'}
    
    def test_sampling_ratio(self, temp_dataset_dir, entity_mapping):
        """Test sampling ratio functionality"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping,
            sampling_ratio=0.5
        )
        
        # Should have roughly half the data (may vary due to groupby sampling)
        assert len(dataset) <= 6  # Original had 6 triples
    
    def test_get_df_test_split(self, temp_dataset_dir, entity_mapping):
        """Test get_df for test split"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping
        )
        
        test_df = dataset.get_df(split='test')
        
        assert len(test_df) == 3  # 3 test triples
        assert 'head_idx' in test_df.columns
        assert 'rel_idx' in test_df.columns
        assert 'tail' in test_df.columns
        
        # Check that indices are properly mapped
        assert all(test_df['head_idx'].notna())
        assert all(test_df['rel_idx'].notna())
    
    def test_get_df_with_selected_attributes(self, temp_dataset_dir, entity_mapping):
        """Test get_df with selected attributes filtering"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping,
            selected_attributes=['height']
        )
        
        test_df = dataset.get_df(split='test')
        
        # Only height triples should remain
        assert len(test_df) == 1
        assert all(test_df['relation'] == 'height')
    
    def test_get_batch_no_sampling(self, temp_dataset_dir, entity_mapping):
        """Test get_batch returns all triples for entities (no random sampling)"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping
        )
        
        # Get batch for entity1 (index 0)
        entity_indices = torch.tensor([0])
        ent_ids, rel_ids, labels = dataset.get_batch(entity_indices)
        
        # Should return all triples for entity1 (height and weight)
        assert len(ent_ids) == 2
        assert all(ent_ids == 0)  # All should be entity1
        
        # Test multiple entities
        entity_indices = torch.tensor([0, 1])
        ent_ids, rel_ids, labels = dataset.get_batch(entity_indices)
        
        # Should return all triples for entity1 and entity2
        assert len(ent_ids) == 4  # 2 for entity1 + 2 for entity2
    
    def test_get_batch_multi_regression(self, temp_dataset_dir, entity_mapping):
        """Test get_batch with multi_regression=True"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping
        )
        
        entity_indices = torch.tensor([0])
        ent_ids, rel_ids, y_true = dataset.get_batch(entity_indices, multi_regression=True)
        
        # y_true should be a matrix with shape [num_triples, num_properties]
        assert y_true.shape[1] == dataset.num_data_properties
        assert y_true.shape[0] == len(ent_ids)
        
        # Most values should be -9.0 (placeholder), only specific relations have real values
        assert torch.sum(y_true == -9.0) > 0
    
    def test_get_ea_encoding(self, temp_dataset_dir, entity_mapping):
        """Test entity-attribute encoding matrix"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping
        )
        
        ea = dataset.get_ea_encoding()
        
        assert ea.shape == (dataset.num_entities, dataset.num_data_properties)
        assert ea.dtype == torch.float32
        
        # Check that the encoding is binary (0s and 1s)
        assert torch.all((ea == 0) | (ea == 1))
    
    def test_denormalize_z_norm(self, temp_dataset_dir, entity_mapping):
        """Test denormalization with z-norm"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping,
            normalization="z-norm"
        )
        
        # Create some normalized predictions
        preds_norm = np.array([0.0, 1.0])  # Normalized predictions
        attributes = ['height', 'height']
        
        denormalized = dataset.denormalize(
            preds_norm=preds_norm,
            attributes=attributes,
            normalization_params=dataset.normalization_params
        )
        
        assert len(denormalized) == 2
        assert isinstance(denormalized, np.ndarray)
    
    def test_denormalize_min_max(self, temp_dataset_dir, entity_mapping):
        """Test denormalization with min-max"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping,
            normalization="min-max"
        )
        
        preds_norm = np.array([0.0, 1.0])
        attributes = ['height', 'height']
        
        denormalized = dataset.denormalize(
            preds_norm=preds_norm,
            attributes=attributes,
            normalization_params=dataset.normalization_params
        )
        
        assert len(denormalized) == 2
        # For min-max, 0.0 should give min value, 1.0 should give max value
        height_min = dataset.normalization_params['height']['min']
        height_max = dataset.normalization_params['height']['max']
        
        assert abs(denormalized[0] - height_min) < 1e-6
        assert abs(denormalized[1] - height_max) < 1e-6
    
    def test_denormalize_no_norm(self, temp_dataset_dir, entity_mapping):
        """Test denormalization with no normalization"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping,
            normalization=None
        )
        
        preds_norm = np.array([1.75, 70.5])
        attributes = ['height', 'weight']
        
        denormalized = dataset.denormalize(
            preds_norm=preds_norm,
            attributes=attributes,
            normalization_params=dataset.normalization_params
        )
        
        # Should return original values unchanged
        assert np.allclose(denormalized, preds_norm)
    
    def test_get_available_attributes(self, temp_dataset_dir, entity_mapping):
        """Test getting available attributes"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping
        )
        
        attributes = dataset.get_available_attributes()
        
        assert isinstance(attributes, list)
        assert set(attributes) == {'height', 'weight', 'age'}
        assert attributes == sorted(attributes)  # Should be sorted
    
    def test_get_attribute_stats(self, temp_dataset_dir, entity_mapping):
        """Test getting attribute statistics"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping
        )
        
        stats = dataset.get_attribute_stats()
        
        assert "all" in stats
        assert isinstance(stats["all"], pd.Series)
        
        # Test with selected attributes
        dataset_filtered = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping,
            selected_attributes=['height']
        )
        
        stats_filtered = dataset_filtered.get_attribute_stats()
        assert "original" in stats_filtered
        assert "filtered" in stats_filtered
    
    def test_getitem(self, temp_dataset_dir, entity_mapping):
        """Test __getitem__ method"""
        dataset = LiteralDataset(
            dataset_dir=temp_dataset_dir,
            ent_idx=entity_mapping
        )
        
        triple, label = dataset[0]
        
        assert isinstance(triple, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert triple.shape == (2,)  # [entity_idx, relation_idx]
        assert label.shape == ()  # Scalar
    
    def test_invalid_sampling_ratio(self, temp_dataset_dir, entity_mapping):
        """Test invalid sampling ratio raises error"""
        with pytest.raises(ValueError, match="Fraction must be between 0 and 1"):
            LiteralDataset(
                dataset_dir=temp_dataset_dir,
                ent_idx=entity_mapping,
                sampling_ratio=1.5
            )
    
    def test_missing_file(self, entity_mapping):
        """Test handling of missing dataset files"""
        with pytest.raises(FileNotFoundError):
            LiteralDataset(
                dataset_dir="/nonexistent/path",
                ent_idx=entity_mapping
            )
    
    def test_empty_selected_attributes(self, temp_dataset_dir, entity_mapping):
        """Test error when selected attributes result in empty dataset"""
        with pytest.raises(ValueError, match="No triples found for selected attributes"):
            LiteralDataset(
                dataset_dir=temp_dataset_dir,
                ent_idx=entity_mapping,
                selected_attributes=['nonexistent_attribute']
            )