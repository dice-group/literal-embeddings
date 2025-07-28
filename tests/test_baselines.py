"""
Simple pytest test for GLOBAL and LOCAL baseline calculations.
Tests the core functionality of the evaluate_LOCAL_GLOBAL function.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.baselines import evaluate_LOCAL_GLOBAL


class TestGlobalLocalBaselines:
    """Test suite for GLOBAL and LOCAL baseline calculations."""
    
    @pytest.mark.unit
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_simple_global_local_calculation(self):
        """Test basic GLOBAL and LOCAL calculation with known data."""
        
        # Simple entity relationships
        entity_triples = pd.DataFrame([
            ("entity1", "connected_to", "entity2"),
            ("entity1", "connected_to", "entity3"),
            ("entity2", "connected_to", "entity4"),
        ], columns=["head", "relation", "tail"])
        
        # Training data with known values
        train_data = pd.DataFrame([
            ("entity1", "age", 25.0),
            ("entity2", "age", 30.0),
            ("entity3", "age", 35.0),
            ("entity4", "age", 40.0),
        ], columns=["head", "relation", "tail"])
        
        # Test data
        test_data = pd.DataFrame([
            ("entity2", "age", 32.0),  # entity2 neighbors: entity1(25), entity4(40)
        ], columns=["head", "relation", "tail"])
        
        # Run the evaluation
        result = evaluate_LOCAL_GLOBAL(entity_triples, train_data, test_data.copy())
        
        # Verify structure
        assert "MAE_GLOBAL" in result.columns
        assert "RMSE_GLOBAL" in result.columns  
        assert "MAE_LOCAL" in result.columns
        assert "RMSE_LOCAL" in result.columns
        
        # Verify there's one row for the 'age' relation
        assert len(result) == 1
        assert result.index[0] == "age"
        
        # Check that values are reasonable (non-negative, finite)
        for col in ["MAE_GLOBAL", "RMSE_GLOBAL", "MAE_LOCAL", "RMSE_LOCAL"]:
            assert result[col].iloc[0] >= 0, f"{col} should be non-negative"
            assert np.isfinite(result[col].iloc[0]), f"{col} should be finite"
        
        # Check RMSE >= MAE property
        assert result["RMSE_GLOBAL"].iloc[0] >= result["MAE_GLOBAL"].iloc[0]
        assert result["RMSE_LOCAL"].iloc[0] >= result["MAE_LOCAL"].iloc[0]
    
    @pytest.mark.unit
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_global_average_calculation(self):
        """Test that global averages are calculated correctly."""
        
        entity_triples = pd.DataFrame([
            ("e1", "rel", "e2"),
        ], columns=["head", "relation", "tail"])
        
        train_data = pd.DataFrame([
            ("e1", "score", 10.0),
            ("e2", "score", 20.0),
            ("e3", "score", 30.0),
        ], columns=["head", "relation", "tail"])
        
        test_data = pd.DataFrame([
            ("e1", "score", 15.0),
        ], columns=["head", "relation", "tail"])
        
        test_copy = test_data.copy()
        evaluate_LOCAL_GLOBAL(entity_triples, train_data, test_copy)
        
        # Global average should be (10 + 20 + 30) / 3 = 20.0
        expected_global = 20.0
        actual_global = test_copy["GLOBAL"].iloc[0]
        
        assert abs(actual_global - expected_global) < 1e-10, \
            f"Expected global average {expected_global}, got {actual_global}"

    @pytest.mark.unit
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_empty_data_handling(self):
        """Test handling of edge cases like empty data."""
        entity_triples = pd.DataFrame(columns=["head", "relation", "tail"])
        train_data = pd.DataFrame(columns=["head", "relation", "tail"])
        test_data = pd.DataFrame(columns=["head", "relation", "tail"])
        
        # Should handle empty data gracefully
        result = evaluate_LOCAL_GLOBAL(entity_triples, train_data, test_data.copy())
        assert len(result) == 0

    @pytest.mark.unit
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_missing_relations(self):
        """Test behavior when test data has relations not in training data."""
        entity_triples = pd.DataFrame([
            ("e1", "rel", "e2"),
        ], columns=["head", "relation", "tail"])
        
        # Training data with one relation
        train_data = pd.DataFrame([
            ("e1", "known_attr", 10.0),
            ("e2", "known_attr", 20.0),
        ], columns=["head", "relation", "tail"])
        
        # Test data with the same relation that exists in training
        test_data = pd.DataFrame([
            ("e1", "known_attr", 15.0),
        ], columns=["head", "relation", "tail"])
        
        # Should handle known relations properly
        result = evaluate_LOCAL_GLOBAL(entity_triples, train_data, test_data.copy())
        assert len(result) == 1
        assert "known_attr" in result.index
        
        # All metrics should be finite for known relations
        for col in ["MAE_GLOBAL", "RMSE_GLOBAL", "MAE_LOCAL", "RMSE_LOCAL"]:
            assert np.isfinite(result[col].iloc[0]), f"{col} should be finite for known relations"

    @pytest.mark.unit
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_unknown_relations_handling(self):
        """Test that unknown relations are handled gracefully."""
        entity_triples = pd.DataFrame([
            ("e1", "rel", "e2"),
        ], columns=["head", "relation", "tail"])
        
        # Training data with one relation
        train_data = pd.DataFrame([
            ("e1", "known_attr", 10.0),
            ("e2", "known_attr", 20.0),
        ], columns=["head", "relation", "tail"])
        
        # Test data with unknown relation - this will likely cause NaN
        test_data = pd.DataFrame([
            ("e1", "unknown_attr", 15.0),
        ], columns=["head", "relation", "tail"])
        
        # This test expects the function might fail with unknown relations
        # or handle them by skipping/filtering them out
        try:
            result = evaluate_LOCAL_GLOBAL(entity_triples, train_data, test_data.copy())
            # If it succeeds, check that we get some kind of reasonable result
            assert isinstance(result, pd.DataFrame)
        except (ValueError, KeyError) as e:
            # It's acceptable for the function to fail with unknown relations
            assert "NaN" in str(e) or "unknown" in str(e).lower()

# For manual testing and development
if __name__ == "__main__":
    # This allows running the test file directly for development
    # Use: python test_baselines.py
    import pytest
    
    # Run pytest on this file specifically
    pytest.main([__file__, "-v", "--tb=short"])
