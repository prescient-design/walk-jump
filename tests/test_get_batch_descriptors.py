from walkjump.metrics._get_batch_descriptors import get_column_info


def test_get_column_info():
    """Test get column info."""
    metrics_column_info = get_column_info("fv_heavy")
    assert "fv_heavy_aromaticity" in metrics_column_info.feature_columns
