"""
Tests for the strapdown_py Python bindings.

This module contains tests for verifying that the Rust functions
exposed via Python bindings work as expected.
"""

import strapdown_py


def test_add_function():
    """Test that the add function works correctly."""
    # Test basic addition
    result = strapdown_py.add(3.5, 2.5)
    assert result == 6

    # Test with negative numbers
    result = strapdown_py.add(-1.5, 2.5)
    assert result == 1

    # Test with zero
    result = strapdown_py.add(0.0, 5.0)
    assert result == 5


# Add more tests as more functionality is added to the strapdown-py bindings
