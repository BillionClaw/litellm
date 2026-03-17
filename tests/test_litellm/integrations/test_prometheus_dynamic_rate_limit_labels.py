"""
Test that failure metrics have correct labels for dynamic rate limits.

This tests the fix for issue #23772: Failure metrics missing/incorrect labels 
for dynamic rate limits - specifically ensuring requested_model_name is correctly 
populated from metadata.model_group.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))


class TestGetModelGroupFromRequestData:
    """Test the get_model_group_from_request_data helper function."""

    def test_model_group_from_metadata(self):
        """
        Test that model_group is extracted from metadata.model_group when available.
        """
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_request_data,
        )

        # Setup request_data with model_group in metadata
        request_data = {
            "model": "",  # Empty model as might happen with rate limit errors
            "metadata": {
                "model_group": "gpt-4",
                "model_info": {"id": "test_model_id"},
            },
        }

        result = get_model_group_from_request_data(request_data)
        assert result == "gpt-4", f"Expected 'gpt-4', got '{result}'"

    def test_model_group_fallback_to_empty(self):
        """
        Test that None is returned when model_group is not in metadata.
        """
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_request_data,
        )

        # Setup request_data without model_group in metadata
        request_data = {
            "model": "gpt-3.5-turbo",
            "metadata": {
                "model_info": {"id": "test_model_id"},
            },
        }

        result = get_model_group_from_request_data(request_data)
        assert result is None, f"Expected None, got '{result}'"

    def test_model_group_with_none_metadata(self):
        """
        Test that None is returned when metadata is None.
        """
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_request_data,
        )

        # Setup request_data without metadata
        request_data = {
            "model": "gpt-3.5-turbo",
        }

        result = get_model_group_from_request_data(request_data)
        assert result is None, f"Expected None, got '{result}'"

    def test_model_group_with_empty_request_data(self):
        """
        Test that None is returned when request_data is empty.
        """
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_request_data,
        )

        result = get_model_group_from_request_data({})
        assert result is None, f"Expected None, got '{result}'"


class TestPrometheusRequestedModelLogic:
    """
    Test the logic for extracting requested_model in prometheus.
    
    This simulates the fix for issue #23772 where the logic is:
    requested_model = get_model_group_from_request_data(request_data) or request_data.get("model", "")
    """

    def test_requested_model_from_metadata_model_group(self):
        """
        Test that requested_model is correctly extracted from metadata.model_group.
        This is the core fix for issue #23772.
        """
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_request_data,
        )

        # Simulate the request data as it would be when dynamic rate limiter fails
        request_data = {
            "model": "",  # Empty model as might happen with rate limit errors
            "metadata": {
                "model_group": "claude-3-opus",  # Set by router/proxy
                "model_info": {"id": "deployment_123"},
                "requester_ip_address": "192.168.1.1",
                "user_agent": "openai-python/1.0",
            },
            "stream": True,
        }

        # This is the fix logic
        requested_model = (
            get_model_group_from_request_data(request_data)
            or request_data.get("model", "")
        )

        assert requested_model == "claude-3-opus", \
            f"Expected requested_model='claude-3-opus', got '{requested_model}'"

    def test_requested_model_fallback_to_request_data_model(self):
        """
        Test that requested_model falls back to request_data["model"] when 
        metadata.model_group is not available.
        """
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_request_data,
        )

        # Setup request_data with model directly (old behavior still works)
        request_data = {
            "model": "gpt-3.5-turbo",
            "metadata": {
                "model_info": {"id": "test_model_id"},
            },
        }

        # This is the fix logic
        requested_model = (
            get_model_group_from_request_data(request_data)
            or request_data.get("model", "")
        )

        assert requested_model == "gpt-3.5-turbo", \
            f"Expected requested_model='gpt-3.5-turbo', got '{requested_model}'"

    def test_dynamic_rate_limiter_scenario(self):
        """
        Test a realistic dynamic rate limiter error scenario.
        
        When dynamic_rate_limiter raises HTTPException 429:
        - request_data["model"] might be empty
        - request_data["metadata"]["model_group"] contains the actual model
        
        The fix ensures requested_model is correctly populated.
        """
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_request_data,
        )

        # Simulate realistic request data when rate limiter fails
        request_data = {
            "model": "",  # May be empty when pre-call hooks fail
            "metadata": {
                "model_group": "gpt-4-turbo",  # Set by proxy before calling rate limiter
                "model_info": {"id": "model_deployment_abc123"},
                "requester_ip_address": "10.0.0.1",
                "user_agent": "MyApp/1.0",
                "user_api_key_hash": "sk-xxx",
            },
            "stream": False,
            "proxy_server_request": {
                "method": "POST",
                "url": "/v1/chat/completions",
            },
        }

        # Apply the fix logic
        requested_model = (
            get_model_group_from_request_data(request_data)
            or request_data.get("model", "")
        )

        # Verify the fix correctly extracts the model
        assert requested_model == "gpt-4-turbo", \
            f"In dynamic rate limiter scenario, expected requested_model='gpt-4-turbo', got '{requested_model}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
