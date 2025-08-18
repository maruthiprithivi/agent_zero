import unittest
from unittest.mock import MagicMock

from agent_zero.monitoring.query_performance import get_current_processes, get_query_duration_stats
from agent_zero.monitoring.resource_usage import get_memory_usage, get_server_sizing


class TestMonitoringTools(unittest.TestCase):
    """Test monitoring tools functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()

    def test_get_current_processes(self):
        """Test get_current_processes function."""
        # Mock client response
        self.mock_client.command.return_value = [
            {"query": "SELECT * FROM table", "elapsed": 1.5},
            {"query": "INSERT INTO table VALUES", "elapsed": 0.8},
        ]
        result = get_current_processes(self.mock_client)
        self.assertIsNotNone(result)

    def test_get_query_duration_stats(self):
        """Test get_query_duration_stats function."""
        # Mock client response
        self.mock_client.command.return_value = [
            {"avg_duration": 1.2, "max_duration": 5.0, "count": 100},
        ]
        result = get_query_duration_stats(self.mock_client, 1, 1000)
        self.assertIsNotNone(result)

    def test_get_memory_usage(self):
        """Test get_memory_usage function."""
        # Mock client response
        self.mock_client.command.return_value = [
            {"metric": "MemoryResident", "value": 1024000000},
            {"metric": "MemoryVirtual", "value": 2048000000},
        ]
        result = get_memory_usage(self.mock_client)
        self.assertIsNotNone(result)

    def test_get_server_sizing(self):
        """Test get_server_sizing function."""
        # Mock client response
        self.mock_client.command.return_value = [
            {"metric": "uptime", "value": 86400},
            {"metric": "queries", "value": 1000},
        ]
        result = get_server_sizing(self.mock_client)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
