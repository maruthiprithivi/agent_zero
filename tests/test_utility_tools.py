"""Tests for the utility tools in mcp_server.py."""

import unittest
from unittest.mock import MagicMock, patch

from clickhouse_connect.driver.client import Client

from agent_zero.mcp_server import (
    generate_table_drop_script,
    list_user_defined_functions,
)


class TestUtilityTools(unittest.TestCase):
    """Test cases for utility tools."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock(spec=Client)

        # Set up the client patcher
        self.client_patcher = patch("agent_zero.mcp_server.create_clickhouse_client")
        self.mock_create_client = self.client_patcher.start()
        self.mock_create_client.return_value = self.mock_client

    def tearDown(self):
        """Tear down test fixtures."""
        self.client_patcher.stop()

    def test_generate_table_drop_script(self):
        """Test generating table drop script."""
        # Mock the generate_drop_tables_script function
        with patch("agent_zero.mcp_server.generate_drop_tables_script") as mock_function:
            # Mock successful execution
            mock_function.return_value = """
DROP TABLE IF EXISTS testdb.table1;
DROP TABLE IF EXISTS testdb.table2;
DROP TABLE IF EXISTS testdb.table3;
"""
            result = generate_table_drop_script("testdb")
            self.assertIsInstance(result, str)
            self.assertIn("DROP TABLE IF EXISTS testdb.table1", result)
            self.assertIn("DROP TABLE IF EXISTS testdb.table2", result)
            self.assertIn("DROP TABLE IF EXISTS testdb.table3", result)
            mock_function.assert_called_once_with(self.mock_client, "testdb")

            # Test error handling
            mock_function.side_effect = Exception("Test exception")
            result = generate_table_drop_script("testdb")
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error generating drop tables script", result)

    def test_list_user_defined_functions(self):
        """Test listing user defined functions."""
        # Mock the get_user_defined_functions function
        with patch("agent_zero.mcp_server.get_user_defined_functions") as mock_function:
            # Mock successful execution
            mock_function.return_value = [
                {
                    "name": "my_sum",
                    "create_query": "CREATE FUNCTION my_sum AS (a, b) -> a + b;",
                    "is_aggregate": False,
                },
                {
                    "name": "my_count",
                    "create_query": "CREATE AGGREGATE FUNCTION my_count AS () -> count();",
                    "is_aggregate": True,
                },
            ]
            result = list_user_defined_functions()
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["name"], "my_sum")
            self.assertEqual(result[1]["name"], "my_count")
            mock_function.assert_called_once_with(self.mock_client)

            # Test error handling
            mock_function.side_effect = Exception("Test exception")
            result = list_user_defined_functions()
            self.assertTrue(isinstance(result, str))
            self.assertIn("Error listing user-defined functions", result)


if __name__ == "__main__":
    unittest.main()
