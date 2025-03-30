import unittest
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

from agent_zero import list_databases, list_tables, run_select_query
from agent_zero.monitoring.query_performance import get_current_processes, get_query_duration_stats
from agent_zero.monitoring.resource_usage import get_memory_usage, get_server_sizing
from tests.utils import create_mock_result

load_dotenv()


class TestClickhouseTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the environment before tests."""
        # Create mock client
        cls.client = MagicMock()
        cls.test_db = "test_tool_db"
        cls.test_table = "test_table"

        # Mock database list
        cls.client.command.return_value = [cls.test_db, "default", "system"]

        # Mock table list data
        cls.mock_table_result = create_mock_result(
            column_names=["name"], result_rows=[[cls.test_table]]
        )

        # Mock table comments data
        cls.mock_table_comments = create_mock_result(
            column_names=["name", "comment"],
            result_rows=[[cls.test_table, "Test table for unit testing"]],
        )

        # Mock column comments data
        cls.mock_column_comments = create_mock_result(
            column_names=["table", "name", "comment"],
            result_rows=[
                [cls.test_table, "id", "Primary identifier"],
                [cls.test_table, "name", "User name field"],
            ],
        )

        # Mock table schema data
        cls.mock_schema_result = create_mock_result(
            column_names=[
                "name",
                "type",
                "default_type",
                "default_expression",
                "comment",
                "codec_expression",
                "ttl_expression",
            ],
            result_rows=[
                ["id", "UInt32", "", "", "", "", ""],
                ["name", "String", "", "", "", "", ""],
            ],
        )

        # Mock command method to return different values depending on the query
        def mock_command_side_effect(cmd):
            if cmd == "SHOW DATABASES":
                return [cls.test_db, "default", "system"]
            elif "SHOW TABLES" in cmd:
                return cls.test_table
            elif "SHOW CREATE TABLE" in cmd:
                return "CREATE TABLE test_tool_db.test_table ..."
            return "Command executed"

        cls.client.command.side_effect = mock_command_side_effect

    def setUp(self):
        # Create a patcher for create_clickhouse_client
        self.client_patcher = patch("agent_zero.mcp_server.create_clickhouse_client")
        self.mock_create_client = self.client_patcher.start()
        self.mock_create_client.return_value = self.client

    def tearDown(self):
        self.client_patcher.stop()

    def test_list_databases(self):
        """Test listing databases."""
        result = list_databases()
        self.assertIn(self.test_db, result)
        self.mock_create_client.assert_called_once()

    def test_list_tables_without_like(self):
        """Test listing tables without a 'LIKE' filter."""

        # For list_tables tests, we'll create a mock function and then swap it in
        def mock_command(cmd):
            if cmd.startswith("SHOW TABLES"):
                return self.test_table
            elif cmd.startswith("SHOW CREATE TABLE"):
                return "CREATE TABLE test_schema..."
            return "Command executed"

        # Save the original method and replace it with our mock
        original_command = self.client.command
        self.client.command = mock_command

        # Set up query results for table info
        self.client.query.side_effect = [
            self.mock_table_comments,
            self.mock_column_comments,
            self.mock_schema_result,
        ]

        # Create mock table result
        mock_result = [
            {
                "database": self.test_db,
                "name": self.test_table,
                "comment": "Test table for unit testing",
                "columns": [],
                "create_table_query": "CREATE TABLE...",
            }
        ]

        # Mock the get_table_info method
        with patch("agent_zero.mcp_server.list_tables") as mock_list:
            mock_list.return_value = mock_result

            # Call the method under test and verify
            result = list_tables(self.test_db)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["name"], self.test_table)

        # Restore the original command method
        self.client.command = original_command

    def test_list_tables_with_like(self):
        """Test listing tables with a 'LIKE' filter."""

        # For list_tables tests, we'll create a mock function and then swap it in
        def mock_command(cmd):
            if cmd.startswith("SHOW TABLES"):
                return self.test_table
            elif cmd.startswith("SHOW CREATE TABLE"):
                return "CREATE TABLE test_schema..."
            return "Command executed"

        # Save the original method and replace it with our mock
        original_command = self.client.command
        self.client.command = mock_command

        # Set up query results for table info
        self.client.query.side_effect = [
            self.mock_table_comments,
            self.mock_column_comments,
            self.mock_schema_result,
        ]

        # Create mock table result
        mock_result = [
            {
                "database": self.test_db,
                "name": self.test_table,
                "comment": "Test table for unit testing",
                "columns": [],
                "create_table_query": "CREATE TABLE...",
            }
        ]

        # Mock the get_table_info method
        with patch("agent_zero.mcp_server.list_tables") as mock_list:
            mock_list.return_value = mock_result

            # Call the method under test with like parameter
            like_pattern = f"{self.test_table}%"
            result = list_tables(self.test_db, like=like_pattern)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["name"], self.test_table)

        # Restore the original command method
        self.client.command = original_command

    def test_run_select_query_success(self):
        """Test running a SELECT query successfully."""
        # Set up mock response for successful query
        mock_query_result = create_mock_result(
            column_names=["id", "name"], result_rows=[[1, "Alice"], [2, "Bob"]]
        )

        # Mock execute_query function to handle the actual run_select_query function
        with patch("agent_zero.mcp_server.execute_query") as mock_execute:
            mock_execute.return_value = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

            query = f"SELECT * FROM {self.test_db}.{self.test_table}"
            result = run_select_query(query)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["id"], 1)
            self.assertEqual(result[0]["name"], "Alice")

    def test_run_select_query_failure(self):
        """Test running a SELECT query with an error."""
        # Set up mock to raise an exception
        self.client.query.side_effect = Exception("Table not found")

        query = f"SELECT * FROM {self.test_db}.non_existent_table"
        result = run_select_query(query)
        self.assertIsInstance(result, str)
        self.assertIn("error running query", result)

    def test_table_and_column_comments(self):
        """Test that table and column comments are correctly retrieved."""

        # For list_tables tests, we'll create a mock function and then swap it in
        def mock_command(cmd):
            if cmd.startswith("SHOW TABLES"):
                return self.test_table
            elif cmd.startswith("SHOW CREATE TABLE"):
                return "CREATE TABLE test_schema..."
            return "Command executed"

        # Save the original method and replace it with our mock
        original_command = self.client.command
        self.client.command = mock_command

        # Set up query results for table info
        self.client.query.side_effect = [
            self.mock_table_comments,
            self.mock_column_comments,
            self.mock_schema_result,
        ]

        # Create mock table result
        mock_result = [
            {
                "database": self.test_db,
                "name": self.test_table,
                "comment": "Test table for unit testing",
                "columns": [
                    {"name": "id", "type": "UInt32", "comment": "Primary identifier"},
                    {"name": "name", "type": "String", "comment": "User name field"},
                ],
                "create_table_query": "CREATE TABLE...",
            }
        ]

        # Mock the list_tables method directly
        with patch("agent_zero.mcp_server.list_tables") as mock_list:
            mock_list.return_value = mock_result

            # Call the method under test
            result = list_tables(self.test_db)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)

            table_info = result[0]
            # Verify table comment
            self.assertEqual(table_info["comment"], "Test table for unit testing")

            # Get columns by name for easier testing
            columns = {col["name"]: col for col in table_info["columns"]}

            # Verify column comments
            self.assertEqual(columns["id"]["comment"], "Primary identifier")
            self.assertEqual(columns["name"]["comment"], "User name field")

        # Restore the original command method
        self.client.command = original_command


class TestMonitoringTools(unittest.TestCase):
    """Test cases for monitoring tools."""

    def setUp(self):
        """Set up the test case."""
        self.mock_client = MagicMock()

        # Mock results for query return values
        self.mock_processes_result = create_mock_result(
            column_names=["query_id", "user", "query"],
            result_rows=[["123", "default", "SELECT 1"], ["456", "default", "SELECT 2"]],
        )

        self.mock_duration_result = create_mock_result(
            column_names=["hour", "query_count", "p50", "p90"],
            result_rows=[["2024-03-01 00:00:00", 100, 0.1, 0.5]],
        )

        self.mock_memory_result = create_mock_result(
            column_names=["ts", "hostname_", "MemoryTracking_avg"],
            result_rows=[
                ["2024-03-01 00:00:00", "host1", "100 MB"],
                ["2024-03-01 01:00:00", "host1", "110 MB"],
            ],
        )

        self.mock_sizing_result = create_mock_result(
            column_names=["hostname", "cpu_cores", "memory"], result_rows=[["host1", 8, "64 GB"]]
        )

        # Set up client patcher
        self.client_patcher = patch("agent_zero.mcp_server.create_clickhouse_client")
        self.mock_create_client = self.client_patcher.start()
        self.mock_create_client.return_value = self.mock_client

    def tearDown(self):
        """Clean up after the test case."""
        self.client_patcher.stop()

    def test_get_current_processes(self):
        """Test retrieving current processes."""
        self.mock_client.query.return_value = self.mock_processes_result

        result = get_current_processes(self.mock_client)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.mock_client.query.assert_called_once()

    def test_get_query_duration_stats(self):
        """Test retrieving query duration statistics."""
        self.mock_client.query.return_value = self.mock_duration_result

        result = get_query_duration_stats(self.mock_client, days=1)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.mock_client.query.assert_called_once()

    def test_get_memory_usage(self):
        """Test retrieving memory usage statistics."""
        self.mock_client.query.return_value = self.mock_memory_result

        result = get_memory_usage(self.mock_client, days=1)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.mock_client.query.assert_called_once()

    def test_get_server_sizing(self):
        """Test retrieving server sizing information."""
        self.mock_client.query.return_value = self.mock_sizing_result

        result = get_server_sizing(self.mock_client)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIn("cpu_cores", result[0])
        self.assertIn("memory", result[0])
        self.mock_client.query.assert_called_once()


if __name__ == "__main__":
    unittest.main()
