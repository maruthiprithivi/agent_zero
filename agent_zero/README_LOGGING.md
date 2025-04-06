# Logging and Tracing in Agent Zero

This document explains how to use and configure the database query logging and MCP server tracing features in Agent Zero.

## Database Query Logging

The database query logging system tracks and logs database queries, their execution times, errors, and warnings. This helps with debugging, performance analysis, and monitoring.

### Configuration Options

Database query logging can be configured using the following environment variables or in your Claude desktop configuration:

| Environment Variable              | Description                                 | Default Value |
| --------------------------------- | ------------------------------------------- | ------------- |
| `CLICKHOUSE_ENABLE_QUERY_LOGGING` | Enable detailed logging of database queries | `false`       |
| `CLICKHOUSE_LOG_QUERY_LATENCY`    | Log query execution times                   | `false`       |
| `CLICKHOUSE_LOG_QUERY_ERRORS`     | Log query errors                            | `true`        |
| `CLICKHOUSE_LOG_QUERY_WARNINGS`   | Log query warnings                          | `true`        |

### How to Enable

To enable database query logging, you can:

1. Set the environment variables in your shell before starting the application:

   ```bash
   export CLICKHOUSE_ENABLE_QUERY_LOGGING=true
   export CLICKHOUSE_LOG_QUERY_LATENCY=true
   ```

2. Or add them to your `.env` file:

   ```
   CLICKHOUSE_ENABLE_QUERY_LOGGING=true
   CLICKHOUSE_LOG_QUERY_LATENCY=true
   ```

3. Or configure them in the Claude desktop configuration.

### Log Output

When enabled, the query logger will output information similar to:

```
2023-07-10 12:34:56 - mcp-db-queries - INFO - Query: SELECT * FROM system.tables WHERE database = 'default'
2023-07-10 12:34:56 - mcp-db-queries - INFO - Query executed in 0.1234s
2023-07-10 12:34:56 - mcp-db-queries - INFO - Query result: 10 rows affected/returned
```

For errors:

```
2023-07-10 12:34:56 - mcp-db-queries - ERROR - Query error: Table 'default.non_existent_table' doesn't exist
2023-07-10 12:34:56 - mcp-db-queries - ERROR - Failed query: SELECT * FROM default.non_existent_table
```

## MCP Server Tracing

The MCP server tracing system tracks and logs MCP tool calls, including the request payload, response, execution time, and any errors. This helps with debugging, monitoring API usage, and performance analysis.

### Configuration Options

MCP server tracing can be configured using the following environment variable or in your Claude desktop configuration:

| Environment Variable | Description                                  | Default Value |
| -------------------- | -------------------------------------------- | ------------- |
| `MCP_ENABLE_TRACING` | Enable tracing for MCP server communications | `false`       |

### How to Enable

To enable MCP server tracing, you can:

1. Set the environment variable in your shell before starting the application:

   ```bash
   export MCP_ENABLE_TRACING=true
   ```

2. Or add it to your `.env` file:

   ```
   MCP_ENABLE_TRACING=true
   ```

3. Or configure it in the Claude desktop configuration.

### Trace Output

When enabled, the MCP tracer will output information similar to:

```
2023-07-10 12:34:56 - mcp-tracing - INFO - TRACE-IN [abc123-1] CALL list_databases - Payload: {}
2023-07-10 12:34:56 - mcp-tracing - INFO - TRACE-OUT [abc123-1] list_databases - Status: 200, Time: 0.1234s, Response: ['default', 'system', ...]
```

For errors:

```
2023-07-10 12:34:56 - mcp-tracing - ERROR - TRACE-ERROR [abc123-2] run_select_query - Error: ClickHouse error: Table 'default.non_existent_table' doesn't exist
```

## Combining with Other Logging

These logging systems work alongside the standard application logging. You can adjust the verbosity by configuring the logging level for the respective loggers:

- `mcp-db-queries` for database query logging
- `mcp-tracing` for MCP server tracing

## Performance Considerations

Both database query logging and MCP server tracing can impact performance, especially with high query volumes or complex requests. It's recommended to enable these features only when needed for debugging or analysis, and to disable them in production environments unless necessary.

For database query logging, consider enabling only specific aspects (e.g., error logging) rather than full query logging when performance is a concern.
