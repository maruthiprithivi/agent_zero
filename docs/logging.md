# Logging and Tracing in Agent Zero

This document explains how to configure logging in Agent Zero.

## Database Query Logging

Tracks database queries, execution times, errors, and warnings.

### Configuration Options

| Environment Variable              | Description                   | Default |
| --------------------------------- | ----------------------------- | ------- |
| `CLICKHOUSE_ENABLE_QUERY_LOGGING` | Enable detailed query logging | `false` |
| `CLICKHOUSE_LOG_QUERY_LATENCY`    | Log query execution times     | `false` |
| `CLICKHOUSE_LOG_QUERY_ERRORS`     | Log query errors              | `true`  |
| `CLICKHOUSE_LOG_QUERY_WARNINGS`   | Log query warnings            | `true`  |

### How to Enable

```bash
# In shell
export CLICKHOUSE_ENABLE_QUERY_LOGGING=true
export CLICKHOUSE_LOG_QUERY_LATENCY=true

# Or in .env file
CLICKHOUSE_ENABLE_QUERY_LOGGING=true
CLICKHOUSE_LOG_QUERY_LATENCY=true
```

## MCP Server Tracing

Tracks MCP tool calls, request payloads, responses, and execution times.

### Configuration

| Environment Variable | Description        | Default |
| -------------------- | ------------------ | ------- |
| `MCP_ENABLE_TRACING` | Enable MCP tracing | `false` |

### How to Enable

```bash
# In shell
export MCP_ENABLE_TRACING=true

# Or in .env file
MCP_ENABLE_TRACING=true
```

## Performance Considerations

- Both logging systems can impact performance with high query volumes
- Enable only when needed for debugging or monitoring
- For production, consider enabling only error logging rather than full query logging

---

For more information, see the [main documentation](README.md) or [standalone server documentation](standalone-server.md).
