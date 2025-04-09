# Testing MCP Endpoints

This guide covers testing the Model Context Protocol (MCP) endpoints in Agent Zero.

## Basic Connection Test

Check if the server is running:

```bash
curl -i http://localhost:8505/health
```

Expected response:

```
HTTP/1.1 200 OK
Content-Type: application/json
{"status": "healthy", "server": "agent-zero", ...}
```

## Testing Core Endpoints

### List Databases

```bash
curl -X POST http://localhost:8505/list_databases \
  -H "Content-Type: application/json" \
  -d '{}'
```

### List Tables

```bash
curl -X POST http://localhost:8505/list_tables \
  -H "Content-Type: application/json" \
  -d '{"database": "system"}'
```

### Run Select Query

```bash
curl -X POST http://localhost:8505/run_select_query \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT name, engine FROM system.databases"}'
```

## Security Testing

### With Authentication

```bash
curl -X POST http://localhost:8505/list_databases \
  -H "Content-Type: application/json" \
  -d '{}' \
  -u admin:your-password
```

### With HTTPS

```bash
curl -X POST https://localhost:8505/list_databases \
  -H "Content-Type: application/json" \
  -d '{}' \
  --cacert /path/to/ca.pem
```

## Automated Testing

Sample Python script:

```python
import requests
import json

base_url = "http://localhost:8505"
auth = None  # ("admin", "password") if auth enabled

# Health check
response = requests.get(f"{base_url}/health", auth=auth)
print(f"Health check: {response.status_code}")

# List databases
response = requests.post(
    f"{base_url}/list_databases",
    json={},
    auth=auth
)
print(f"Databases: {response.json()}")
```

## Error Testing

```bash
# Invalid query test
curl -X POST http://localhost:8505/run_select_query \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT * FROM non_existent_table"}'

# Auth failure test
curl -X POST http://localhost:8505/list_databases \
  -u admin:wrong-password
```

---

For more information, see the [testing documentation index](README.md) or [standalone server documentation](../standalone-server.md).
