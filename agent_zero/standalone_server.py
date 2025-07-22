"""Standalone MCP server implementation with HTTP/WebSocket support.

This module provides a standalone MCP server that can run independently
and serve multiple clients over HTTP and WebSocket transports. Designed
for enterprise deployments and remote IDE integrations.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, Set
from collections import defaultdict

try:
    import aiohttp
    from aiohttp import web, WSMsgType
    from aiohttp.web_middlewares import cors_handler
    from aiohttp_cors import setup as cors_setup, ResourceOptions
except ImportError:
    aiohttp = None
    web = None
    WSMsgType = None
    cors_handler = None
    cors_setup = None
    ResourceOptions = None

from .server_config import ServerConfig, DeploymentMode, TransportType
from .mcp_server import mcp, logger as mcp_logger

logger = logging.getLogger("mcp-standalone-server")


class RateLimiter:
    """Simple rate limiter for MCP requests."""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 1):
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if client is allowed to make a request."""
        now = time.time()
        client_requests = self.requests[client_id]
        
        # Remove old requests outside the window
        while client_requests and client_requests[0] < now - self.window_seconds:
            client_requests.pop(0)
        
        # Check if under limit
        if len(client_requests) < self.max_requests:
            client_requests.append(now)
            return True
        
        return False


class HealthChecker:
    """Health check service for the standalone server."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.active_connections = 0
    
    def record_request(self):
        """Record a successful request."""
        self.request_count += 1
    
    def record_error(self):
        """Record an error."""
        self.error_count += 1
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        uptime = datetime.now() - self.start_time
        return {
            "status": "healthy",
            "uptime_seconds": int(uptime.total_seconds()),
            "start_time": self.start_time.isoformat(),
            "requests_total": self.request_count,
            "errors_total": self.error_count,
            "active_connections": self.active_connections,
            "version": "0.0.1",
            "deployment_mode": "standalone"
        }


class MetricsCollector:
    """Metrics collection service."""
    
    def __init__(self):
        self.tool_usage = defaultdict(int)
        self.response_times = defaultdict(list)
        self.client_connections = defaultdict(int)
    
    def record_tool_usage(self, tool_name: str):
        """Record tool usage."""
        self.tool_usage[tool_name] += 1
    
    def record_response_time(self, tool_name: str, response_time: float):
        """Record response time for a tool."""
        self.response_times[tool_name].append(response_time)
        # Keep only last 100 response times
        if len(self.response_times[tool_name]) > 100:
            self.response_times[tool_name] = self.response_times[tool_name][-100:]
    
    def record_client_connection(self, client_id: str):
        """Record client connection."""
        self.client_connections[client_id] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        avg_response_times = {}
        for tool, times in self.response_times.items():
            if times:
                avg_response_times[tool] = sum(times) / len(times)
        
        return {
            "tool_usage": dict(self.tool_usage),
            "average_response_times": avg_response_times,
            "client_connections": dict(self.client_connections),
            "total_tools_available": len(mcp.list_tools()) if hasattr(mcp, 'list_tools') else 0,
        }


class StandaloneMCPServer:
    """Standalone MCP server with HTTP/WebSocket support."""
    
    def __init__(self, server_config: ServerConfig):
        self.config = server_config
        self.rate_limiter = RateLimiter(
            max_requests=server_config.rate_limit_requests,
        ) if server_config.rate_limit_enabled else None
        self.health_checker = HealthChecker() if server_config.enable_health_check else None
        self.metrics = MetricsCollector() if server_config.enable_metrics else None
        self.active_websockets: Set[web.WebSocketResponse] = set()
        self.app = None
    
    async def handle_mcp_request(self, request_data: Dict[str, Any], client_id: str = None) -> Dict[str, Any]:
        """Handle an MCP request and return the response."""
        start_time = time.time()
        
        try:
            # Rate limiting
            if self.rate_limiter and not self.rate_limiter.is_allowed(client_id or "anonymous"):
                return {
                    "error": {
                        "code": 429,
                        "message": "Rate limit exceeded"
                    }
                }
            
            # Extract method and params
            method = request_data.get("method")
            params = request_data.get("params", {})
            request_id = request_data.get("id")
            
            if not method:
                return {
                    "error": {
                        "code": -32600,
                        "message": "Invalid request - missing method"
                    },
                    "id": request_id
                }
            
            # Handle MCP methods
            if method == "tools/list":
                # List available tools
                tools = await self._list_tools()\n                response_data = {"tools": tools}\n            elif method == "tools/call":\n                # Call a specific tool\n                tool_name = params.get("name")\n                tool_arguments = params.get("arguments", {})\n                \n                if not tool_name:\n                    return {\n                        "error": {\n                            "code": -32602,\n                            "message": "Invalid params - missing tool name"\n                        },\n                        "id": request_id\n                    }\n                \n                response_data = await self._call_tool(tool_name, tool_arguments)\n                \n                # Record metrics\n                if self.metrics:\n                    self.metrics.record_tool_usage(tool_name)\n            \n            elif method == "resources/list":\n                # List available resources\n                resources = await self._list_resources()\n                response_data = {"resources": resources}\n            \n            elif method == "resources/read":\n                # Read a specific resource\n                resource_uri = params.get("uri")\n                if not resource_uri:\n                    return {\n                        "error": {\n                            "code": -32602,\n                            "message": "Invalid params - missing resource URI"\n                        },\n                        "id": request_id\n                    }\n                \n                response_data = await self._read_resource(resource_uri)\n            \n            elif method == "prompts/list":\n                # List available prompts\n                prompts = await self._list_prompts()\n                response_data = {"prompts": prompts}\n            \n            elif method == "prompts/get":\n                # Get a specific prompt\n                prompt_name = params.get("name")\n                prompt_arguments = params.get("arguments", {})\n                \n                if not prompt_name:\n                    return {\n                        "error": {\n                            "code": -32602,\n                            "message": "Invalid params - missing prompt name"\n                        },\n                        "id": request_id\n                    }\n                \n                response_data = await self._get_prompt(prompt_name, prompt_arguments)\n            \n            elif method == "initialize":\n                # Initialize the connection\n                response_data = {\n                    "protocolVersion": "2024-11-05",\n                    "capabilities": {\n                        "tools": {"listChanged": True},\n                        "resources": {"subscribe": True, "listChanged": True},\n                        "prompts": {"listChanged": True},\n                        "logging": {},\n                    },\n                    "serverInfo": {\n                        "name": "agent-zero",\n                        "version": "0.0.1"\n                    }\n                }\n            \n            else:\n                return {\n                    "error": {\n                        "code": -32601,\n                        "message": f"Method not found: {method}"\n                    },\n                    "id": request_id\n                }\n            \n            # Record metrics\n            if self.metrics:\n                response_time = time.time() - start_time\n                self.metrics.record_response_time(method, response_time)\n            \n            if self.health_checker:\n                self.health_checker.record_request()\n            \n            return {\n                "result": response_data,\n                "id": request_id\n            }\n        \n        except Exception as e:\n            mcp_logger.error(f"Error handling MCP request: {e}", exc_info=True)\n            \n            if self.health_checker:\n                self.health_checker.record_error()\n            \n            return {\n                "error": {\n                    "code": -32603,\n                    "message": f"Internal error: {str(e)}"\n                },\n                "id": request_data.get("id")\n            }\n    \n    async def _list_tools(self) -> list:\n        """List available MCP tools."""\n        # Get tools from the mcp instance\n        tools = []\n        \n        # This is a simplified version - in a real implementation,\n        # you'd introspect the mcp object to get the actual tools\n        sample_tools = [\n            {\n                "name": "list_databases",\n                "description": "List all databases in the ClickHouse server",\n                "inputSchema": {\n                    "type": "object",\n                    "properties": {},\n                    "required": []\n                }\n            },\n            {\n                "name": "monitor_current_processes",\n                "description": "Get information about currently running processes",\n                "inputSchema": {\n                    "type": "object",\n                    "properties": {},\n                    "required": []\n                }\n            },\n            {\n                "name": "run_select_query",\n                "description": "Execute a read-only SELECT query",\n                "inputSchema": {\n                    "type": "object",\n                    "properties": {\n                        "query": {\n                            "type": "string",\n                            "description": "The SQL query to execute"\n                        }\n                    },\n                    "required": ["query"]\n                }\n            }\n        ]\n        \n        # Apply tool limit\n        if self.config.tool_limit:\n            sample_tools = sample_tools[:self.config.tool_limit]\n        \n        return sample_tools\n    \n    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:\n        """Call a specific MCP tool."""\n        try:\n            # This would integrate with the actual mcp tools\n            # For now, return a placeholder response\n            return {\n                "content": [\n                    {\n                        "type": "text",\n                        "text": f"Tool {tool_name} called with arguments: {arguments}"\n                    }\n                ],\n                "isError": False\n            }\n        except Exception as e:\n            return {\n                "content": [\n                    {\n                        "type": "text",\n                        "text": f"Error calling tool {tool_name}: {str(e)}"\n                    }\n                ],\n                "isError": True\n            }\n    \n    async def _list_resources(self) -> list:\n        """List available MCP resources."""\n        return []\n    \n    async def _read_resource(self, uri: str) -> Dict[str, Any]:\n        """Read a specific MCP resource."""\n        return {\n            "contents": [\n                {\n                    "uri": uri,\n                    "mimeType": "text/plain",\n                    "text": f"Resource content for {uri}"\n                }\n            ]\n        }\n    \n    async def _list_prompts(self) -> list:\n        """List available MCP prompts."""\n        return []\n    \n    async def _get_prompt(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:\n        """Get a specific MCP prompt."""\n        return {\n            "description": f"Prompt {name}",\n            "messages": [\n                {\n                    "role": "user",\n                    "content": {\n                        "type": "text",\n                        "text": f"Prompt {name} with arguments: {arguments}"\n                    }\n                }\n            ]\n        }\n    \n    async def handle_http_request(self, request: web.Request) -> web.Response:\n        """Handle HTTP MCP request."""\n        try:\n            # Get client identifier\n            client_id = request.remote or "unknown"\n            \n            if self.metrics:\n                self.metrics.record_client_connection(client_id)\n            \n            # Parse JSON request\n            request_data = await request.json()\n            \n            # Handle the MCP request\n            response_data = await self.handle_mcp_request(request_data, client_id)\n            \n            return web.json_response(response_data)\n        \n        except json.JSONDecodeError:\n            return web.json_response(\n                {\n                    "error": {\n                        "code": -32700,\n                        "message": "Parse error - invalid JSON"\n                    }\n                },\n                status=400\n            )\n        except Exception as e:\n            logger.error(f"Error handling HTTP request: {e}", exc_info=True)\n            return web.json_response(\n                {\n                    "error": {\n                        "code": -32603,\n                        "message": "Internal error"\n                    }\n                },\n                status=500\n            )\n    \n    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:\n        """Handle WebSocket MCP connection."""\n        ws = web.WebSocketResponse()\n        await ws.prepare(request)\n        \n        self.active_websockets.add(ws)\n        if self.health_checker:\n            self.health_checker.active_connections += 1\n        \n        client_id = request.remote or "unknown"\n        \n        if self.metrics:\n            self.metrics.record_client_connection(client_id)\n        \n        logger.info(f"WebSocket connection established from {client_id}")\n        \n        try:\n            async for msg in ws:\n                if msg.type == WSMsgType.TEXT:\n                    try:\n                        request_data = json.loads(msg.data)\n                        response_data = await self.handle_mcp_request(request_data, client_id)\n                        await ws.send_str(json.dumps(response_data))\n                    except json.JSONDecodeError:\n                        error_response = {\n                            "error": {\n                                "code": -32700,\n                                "message": "Parse error - invalid JSON"\n                            }\n                        }\n                        await ws.send_str(json.dumps(error_response))\n                elif msg.type == WSMsgType.ERROR:\n                    logger.error(f"WebSocket error: {ws.exception()}")\n                    break\n        \n        except Exception as e:\n            logger.error(f"Error in WebSocket handler: {e}", exc_info=True)\n        \n        finally:\n            self.active_websockets.discard(ws)\n            if self.health_checker:\n                self.health_checker.active_connections -= 1\n            logger.info(f"WebSocket connection closed for {client_id}")\n        \n        return ws\n    \n    async def handle_sse(self, request: web.Request) -> web.StreamResponse:\n        """Handle Server-Sent Events MCP connection."""\n        resp = web.StreamResponse(\n            status=200,\n            reason='OK',\n            headers={\n                'Content-Type': 'text/event-stream',\n                'Cache-Control': 'no-cache',\n                'Connection': 'keep-alive',\n            }\n        )\n        await resp.prepare(request)\n        \n        client_id = request.remote or "unknown"\n        \n        if self.metrics:\n            self.metrics.record_client_connection(client_id)\n        \n        if self.health_checker:\n            self.health_checker.active_connections += 1\n        \n        logger.info(f"SSE connection established from {client_id}")\n        \n        try:\n            # Send initial connection message\n            await resp.write(b'data: {"type": "connection", "status": "connected"}\\n\\n')\n            \n            # Keep connection alive\n            while True:\n                await asyncio.sleep(30)  # Send keepalive every 30 seconds\n                await resp.write(b'data: {"type": "keepalive"}\\n\\n')\n        \n        except Exception as e:\n            logger.error(f"Error in SSE handler: {e}", exc_info=True)\n        \n        finally:\n            if self.health_checker:\n                self.health_checker.active_connections -= 1\n            logger.info(f"SSE connection closed for {client_id}")\n        \n        return resp\n    \n    async def handle_health_check(self, request: web.Request) -> web.Response:\n        """Handle health check request."""\n        if not self.health_checker:\n            return web.json_response({"status": "disabled"}, status=404)\n        \n        health_status = self.health_checker.get_health_status()\n        return web.json_response(health_status)\n    \n    async def handle_metrics(self, request: web.Request) -> web.Response:\n        """Handle metrics request."""\n        if not self.metrics:\n            return web.json_response({"error": "Metrics disabled"}, status=404)\n        \n        metrics_data = self.metrics.get_metrics()\n        return web.json_response(metrics_data)\n    \n    def create_app(self) -> web.Application:\n        """Create the aiohttp application."""\n        app = web.Application()\n        \n        # Add CORS support\n        if cors_setup:\n            cors = cors_setup(app, defaults={\n                "*": ResourceOptions(\n                    allow_credentials=True,\n                    expose_headers="*",\n                    allow_headers="*",\n                    allow_methods="*"\n                )\n            })\n        \n        # MCP endpoints\n        app.router.add_post('/mcp', self.handle_http_request)\n        app.router.add_get('/mcp/websocket', self.handle_websocket)\n        app.router.add_get('/mcp/sse', self.handle_sse)\n        \n        # Health and metrics endpoints\n        if self.config.enable_health_check:\n            app.router.add_get('/health', self.handle_health_check)\n        \n        if self.config.enable_metrics:\n            app.router.add_get('/metrics', self.handle_metrics)\n        \n        # Add authentication middleware if configured\n        if self.config.get_auth_config() or self.config.get_oauth_config():\n            app.middlewares.append(self.auth_middleware)\n        \n        self.app = app\n        return app\n    \n    async def auth_middleware(self, request: web.Request, handler):\n        """Authentication middleware."""\n        # Skip auth for health check\n        if request.path == '/health':\n            return await handler(request)\n        \n        auth_config = self.config.get_auth_config()\n        if auth_config:\n            # Basic authentication\n            auth_header = request.headers.get('Authorization', '')\n            if not auth_header.startswith('Basic '):\n                return web.Response(status=401, text='Authentication required')\n            \n            try:\n                import base64\n                encoded_credentials = auth_header[6:]\n                credentials = base64.b64decode(encoded_credentials).decode()\n                username, password = credentials.split(':', 1)\n                \n                if username != auth_config['username'] or password != auth_config['password']:\n                    return web.Response(status=401, text='Invalid credentials')\n            except Exception:\n                return web.Response(status=401, text='Invalid authorization header')\n        \n        # OAuth 2.0 support would be implemented here\n        oauth_config = self.config.get_oauth_config()\n        if oauth_config:\n            # Simplified OAuth check - in production, you'd validate the token\n            token_header = request.headers.get('Authorization', '')\n            if not token_header.startswith('Bearer '):\n                return web.Response(status=401, text='Bearer token required')\n        \n        return await handler(request)\n    \n    async def start_server(self):\n        """Start the standalone MCP server."""\n        if not aiohttp:\n            raise RuntimeError(\"aiohttp is required for standalone server mode. Install with: pip install aiohttp aiohttp-cors\")\n        \n        app = self.create_app()\n        \n        # Create runner\n        runner = web.AppRunner(app)\n        await runner.setup()\n        \n        # Create site\n        ssl_context = None\n        if self.config.get_ssl_config():\n            import ssl\n            ssl_config = self.config.get_ssl_config()\n            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)\n            ssl_context.load_cert_chain(ssl_config['certfile'], ssl_config['keyfile'])\n        \n        site = web.TCPSite(runner, self.config.host, self.config.port, ssl_context=ssl_context)\n        await site.start()\n        \n        protocol = 'https' if ssl_context else 'http'\n        logger.info(f\"Standalone MCP server started on {protocol}://{self.config.host}:{self.config.port}\")\n        logger.info(f\"Available endpoints:\")\n        logger.info(f\"  - HTTP MCP: {protocol}://{self.config.host}:{self.config.port}/mcp\")\n        logger.info(f\"  - WebSocket MCP: {protocol.replace('http', 'ws')}://{self.config.host}:{self.config.port}/mcp/websocket\")\n        logger.info(f\"  - SSE MCP: {protocol}://{self.config.host}:{self.config.port}/mcp/sse\")\n        \n        if self.config.enable_health_check:\n            logger.info(f\"  - Health Check: {protocol}://{self.config.host}:{self.config.port}/health\")\n        \n        if self.config.enable_metrics:\n            logger.info(f\"  - Metrics: {protocol}://{self.config.host}:{self.config.port}/metrics\")\n        \n        # Keep server running\n        try:\n            while True:\n                await asyncio.sleep(3600)  # Sleep for 1 hour\n        except KeyboardInterrupt:\n            logger.info(\"Shutting down server...\")\n        finally:\n            await runner.cleanup()\n\n\nasync def run_standalone_server(server_config: ServerConfig):\n    \"\"\"Run the standalone MCP server.\"\"\"\n    server = StandaloneMCPServer(server_config)\n    await server.start_server()\n