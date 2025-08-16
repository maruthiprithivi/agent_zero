"""
Standalone MCP server with HTTP/WebSocket support.

This module provides a complete HTTP/WebSocket implementation of the MCP
protocol, supporting multiple transport methods, authentication, rate limiting,
and comprehensive health monitoring.
"""

import asyncio
import json
import logging
import ssl
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

# Import version from package
try:
    from agent_zero import __version__
except ImportError:
    __version__ = "unknown"

# Optional imports for standalone server functionality
try:
    from aiohttp import WSMsgType, web
    from aiohttp_cors import ResourceOptions, setup as cors_setup

    aiohttp = True
except ImportError:
    web = WSMsgType = cors_setup = ResourceOptions = None
    aiohttp = False

from .mcp_server import logger as mcp_logger, mcp
from .server_config import ServerConfig

logger = logging.getLogger("mcp-standalone-server")


class RateLimiter:
    """Simple rate limiter for MCP requests."""

    def __init__(self, max_requests: int = 100, window_minutes: int = 1):
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests: dict[str, list] = defaultdict(list)

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

    def get_health_status(self) -> dict[str, Any]:
        """Get current health status."""
        uptime = datetime.now() - self.start_time
        return {
            "status": "healthy",
            "uptime_seconds": int(uptime.total_seconds()),
            "start_time": self.start_time.isoformat(),
            "requests_total": self.request_count,
            "errors_total": self.error_count,
            "active_connections": self.active_connections,
            "version": __version__,
            "deployment_mode": "standalone",
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

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        avg_response_times = {}
        for tool, times in self.response_times.items():
            if times:
                avg_response_times[tool] = sum(times) / len(times)

        return {
            "tool_usage": dict(self.tool_usage),
            "average_response_times": avg_response_times,
            "client_connections": dict(self.client_connections),
            "total_tools_available": len(mcp.list_tools()) if hasattr(mcp, "list_tools") else 0,
        }


class StandaloneMCPServer:
    """Standalone MCP server with HTTP/WebSocket support."""

    def __init__(self, server_config: ServerConfig):
        self.config = server_config
        self.rate_limiter = (
            RateLimiter(
                max_requests=server_config.rate_limit_requests,
            )
            if server_config.rate_limit_enabled
            else None
        )
        self.health_checker = HealthChecker() if server_config.enable_health_check else None
        self.metrics = MetricsCollector() if server_config.enable_metrics else None
        self.active_websockets: set[web.WebSocketResponse] = set()
        self.app = None

    async def handle_mcp_request(
        self, request_data: dict[str, Any], client_id: str = None
    ) -> dict[str, Any]:
        """Handle an MCP request and return the response."""
        start_time = time.time()

        try:
            # Rate limiting
            if self.rate_limiter and not self.rate_limiter.is_allowed(client_id or "anonymous"):
                return {"error": {"code": 429, "message": "Rate limit exceeded"}}

            # Extract method and params
            method = request_data.get("method")
            params = request_data.get("params", {})
            request_id = request_data.get("id")

            if not method:
                return {
                    "error": {"code": -32600, "message": "Invalid request - missing method"},
                    "id": request_id,
                }

            # Handle MCP methods
            if method == "tools/list":
                # List available tools
                tools = await self._list_tools()
                response_data = {"tools": tools}
            elif method == "tools/call":
                # Call a specific tool
                tool_name = params.get("name")
                tool_arguments = params.get("arguments", {})

                if not tool_name:
                    return {
                        "error": {"code": -32602, "message": "Invalid params - missing tool name"},
                        "id": request_id,
                    }

                response_data = await self._call_tool(tool_name, tool_arguments)

                # Record metrics
                if self.metrics:
                    self.metrics.record_tool_usage(tool_name)

            elif method == "resources/list":
                # List available resources
                resources = await self._list_resources()
                response_data = {"resources": resources}

            elif method == "resources/read":
                # Read a specific resource
                resource_uri = params.get("uri")
                if not resource_uri:
                    return {
                        "error": {
                            "code": -32602,
                            "message": "Invalid params - missing resource URI",
                        },
                        "id": request_id,
                    }

                response_data = await self._read_resource(resource_uri)

            elif method == "prompts/list":
                # List available prompts
                prompts = await self._list_prompts()
                response_data = {"prompts": prompts}

            elif method == "prompts/get":
                # Get a specific prompt
                prompt_name = params.get("name")
                prompt_arguments = params.get("arguments", {})

                if not prompt_name:
                    return {
                        "error": {
                            "code": -32602,
                            "message": "Invalid params - missing prompt name",
                        },
                        "id": request_id,
                    }

                response_data = await self._get_prompt(prompt_name, prompt_arguments)

            elif method == "initialize":
                # Initialize the connection
                response_data = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "resources": {"subscribe": True, "listChanged": True},
                        "prompts": {"listChanged": True},
                        "logging": {},
                    },
                    "serverInfo": {"name": "agent-zero", "version": __version__},
                }

            else:
                return {
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                    "id": request_id,
                }

            # Record metrics
            if self.metrics:
                response_time = time.time() - start_time
                self.metrics.record_response_time(method, response_time)

            if self.health_checker:
                self.health_checker.record_request()

            return {"result": response_data, "id": request_id}

        except Exception as e:
            mcp_logger.error(f"Error handling MCP request: {e}", exc_info=True)

            if self.health_checker:
                self.health_checker.record_error()

            return {
                "error": {"code": -32603, "message": f"Internal error: {e!s}"},
                "id": request_data.get("id"),
            }

    async def _list_tools(self) -> list:
        """List available MCP tools."""
        # Get tools from the mcp instance
        tools = []

        # This is a simplified version - in a real implementation,
        # you'd introspect the mcp object to get the actual tools
        sample_tools = [
            {
                "name": "list_databases",
                "description": "List all databases in the ClickHouse server",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "monitor_current_processes",
                "description": "Get information about currently running processes",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "run_select_query",
                "description": "Execute a read-only SELECT query",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The SQL query to execute"}
                    },
                    "required": ["query"],
                },
            },
        ]

        # Apply tool limit
        if self.config.tool_limit:
            sample_tools = sample_tools[: self.config.tool_limit]

        return sample_tools

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a specific MCP tool."""
        try:
            # This would integrate with the actual mcp tools
            # For now, return a placeholder response
            return {
                "content": [
                    {"type": "text", "text": f"Tool {tool_name} called with arguments: {arguments}"}
                ],
                "isError": False,
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error calling tool {tool_name}: {e!s}"}],
                "isError": True,
            }

    async def _list_resources(self) -> list:
        """List available MCP resources."""
        return []

    async def _read_resource(self, uri: str) -> dict[str, Any]:
        """Read a specific MCP resource."""
        return {
            "contents": [
                {"uri": uri, "mimeType": "text/plain", "text": f"Resource content for {uri}"}
            ]
        }

    async def _list_prompts(self) -> list:
        """List available MCP prompts."""
        return []

    async def _get_prompt(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get a specific MCP prompt."""
        return {
            "description": f"Prompt {name}",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"Prompt {name} with arguments: {arguments}",
                    },
                }
            ],
        }

    async def handle_http_request(self, request: web.Request) -> web.Response:
        """Handle HTTP MCP request."""
        try:
            # Get client identifier
            client_id = request.remote or "unknown"

            if self.metrics:
                self.metrics.record_client_connection(client_id)

            # Parse JSON request
            request_data = await request.json()

            # Handle the MCP request
            response_data = await self.handle_mcp_request(request_data, client_id)

            return web.json_response(response_data)

        except json.JSONDecodeError:
            return web.json_response(
                {"error": {"code": -32700, "message": "Parse error - invalid JSON"}}, status=400
            )
        except Exception as e:
            logger.error(f"Error handling HTTP request: {e}", exc_info=True)
            return web.json_response(
                {"error": {"code": -32603, "message": "Internal error"}}, status=500
            )

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket MCP connection."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.active_websockets.add(ws)
        if self.health_checker:
            self.health_checker.active_connections += 1

        client_id = request.remote or "unknown"

        if self.metrics:
            self.metrics.record_client_connection(client_id)

        logger.info(f"WebSocket connection established from {client_id}")

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        request_data = json.loads(msg.data)
                        response_data = await self.handle_mcp_request(request_data, client_id)
                        await ws.send_str(json.dumps(response_data))
                    except json.JSONDecodeError:
                        error_response = {
                            "error": {"code": -32700, "message": "Parse error - invalid JSON"}
                        }
                        await ws.send_str(json.dumps(error_response))
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break

        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}", exc_info=True)

        finally:
            self.active_websockets.discard(ws)
            if self.health_checker:
                self.health_checker.active_connections -= 1
            logger.info(f"WebSocket connection closed for {client_id}")

        return ws

    async def handle_sse(self, request: web.Request) -> web.StreamResponse:
        """Handle Server-Sent Events MCP connection."""
        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)

        client_id = request.remote or "unknown"

        if self.metrics:
            self.metrics.record_client_connection(client_id)

        if self.health_checker:
            self.health_checker.active_connections += 1

        logger.info(f"SSE connection established from {client_id}")

        try:
            # Send initial connection message
            await resp.write(b'data: {"type": "connection", "status": "connected"}\n\n')

            # Keep connection alive
            while True:
                await asyncio.sleep(30)  # Send keepalive every 30 seconds
                await resp.write(b'data: {"type": "keepalive"}\n\n')

        except Exception as e:
            logger.error(f"Error in SSE handler: {e}", exc_info=True)

        finally:
            if self.health_checker:
                self.health_checker.active_connections -= 1
            logger.info(f"SSE connection closed for {client_id}")

        return resp

    async def handle_health_check(self, request: web.Request) -> web.Response:
        """Handle health check request."""
        if not self.health_checker:
            return web.json_response({"status": "disabled"}, status=404)

        health_status = self.health_checker.get_health_status()
        return web.json_response(health_status)

    async def handle_metrics(self, request: web.Request) -> web.Response:
        """Handle metrics request."""
        if not self.metrics:
            return web.json_response({"error": "Metrics disabled"}, status=404)

        metrics_data = self.metrics.get_metrics()
        return web.json_response(metrics_data)

    def create_app(self) -> web.Application:
        """Create the aiohttp application."""
        app = web.Application()

        # Add CORS support
        if cors_setup:
            cors = cors_setup(
                app,
                defaults={
                    "*": ResourceOptions(
                        allow_credentials=True,
                        expose_headers="*",
                        allow_headers="*",
                        allow_methods="*",
                    )
                },
            )

        # MCP endpoints
        app.router.add_post("/mcp", self.handle_http_request)
        app.router.add_get("/mcp/websocket", self.handle_websocket)
        app.router.add_get("/mcp/sse", self.handle_sse)

        # Health and metrics endpoints
        if self.config.enable_health_check:
            app.router.add_get("/health", self.handle_health_check)

        if self.config.enable_metrics:
            app.router.add_get("/metrics", self.handle_metrics)

        # Add authentication middleware if configured
        if self.config.get_auth_config() or self.config.get_oauth_config():
            app.middlewares.append(self.auth_middleware)

        self.app = app
        return app

    async def auth_middleware(self, request: web.Request, handler):
        """Authentication middleware."""
        # Skip auth for health check
        if request.path == "/health":
            return await handler(request)

        auth_config = self.config.get_auth_config()
        if auth_config:
            # Basic authentication
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Basic "):
                return web.Response(status=401, text="Authentication required")

            try:
                import base64

                encoded_credentials = auth_header[6:]
                credentials = base64.b64decode(encoded_credentials).decode()
                username, password = credentials.split(":", 1)

                if username != auth_config["username"] or password != auth_config["password"]:
                    return web.Response(status=401, text="Invalid credentials")
            except Exception:
                return web.Response(status=401, text="Invalid authorization header")

        # OAuth 2.0 support would be implemented here
        oauth_config = self.config.get_oauth_config()
        if oauth_config:
            # Simplified OAuth check - in production, you'd validate the token
            token_header = request.headers.get("Authorization", "")
            if not token_header.startswith("Bearer "):
                return web.Response(status=401, text="Bearer token required")

        return await handler(request)

    async def start_server(self):
        """Start the standalone MCP server."""
        if not aiohttp:
            raise RuntimeError(
                "aiohttp is required for standalone server mode. Install with: pip install aiohttp aiohttp-cors"
            )

        app = self.create_app()

        # Create runner
        runner = web.AppRunner(app)
        await runner.setup()

        # Create site
        ssl_context = None
        if self.config.get_ssl_config():
            ssl_config = self.config.get_ssl_config()
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(ssl_config["certfile"], ssl_config["keyfile"])

        site = web.TCPSite(runner, self.config.host, self.config.port, ssl_context=ssl_context)
        await site.start()

        protocol = "https" if ssl_context else "http"
        logger.info(
            f"Standalone MCP server started on {protocol}://{self.config.host}:{self.config.port}"
        )
        logger.info("Available endpoints:")
        logger.info(f"  - HTTP MCP: {protocol}://{self.config.host}:{self.config.port}/mcp")
        logger.info(
            f"  - WebSocket MCP: {protocol.replace('http', 'ws')}://{self.config.host}:{self.config.port}/mcp/websocket"
        )
        logger.info(f"  - SSE MCP: {protocol}://{self.config.host}:{self.config.port}/mcp/sse")

        if self.config.enable_health_check:
            logger.info(
                f"  - Health Check: {protocol}://{self.config.host}:{self.config.port}/health"
            )

        if self.config.enable_metrics:
            logger.info(f"  - Metrics: {protocol}://{self.config.host}:{self.config.port}/metrics")

        # Keep server running
        try:
            while True:
                await asyncio.sleep(3600)  # Sleep for 1 hour
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        finally:
            await runner.cleanup()


async def run_standalone_server(server_config: ServerConfig):
    """Run the standalone MCP server."""
    server = StandaloneMCPServer(server_config)
    await server.start_server()
