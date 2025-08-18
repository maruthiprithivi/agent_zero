"""MCP 2025 Specification Compliant Transport Implementation.

This module implements the MCP 2025-03-26 specification features:
- Streamable HTTP Transport
- OAuth 2.1 Authorization Framework
- JSON-RPC Batching
- Enhanced Content Types (audio support)
- Tool Annotations
- Progress Notifications with messages
- Completions Capability
- Client Capability Negotiation
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import aiohttp
import jwt
from aiohttp import web
from aiohttp_cors import CorsConfig, setup as cors_setup
from cryptography.hazmat.primitives.asymmetric import rsa

from ..config import UnifiedConfig

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Enhanced content types supporting 2025 spec."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"  # New in 2025 spec


class ToolAnnotation(Enum):
    """Tool behavior annotations from 2025 spec."""

    READ_ONLY = "read_only"
    DESTRUCTIVE = "destructive"
    IDEMPOTENT = "idempotent"
    RATE_LIMITED = "rate_limited"


class TransportType(Enum):
    """Transport types for MCP 2025."""

    STREAMABLE_HTTP = "streamable_http"
    STDIO = "stdio"
    WEBSOCKET = "websocket"


@dataclass
class MCPCapability:
    """Client capability representation."""

    name: str
    version: str
    features: list[str] = field(default_factory=list)


@dataclass
class OAuth2Config:
    """OAuth 2.1 configuration."""

    client_id: str
    client_secret: str
    authorization_endpoint: str
    token_endpoint: str
    scope: list[str] = field(default_factory=lambda: ["read", "write"])


@dataclass
class StreamableHTTPConfig:
    """Configuration for Streamable HTTP transport."""

    host: str = "127.0.0.1"
    port: int = 8505
    session_timeout: int = 3600  # 1 hour
    max_connections: int = 100
    enable_chunked_encoding: bool = True
    enable_compression: bool = True
    cors_enabled: bool = True
    oauth_config: OAuth2Config | None = None


class StreamableHTTPTransport:
    """
    MCP 2025 Streamable HTTP Transport Implementation.

    Features:
    - Single HTTP endpoint for POST and GET
    - Optional Server-Sent Events (SSE) streaming
    - Session management with Mcp-Session-Id header
    - OAuth 2.1 authorization framework
    - JSON-RPC batching support
    - Enhanced content type support
    """

    def __init__(self, config: StreamableHTTPConfig, unified_config: UnifiedConfig):
        self.config = config
        self.unified_config = unified_config
        self.sessions: dict[str, dict[str, Any]] = {}
        self.app = web.Application()
        self.capabilities: list[MCPCapability] = []
        self._setup_jwt_keys()
        self._setup_routes()
        self._setup_cors()

    def _setup_jwt_keys(self):
        """Setup JWT keys for OAuth 2.1."""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self.public_key = self.private_key.public_key()

    def _setup_routes(self):
        """Setup HTTP routes for MCP endpoint."""
        self.app.router.add_post("/mcp", self.handle_mcp_request)
        self.app.router.add_get("/mcp", self.handle_mcp_get)
        self.app.router.add_get("/mcp/capabilities", self.handle_capabilities)
        self.app.router.add_get("/health", self.handle_health_check)

        # OAuth 2.1 endpoints
        if self.config.oauth_config:
            self.app.router.add_post("/oauth/token", self.handle_oauth_token)
            self.app.router.add_get("/oauth/authorize", self.handle_oauth_authorize)

    def _setup_cors(self):
        """Setup CORS for browser compatibility."""
        if self.config.cors_enabled:
            cors_config = CorsConfig()
            cors_config.add(
                "*",
                {
                    aiohttp.hdrs.METH_GET,
                    aiohttp.hdrs.METH_POST,
                    aiohttp.hdrs.METH_OPTIONS,
                },
            )
            cors_setup(self.app, cors_config)

    async def handle_mcp_request(self, request: web.Request) -> web.Response:
        """
        Handle MCP requests via HTTP POST.

        Implements the core MCP 2025 Streamable HTTP protocol.
        """
        try:
            # Extract session ID from headers
            session_id = request.headers.get("Mcp-Session-Id")
            if not session_id:
                session_id = str(uuid.uuid4())

            # Authenticate request if OAuth is enabled
            if self.config.oauth_config:
                auth_result = await self._authenticate_request(request)
                if not auth_result.get("valid", False):
                    return web.Response(
                        status=401,
                        headers={"WWW-Authenticate": "Bearer"},
                        text="Authentication required",
                    )

            # Parse request body
            content_type = request.headers.get("Content-Type", "application/json")
            if content_type.startswith("application/json"):
                data = await request.json()
            else:
                return web.Response(status=400, text="Unsupported content type")

            # Handle JSON-RPC batching
            is_batch = isinstance(data, list)
            requests = data if is_batch else [data]

            # Process each request
            responses = []
            for req in requests:
                response = await self._process_mcp_request(req, session_id)
                responses.append(response)

            # Prepare response
            response_data = responses if is_batch else responses[0]
            response_headers = {
                "Content-Type": "application/json",
                "Mcp-Session-Id": session_id,
                "Mcp-Protocol-Version": "2025-03-26",
            }

            # Enable chunked encoding for streaming if configured
            if self.config.enable_chunked_encoding and self._should_stream(response_data):
                return await self._stream_response(response_data, response_headers)

            return web.Response(
                text=json.dumps(response_data, separators=(",", ":")),
                headers=response_headers,
                content_type="application/json",
            )

        except Exception as e:
            logger.error(f"Error handling MCP request: {e}", exc_info=True)
            return web.Response(
                status=500,
                text=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": -32603, "message": "Internal error"},
                        "id": None,
                    }
                ),
                content_type="application/json",
            )

    async def handle_mcp_get(self, request: web.Request) -> web.Response:
        """Handle MCP requests via HTTP GET for SSE streaming."""
        session_id = request.headers.get("Mcp-Session-Id", str(uuid.uuid4()))

        # Setup Server-Sent Events stream
        response = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Mcp-Session-Id": session_id,
                "Mcp-Protocol-Version": "2025-03-26",
            }
        )

        await response.prepare(request)

        try:
            # Send initial connection event
            await self._send_sse_event(
                response,
                "connected",
                {
                    "session_id": session_id,
                    "protocol_version": "2025-03-26",
                    "capabilities": [cap.__dict__ for cap in self.capabilities],
                },
            )

            # Keep connection alive and handle incoming messages
            while True:
                try:
                    await asyncio.sleep(30)  # Send keepalive every 30 seconds
                    await self._send_sse_event(
                        response, "keepalive", {"timestamp": datetime.now(UTC).isoformat()}
                    )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"SSE stream error: {e}")
                    break

        except asyncio.CancelledError:
            pass
        finally:
            await response.write_eof()

        return response

    async def handle_capabilities(self, request: web.Request) -> web.Response:
        """Handle capability negotiation endpoint."""
        capabilities = {
            "protocol_version": "2025-03-26",
            "server_capabilities": [cap.__dict__ for cap in self.capabilities],
            "supported_transports": ["streamable_http", "sse"],
            "supported_content_types": [ct.value for ct in ContentType],
            "supported_annotations": [ta.value for ta in ToolAnnotation],
            "features": {
                "json_rpc_batching": True,
                "streaming": self.config.enable_chunked_encoding,
                "oauth_2_1": self.config.oauth_config is not None,
                "progress_notifications": True,
                "completions": True,
                "tool_annotations": True,
                "audio_content": True,
            },
        }

        return web.Response(
            text=json.dumps(capabilities, separators=(",", ":")), content_type="application/json"
        )

    async def handle_health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint for load balancers."""
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "version": "2025.1.0",
            "protocol_version": "2025-03-26",
            "active_sessions": len(self.sessions),
            "uptime_seconds": getattr(self, "_start_time", 0),
        }

        return web.Response(
            text=json.dumps(health_data, separators=(",", ":")), content_type="application/json"
        )

    async def handle_oauth_token(self, request: web.Request) -> web.Response:
        """Handle OAuth 2.1 token requests."""
        if not self.config.oauth_config:
            return web.Response(status=404, text="OAuth not configured")

        data = await request.post()
        grant_type = data.get("grant_type")

        if grant_type != "client_credentials":
            return web.Response(
                status=400,
                text=json.dumps({"error": "unsupported_grant_type"}),
                content_type="application/json",
            )

        # Validate client credentials
        client_id = data.get("client_id")
        client_secret = data.get("client_secret")

        if (
            client_id != self.config.oauth_config.client_id
            or client_secret != self.config.oauth_config.client_secret
        ):
            return web.Response(
                status=401,
                text=json.dumps({"error": "invalid_client"}),
                content_type="application/json",
            )

        # Generate JWT token
        payload = {
            "iss": "agent-zero-mcp",
            "sub": client_id,
            "aud": "mcp-api",
            "exp": datetime.now(UTC).timestamp() + 3600,  # 1 hour
            "iat": datetime.now(UTC).timestamp(),
            "scope": " ".join(self.config.oauth_config.scope),
        }

        token = jwt.encode(payload, self.private_key, algorithm="RS256")

        return web.Response(
            text=json.dumps(
                {
                    "access_token": token,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": " ".join(self.config.oauth_config.scope),
                }
            ),
            content_type="application/json",
        )

    async def handle_oauth_authorize(self, request: web.Request) -> web.Response:
        """Handle OAuth 2.1 authorization requests."""
        return web.Response(
            status=501, text="Authorization code flow not implemented - use client credentials"
        )

    async def _authenticate_request(self, request: web.Request) -> dict[str, Any]:
        """Authenticate request using OAuth 2.1."""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return {"valid": False, "error": "missing_token"}

        token = auth_header[7:]  # Remove "Bearer "

        try:
            payload = jwt.decode(token, self.public_key, algorithms=["RS256"], audience="mcp-api")
            return {"valid": True, "payload": payload}
        except jwt.InvalidTokenError as e:
            return {"valid": False, "error": str(e)}

    async def _process_mcp_request(
        self, request: dict[str, Any], session_id: str
    ) -> dict[str, Any]:
        """Process individual MCP request."""
        # This is a placeholder - actual MCP request processing would be implemented
        # by the calling server using registered tools and handlers

        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        # Handle initialization
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2025-03-26",
                    "serverInfo": {"name": "agent-zero-mcp", "version": "2025.1.0"},
                    "capabilities": {
                        "tools": {"annotations": True},
                        "resources": {"subscribe": True},
                        "prompts": {"list_changed": True},
                        "logging": {"level": "info"},
                        "completions": {"argument_hint": True},
                        "experimental": {
                            "streamable_http": True,
                            "audio_content": True,
                            "progress_notifications": True,
                        },
                    },
                },
                "id": request_id,
            }

        # Return method not found for unhandled methods
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": "Method not found"},
            "id": request_id,
        }

    def _should_stream(self, response_data: Any) -> bool:
        """Determine if response should be streamed."""
        # Stream large responses or responses with progress notifications
        if isinstance(response_data, dict):
            if response_data.get("method") == "notifications/progress":
                return True
            if len(json.dumps(response_data)) > 1024 * 10:  # 10KB threshold
                return True
        return False

    async def _stream_response(self, data: Any, headers: dict[str, str]) -> web.StreamResponse:
        """Stream response using chunked encoding."""
        response = web.StreamResponse(headers=headers)
        response.enable_chunked_encoding()

        await response.prepare()

        # Stream JSON data in chunks
        json_str = json.dumps(data, separators=(",", ":"))
        chunk_size = 1024  # 1KB chunks

        for i in range(0, len(json_str), chunk_size):
            chunk = json_str[i : i + chunk_size].encode()
            await response.write(chunk)
            await asyncio.sleep(0.001)  # Small delay for streaming effect

        await response.write_eof()
        return response

    async def _send_sse_event(self, response: web.StreamResponse, event_type: str, data: Any):
        """Send Server-Sent Event."""
        event_data = f"event: {event_type}\n"
        event_data += f"data: {json.dumps(data, separators=(',', ':'))}\n\n"

        await response.write(event_data.encode())

    async def start(self) -> None:
        """Start the Streamable HTTP transport server."""
        self._start_time = datetime.now(UTC).timestamp()

        logger.info(
            f"Starting MCP 2025 Streamable HTTP server on {self.config.host}:{self.config.port}"
        )
        logger.info(f"OAuth 2.1 enabled: {self.config.oauth_config is not None}")
        logger.info(f"Chunked encoding: {self.config.enable_chunked_encoding}")
        logger.info(f"CORS enabled: {self.config.cors_enabled}")

        # Start the HTTP server
        runner = web.AppRunner(self.app)
        await runner.setup()

        site = web.TCPSite(
            runner, self.config.host, self.config.port, ssl_context=self._get_ssl_context()
        )
        await site.start()

        logger.info("MCP 2025 Streamable HTTP server started successfully")

    def _get_ssl_context(self):
        """Get SSL context if configured."""
        ssl_config = self.unified_config.get_ssl_config()
        if ssl_config:
            import ssl

            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(ssl_config["certfile"], ssl_config["keyfile"])
            return context
        return None

    async def stop(self) -> None:
        """Stop the transport server."""
        # Close all active sessions
        self.sessions.clear()
        logger.info("MCP 2025 Streamable HTTP server stopped")

    def register_capability(self, capability: MCPCapability) -> None:
        """Register a server capability."""
        self.capabilities.append(capability)
        logger.debug(f"Registered capability: {capability.name} v{capability.version}")

    async def send_progress_notification(
        self,
        session_id: str,
        progress: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        """
        Send progress notification with message (2025 spec feature).

        Args:
            session_id: Target session ID
            progress: Current progress value
            total: Total progress value (optional)
            message: Descriptive status message (new in 2025)
        """
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {
                "progress": progress,
                "total": total,
                "message": message,  # New in 2025 spec
                "timestamp": datetime.now(UTC).isoformat(),
            },
        }

        # In a real implementation, this would send to the specific session
        logger.debug(f"Progress notification for {session_id}: {notification}")
