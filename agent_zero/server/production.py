"""Production-grade server enhancements for Agent Zero MCP Server.

This module integrates all production operations capabilities including health checks,
metrics, logging, tracing, performance monitoring, and backup systems into a unified
production-ready server implementation.
"""

import asyncio
import json
import ssl
import time
from typing import Any

from aiohttp import WSMsgType, web
from aiohttp_cors import ResourceOptions, setup as cors_setup

# Import version from package
try:
    from agent_zero import __version__
except ImportError:
    __version__ = "unknown"

from .backup import BackupConfig, get_backup_manager
from .health import get_health_manager
from .logging import LogContext, correlation_id, get_logger, set_correlation_id
from .metrics import MetricsConfig, get_metrics_manager
from .performance import get_performance_monitor
from .tracing import TracingConfig, get_tracing_manager

logger = get_logger(__name__)


class ProductionMCPServer:
    """Production-grade MCP server with comprehensive monitoring and operations."""

    def __init__(self, config: dict[str, Any]):
        """Initialize production MCP server.

        Args:
            config: Server configuration
        """
        self.config = config
        self.app: web.Application | None = None

        # Initialize production subsystems
        self._init_monitoring_systems()

        # Server state
        self.active_websockets = set()
        self._shutdown_event = asyncio.Event()

    def _init_monitoring_systems(self) -> None:
        """Initialize all monitoring and operational systems."""
        # Initialize health check manager
        clickhouse_factory = self.config.get("clickhouse_client_factory")
        self.health_manager = get_health_manager(clickhouse_factory)

        # Initialize metrics manager
        metrics_config = MetricsConfig(
            enabled=self.config.get("enable_metrics", True),
            prefix=self.config.get("metrics_prefix", "agent_zero"),
        )
        self.metrics_manager = get_metrics_manager(metrics_config)

        # Initialize tracing manager
        tracing_config = TracingConfig(
            enabled=self.config.get("enable_tracing", True),
            service_name=self.config.get("service_name", "agent-zero-mcp"),
            environment=self.config.get("environment", "production"),
            export_endpoint=self.config.get("tracing_endpoint"),
        )
        self.tracing_manager = get_tracing_manager(tracing_config)

        # Initialize performance monitor
        perf_config = {"enabled": self.config.get("enable_performance_monitoring", True)}
        self.performance_monitor = get_performance_monitor(perf_config)

        # Initialize backup manager
        backup_config = BackupConfig(
            enabled=self.config.get("enable_backups", True),
            backup_dir=self.config.get("backup_dir", "/tmp/agent_zero_backups"),
            retention_days=self.config.get("backup_retention_days", 30),
        )
        self.backup_manager = get_backup_manager(backup_config, clickhouse_factory)

    async def start_monitoring(self) -> None:
        """Start all monitoring systems."""
        try:
            # Start health check background monitoring
            await self.health_manager.start_background_checks()

            logger.info("Production monitoring systems started successfully")

        except Exception as e:
            logger.error("Failed to start monitoring systems", exception=e)
            raise

    async def create_app(self) -> web.Application:
        """Create the production aiohttp application."""
        app = web.Application(
            middlewares=[
                self.correlation_middleware,
                self.metrics_middleware,
                self.performance_middleware,
                self.error_handling_middleware,
            ]
        )

        # Setup CORS
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
        app.router.add_post("/mcp", self._handle_mcp_request)
        app.router.add_get("/mcp/websocket", self._handle_websocket)
        app.router.add_get("/mcp/sse", self._handle_sse)

        # Production operations endpoints
        app.router.add_get("/health", self._handle_health_check)
        app.router.add_get("/health/ready", self._handle_readiness_check)
        app.router.add_get("/health/live", self._handle_liveness_check)
        app.router.add_get("/metrics", self._handle_metrics)
        app.router.add_get("/performance", self._handle_performance_summary)
        app.router.add_get("/backup/status", self._handle_backup_status)
        app.router.add_post("/backup/create", self._handle_create_backup)
        app.router.add_get("/admin/info", self._handle_admin_info)

        self.app = app
        return app

    @web.middleware
    async def correlation_middleware(self, request: web.Request, handler):
        """Middleware to set correlation ID for request tracing."""
        # Generate or extract correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = set_correlation_id()
        else:
            set_correlation_id(correlation_id)

        # Add correlation ID to response headers
        response = await handler(request)
        response.headers["X-Correlation-ID"] = correlation_id

        return response

    @web.middleware
    async def metrics_middleware(self, request: web.Request, handler):
        """Middleware to collect request metrics."""
        start_time = time.time()
        method = request.method
        endpoint = self._get_endpoint_name(request.path)

        try:
            response = await handler(request)
            status_code = response.status

            # Record successful request
            duration = time.time() - start_time
            self.metrics_manager.record_http_request(method, endpoint, status_code, duration)
            self.performance_monitor.record_request(f"{method}:{endpoint}", True)

            return response

        except Exception as e:
            # Record failed request
            duration = time.time() - start_time
            self.metrics_manager.record_http_request(method, endpoint, 500, duration)
            self.metrics_manager.record_error(type(e).__name__, "http_handler")
            self.performance_monitor.record_request(f"{method}:{endpoint}", False)
            raise

    @web.middleware
    async def performance_middleware(self, request: web.Request, handler):
        """Middleware to track performance metrics."""
        start_time = time.time()
        operation = f"{request.method}:{self._get_endpoint_name(request.path)}"

        try:
            response = await handler(request)

            # Record operation time
            duration_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_operation_time(operation, duration_ms)

            return response

        except Exception:
            # Still record operation time for failed requests
            duration_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_operation_time(operation, duration_ms)
            raise

    @web.middleware
    async def error_handling_middleware(self, request: web.Request, handler):
        """Middleware for centralized error handling and logging."""
        try:
            return await handler(request)

        except web.HTTPException:
            # Re-raise HTTP exceptions (they're handled by aiohttp)
            raise

        except Exception as e:
            # Log unexpected errors with context
            context = LogContext(
                component="http_handler",
                operation=f"{request.method}:{request.path}",
                client_ip=request.remote,
            )

            logger.error(
                f"Unhandled error in {request.method} {request.path}",
                context=context,
                exception=e,
                request_path=request.path,
                request_method=request.method,
            )

            # Record in tracing
            self.tracing_manager.record_exception(e)

            return web.json_response(
                {"error": "Internal server error", "correlation_id": correlation_id.get()},
                status=500,
            )

    def _get_endpoint_name(self, path: str) -> str:
        """Extract endpoint name from path for metrics."""
        if path.startswith("/mcp"):
            if "websocket" in path:
                return "mcp_websocket"
            elif "sse" in path:
                return "mcp_sse"
            else:
                return "mcp_http"
        elif path.startswith("/health"):
            if "ready" in path:
                return "health_ready"
            elif "live" in path:
                return "health_live"
            else:
                return "health"
        elif path.startswith("/metrics"):
            return "metrics"
        elif path.startswith("/performance"):
            return "performance"
        elif path.startswith("/backup"):
            return "backup"
        elif path.startswith("/admin"):
            return "admin"
        else:
            return "unknown"

    async def _handle_mcp_request(self, request: web.Request) -> web.Response:
        """Handle HTTP MCP requests with full tracing and monitoring."""
        with self.tracing_manager.trace_operation("mcp.http.request"):
            try:
                # Parse request
                request_data = await request.json()

                # Add trace attributes
                self.tracing_manager.add_span_attribute(
                    "mcp.method", request_data.get("method", "unknown")
                )
                self.tracing_manager.add_span_attribute(
                    "mcp.request_id", request_data.get("id", "")
                )

                # Process MCP request (this would integrate with your existing MCP handler)
                response_data = await self._process_mcp_request(request_data)

                return web.json_response(response_data)

            except json.JSONDecodeError:
                self.tracing_manager.add_span_attribute("error.type", "json_decode_error")
                return web.json_response(
                    {"error": {"code": -32700, "message": "Parse error - invalid JSON"}}, status=400
                )

    async def _process_mcp_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Process MCP request with monitoring."""
        method = request_data.get("method", "")

        # This is a placeholder - integrate with your actual MCP processing logic
        with self.tracing_manager.trace_operation(f"mcp.method.{method}"):
            # Record tool usage if this is a tool call
            if method == "tools/call":
                tool_name = request_data.get("params", {}).get("name", "unknown")
                self.metrics_manager.record_mcp_tool_call(tool_name, "success", 0.1)

            # Placeholder response
            return {"result": {"message": f"Processed {method}"}, "id": request_data.get("id")}

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections with monitoring."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.active_websockets.add(ws)
        self.metrics_manager.set_active_connections("websocket", len(self.active_websockets))

        client_id = request.remote or "unknown"

        logger.info("WebSocket connection established", client_ip=client_id)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    with self.tracing_manager.trace_operation("mcp.websocket.message"):
                        try:
                            request_data = json.loads(msg.data)
                            response_data = await self._process_mcp_request(request_data)
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
            logger.error("Error in WebSocket handler", exception=e, client_ip=client_id)
        finally:
            self.active_websockets.discard(ws)
            self.metrics_manager.set_active_connections("websocket", len(self.active_websockets))
            logger.info("WebSocket connection closed", client_ip=client_id)

        return ws

    async def _handle_sse(self, request: web.Request) -> web.StreamResponse:
        """Handle SSE connections with monitoring."""
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
        self.metrics_manager.set_active_connections("sse", 1)  # Simplified tracking

        logger.info("SSE connection established", client_ip=client_id)

        try:
            # Send initial connection message
            await resp.write(b'data: {"type": "connection", "status": "connected"}\n\n')

            # Keep connection alive
            while True:
                await asyncio.sleep(30)  # Send keepalive every 30 seconds
                await resp.write(b'data: {"type": "keepalive"}\n\n')

        except Exception as e:
            logger.error("Error in SSE handler", exception=e, client_ip=client_id)
        finally:
            self.metrics_manager.set_active_connections("sse", 0)
            logger.info("SSE connection closed", client_ip=client_id)

        return resp

    async def _handle_health_check(self, request: web.Request) -> web.Response:
        """Handle comprehensive health check requests."""
        with self.tracing_manager.trace_operation("health.check"):
            health_status = await self.health_manager.get_health_status()

            status_code = 200 if health_status.status.value == "healthy" else 503

            return web.json_response(
                {
                    "status": health_status.status.value,
                    "timestamp": health_status.timestamp.isoformat(),
                    "uptime_seconds": health_status.uptime_seconds,
                    "version": health_status.version,
                    "deployment_mode": health_status.deployment_mode,
                    "services": [
                        {
                            "name": service.name,
                            "status": service.status.value,
                            "message": service.message,
                            "response_time_ms": service.response_time_ms,
                            "details": service.details,
                        }
                        for service in health_status.services
                    ],
                    "metrics": health_status.metrics,
                },
                status=status_code,
            )

    async def _handle_readiness_check(self, request: web.Request) -> web.Response:
        """Handle readiness probe for load balancers."""
        readiness = await self.health_manager.get_readiness_status()
        status_code = 200 if readiness["ready"] else 503

        return web.json_response(readiness, status=status_code)

    async def _handle_liveness_check(self, request: web.Request) -> web.Response:
        """Handle liveness probe for orchestrators."""
        liveness = await self.health_manager.get_liveness_status()

        return web.json_response(liveness, status=200)

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """Handle Prometheus metrics export."""
        metrics_data = self.metrics_manager.export_metrics()

        return web.Response(text=metrics_data, content_type=self.metrics_manager.get_content_type())

    async def _handle_performance_summary(self, request: web.Request) -> web.Response:
        """Handle performance summary requests."""
        summary = self.performance_monitor.get_performance_summary()

        return web.json_response(summary)

    async def _handle_backup_status(self, request: web.Request) -> web.Response:
        """Handle backup status requests."""
        summary = self.backup_manager.get_backup_summary()

        return web.json_response(summary)

    async def _handle_create_backup(self, request: web.Request) -> web.Response:
        """Handle backup creation requests."""
        try:
            data = await request.json()
            backup_type = data.get("type", "configuration")
            tags = data.get("tags", {})

            from .backup import BackupType

            backup_type_enum = BackupType(backup_type)

            backup_id = await self.backup_manager.create_backup(backup_type_enum, tags)

            return web.json_response({"backup_id": backup_id, "status": "started"})

        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def _handle_admin_info(self, request: web.Request) -> web.Response:
        """Handle admin information requests."""
        info = {
            "service_name": self.config.get("service_name", "agent-zero-mcp"),
            "version": __version__,
            "environment": self.config.get("environment", "production"),
            "deployment_mode": "production",
            "monitoring": {
                "health_checks_enabled": True,
                "metrics_enabled": self.metrics_manager.enabled,
                "tracing_enabled": self.tracing_manager.enabled,
                "performance_monitoring_enabled": self.performance_monitor.enabled,
                "backups_enabled": self.backup_manager.config.enabled,
            },
            "uptime_seconds": (time.time() - self.health_manager.start_time.timestamp()),
            "active_connections": {
                "websocket": len(self.active_websockets),
                "total": len(self.active_websockets),
            },
        }

        return web.json_response(info)

    async def start(self, host: str = "0.0.0.0", port: int = 8505) -> None:
        """Start the production server."""
        # Start monitoring systems
        await self.start_monitoring()

        # Create application
        app = await self.create_app()

        # Setup SSL if configured
        ssl_context = None
        ssl_config = self.config.get("ssl_config")
        if ssl_config:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(ssl_config["certfile"], ssl_config["keyfile"])

        # Create runner
        runner = web.AppRunner(app)
        await runner.setup()

        # Create site
        site = web.TCPSite(runner, host, port, ssl_context=ssl_context)
        await site.start()

        protocol = "https" if ssl_context else "http"
        logger.info(f"Production MCP server started on {protocol}://{host}:{port}")
        logger.info("Available endpoints:")
        logger.info(f"  - MCP HTTP: {protocol}://{host}:{port}/mcp")
        logger.info(
            f"  - WebSocket: {protocol.replace('http', 'ws')}://{host}:{port}/mcp/websocket"
        )
        logger.info(f"  - SSE: {protocol}://{host}:{port}/mcp/sse")
        logger.info(f"  - Health: {protocol}://{host}:{port}/health")
        logger.info(f"  - Metrics: {protocol}://{host}:{port}/metrics")
        logger.info(f"  - Performance: {protocol}://{host}:{port}/performance")
        logger.info(f"  - Backup: {protocol}://{host}:{port}/backup/status")

        # Wait for shutdown signal
        try:
            await self._shutdown_event.wait()
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        finally:
            await self.shutdown()
            await runner.cleanup()

    async def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        logger.info("Initiating graceful shutdown...")

        # Close active WebSocket connections
        for ws in list(self.active_websockets):
            if not ws.closed:
                await ws.close()

        # Stop monitoring systems
        await self.health_manager.stop_background_checks()
        await self.performance_monitor.cleanup()

        # Set shutdown event
        self._shutdown_event.set()

        logger.info("Server shutdown completed")


async def run_production_server(config: dict[str, Any]) -> None:
    """Run the production MCP server.

    Args:
        config: Server configuration
    """
    server = ProductionMCPServer(config)

    host = config.get("host", "0.0.0.0")
    port = config.get("port", 8505)

    await server.start(host, port)
