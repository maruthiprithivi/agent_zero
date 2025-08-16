"""Production-grade backup and disaster recovery for Agent Zero MCP Server.

This module implements comprehensive backup and disaster recovery capabilities
including configuration backup, data export/import, failover strategies, and
automated recovery procedures following 2025 best practices.
"""

import asyncio
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

# Import version from package
try:
    from agent_zero import __version__
except ImportError:
    __version__ = "unknown"

logger = logging.getLogger(__name__)

# Optional imports for cloud storage
try:
    import boto3

    boto3_available = True
except ImportError:
    logger.debug("boto3 not available. Cloud backup features will be disabled.")
    boto3_available = False
    boto3 = None


class BackupType(Enum):
    """Backup type enumeration."""

    CONFIGURATION = "configuration"
    DATA_EXPORT = "data_export"
    FULL_SYSTEM = "full_system"
    INCREMENTAL = "incremental"


class BackupStatus(Enum):
    """Backup status enumeration."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class StorageType(Enum):
    """Storage type enumeration."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


@dataclass
class BackupConfig:
    """Backup configuration."""

    enabled: bool = True
    backup_dir: str = "/tmp/agent_zero_backups"
    retention_days: int = 30
    max_backups: int = 100
    compression: bool = True
    encryption: bool = False
    encryption_key_file: str | None = None
    storage_type: StorageType = StorageType.LOCAL
    cloud_config: dict[str, Any] = field(default_factory=dict)
    schedule_interval_hours: int = 24
    auto_cleanup: bool = True


@dataclass
class BackupMetadata:
    """Backup metadata information."""

    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    created_at: datetime
    completed_at: datetime | None = None
    file_path: str = ""
    file_size_bytes: int = 0
    checksum: str = ""
    compression_ratio: float = 0.0
    error_message: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        return data


@dataclass
class RestoreConfig:
    """Restore configuration."""

    backup_id: str
    target_location: str | None = None
    partial_restore: bool = False
    restore_items: list[str] = field(default_factory=list)
    verify_integrity: bool = True
    create_backup_before_restore: bool = True


class CloudStorageManager:
    """Manages cloud storage operations for backups."""

    def __init__(self, storage_type: StorageType, config: dict[str, Any]):
        """Initialize cloud storage manager.

        Args:
            storage_type: Type of cloud storage
            config: Storage configuration
        """
        self.storage_type = storage_type
        self.config = config
        self.client = None

        if storage_type == StorageType.S3 and boto3_available:
            self._init_s3_client()

    def _init_s3_client(self) -> None:
        """Initialize S3 client."""
        try:
            self.client = boto3.client(
                "s3",
                aws_access_key_id=self.config.get("access_key_id"),
                aws_secret_access_key=self.config.get("secret_access_key"),
                region_name=self.config.get("region", "us-east-1"),
            )
            logger.info("S3 client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.client = None

    async def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload file to cloud storage.

        Args:
            local_path: Local file path
            remote_path: Remote file path

        Returns:
            True if upload successful
        """
        if self.storage_type == StorageType.S3 and self.client:
            return await self._upload_s3(local_path, remote_path)

        logger.warning(f"Cloud upload not supported for {self.storage_type}")
        return False

    async def _upload_s3(self, local_path: str, remote_path: str) -> bool:
        """Upload file to S3.

        Args:
            local_path: Local file path
            remote_path: S3 key

        Returns:
            True if upload successful
        """
        try:
            bucket_name = self.config.get("bucket_name")
            if not bucket_name:
                logger.error("S3 bucket name not configured")
                return False

            await asyncio.to_thread(self.client.upload_file, local_path, bucket_name, remote_path)
            logger.info(f"Successfully uploaded {local_path} to s3://{bucket_name}/{remote_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False

    async def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file from cloud storage.

        Args:
            remote_path: Remote file path
            local_path: Local file path

        Returns:
            True if download successful
        """
        if self.storage_type == StorageType.S3 and self.client:
            return await self._download_s3(remote_path, local_path)

        logger.warning(f"Cloud download not supported for {self.storage_type}")
        return False

    async def _download_s3(self, remote_path: str, local_path: str) -> bool:
        """Download file from S3.

        Args:
            remote_path: S3 key
            local_path: Local file path

        Returns:
            True if download successful
        """
        try:
            bucket_name = self.config.get("bucket_name")
            if not bucket_name:
                logger.error("S3 bucket name not configured")
                return False

            await asyncio.to_thread(self.client.download_file, bucket_name, remote_path, local_path)
            logger.info(f"Successfully downloaded s3://{bucket_name}/{remote_path} to {local_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return False

    async def list_files(self, prefix: str = "") -> list[str]:
        """List files in cloud storage.

        Args:
            prefix: File prefix filter

        Returns:
            List of file paths
        """
        if self.storage_type == StorageType.S3 and self.client:
            return await self._list_s3_files(prefix)

        return []

    async def _list_s3_files(self, prefix: str = "") -> list[str]:
        """List files in S3 bucket.

        Args:
            prefix: Key prefix filter

        Returns:
            List of S3 keys
        """
        try:
            bucket_name = self.config.get("bucket_name")
            if not bucket_name:
                return []

            response = await asyncio.to_thread(
                self.client.list_objects_v2, Bucket=bucket_name, Prefix=prefix
            )

            files = []
            for obj in response.get("Contents", []):
                files.append(obj["Key"])

            return files

        except Exception as e:
            logger.error(f"Failed to list S3 files: {e}")
            return []


class BackupManager:
    """Manages backup and disaster recovery operations."""

    def __init__(
        self, config: BackupConfig | None = None, clickhouse_client_factory: Callable | None = None
    ):
        """Initialize backup manager.

        Args:
            config: Backup configuration
            clickhouse_client_factory: Factory for ClickHouse clients
        """
        self.config = config or BackupConfig()
        self.clickhouse_client_factory = clickhouse_client_factory

        # Ensure backup directory exists
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)

        # Initialize cloud storage if configured
        self.cloud_storage = None
        if self.config.storage_type != StorageType.LOCAL:
            self.cloud_storage = CloudStorageManager(
                self.config.storage_type, self.config.cloud_config
            )

        # Backup tracking
        self.active_backups: dict[str, BackupMetadata] = {}
        self.completed_backups: list[BackupMetadata] = []

        # Load existing backup metadata
        self._load_backup_metadata()

        # Start background tasks
        if self.config.enabled:
            self._backup_task = asyncio.create_task(self._scheduled_backup_task())
            if self.config.auto_cleanup:
                self._cleanup_task = asyncio.create_task(self._cleanup_task())

    def _load_backup_metadata(self) -> None:
        """Load existing backup metadata from disk."""
        metadata_file = Path(self.config.backup_dir) / "backup_metadata.json"

        if metadata_file.exists():
            try:
                with metadata_file.open() as f:
                    data = json.load(f)

                for backup_data in data.get("backups", []):
                    metadata = BackupMetadata(
                        backup_id=backup_data["backup_id"],
                        backup_type=BackupType(backup_data["backup_type"]),
                        status=BackupStatus(backup_data["status"]),
                        created_at=datetime.fromisoformat(backup_data["created_at"]),
                        completed_at=(
                            datetime.fromisoformat(backup_data["completed_at"])
                            if backup_data.get("completed_at")
                            else None
                        ),
                        file_path=backup_data.get("file_path", ""),
                        file_size_bytes=backup_data.get("file_size_bytes", 0),
                        checksum=backup_data.get("checksum", ""),
                        compression_ratio=backup_data.get("compression_ratio", 0.0),
                        error_message=backup_data.get("error_message", ""),
                        tags=backup_data.get("tags", {}),
                    )
                    self.completed_backups.append(metadata)

                logger.info(f"Loaded {len(self.completed_backups)} backup records")

            except Exception as e:
                logger.error(f"Failed to load backup metadata: {e}")

    def _save_backup_metadata(self) -> None:
        """Save backup metadata to disk."""
        metadata_file = Path(self.config.backup_dir) / "backup_metadata.json"

        try:
            all_backups = list(self.active_backups.values()) + self.completed_backups
            data = {
                "backups": [backup.to_dict() for backup in all_backups],
                "last_updated": datetime.now().isoformat(),
            }

            with metadata_file.open("w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")

    async def create_backup(
        self, backup_type: BackupType, tags: dict[str, str] | None = None
    ) -> str:
        """Create a new backup.

        Args:
            backup_type: Type of backup to create
            tags: Optional tags for the backup

        Returns:
            Backup ID
        """
        backup_id = f"{backup_type.value}_{int(time.time())}"

        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            status=BackupStatus.PENDING,
            created_at=datetime.now(),
            tags=tags or {},
        )

        self.active_backups[backup_id] = metadata

        # Start backup process
        backup_task = asyncio.create_task(self._perform_backup(metadata))
        # Store task reference to prevent garbage collection
        self._active_tasks = getattr(self, '_active_tasks', set())
        self._active_tasks.add(backup_task)
        backup_task.add_done_callback(self._active_tasks.discard)

        logger.info(f"Started backup {backup_id} of type {backup_type.value}")
        return backup_id

    async def _perform_backup(self, metadata: BackupMetadata) -> None:
        """Perform the actual backup operation.

        Args:
            metadata: Backup metadata
        """
        try:
            metadata.status = BackupStatus.IN_PROGRESS

            if metadata.backup_type == BackupType.CONFIGURATION:
                await self._backup_configuration(metadata)
            elif metadata.backup_type == BackupType.DATA_EXPORT:
                await self._backup_data_export(metadata)
            elif metadata.backup_type == BackupType.FULL_SYSTEM:
                await self._backup_full_system(metadata)
            else:
                raise ValueError(f"Unsupported backup type: {metadata.backup_type}")

            metadata.status = BackupStatus.COMPLETED
            metadata.completed_at = datetime.now()

            # Upload to cloud if configured
            if self.cloud_storage and metadata.file_path:
                cloud_path = f"backups/{metadata.backup_id}/{Path(metadata.file_path).name}"
                await self.cloud_storage.upload_file(metadata.file_path, cloud_path)

        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.error_message = str(e)
            logger.error(f"Backup {metadata.backup_id} failed: {e}")

        finally:
            # Move from active to completed
            if metadata.backup_id in self.active_backups:
                del self.active_backups[metadata.backup_id]
            self.completed_backups.append(metadata)
            self._save_backup_metadata()

    async def _backup_configuration(self, metadata: BackupMetadata) -> None:
        """Backup system configuration.

        Args:
            metadata: Backup metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"config_backup_{timestamp}.tar.gz"
        backup_path = Path(self.config.backup_dir) / backup_filename

        # Create temporary directory for configuration files
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "configuration"
            config_dir.mkdir()

            # Collect configuration files
            config_files = [
                "agent_zero/config/",
                "configs/",
                ".env",
                "pyproject.toml",
            ]

            # Copy configuration files
            for config_file in config_files:
                src_path = Path(config_file)
                if src_path.exists():
                    if src_path.is_file():
                        shutil.copy2(src_path, config_dir / src_path.name)
                    else:
                        shutil.copytree(src_path, config_dir / src_path.name, dirs_exist_ok=True)

            # Create metadata file
            metadata_file = config_dir / "backup_info.json"
            with metadata_file.open("w") as f:
                json.dump(
                    {
                        "backup_id": metadata.backup_id,
                        "backup_type": metadata.backup_type.value,
                        "created_at": metadata.created_at.isoformat(),
                        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
                        "version": __version__,
                    },
                    f,
                    indent=2,
                )

            # Create compressed archive
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(config_dir, arcname="configuration")

        # Update metadata
        metadata.file_path = str(backup_path)
        metadata.file_size_bytes = backup_path.stat().st_size
        metadata.checksum = await self._calculate_checksum(backup_path)

        logger.info(f"Configuration backup created: {backup_path}")

    async def _backup_data_export(self, metadata: BackupMetadata) -> None:
        """Backup ClickHouse data export.

        Args:
            metadata: Backup metadata
        """
        if not self.clickhouse_client_factory:
            raise ValueError("ClickHouse client factory not configured")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"data_export_{timestamp}.tar.gz"
        backup_path = Path(self.config.backup_dir) / backup_filename

        # Create temporary directory for data export
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "data_export"
            export_dir.mkdir()

            # Get ClickHouse client
            client = self.clickhouse_client_factory()

            # Export database schema
            schema_file = export_dir / "schema.sql"
            with schema_file.open("w") as f:
                # Export table schemas
                tables_result = client.query("SHOW TABLES").result_rows
                for (table_name,) in tables_result:
                    create_result = client.query(f"SHOW CREATE TABLE {table_name}").result_rows
                    if create_result:
                        f.write(f"-- Table: {table_name}\n")
                        f.write(create_result[0][0] + ";\n\n")

            # Export sample data (limited for backup purposes)
            data_dir = export_dir / "sample_data"
            data_dir.mkdir()

            for (table_name,) in tables_result:
                try:
                    # Export limited sample data
                    sample_result = client.query(
                        f"SELECT * FROM {table_name} LIMIT 1000"
                    ).result_rows

                    sample_file = data_dir / f"{table_name}.json"
                    with sample_file.open("w") as f:
                        json.dump(
                            {
                                "table": table_name,
                                "sample_rows": len(sample_result),
                                "data": sample_result,
                            },
                            f,
                            indent=2,
                            default=str,
                        )

                except Exception as e:
                    logger.warning(f"Failed to export sample data for {table_name}: {e}")

            # Create compressed archive
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(export_dir, arcname="data_export")

        # Update metadata
        metadata.file_path = str(backup_path)
        metadata.file_size_bytes = backup_path.stat().st_size
        metadata.checksum = await self._calculate_checksum(backup_path)

        logger.info(f"Data export backup created: {backup_path}")

    async def _backup_full_system(self, metadata: BackupMetadata) -> None:
        """Backup full system state.

        Args:
            metadata: Backup metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"full_backup_{timestamp}.tar.gz"
        backup_path = Path(self.config.backup_dir) / backup_filename

        # Create temporary directory for full backup
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_root = Path(temp_dir) / "full_backup"
            backup_root.mkdir()

            # Backup configuration
            config_metadata = BackupMetadata(
                backup_id=f"{metadata.backup_id}_config",
                backup_type=BackupType.CONFIGURATION,
                status=BackupStatus.IN_PROGRESS,
                created_at=datetime.now(),
            )
            await self._backup_configuration(config_metadata)

            # Backup data export
            data_metadata = BackupMetadata(
                backup_id=f"{metadata.backup_id}_data",
                backup_type=BackupType.DATA_EXPORT,
                status=BackupStatus.IN_PROGRESS,
                created_at=datetime.now(),
            )
            if self.clickhouse_client_factory:
                await self._backup_data_export(data_metadata)

            # Copy component backups
            if config_metadata.file_path and Path(config_metadata.file_path).exists():
                shutil.copy2(config_metadata.file_path, backup_root / "configuration.tar.gz")

            if data_metadata.file_path and Path(data_metadata.file_path).exists():
                shutil.copy2(data_metadata.file_path, backup_root / "data_export.tar.gz")

            # Create full backup manifest
            manifest_file = backup_root / "backup_manifest.json"
            with manifest_file.open("w") as f:
                json.dump(
                    {
                        "backup_id": metadata.backup_id,
                        "backup_type": "full_system",
                        "created_at": metadata.created_at.isoformat(),
                        "components": [
                            {"type": "configuration", "file": "configuration.tar.gz"},
                            {"type": "data_export", "file": "data_export.tar.gz"},
                        ],
                    },
                    f,
                    indent=2,
                )

            # Create compressed archive
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(backup_root, arcname="full_backup")

        # Update metadata
        metadata.file_path = str(backup_path)
        metadata.file_size_bytes = backup_path.stat().st_size
        metadata.checksum = await self._calculate_checksum(backup_path)

        logger.info(f"Full system backup created: {backup_path}")

    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum.

        Args:
            file_path: File to checksum

        Returns:
            SHA256 checksum
        """
        import hashlib

        sha256_hash = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    async def restore_backup(self, config: RestoreConfig) -> bool:
        """Restore from backup.

        Args:
            config: Restore configuration

        Returns:
            True if restore successful
        """
        try:
            # Find backup metadata
            backup_metadata = None
            for backup in self.completed_backups:
                if backup.backup_id == config.backup_id:
                    backup_metadata = backup
                    break

            if not backup_metadata:
                logger.error(f"Backup {config.backup_id} not found")
                return False

            if backup_metadata.status != BackupStatus.COMPLETED:
                logger.error(f"Backup {config.backup_id} is not in completed state")
                return False

            # Create pre-restore backup if requested
            if config.create_backup_before_restore:
                pre_restore_id = await self.create_backup(
                    BackupType.CONFIGURATION,
                    tags={"pre_restore": "true", "restore_backup_id": config.backup_id},
                )
                logger.info(f"Created pre-restore backup: {pre_restore_id}")

            # Verify backup integrity
            if config.verify_integrity and not await self._verify_backup_integrity(backup_metadata):
                logger.error(f"Backup integrity verification failed for {config.backup_id}")
                return False

            # Perform restore
            success = await self._perform_restore(backup_metadata, config)

            if success:
                logger.info(f"Successfully restored backup {config.backup_id}")
            else:
                logger.error(f"Failed to restore backup {config.backup_id}")

            return success

        except Exception as e:
            logger.error(f"Error during restore: {e}")
            return False

    async def _verify_backup_integrity(self, metadata: BackupMetadata) -> bool:
        """Verify backup file integrity.

        Args:
            metadata: Backup metadata

        Returns:
            True if integrity check passes
        """
        if not metadata.file_path or not Path(metadata.file_path).exists():
            logger.error(f"Backup file not found: {metadata.file_path}")
            return False

        # Verify checksum
        current_checksum = await self._calculate_checksum(Path(metadata.file_path))
        if current_checksum != metadata.checksum:
            logger.error(f"Checksum mismatch for backup {metadata.backup_id}")
            return False

        # Verify archive can be opened
        try:
            with tarfile.open(metadata.file_path, "r:gz") as tar:
                tar.getnames()  # Try to read archive structure
            return True
        except Exception as e:
            logger.error(f"Backup archive verification failed: {e}")
            return False

    async def _perform_restore(self, metadata: BackupMetadata, config: RestoreConfig) -> bool:
        """Perform the actual restore operation.

        Args:
            metadata: Backup metadata
            config: Restore configuration

        Returns:
            True if restore successful
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract backup
                with tarfile.open(metadata.file_path, "r:gz") as tar:
                    tar.extractall(temp_dir)

                if metadata.backup_type == BackupType.CONFIGURATION:
                    return await self._restore_configuration(temp_dir, config)
                elif metadata.backup_type == BackupType.FULL_SYSTEM:
                    return await self._restore_full_system(temp_dir, config)
                else:
                    logger.warning(
                        f"Restore not implemented for backup type: {metadata.backup_type}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Error performing restore: {e}")
            return False

    async def _restore_configuration(self, temp_dir: str, config: RestoreConfig) -> bool:
        """Restore configuration files.

        Args:
            temp_dir: Temporary directory with extracted backup
            config: Restore configuration

        Returns:
            True if restore successful
        """
        try:
            config_dir = Path(temp_dir) / "configuration"
            if not config_dir.exists():
                logger.error("Configuration directory not found in backup")
                return False

            # Restore configuration files
            target_base = Path(config.target_location) if config.target_location else Path()

            for item in config_dir.iterdir():
                if item.name == "backup_info.json":
                    continue  # Skip metadata file

                target_path = target_base / item.name

                if config.partial_restore and item.name not in config.restore_items:
                    continue

                if item.is_file():
                    # Backup existing file
                    if target_path.exists():
                        backup_path = target_path.with_suffix(target_path.suffix + ".backup")
                        shutil.copy2(target_path, backup_path)

                    shutil.copy2(item, target_path)
                    logger.info(f"Restored file: {target_path}")

                elif item.is_dir():
                    # Backup existing directory
                    if target_path.exists():
                        backup_path = target_path.with_suffix(".backup")
                        if backup_path.exists():
                            shutil.rmtree(backup_path)
                        shutil.move(target_path, backup_path)

                    shutil.copytree(item, target_path)
                    logger.info(f"Restored directory: {target_path}")

            return True

        except Exception as e:
            logger.error(f"Error restoring configuration: {e}")
            return False

    async def _restore_full_system(self, temp_dir: str, config: RestoreConfig) -> bool:
        """Restore full system backup.

        Args:
            temp_dir: Temporary directory with extracted backup
            config: Restore configuration

        Returns:
            True if restore successful
        """
        try:
            backup_dir = Path(temp_dir) / "full_backup"
            if not backup_dir.exists():
                logger.error("Full backup directory not found")
                return False

            # Read manifest
            manifest_file = backup_dir / "backup_manifest.json"
            if not manifest_file.exists():
                logger.error("Backup manifest not found")
                return False

            with manifest_file.open() as f:
                manifest = json.load(f)

            # Restore components
            success = True
            for component in manifest.get("components", []):
                component_file = backup_dir / component["file"]
                if component_file.exists():
                    # Extract and restore component
                    with tempfile.TemporaryDirectory() as comp_temp_dir:
                        with tarfile.open(component_file, "r:gz") as tar:
                            tar.extractall(comp_temp_dir)

                        if component["type"] == "configuration":
                            comp_success = await self._restore_configuration(comp_temp_dir, config)
                            success = success and comp_success

            return success

        except Exception as e:
            logger.error(f"Error restoring full system: {e}")
            return False

    async def _scheduled_backup_task(self) -> None:
        """Background task for scheduled backups."""
        while self.config.enabled:
            try:
                await asyncio.sleep(self.config.schedule_interval_hours * 3600)

                # Create scheduled backup
                backup_id = await self.create_backup(
                    BackupType.CONFIGURATION, tags={"scheduled": "true", "automated": "true"}
                )
                logger.info(f"Created scheduled backup: {backup_id}")

            except Exception as e:
                logger.error(f"Error in scheduled backup task: {e}")

    async def _cleanup_task(self) -> None:
        """Background task for backup cleanup."""
        while self.config.enabled:
            try:
                await asyncio.sleep(24 * 3600)  # Run daily

                # Clean up old backups
                cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)

                backups_to_remove = []
                for backup in self.completed_backups:
                    if backup.created_at < cutoff_date:
                        backups_to_remove.append(backup)

                # Also check if we exceed max backup count
                if len(self.completed_backups) > self.config.max_backups:
                    # Sort by creation date and remove oldest
                    sorted_backups = sorted(self.completed_backups, key=lambda x: x.created_at)
                    excess_count = len(self.completed_backups) - self.config.max_backups
                    backups_to_remove.extend(sorted_backups[:excess_count])

                # Remove old backups
                for backup in backups_to_remove:
                    await self._remove_backup(backup)

                if backups_to_remove:
                    logger.info(f"Cleaned up {len(backups_to_remove)} old backups")

            except Exception as e:
                logger.error(f"Error in backup cleanup task: {e}")

    async def _remove_backup(self, metadata: BackupMetadata) -> None:
        """Remove a backup and its files.

        Args:
            metadata: Backup metadata
        """
        try:
            # Remove local file
            if metadata.file_path and Path(metadata.file_path).exists():
                Path(metadata.file_path).unlink()

            # Remove from cloud storage if configured
            if self.cloud_storage:
                # Note: Cloud deletion would need to be implemented per storage type
                # cloud_path = f"backups/{metadata.backup_id}/{Path(metadata.file_path).name}"
                pass

            # Remove from tracking
            if metadata in self.completed_backups:
                self.completed_backups.remove(metadata)

            logger.info(f"Removed backup {metadata.backup_id}")

        except Exception as e:
            logger.error(f"Error removing backup {metadata.backup_id}: {e}")

    def get_backup_status(self, backup_id: str) -> BackupMetadata | None:
        """Get backup status.

        Args:
            backup_id: Backup ID

        Returns:
            Backup metadata or None if not found
        """
        # Check active backups
        if backup_id in self.active_backups:
            return self.active_backups[backup_id]

        # Check completed backups
        for backup in self.completed_backups:
            if backup.backup_id == backup_id:
                return backup

        return None

    def list_backups(
        self,
        backup_type: BackupType | None = None,
        status: BackupStatus | None = None,
        limit: int = 100,
    ) -> list[BackupMetadata]:
        """List backups with optional filtering.

        Args:
            backup_type: Filter by backup type
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of backup metadata
        """
        all_backups = list(self.active_backups.values()) + self.completed_backups

        # Apply filters
        filtered_backups = []
        for backup in all_backups:
            if backup_type and backup.backup_type != backup_type:
                continue
            if status and backup.status != status:
                continue
            filtered_backups.append(backup)

        # Sort by creation date (newest first) and limit
        filtered_backups.sort(key=lambda x: x.created_at, reverse=True)
        return filtered_backups[:limit]

    def get_backup_summary(self) -> dict[str, Any]:
        """Get backup system summary.

        Returns:
            Backup system summary
        """
        total_backups = len(self.completed_backups)
        active_backups = len(self.active_backups)

        # Calculate total size
        total_size = sum(backup.file_size_bytes for backup in self.completed_backups)

        # Status counts
        status_counts = {}
        for status in BackupStatus:
            count = sum(1 for backup in self.completed_backups if backup.status == status)
            status_counts[status.value] = count

        # Type counts
        type_counts = {}
        for backup_type in BackupType:
            count = sum(1 for backup in self.completed_backups if backup.backup_type == backup_type)
            type_counts[backup_type.value] = count

        return {
            "total_backups": total_backups,
            "active_backups": active_backups,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "status_counts": status_counts,
            "type_counts": type_counts,
            "retention_days": self.config.retention_days,
            "storage_type": self.config.storage_type.value,
            "enabled": self.config.enabled,
        }


# Global backup manager
backup_manager: BackupManager | None = None


def get_backup_manager(
    config: BackupConfig | None = None, clickhouse_client_factory: Callable | None = None
) -> BackupManager:
    """Get or create the global backup manager.

    Args:
        config: Backup configuration
        clickhouse_client_factory: ClickHouse client factory

    Returns:
        Global backup manager instance
    """
    global backup_manager
    if backup_manager is None:
        backup_manager = BackupManager(config, clickhouse_client_factory)
    return backup_manager


def reset_backup_manager() -> None:
    """Reset the global backup manager (useful for testing)."""
    global backup_manager
    backup_manager = None
