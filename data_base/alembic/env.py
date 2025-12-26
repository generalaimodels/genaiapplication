# ==============================================================================
# ALEMBIC ENVIRONMENT - Migration Configuration
# ==============================================================================
# Database migration environment for SQLAlchemy models
# ==============================================================================

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Import models and settings
from app.core.settings import settings, DatabaseType
from app.domain_models.base import SQLBase

# Import all models to register with metadata
from app.domain_models import user, chat, transaction, product, order, project, course  # noqa

# Alembic Config object
config = context.config

# Configure logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata
target_metadata = SQLBase.metadata


def get_url() -> str:
    """Get database URL for migrations."""
    if settings.DATABASE_TYPE == DatabaseType.SQLITE:
        # Use sync SQLite for Alembic
        return settings.SQLITE_URL
    elif settings.DATABASE_TYPE == DatabaseType.POSTGRESQL:
        return settings.postgres_sync_url
    else:
        raise ValueError(f"Alembic only supports SQL databases, not {settings.DATABASE_TYPE}")


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    
    Generates SQL script without connecting to database.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with active connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations in 'online' mode with async engine.
    """
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
