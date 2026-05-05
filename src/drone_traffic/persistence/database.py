from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

_ASYNC_ENGINE = None
_ASYNC_SESSION_FACTORY = None


def init_db(db_url: str, echo: bool = False) -> None:
    global _ASYNC_ENGINE, _ASYNC_SESSION_FACTORY
    _ASYNC_ENGINE = create_async_engine(db_url, echo=echo, pool_size=5, max_overflow=10)
    _ASYNC_SESSION_FACTORY = async_sessionmaker(
        _ASYNC_ENGINE, class_=AsyncSession, expire_on_commit=False
    )


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _ASYNC_SESSION_FACTORY is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _ASYNC_SESSION_FACTORY


async def create_tables() -> None:
    if _ASYNC_ENGINE is None:
        raise RuntimeError("Database not initialized.")
    from drone_traffic.persistence.models import Base
    async with _ASYNC_ENGINE.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    global _ASYNC_ENGINE, _ASYNC_SESSION_FACTORY
    if _ASYNC_ENGINE is not None:
        await _ASYNC_ENGINE.dispose()
    _ASYNC_ENGINE = None
    _ASYNC_SESSION_FACTORY = None
