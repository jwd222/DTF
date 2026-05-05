from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from drone_traffic.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Drone Traffic Monitoring API",
        version="0.1.0",
        description="API for querying multi-camera traffic monitoring data",
    )

    app.include_router(router, prefix="/api/v1")

    @app.on_event("startup")
    async def startup() -> None:
        pass

    @app.on_event("shutdown")
    async def shutdown() -> None:
        from drone_traffic.persistence.database import close_db
        await close_db()

    return app
