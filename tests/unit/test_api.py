"""Unit tests for the FastAPI server."""

import pytest
from httpx import ASGITransport, AsyncClient

from simverse.api.app import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_root(client: AsyncClient) -> None:
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "SimVerse API"
    assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_list_robots(client: AsyncClient) -> None:
    resp = await client.get("/api/robots")
    assert resp.status_code == 200
    robots = resp.json()
    assert isinstance(robots, list)
    assert len(robots) >= 1
    assert any(r["name"] == "panda" for r in robots)


@pytest.mark.asyncio
async def test_list_simulations_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/simulations")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_list_training_runs_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/training")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
