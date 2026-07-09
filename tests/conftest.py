"""Pytest fixtures and configuration for dependency-director tests."""

import asyncio
import inspect
from collections.abc import AsyncGenerator, Callable, Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx as httpx_mod
import pytest

from dependency_director.config import DEFAULT_BOTS, BotConfig
from dependency_director.tools import GitHubClient, ToolFn, _make_write_tools


@pytest.fixture
def mock_client() -> MagicMock:
    """Shared MagicMock for GitHubClient."""
    return MagicMock(spec=GitHubClient)


@pytest.fixture
def wait_tool(mock_client: MagicMock) -> Callable[..., ToolFn]:
    """Create a factory fixture for wait_for_reviews with configurable review_wait.

    Usage::

        async def test_something(wait_tool):
            wait = wait_tool()           # default review_wait=1
            wait = wait_tool(review_wait=5)
            wait = wait_tool(bots=[...], dry_run=True)
    """

    def _make(
        *,
        review_wait: int = 1,
        bots: list[BotConfig] = DEFAULT_BOTS,
        dry_run: bool = False,
    ) -> ToolFn:
        _, _, wait = _make_write_tools(
            client=mock_client,
            bots=bots,
            dry_run=dry_run,
            review_wait=review_wait,
        )
        return wait

    return _make


def make_client_with_status(status_code: int, url: str, token: str | None = None) -> GitHubClient:
    """Create a GitHubClient whose transport always returns the given HTTP status.

    Useful for testing the event-hook error-classification logic without
    making real network calls.
    """
    token_str = token or "placeholder"
    c = GitHubClient(token=token_str)
    mock_transport = MagicMock()
    mock_transport.aclose = AsyncMock()
    mock_transport.handle_async_request = AsyncMock(
        return_value=httpx_mod.Response(
            status_code=status_code,
            request=httpx_mod.Request("GET", url),
        ),
    )
    c.client._transport = mock_transport
    return c


async def _mock_chunks_async_generator() -> AsyncGenerator[None]:
    return
    yield


@pytest.fixture
def mock_agent_class() -> Generator[MagicMock]:
    """Shared fixture that patches main Agent and sets up generic mocks/defaults."""
    with patch("dependency_director.main.Agent") as mock_cls:
        mock_agent_instance = mock_cls.return_value
        mock_agent_instance.__aenter__.return_value = mock_agent_instance

        # Default mock response with empty chunks and Done text
        mock_response = MagicMock()
        mock_response.chunks = _mock_chunks_async_generator()
        mock_response.text = AsyncMock(return_value="Done.")
        mock_agent_instance.chat = AsyncMock(return_value=mock_response)

        # Default mock usage metrics
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 0
        mock_usage.cached_content_token_count = 0
        mock_usage.candidates_token_count = 0
        mock_usage.thoughts_token_count = 0
        mock_usage.total_token_count = 0
        mock_agent_instance.conversation.total_usage = mock_usage

        yield mock_cls


@pytest.fixture(autouse=True)
def mock_list_open_prs() -> Generator[MagicMock]:
    """Mock list_open_prs globally to return a placeholder bot PR in tests."""
    with patch(
        "dependency_director.tools.GitHubClient.list_open_prs",
        new_callable=AsyncMock,
    ) as mock:

        async def side_effect(_owner: str, _repo: str) -> list[dict[str, str | int]]:
            allowed_authors = ["dependabot[bot]"]
            frame = inspect.currentframe()
            while frame:
                if frame.f_code.co_name == "run_agent_for_repo":
                    settings = frame.f_locals.get("settings")
                    if settings and hasattr(settings, "bots"):
                        allowed_authors = [b.author for b in settings.bots]
                    break
                frame = frame.f_back

            return [
                {
                    "number": 12 + i,
                    "title": f"bump foo for {author}",
                    "author": author,
                    "created_at": "2026-06-08T00:00:00Z",
                }
                for i, author in enumerate(allowed_authors)
            ]

        mock.side_effect = side_effect
        yield mock


@pytest.fixture
def github_token() -> str:
    """Return a placeholder token to satisfy S106 security checks."""
    return "placeholder"


class AsyncFSHelper:
    """Helper class providing async wrappers for blocking filesystem operations."""

    @staticmethod
    async def exists(path: Path | str) -> bool:
        """Check if a path exists asynchronously."""
        return await asyncio.to_thread(Path(path).exists)

    @staticmethod
    async def unlink(path: Path | str) -> None:
        """Remove a file asynchronously."""
        await asyncio.to_thread(Path(path).unlink, missing_ok=True)

    @staticmethod
    async def mkdir(path: Path | str) -> None:
        """Create a directory asynchronously."""
        await asyncio.to_thread(Path(path).mkdir, parents=True, exist_ok=True)

    @staticmethod
    async def write_text(path: Path | str, content: str) -> None:
        """Write text to a file asynchronously."""

        def _write() -> None:
            Path(path).write_text(content)

        await asyncio.to_thread(_write)

    @staticmethod
    async def read_text(path: Path | str) -> str:
        """Read text from a file asynchronously."""
        return await asyncio.to_thread(Path(path).read_text)

    @staticmethod
    async def is_symlink(path: Path | str) -> bool:
        """Check if path is a symlink asynchronously."""
        return await asyncio.to_thread(Path(path).is_symlink)

    @staticmethod
    async def rmdir(path: Path | str) -> None:
        """Remove a directory asynchronously."""
        await asyncio.to_thread(Path(path).rmdir)

    @staticmethod
    async def expanduser(path: str) -> str:
        """Expand user home directory symbol in path asynchronously."""
        return await asyncio.to_thread(lambda: str(Path(path).expanduser()))


@pytest.fixture
def async_fs() -> type[AsyncFSHelper]:
    """Fixture providing async filesystem helpers to avoid blocking event loop."""
    return AsyncFSHelper
