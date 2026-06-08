from collections.abc import AsyncGenerator, Callable, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx as httpx_mod
import pytest

from dependency_director.config import DEFAULT_BOTS, BotConfig
from dependency_director.tools import GitHubClient, ToolFn, _create_write_tools


@pytest.fixture
def mock_client() -> MagicMock:
    """Shared MagicMock for GitHubClient."""
    return MagicMock(spec=GitHubClient)


@pytest.fixture
def wait_tool(mock_client: MagicMock) -> Callable[..., ToolFn]:
    """Factory fixture for wait_for_reviews with configurable review_wait.

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
        _, _, wait = _create_write_tools(
            client=mock_client,
            bots=bots,
            dry_run=dry_run,
            review_wait=review_wait,
        )
        return wait

    return _make


def make_client_with_status(status_code: int, url: str) -> GitHubClient:
    """Create a GitHubClient whose transport always returns the given HTTP status.

    Useful for testing the event-hook error-classification logic without
    making real network calls.
    """
    c = GitHubClient(token="dummy")
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
        mock_usage.candidates_token_count = 0
        mock_usage.thoughts_token_count = 0
        mock_usage.total_token_count = 0
        mock_agent_instance.conversation.total_usage = mock_usage

        yield mock_cls
