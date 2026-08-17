"""dependency-director: Autonomous dependency triage and patching agent."""

import asyncio
import hashlib
import logging
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import click
import httpx
from google.antigravity import Agent, LocalAgentConfig, types
from google.antigravity.hooks import hooks, policy
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme

from dependency_director.config import (
    BotConfig,
    Settings,
    get_dry_run_policies,
    get_safety_policies,
)
from dependency_director.instructions import get_system_instructions
from dependency_director.schemas import PullRequest
from dependency_director.tools import (
    GitHubClient,
    GitHubClientError,
    create_agent_tools,
    create_run_command_tool,
    is_ripgrep_available,
    is_srt_available,
)

MAX_ARGS_DISPLAY_LEN = 80

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SKILLS_PATH = PROJECT_ROOT / ".agents" / "skills"

console = Console(
    theme=Theme(
        {
            "thinking": "dim",
            "tool.name": "bold yellow",
            "tool.args": "dim",
            "tool.ok": "green",
            "tool.fail": "bold red",
            "banner": "bold cyan",
            "status": "cyan",
        },
    ),
)


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("dependency_director")
logger.setLevel(logging.INFO)


async def get_repositories(owner: str, token: str) -> list[str]:
    """Fetch non-forked repositories from GitHub API."""
    client = GitHubClient(token=token)
    try:
        return await client.get_repositories(owner)
    finally:
        await client.close()


def _prepare_workspace(workspace_dir: str) -> None:
    """Prepare a clean workspace directory synchronously."""
    path = Path(workspace_dir)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _cleanup_workspace(workspace_dir: str) -> None:
    """Clean up a workspace directory synchronously."""
    path = Path(workspace_dir)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


async def _check_open_bot_prs(
    owner: str,
    repo_name: str,
    client: GitHubClient,
    bots: list[BotConfig],
) -> list[PullRequest]:
    open_prs = await client.list_open_prs(owner, repo_name)
    allowed_authors = {b.author for b in bots}
    return [pr for pr in open_prs if pr.author in allowed_authors]


class _SdkIssueCollector(logging.Handler):
    """Collect warning-or-worse log records emitted by anything but us.

    The SDK reports some terminal conditions by logging rather than raising or
    yielding a chunk — a loop detected in the model's output arrives as
    'System step error (HTTP 0): ...' on the root logger. The turn then ends
    quietly with truncated output, and the caller has no way to tell it apart
    from a clean run.
    """

    def __init__(self) -> None:
        """Collect at WARNING and above."""
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Record the message unless we logged it ourselves."""
        if record.name.startswith("dependency_director"):
            return
        self.messages.append(record.getMessage())


async def _render_agent_response(response: types.ChatResponse) -> None:
    text_buffer: list[str] = []

    def _flush_text() -> None:
        if text_buffer:
            console.print(Markdown("".join(text_buffer)))
            text_buffer.clear()

    async for chunk in response.chunks:
        match chunk:
            case types.Thought(text=text) if text.strip():
                _flush_text()
                console.print(
                    Markdown(text.strip()),
                    style="thinking",
                    width=100,
                )
            case types.Text(text=text):
                text_buffer.append(text)
            case types.ToolCall(name=name, args=args):
                _flush_text()
                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                if len(args_str) > MAX_ARGS_DISPLAY_LEN:
                    args_str = args_str[: MAX_ARGS_DISPLAY_LEN - 3] + "..."
                console.print(
                    f"  🔧 [tool.name]{name}[/tool.name]([tool.args]{args_str}[/tool.args])",
                )
            case types.ToolResult(name=name, error=error) if error:
                _flush_text()
                console.print(
                    f"  ❌ [tool.fail]{name} failed: {error}[/tool.fail]",
                )
            case types.ToolResult(name=name):
                _flush_text()
                console.print(f"  ✅ [tool.ok]{name}[/tool.ok]")
    _flush_text()


async def _prepare_agent_environment(
    repo: str,
    settings: Settings,
    *,
    dry_run: bool,
) -> tuple[str, list[Any], Any]:
    policies = get_safety_policies()
    if dry_run:
        dry_run_policies = get_dry_run_policies()
        for p in reversed(dry_run_policies):
            policies.insert(0, p)

    repo_hash = hashlib.sha256(repo.encode()).hexdigest()[:8]
    workspace_tmp = str(
        Path(tempfile.gettempdir()) / f"dependency-director-{repo_hash}",
    )
    await asyncio.to_thread(_prepare_workspace, workspace_tmp)

    # srt only sandboxes run_command_sandboxed. The SDK's own view/edit/create
    # file tools bypass it, so under allow("*") they can reach any path on the
    # host. Listing workspaces does not bound them; this policy does.
    policies = [*policy.workspace_only([str(SKILLS_PATH), workspace_tmp]), *policies]

    if settings.no_sandbox:
        run_command = None
    else:
        run_command = create_run_command_tool(
            workspace_tmp,
            srt_settings_path=settings.srt_settings,
            github_token=settings.github_token,
            command_timeout=settings.command_timeout,
            output_limits=settings.output_limits,
            cache_dir=settings.cache_dir,
        )
    return workspace_tmp, policies, run_command


def _build_agent_prompt(
    owner: str,
    repo_name: str,
    settings: Settings,
    workspace_tmp: str,
    *,
    dry_run: bool,
    hint: str | None,
) -> str:
    """Build the prompt string to send to the agent."""
    bot_names = ", ".join(b.author for b in settings.bots)
    prompt = f"Process all open dependency update PRs (authored by {bot_names}) for '{owner}/{repo_name}'."
    if not settings.no_sandbox:
        prompt += f" Workspace directory: {workspace_tmp}"
    if dry_run:
        prompt += " Perform this run in DRY-RUN mode (simulate all merge and push actions)."
    if hint:
        prompt += f" Additional context: {hint}"
    return prompt


async def run_agent_for_repo(
    repo: str,
    settings: Settings,
    max_attempts: int,
    *,
    dry_run: bool = False,
    auto_merge: bool = False,
    verify_all: bool = False,
    standalone_fix: bool = False,
    fix_base: bool = False,
    review_wait: int = 0,
    hint: str | None = None,
    model: str | None = None,
) -> None:
    """Run the triage agent for a single GitHub repository."""
    client = GitHubClient(token=settings.github_token)
    (
        create_pr,
        get_branch_ci_status,
        get_commit_details,
        get_file_contents,
        get_pr_diff,
        get_pr_files,
        get_pr_status,
        get_pr_workflow_run_logs,
        list_bot_prs,
        list_branches,
        list_commits,
        merge_bot_pr,
        rebase_bot_pr,
        wait_for_ci,
        wait_for_reviews,
    ) = create_agent_tools(
        client=client,
        bots=settings.bots,
        dry_run=dry_run,
        review_wait=review_wait,
    )

    workspace_tmp: str | None = None
    run_command: Any | None = None
    try:
        # Check for work before building anything: a repo with no bot PRs should
        # cost one API call, not a workspace and a sandbox configuration.
        owner, repo_name = repo.split("/", 1)
        bot_prs = await _check_open_bot_prs(owner, repo_name, client, settings.bots)

        if not bot_prs:
            click.echo("Open Pull Requests (Initial List)\n")
            click.echo(f" • No open dependency update PRs were found for {repo}.\n")
            return

        workspace_tmp, policies, run_command = await _prepare_agent_environment(
            repo,
            settings,
            dry_run=dry_run,
        )

        # Get agent system instructions
        system_instructions = get_system_instructions(
            max_attempts=max_attempts,
            verify_all=verify_all,
            auto_merge=auto_merge,
            dry_run=dry_run,
            standalone_fix=standalone_fix,
            fix_base=fix_base,
            review_wait=review_wait,
            bots=settings.bots,
            no_sandbox=settings.no_sandbox,
        )

        class RepoToolErrorHook(hooks.OnToolErrorHook):  # type: ignore[misc]
            async def run(self, context: hooks.HookContext | None, data: Exception) -> None:
                _ = context
                console.print(f"  [tool.fail]Tool error: {data!r}[/tool.fail]")

            async def __call__(self, error: Exception) -> None:
                await self.run(None, error)

        skills_path = str(SKILLS_PATH)

        agent_tools: list[Any] = [
            create_pr,
            get_branch_ci_status,
            get_commit_details,
            get_file_contents,
            get_pr_diff,
            get_pr_files,
            get_pr_status,
            get_pr_workflow_run_logs,
            list_bot_prs,
            list_branches,
            list_commits,
            merge_bot_pr,
            rebase_bot_pr,
            wait_for_ci,
            wait_for_reviews,
        ]
        if run_command is not None:
            agent_tools.append(run_command)

        config = LocalAgentConfig(
            model=model or settings.model,
            vertex=settings.vertex or None,
            project=(settings.google_cloud_project or None) if settings.vertex else None,
            location=(settings.google_cloud_location or None) if settings.vertex else None,
            system_instructions=system_instructions,
            policies=policies,
            hooks=[RepoToolErrorHook()],
            tools=agent_tools,
            skills_paths=[skills_path],
            # Workspaces are what file tools may touch. Grant the clone and the
            # skill, not PROJECT_ROOT: our checkout holds this project's source
            # and the .env our template puts there, and the skill is loaded via
            # skills_paths regardless.
            workspaces=[skills_path, workspace_tmp],
            capabilities=types.CapabilitiesConfig(
                enable_subagents=False,
                disabled_tools=[types.BuiltinTools.RUN_COMMAND],
            ),
        )

        effective_model = model or settings.model
        mode_str = "Vertex AI" if settings.vertex else "Developer API"
        click.secho(
            f"🚀 Spawning Agent for {repo} [model: {effective_model} | mode: {mode_str}]...",
            fg="cyan",
            bold=True,
        )
        async with Agent(config=config) as agent:
            prompt = _build_agent_prompt(
                owner,
                repo_name,
                settings,
                workspace_tmp,
                dry_run=dry_run,
                hint=hint,
            )
            wrapped_prompt = textwrap.fill(
                prompt,
                width=100,
                initial_indent="💬 Agent Prompt: ",
                subsequent_indent="                 ",
            )
            click.secho(wrapped_prompt, fg="blue")

            issues = _SdkIssueCollector()
            root_logger = logging.getLogger()
            root_logger.addHandler(issues)
            try:
                response = await agent.chat(prompt)
                await _render_agent_response(response)
            except Exception:
                logger.exception("Agent execution failed for %s", repo)
                console.print(
                    f"\n[bold red]❌ Agent execution failed for {repo} (see log above).[/bold red]",
                )
                return
            finally:
                root_logger.removeHandler(issues)

            if issues.messages:
                console.print(
                    f"\n[bold yellow]⚠ Agent execution for {repo} ended with "
                    f"{len(issues.messages)} error(s) the SDK only logged; "
                    f"the run may be incomplete.[/bold yellow]",
                )
                for message in issues.messages:
                    console.print(f"  [yellow]{message}[/yellow]")
            else:
                console.print(
                    f"\n[bold green]✨ Agent execution completed for {repo}.[/bold green]",
                )

            usage = agent.conversation.total_usage
            cached = usage.cached_content_token_count or 0
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            thinking = usage.thoughts_token_count or 0
            total = usage.total_token_count or (input_tokens + output_tokens)
            console.print(
                Panel(
                    f"Input: {input_tokens:,} (Cached: {cached:,})  |  "
                    f"Output: {output_tokens:,} (Thinking: {thinking:,})"
                    f"  |  Total: {total:,}",
                    title=f"Token Usage — {repo}",
                    style="dim cyan",
                ),
            )
    finally:
        await client.close()
        cleanup_sandbox = getattr(run_command, "cleanup", None)
        if cleanup_sandbox:
            cleanup_sandbox()
        if workspace_tmp:
            await asyncio.to_thread(_cleanup_workspace, workspace_tmp)


def _check_api_keys(settings: Settings) -> None:
    if settings.vertex:
        if not settings.gemini_api_key and not (settings.google_cloud_project and settings.google_cloud_location):
            click.secho(
                "❌ Vertex AI mode requires GOOGLE_CLOUD_PROJECT and "
                "GOOGLE_CLOUD_LOCATION (or a GEMINI_API_KEY for Express Mode).",
                fg="red",
                bold=True,
            )
            sys.exit(1)
    elif not settings.gemini_api_key:
        click.secho(
            "❌ GEMINI_API_KEY is required. Get one at "
            "https://aistudio.google.com/app/api-keys\n"
            "   Or set GOOGLE_GENAI_USE_VERTEXAI=TRUE to use Vertex AI "
            "with Application Default Credentials.",
            fg="red",
            bold=True,
        )
        sys.exit(1)


async def _validate_repo_accessibility(repo: str, token: str | None) -> None:
    owner_name, repo_name = repo.split("/", 1)
    client = GitHubClient(token=token)
    try:
        url = f"https://api.github.com/repos/{owner_name}/{repo_name}"
        response = await client.client.get(url, headers=client.headers)
        response.raise_for_status()
    except (httpx.HTTPError, GitHubClientError) as e:
        click.secho(
            f"❌ Failed to access repository '{repo}': {e}",
            fg="red",
            bold=True,
        )
        sys.exit(1)
    finally:
        await client.close()


async def run_agent(
    owner: str,
    concurrency: int,
    max_attempts: int,
    repo: str | None = None,
    *,
    dry_run: bool = False,
    auto_merge: bool = False,
    verify_all: bool = False,
    standalone_fix: bool = False,
    fix_base: bool = False,
    review_wait: int = 0,
    hint: str | None = None,
    no_sandbox: bool = False,
    model: str | None = None,
) -> None:
    """Run dependency-director triage and patch execution across repositories."""
    click.secho("⚙️  Initializing dependency-director configuration...", fg="cyan")
    settings = Settings()
    if no_sandbox:
        settings.no_sandbox = True

    _check_sandbox_requirements(verify_all=verify_all, no_sandbox=settings.no_sandbox)
    _check_api_keys(settings)

    if not settings.github_token:
        click.secho(
            "⚠️ GITHUB_TOKEN is not set. API calls to private repos will fail.\n"
            "   Get a token: run 'gh auth token' or visit https://github.com/settings/tokens",
            fg="yellow",
        )

    if repo:
        click.secho(
            f"📦 Restricting triage run to repository: {repo}",
            fg="magenta",
            bold=True,
        )
        await _validate_repo_accessibility(repo, settings.github_token)

        repo_kwargs: dict[str, Any] = {
            "dry_run": dry_run,
            "auto_merge": auto_merge,
            "verify_all": verify_all,
            "standalone_fix": standalone_fix,
            "fix_base": fix_base,
            "review_wait": review_wait,
            "hint": hint,
        }
        if model is not None:
            repo_kwargs["model"] = model

        await run_agent_for_repo(
            repo,
            settings,
            max_attempts,
            **repo_kwargs,
        )
    else:
        click.secho(f"🔍 Fetching repositories for owner '{owner}'...", fg="cyan")
        try:
            repos = await get_repositories(owner, settings.github_token)
        except Exception as e:  # noqa: BLE001
            click.secho(
                f"❌ Failed to fetch repositories for owner '{owner}': {e}",
                fg="red",
                bold=True,
            )
            sys.exit(1)

        if not repos:
            click.secho(
                f"ℹ️  No active repositories found for owner '{owner}'.",
                fg="yellow",
            )
            return

        click.secho(
            f"📚 Found {len(repos)} repositories to process with concurrency={concurrency}.",
            fg="green",
        )

        semaphore = asyncio.Semaphore(concurrency)

        async def worker(r: str) -> None:
            async with semaphore:
                click.secho(f"▶️  Starting processing for repository: {r}", fg="yellow")
                try:
                    r_kwargs: dict[str, Any] = {
                        "dry_run": dry_run,
                        "auto_merge": auto_merge,
                        "verify_all": verify_all,
                        "standalone_fix": standalone_fix,
                        "fix_base": fix_base,
                        "review_wait": review_wait,
                        "hint": hint,
                    }
                    if model is not None:
                        r_kwargs["model"] = model

                    await run_agent_for_repo(
                        r,
                        settings,
                        max_attempts,
                        **r_kwargs,
                    )
                    click.secho(
                        f"✅ Finished processing for repository: {r}",
                        fg="green",
                        bold=True,
                    )
                except Exception as e:  # noqa: BLE001
                    click.secho(
                        f"❌ Error processing repository {r}: {e}",
                        fg="red",
                        bold=True,
                    )

        await asyncio.gather(*(worker(r) for r in repos))


def print_banner() -> None:
    """Print the dependency-director ascii banner and initialization messages."""
    banner = r"""
    ____                            __                      ____  _                __
   / __ \___  ____  ___  ____  ____/ /__  ____  _______  __/ __ \(_)_______  _____/ /_____  _____
  / / / / _ \/ __ \/ _ \/ __ \/ __  / _ \/ __ \/ ___/ / / / / / / / ___/ _ \/ ___/ __/ __ \/ ___/
 / /_/ /  __/ /_/ /  __/ / / / /_/ /  __/ / / / /__/ /_/ / /_/ / / /  /  __/ /__/ /_/ /_/ / /
/_____/\___/ .___/\___/_/ /_/\__,_/\___/_/ /_/\___/\__, /_____/_/_/   \___/\___/\__/\____/_/
          /_/                                     /____/
"""
    console.print(banner, style="banner", highlight=False)
    console.print("Autonomous Dependency Triage Agent", style="status")
    console.rule(style="dim")


def _check_sandbox_requirements(*, verify_all: bool, no_sandbox: bool) -> None:
    if verify_all and no_sandbox:
        click.secho(
            "❌ Error: --verify-all is not allowed in --no-sandbox mode.\n"
            "   Local test verification requires OS-level sandboxing (sandbox-runtime) "
            "to safely execute untrusted package dependencies.",
            fg="red",
            bold=True,
        )
        sys.exit(1)

    if not no_sandbox and not is_srt_available():
        srt_installed = shutil.which("srt") is not None
        if srt_installed:
            if sys.platform == "linux" and not is_ripgrep_available():
                click.secho(
                    "❌ Error: sandbox-runtime (srt) is installed, but ripgrep (rg) is missing.\n"
                    "   srt requires ripgrep on Linux to enforce write restrictions.\n"
                    "   Please install ripgrep (e.g. sudo apt-get install ripgrep).",
                    fg="red",
                    bold=True,
                )
            else:
                click.secho(
                    "❌ Error: sandbox-runtime (srt) is installed but not functioning properly.\n"
                    "   Ensure bubblewrap and socat are installed on Linux.",
                    fg="red",
                    bold=True,
                )
        else:
            click.secho(
                "❌ Error: sandbox-runtime (srt) is not available.\n"
                "   Install it with: npm install -g @anthropic-ai/sandbox-runtime\n"
                "   Or pass --no-sandbox to run without sandboxing (run cautiously).",
                fg="red",
                bold=True,
            )
        sys.exit(1)

    if no_sandbox:
        click.secho(
            "⚠️ Warning: Running in --no-sandbox mode. Commands will run with host privileges.",
            fg="yellow",
        )


def _parse_target_string(s: str, target: str) -> tuple[str, str | None]:
    """Parse a cleaned target string into (owner, full_repo | None).

    Handles SSH (git@), URL (https://), bare github.com/, and plain owner/repo formats.
    Raises click.UsageError on invalid input.
    """
    if "git@" in s and ":" in s:
        # e.g., git@github.com:owner/repo
        _, path = s.split(":", 1)
        target_path = path
    elif "://" in s:
        # e.g., https://github.com/owner/repo
        parsed = urlparse(s)
        target_path = parsed.path.lstrip("/")
    elif s.startswith("github.com/"):
        target_path = s[len("github.com/") :]
    else:
        target_path = s

    if "/" in target_path:
        owner, repo_part = target_path.split("/", 1)
        if not repo_part:
            msg = f"Invalid target '{target}'. Use 'owner/repo' format or just 'owner' to scan all repos."
            raise click.UsageError(msg)
        # Strip trailing slash but reject further nested paths
        repo_part = repo_part.rstrip("/")
        if not repo_part or "/" in repo_part:
            msg = f"Invalid target '{target}'. Use 'owner/repo' format or just 'owner' to scan all repos."
            raise click.UsageError(msg)
        return owner, f"{owner}/{repo_part}"

    if not target_path:
        msg = f"Invalid target '{target}'."
        raise click.UsageError(msg)

    return target_path, None


def _resolve_target(target: str | None, default_owner: str | None) -> tuple[str, str | None]:
    """Resolve the owner and repository targets from the input string."""
    if not target:
        if default_owner:
            return default_owner, None
        msg = (
            "No target specified. Provide a GitHub user/org or owner/repo "
            "as an argument, or set DEPDIRECTOR_OWNER in your environment."
        )
        raise click.UsageError(msg)

    # Clean target string (remove .git suffix)
    s = target.strip()
    s = s.removesuffix(".git")

    # Normalize trailing slash for URL-like formats only
    is_url = "://" in s or "git@" in s or s.startswith("github.com/")
    if is_url:
        s = s.rstrip("/")

    return _parse_target_string(s, target)


@click.command()
@click.argument("target", required=False)
@click.option(
    "--concurrency",
    "-c",
    type=int,
    default=None,
    help="Number of repository tasks to run concurrently.",
)
@click.option(
    "--max-attempts",
    "-m",
    type=int,
    default=None,
    help="Maximum fix-and-test attempts per failing PR before it is skipped.",
)
@click.option(
    "--dry-run",
    "-d",
    is_flag=True,
    help="Enable dry-run mode (skip writing pull requests or merging).",
)
@click.option(
    "--auto-merge",
    "-a",
    is_flag=True,
    help="Automatically merge successful PRs (if verified and all status checks pass).",
)
@click.option(
    "--verify-all",
    "-v",
    is_flag=True,
    help="Always verify fixes locally by executing tests, even if bot-authored verification succeeded.",
)
@click.option(
    "--standalone-fix",
    is_flag=True,
    help="Run in standalone fix mode (bypasses dependency checker and attempts directly).",
)
@click.option(
    "--fix-base",
    is_flag=True,
    help=(
        "When the base branch is already failing CI, fix it in a separate PR "
        "against the base instead of only reporting it. Never merged automatically."
    ),
)
@click.option(
    "--review-wait",
    "-w",
    type=int,
    default=None,
    help="Number of minutes to wait for PR review approval checks to pass.",
)
@click.option(
    "--hint",
    "-H",
    type=str,
    default=None,
    help=("Extra context appended to the agent prompt (e.g., 'skip PR #5, known upstream issue')."),
)
@click.option(
    "--no-sandbox",
    is_flag=True,
    help="Disable sandbox-runtime (srt) sandboxing (runs with host privileges).",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Gemini model identifier (defaults to gemini-3.7-flash).",
)
def cli(
    target: str | None,
    concurrency: int | None,
    max_attempts: int | None,
    *,
    dry_run: bool,
    auto_merge: bool,
    verify_all: bool,
    standalone_fix: bool,
    fix_base: bool,
    review_wait: int | None,
    hint: str | None,
    no_sandbox: bool,
    model: str | None = None,
) -> None:
    """dependency-director: Autonomous dependency triage and patching agent.

    TARGET is a GitHub user/org (scan all repos) or owner/repo (scan one repo).
    If omitted, uses DEPDIRECTOR_OWNER from env.
    """
    print_banner()
    settings = Settings()

    no_sandbox_val = no_sandbox or settings.no_sandbox
    _check_sandbox_requirements(verify_all=verify_all, no_sandbox=no_sandbox_val)
    owner, repo = _resolve_target(target, settings.owner)

    concurrency_val = concurrency if concurrency is not None else settings.concurrency
    max_attempts_val = max_attempts if max_attempts is not None else settings.max_fix_attempts
    review_wait_val = review_wait if review_wait is not None else settings.review_wait

    try:
        run_kwargs: dict[str, Any] = {
            "dry_run": dry_run,
            "auto_merge": auto_merge,
            "verify_all": verify_all,
            "standalone_fix": standalone_fix,
            "fix_base": fix_base,
            "review_wait": review_wait_val,
            "hint": hint,
            "no_sandbox": no_sandbox_val,
        }
        if model is not None:
            run_kwargs["model"] = model

        asyncio.run(
            run_agent(
                owner,
                concurrency_val,
                max_attempts_val,
                repo,
                **run_kwargs,
            ),
        )
    except* KeyboardInterrupt:
        click.secho(
            "\n🛑 Execution interrupted by user. Shutting down...",
            fg="yellow",
            bold=True,
        )
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    cli()
