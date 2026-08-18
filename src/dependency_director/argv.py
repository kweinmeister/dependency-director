"""Argv parsing primitives shared by the sandbox validator and safety policies.

These helpers answer one question: given a raw argv list, which executable
actually runs, and with what arguments?  Answering it correctly requires
looking past ``KEY=val`` prefixes, ``env(1)`` wrappers, and ``&&``/``||``
compounds — a caller that inspects ``argv[0]`` alone is trivially evaded.

This module is deliberately free of security policy.  It reports what a
command *is*; deciding whether that is allowed belongs to the caller.
"""

from pathlib import Path
from typing import NamedTuple

ENV_FLAGS_WITH_ARG = {"-u", "--unset"}

# git's own options, consumed before the subcommand. Those listed here take a
# separate following argument, so the subcommand sits two tokens later.
GIT_GLOBAL_FLAGS_WITH_ARG = {
    "-C",
    "-c",
    "--config-env",
    "--exec-path",
    "--git-dir",
    "--namespace",
    "--super-prefix",
    "--work-tree",
}

COMPOUND_OPERATORS = {"&&", "||"}

# Every spelling by which 'git push' asks to overwrite the remote branch.
FORCE_PUSH_FLAGS = {"-f", "--force", "--force-with-lease", "--force-if-includes"}


class CompoundPart(NamedTuple):
    """A sub-command extracted from a compound argv list.

    Attributes:
        argv: The sub-command argument list.
        operator: The operator preceding this sub-command (None for the first).

    """

    argv: list[str]
    operator: str | None


class ResolvedExe(NamedTuple):
    """The executable an argv list actually invokes.

    Attributes:
        wrapper_idx: Index of the outermost executable token, which is the
            ``env`` wrapper when one is present.
        exe_idx: Index of the executable that ultimately runs, with any
            ``env`` wrapper unwrapped.
        name: Basename of that executable, lowercased.
        env_assignments: Every ``KEY=val`` token applied to the command,
            whether written bare or passed through ``env``.

    """

    wrapper_idx: int
    exe_idx: int
    name: str
    env_assignments: list[str]

    @property
    def env_wrapped(self) -> bool:
        """Return True if the executable was reached through an ``env`` wrapper."""
        return self.wrapper_idx != self.exe_idx


def split_compound_argv(argv: list[str]) -> list[CompoundPart]:
    """Split an argv list on ``&&`` and ``||`` tokens into sub-commands.

    Returns a list of CompoundPart(argv, operator) tuples.  The first
    part has operator=None; subsequent parts record the joining operator.
    """
    parts: list[CompoundPart] = []
    current: list[str] = []
    current_op: str | None = None

    for token in argv:
        if token in COMPOUND_OPERATORS:
            parts.append(CompoundPart(argv=current, operator=current_op))
            current = []
            current_op = token
        else:
            current.append(token)

    parts.append(CompoundPart(argv=current, operator=current_op))
    return parts


def _unwrap_env(argv: list[str], idx: int) -> tuple[int, str, list[str]]:
    """Resolve the executable wrapped by an ``env`` invocation at ``idx``.

    Returns (exe_idx, exe_name, env_assignments).  Falls back to ``env``
    itself when no wrapped command follows.
    """
    assignments: list[str] = []
    j = idx + 1
    while j < len(argv):
        token = argv[j]
        if token == "--":
            j += 1
            if j < len(argv):
                return j, Path(argv[j]).name.lower(), assignments
            break
        if token in ENV_FLAGS_WITH_ARG:
            j += 2
        elif token.startswith("-"):
            j += 1
        elif "=" in token:
            assignments.append(token)
            j += 1
        else:
            return j, Path(token).name.lower(), assignments
    return idx, "env", assignments


def resolve_exe(argv: list[str]) -> ResolvedExe | None:
    """Identify the executable an argv list runs, seeing through wrappers.

    Walks leading ``KEY=val`` assignments and unwraps a single ``env``
    invocation.  Returns None when the list contains no executable at all.
    """
    assignments: list[str] = []
    for i, token in enumerate(argv):
        if "=" in token and not token.startswith("-"):
            assignments.append(token)
            continue

        exe_name = Path(token).name.lower()
        if exe_name == "env":
            exe_idx, exe_name, env_assignments = _unwrap_env(argv, i)
            assignments.extend(env_assignments)
            return ResolvedExe(i, exe_idx, exe_name, assignments)
        return ResolvedExe(i, i, exe_name, assignments)
    return None


def git_subcommand(argv: list[str], start: int) -> str | None:
    """Return the git subcommand at or after ``start``, skipping global options.

    ``git -C dir push`` yields ``push``; ``git commit -m push`` yields
    ``commit``, so a caller looking for a push is not fooled by an argument
    that merely spells one.
    """
    i = start
    while i < len(argv):
        token = argv[i]
        if token in GIT_GLOBAL_FLAGS_WITH_ARG:
            i += 2
        elif token.startswith("-"):
            i += 1
        else:
            return token.lower()
    return None


def is_git_push_argv(argv: list[str]) -> bool:
    """Return True if any sub-command in ``argv`` invokes ``git push``.

    Sees through ``env`` wrappers, ``KEY=val`` prefixes, absolute paths, and
    ``&&``/``||`` compounds.
    """
    return any(_push_start(part.argv) is not None for part in split_compound_argv(argv))


def _push_start(argv: list[str]) -> int | None:
    """Return where a ``git push``'s arguments begin, or None if it is not one."""
    resolved = resolve_exe(argv)
    if resolved is None or resolved.name != "git":
        return None
    start = resolved.exe_idx + 1
    return start if git_subcommand(argv, start) == "push" else None


def _push_is_forced(argv: list[str], start: int) -> bool:
    """Report whether a push's arguments ask for a forced update.

    Covers the flag spellings, short-flag bundles such as ``-qf``, and the
    ``+refspec`` form, which forces the update without naming a flag at all.
    ``--force-with-lease`` counts: it only checks the remote against what the
    caller last fetched, so a fetch immediately beforehand — which is what a
    rejected push prompts — makes it overwrite just as freely.
    """
    for token in argv[start:]:
        base = token.split("=", 1)[0]
        if base in FORCE_PUSH_FLAGS:
            return True
        if token.startswith("-") and not token.startswith("--") and "f" in token[1:]:
            return True
        # After 'push', a leading '+' marks a refspec as forced.
        if token.startswith("+"):
            return True
    return False


def is_forced_git_push_argv(argv: list[str]) -> bool:
    """Return True if any sub-command in ``argv`` force-pushes.

    A forced push discards whatever the remote branch already carried, which
    for a branch under review is a reviewer's commits.
    """
    for part in split_compound_argv(argv):
        start = _push_start(part.argv)
        if start is not None and _push_is_forced(part.argv, start):
            return True
    return False
