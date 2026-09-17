# Contributing

Fork this repository, create a branch in your fork, and open a pull request against
`main`. Do not include private `.env` files, credentials, source archives, or local
machine identifiers. Follow the setup and verification commands in
[COMMANDS.md](docs/COMMANDS.md).

All changes to this repository's `main` must use a pull request. The repository
owner is the sole code owner and merger. Other people may leave reviews, but only
the owner's approval satisfies the required code-owner review. New commits dismiss
stale approvals, and review conversations must be resolved before merging.

The owner may merge their own PR using the owner-only review bypass because GitHub
does not support self-approval. Use that bypass only for owner-authored PRs. GitHub
assigns bypass to an actor, not to a PR-author condition, so the owner technically
also has that ability on other PRs. Direct pushes, force pushes, and deletion of
`main` remain blocked by a separate ruleset with no bypass actors.

The public repository has a fresh sanitized history. Original development history,
source datasets, model artifacts, and private configuration are not included.
