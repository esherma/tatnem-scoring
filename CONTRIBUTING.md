# Contributing

Thanks for your interest. **This repository is currently published read-only**: issues and
discussion are welcome, but external pull requests are not merged until the contributor
terms (DCO or CLA) are settled. Please open an issue first.

## Development

```bash
uv sync                      # Python and dev tools, pinned by uv.lock
uv run pre-commit install    # gitleaks, ruff, basic hygiene on every commit
uv run pytest                # unit, worked-example, property and verify tests
uv run pytest -m slow        # simulation tests S1-S8 (a few minutes)
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

Hypothesis profiles: `HYPOTHESIS_PROFILE=dev|ci|nightly`.

## Rules for changes to scoring

* A released rule module (`src/fbscore/rules/v1.py`, ...) is frozen. Behavior changes go in
  a new module registered in `fbscore.rules.RULE_MODULES` with a new package version.
* Abuse-relevant parameters (`fbscore.ABUSE_PARAMETER_NAMES`) are always required
  arguments with no default. Tests use deliberately unrealistic `EXAMPLE_*` values.
* `spec/scoring.md` is normative; code and spec change together.
