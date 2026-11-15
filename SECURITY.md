# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
**Security → Report a vulnerability** on this repository
([private vulnerability reporting](https://github.com/esherma/tatnem-scoring/security/advisories/new)).
Do not open a public issue for a suspected vulnerability.

In scope: any way the published scoring rules, stake locks, settlement or `fbscore verify`
compute a result that contradicts `spec/scoring.md`, or any defect in the release pipeline.
We aim to acknowledge reports within 5 business days.

## Supported versions

Every released rule module (`fbscore.rules.v1`, ...) stays supported while any question
on the platform is frozen at it. Fixes never change a released rule's behavior in place;
see `spec/scoring.md` §10.

## Verifying a release

Releases are published from GitHub Actions with PyPI trusted publishing, so no API token
exists. Each release carries PEP 740 attestations on PyPI and a GitHub build-provenance
attestation:

```bash
gh attestation verify tatnem_scoring-<version>-py3-none-any.whl --repo esherma/tatnem-scoring
```
