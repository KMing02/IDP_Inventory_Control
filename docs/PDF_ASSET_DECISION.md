# PDF Asset Decision

Date: 2026-08-02

Rule for this repository: do not commit third-party papers unless redistribution permission is clear. Prefer bibliographic citations, DOI/arXiv links, or short local notes.

## Files Inspected

### `docs/1-s2.0-S0893608025011591-main.pdf`

- Size: 3.4 MB.
- Authored by this project: no.
- Type: third-party literature paper.
- Redistribution rights known: no.
- Recommendation: do not commit. Keep as local reading material or replace with citation/link and extracted notes.

### `docs/2507.22040v1.pdf`

- Size: 2.5 MB.
- Authored by this project: no.
- Type: third-party literature paper or arXiv manuscript.
- Redistribution rights known: no.
- Recommendation: do not commit. Keep as local reading material or replace with citation/link and extracted notes.

### `docs/Structure-Informed Neural Policies for Lost-Sales Inventory Control.pdf`

- Size: 17 MB.
- Authored by this project: unclear from filename alone; appears to be a paper/reference PDF rather than a generated project report.
- Type: likely third-party or supervisor-provided reference.
- Redistribution rights known: no.
- Recommendation: do not commit unless the team confirms it is an intentionally versioned project artifact with redistribution permission. Prefer committing `docs/MAGGIAR_2025_CODEX_REFERENCE.md` and citation metadata instead.

## Git Policy

`docs/*.pdf` should be treated as local literature by default and ignored. If the team intentionally wants to version a project-authored PDF report, place it under an explicitly documented report path and update `.gitignore` with a narrow exception.

Existing tracked generated artifact:

- `output/pdf/idp_monotonic_inventory_report.pdf`

Recommendation: leave it untouched in this cleanup. Decide in a separate commit whether generated reports should remain tracked.
