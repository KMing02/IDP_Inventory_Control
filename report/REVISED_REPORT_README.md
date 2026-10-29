# Approved report source and compilation

The authoritative approved report is `report/IDP_Final_Report.pdf`. Its matching source is
`report/revised/main.tex`, with six PDFs in `report/revised/figures/` and the notice-bearing `IEEEtran.cls`.
These files were copied unchanged from the reviewed submission package. This sync preserves the existing IEEE
conference layout; it does not imply conference submission or resolve any separate IDP formatting requirement.

From `report/revised/`, run `pdflatex main.tex` twice, or `tectonic main.tex`. Alternatively upload that directory
to Overleaf and select `main.tex`. The other TeX packages declared in the source must be available.
Compilation produces `main.pdf`. Compiler/package differences can change PDF bytes or pagination; retain the
approved PDF unless intentionally replacing it after review.

The older `report/build_report.py`, `template.html`, `style.css`, `results.py` and `figures.py` remain for historical
provenance. They do not maintain the revised narrative. The old generator writes to `report/IDP_Final_Report.pdf`,
so do not run it to regenerate the approved report. No training or historical report generator is needed to
compile the supplied LaTeX source. See `RESULTS_PROVENANCE.md` for evidence availability.
