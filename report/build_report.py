"""Build the final report: figures, HTML, and a PDF printed by headless Chrome (or Edge).

    python report/build_report.py [--browser PATH]

Every number in the text comes from ``results.py`` through ``{{expression|format}}`` placeholders in
``template.html``; the build fails if a placeholder is left unresolved. The table of contents gets its page numbers
from a first print: Chrome writes every ``id`` that a link points to as a named destination, and the page of each
destination is read back from that PDF before the final print.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import figures  # noqa: E402
import results as R  # noqa: E402

BUILD = HERE / "build"
OUTPUT = HERE / "IDP_Final_Report.pdf"
BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chromium",
    "msedge",
)
PLACEHOLDER = re.compile(r"\{\{\s*(.+?)\s*(?:\|\s*([^}]*?))?\s*\}\}")
HEADING = re.compile(r'<h([12]) id="([^"]+)"[^>]*>(.*?)</h\1>', re.S)


def _number(value, spec: str) -> str:
    text = format(value, spec) if spec else str(value)
    return text.replace("-", "\u2212", 1) if text.startswith("-") else text


def _verdict(ours: float, classical: float) -> str:
    gap = ours - classical
    return "about the same" if abs(gap) < 0.15 else ("better" if gap < 0 else "worse")


def _context() -> dict:
    paired = R.DIRECT_PAIRED
    breaks = [value for name in ("kan", "kan_open") for value in R.DIRECT[name]["violation_by_seed"]]
    cap_rise = [R.UNCAPPED_REGRET[name] - R.DIRECT[name]["regret"] for name in ("monokan_constant", "kan", "monokan")]
    return {
        "R": R,
        "D": R.DIRECT,
        "C": R.CLASSICAL,
        "M": R.TS_MONOKAN,
        "K": R.TS_KAN,
        "TP": R.TS_PAIRED,
        "DP": paired,
        "DS": R.DIRECT_START,
        "UP": R.UNCAPPED_PAIRED,
        "IC": R.DIRECT_IMPLIED_CORR,
        "MA": R.MATCHED,
        "KS": R.DIRECT["kan"]["violation_by_seed"],
        "KO": R.DIRECT["kan_open"]["violation_by_seed"],
        "BREAKS": (min(breaks), max(breaks)),
        "CAP_RISE": (min(cap_rise), max(cap_rise)),
        "cost_share": -paired["constant"]["mean"] / R.DIRECT["monokan_constant"]["reward"] * 100,
        "max_reaction": max(-arm["slope"] for arm in R.DIRECT.values()),
        **{f"verdict_{window}": _verdict(R.TS_MONOKAN[window][1], R.CLASSICAL["percentile"][window][1])
           for window in R.WINDOWS},
    }


def _signed(value: float) -> str:
    return _number(value, "+.2f")


def _paired_row(label: str, row: dict, reading: str | None = None) -> str:
    cells = (f'<td>{label}</td><td class="num">{_signed(row["mean"])}</td>'
             f'<td class="num">[{_signed(row["ci"][0])}, {_signed(row["ci"][1])}]</td>'
             f'<td class="num">{html.escape(row["p_t"])}</td><td class="num">{html.escape(row["p_w"])}</td>'
             f'<td class="num">{row["ahead"]}%</td>')
    return f"<tr>{cells}</tr>" if reading is None else f"<tr>{cells}<td>{reading}</td></tr>"


def _tables() -> dict[str, str]:
    direct = "\n".join(
        f'<tr{" class=\"ours\"" if name.startswith("monokan") else ""}><td>{arm["label"]}</td><td>{arm["start"]}</td>'
        f'<td class="num">{arm["reward"]:,.2f} ± {arm["reward_sd"]:.2f}</td><td class="num">{arm["regret"]:.2f}%</td>'
        f'<td class="num">{arm["violation"]:.1f}%</td><td class="num">{_number(-arm["slope"], ".2f")}</td>'
        f'<td class="num">{arm["params"]:,}</td></tr>'
        for name, arm in R.DIRECT.items()
    )
    direct_labels = {
        "constant": "Both constant",
        "open": "Both open",
        "first": "MonoKAN open, KAN constant (first runs)",
    }
    direct_paired = "\n".join(_paired_row(direct_labels[key], row) for key, row in R.DIRECT_PAIRED.items())
    labels = {
        "expanding": "Final setup, all demand seen so far",
        "rolling": "Final setup, last 32 days",
        "fixed": "Final setup, first 32 days",
        "matched": "1,000 products, profit objective, last 32 days",
    }
    paired = "\n".join(_paired_row(labels[key], row, row["reading"]) for key, row in R.TS_PAIRED.items())
    confirmation = "\n".join(
        f'<tr{" class=\"ours\"" if "5,000" in name else ""}><td>{name}</td>'
        f'<td class="num">{reward:,.2f}</td><td class="num">{regret:.2f}%</td></tr>'
        for name, reward, regret in R.CONFIRMATION_3000
    )
    sample = "\n".join(
        f"<tr><td>{name}</td>"
        + "".join(f'<td class="num">{reward:,.2f} / {regret:.2f}%</td>'
                  for reward, regret in zip(R.SAMPLE_REWARD[name], R.SAMPLE_REGRET[name]))
        + "</tr>"
        for name in R.SAMPLE_REGRET
    )
    return {"DIRECT_ROWS": direct, "DIRECT_PAIRED_ROWS": direct_paired, "TS_PAIRED_ROWS": paired,
            "CONFIRMATION_ROWS": confirmation, "SAMPLE_ROWS": sample}


def _toc(template: str, pages: dict[str, int]) -> str:
    items = []
    for level, anchor, title in HEADING.findall(template):
        if anchor == "abstract":
            continue
        title = re.sub(r"<[^>]+>", "", title)
        title = PLACEHOLDER.sub(lambda match: _number(eval(match.group(1), {}, _context()), match.group(2) or ""),
                                title)
        css = "top" if level == "1" else "sub"
        page = pages.get(anchor, "")
        items.append(f'<li class="{css}"><a class="label" href="#{anchor}">{title}</a><span class="dots"></span>'
                     f'<span class="page">{page}</span></li>')
    return "<ol>\n" + "\n".join(items) + "\n</ol>"


def render(pages: dict[str, int]) -> str:
    template = (HERE / "template.html").read_text(encoding="utf-8")
    blocks = {"TOC": _toc(template, pages), **_tables()}
    context = _context()

    def substitute(match: re.Match) -> str:
        expression, spec = match.group(1), match.group(2) or ""
        if expression in blocks:
            return blocks[expression]
        return html.escape(_number(eval(expression, {}, context), spec), quote=False)

    page = PLACEHOLDER.sub(substitute, template)
    leftover = re.findall(r"\{\{.*?\}\}", page)
    if leftover:
        raise ValueError(f"unresolved placeholders: {leftover[:3]}")
    return page


def _browser(requested: str | None) -> str:
    for candidate in ([requested] if requested else []) + list(BROWSERS):
        found = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if found:
            return found
    raise FileNotFoundError("no Chrome or Edge found; pass --browser")


def print_pdf(browser: str, html_path: Path, pdf_path: Path) -> None:
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
             f"--user-data-dir={profile}", "--no-pdf-header-footer", "--generate-pdf-document-outline",
             f"--print-to-pdf={pdf_path}", html_path.as_uri()],
            check=True, capture_output=True, timeout=180,
        )
    if not pdf_path.exists():
        raise RuntimeError("the browser did not write a PDF")


def _pdf_objects(data: bytes) -> dict[int, bytes]:
    objects = {int(m.group(1)): m.group(2) for m in re.finditer(rb"(\d+) 0 obj(.*?)endobj", data, re.S)}
    # Objects packed into compressed object streams.
    for body in list(objects.values()):
        if b"/ObjStm" not in body:
            continue
        stream = re.search(rb"stream\r?\n(.*?)\r?\nendstream", body, re.S)
        first = int(re.search(rb"/First (\d+)", body).group(1))
        content = zlib.decompress(stream.group(1))
        header = list(map(int, content[:first].split()))
        offsets = list(zip(header[0::2], header[1::2]))
        for index, (number, offset) in enumerate(offsets):
            end = offsets[index + 1][1] if index + 1 < len(offsets) else len(content) - first
            objects.setdefault(number, content[first + offset:first + end])
    return objects


def destination_pages(pdf_path: Path) -> dict[str, int]:
    """Page number (1-based) of every named destination in a PDF written by Chrome."""
    data = pdf_path.read_bytes()
    objects = _pdf_objects(data)

    def resolve(reference: bytes) -> bytes:
        match = re.fullmatch(rb"\s*(\d+) 0 R\s*", reference)
        return objects[int(match.group(1))] if match else reference

    catalog = next(body for body in objects.values() if b"/Type /Catalog" in body)
    pages_root = int(re.search(rb"/Pages (\d+) 0 R", catalog).group(1))
    order: list[int] = []

    def walk(number: int) -> None:
        body = objects[number]
        if b"/Type /Pages" in body:
            for kid in re.findall(rb"(\d+) 0 R", re.search(rb"/Kids\s*\[(.*?)\]", body, re.S).group(1)):
                walk(int(kid))
        else:
            order.append(number)

    walk(pages_root)
    page_of = {number: index + 1 for index, number in enumerate(order)}
    pages: dict[str, int] = {}
    pairs = []
    dests = re.search(rb"/Dests (\d+) 0 R", catalog)
    if dests:
        pairs += re.findall(rb"/([^\s/\[\]<>()]+)\s*\[\s*(\d+) 0 R", objects[int(dests.group(1))])
    names = re.search(rb"/Names\s*<<.*?/Dests (\d+) 0 R", catalog, re.S)
    if names:
        stack = [int(names.group(1))]
        while stack:
            body = objects[stack.pop()]
            kids = re.search(rb"/Kids\s*\[(.*?)\]", body, re.S)
            if kids:
                stack += [int(kid) for kid in re.findall(rb"(\d+) 0 R", kids.group(1))]
            entries = re.search(rb"/Names\s*\[(.*)\]", body, re.S)
            if entries:
                pairs += re.findall(rb"\(([^)]*)\)\s*\[\s*(\d+) 0 R", entries.group(1))
                for name, ref in re.findall(rb"\(([^)]*)\)\s*(\d+ 0 R)", entries.group(1)):
                    target = re.search(rb"\[\s*(\d+) 0 R", resolve(ref))
                    if target:
                        pairs.append((name, target.group(1)))
    for name, page_object in pairs:
        if int(page_object) in page_of:
            pages[name.decode("latin-1")] = page_of[int(page_object)]
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--browser", help="path to Chrome or Edge")
    args = parser.parse_args()
    browser = _browser(args.browser)

    if BUILD.exists():
        shutil.rmtree(BUILD)
    figures.make_all(BUILD / "figures")
    shutil.copy(HERE / "style.css", BUILD / "style.css")
    html_path, pdf_path = BUILD / "report.html", BUILD / "report.pdf"

    pages: dict[str, int] = {}
    for attempt in range(3):
        html_path.write_text(render(pages), encoding="utf-8")
        pdf_path.unlink(missing_ok=True)
        print_pdf(browser, html_path, pdf_path)
        found = destination_pages(pdf_path)
        if not found:
            raise RuntimeError("no named destinations found in the PDF; cannot number the contents")
        if found == pages:
            break
        pages = found
    else:
        html_path.write_text(render(pages), encoding="utf-8")
        print_pdf(browser, html_path, pdf_path)
    shutil.copy(pdf_path, OUTPUT)
    count = len(re.findall(rb"/Type\s*/Page[^s]", pdf_path.read_bytes()))
    print(f"wrote {OUTPUT} ({count} pages, contents numbered from {len(pages)} destinations)")


if __name__ == "__main__":
    main()
