#!/usr/bin/env python3
"""Convert a markdown file into a deliverable PDF, via headless Chrome.

Nothing is installed. Chrome is already on the machine and prints HTML to PDF with
better quality than the Python PDF libraries, which break wide tables and mangle
accented characters.

An isolated profile is mandatory: calling Chrome without `--user-data-dir` uses the
user's real profile, and with the browser open the call either fails or alters the
session state. The temporary profile is created in the working directory.

Tables are the element that suffers most under pagination. The CSS below forbids
breaking inside a table and forbids an orphaned heading at the foot of a page;
without it the PDF ends up with a section title alone in the footer and the table on
the next page.

Usage: python scripts/md_para_pdf.py input.md [-o output.pdf]
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 17mm 15mm 18mm 15mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Source Sans 3", "Segoe UI", Carlito, Calibri, sans-serif;
       font-size: 10.2pt; line-height: 1.45; color: #1b1b1b; margin: 0;
       orphans: 2; widows: 2; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 1em 0;
     padding-bottom: .35em; border-bottom: 2px solid #1D6B45; color: #14472e; }
h2 { font-size: 12.4pt; margin: 1.3em 0 .5em 0; color: #14472e;
     page-break-after: avoid; break-after: avoid; }
h2 + p, h2 + table { page-break-before: avoid; break-before: avoid; }
/* heading, context sentence and table travel together, otherwise the heading is
   left alone with one line at the foot of the page and the table moves to the next */
h2 + p + table { page-break-before: avoid; break-before: avoid; }
p { margin: 0 0 .68em 0; text-align: justify; hyphens: auto; }
table { border-collapse: collapse; width: 100%; margin: .45em 0 .9em 0;
        font-size: 9.3pt; page-break-inside: avoid; break-inside: avoid; }
th, td { border: 1px solid #c8cdd2; padding: 4.5px 8px; text-align: left;
         vertical-align: top; }
th { background: #eef3f0; font-weight: 600; }
tbody tr:nth-child(even) { background: #fafbfa; }
/* numeric columns are right-aligned, otherwise the F1 table becomes unreadable */
td, th { text-align: left; }
td.num, th.num { text-align: right; }
/* figure and caption form one block that must not split across pages */
figure { margin: .9em 0 1.1em 0; page-break-inside: avoid; break-inside: avoid; text-align: center; }
/* The HEIGHT cap exists because of the nearly square figures. Stretched to the
   page width they exceed 190 mm, i.e. three quarters of the usable height, and push
   everything that follows onto the next page, leaving a huge gap. The scatter plot
   was designed for a single column, and with the cap it returns to that size. Wide
   figures never come close to this limit. */
figure img { max-width: 100%; max-height: 104mm; height: auto; width: auto; }
figcaption { font-size: 8.8pt; color: #3c3c3c; text-align: left; margin-top: .35em; }
ul { margin: .3em 0 1em 0; padding-left: 1.15em; }
li { margin: .3em 0; page-break-inside: avoid; break-inside: avoid; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: .86em;
       background: #f1f3f4; padding: .1em .3em; border-radius: 3px; }
strong { color: #0f0f0f; }
"""


NUM = re.compile(r"^\**\s*[\d]+(?:[.,]\d+)*\s*(?:%|ha|m|m²|pts/m²|árvores)?\s*\**$")


def texto_da_celula(c: str) -> str:
    return re.sub(r"<[^>]+>", "", c).replace("&nbsp;", " ").strip()


def alinhar_colunas_numericas(html: str) -> str:
    """Right-align the numeric columns, deciding column by column.

    The decision is per column rather than per whole table: in a table with long
    labels in the first column and short numbers in the rest, the global fraction of
    short cells falls below the threshold and the entire table would be treated as
    text, making it inconsistent with its neighbours.

    Right alignment puts the decimal separator at the same vertical position, which
    makes a column of F1 scores easier to compare.
    """
    def por_tabela(m):
        corpo = m.group(0)
        linhas = re.findall(r"<tr>(.*?)</tr>", corpo, re.S)
        if not linhas:
            return corpo
        grade = [re.findall(r"<t[hd]>(.*?)</t[hd]>", l, re.S) for l in linhas]
        n = max((len(l) for l in grade), default=0)
        numerica = []
        for j in range(n):
            col = [texto_da_celula(l[j]) for l in grade[1:] if j < len(l)]
            col = [c for c in col if c]
            # the first column is never right-aligned, even when made entirely of
            # digits: it is a row label, not a value. Stand identifiers and figure
            # indices are numeric and would end up far from the text they describe,
            # opening a gap that associates them with the wrong column.
            numerica.append(j > 0 and bool(col) and
                            sum(bool(NUM.match(c)) for c in col) / len(col) >= 0.6)

        def marca(linha):
            k = [0]

            def sub(mm):
                j = k[0]
                k[0] += 1
                tag = mm.group(1)
                cls = ' class="num"' if j < n and numerica[j] else ""
                return f"<{tag}{cls}>{mm.group(2)}</{tag}>"

            return re.sub(r"<(t[hd])>(.*?)</t[hd]>", sub, linha, flags=re.S)

        for l in linhas:
            corpo = corpo.replace(f"<tr>{l}</tr>", f"<tr>{marca(l)}</tr>", 1)
        return corpo

    return re.sub(r"<table>.*?</table>", por_tabela, html, flags=re.S)


def embute_figuras(html: str, base: Path) -> str:
    """Inline the images as base64 and pair each one with the caption right below it.

    Inlining is mandatory: the HTML is written to a temporary directory for Chrome to
    print, so a relative image path points to the wrong place and the PDF comes out
    with an empty square where each figure should be. Resolving against the markdown
    folder and embedding also makes the PDF self-contained.

    The caption must stay attached to the figure: markdown emits the image in one
    paragraph and the caption in the next, two independent blocks that Chrome may
    separate at a page break. Joining them inside a `<figure>` makes
    `break-inside: avoid` apply to the pair, not just to the image.
    """
    import base64
    import mimetypes

    def dados(src: str) -> str | None:
        cam = (base / src).resolve()
        if not cam.exists():
            return None
        tipo = mimetypes.guess_type(cam.name)[0] or "image/png"
        return f"data:{tipo};base64,{base64.b64encode(cam.read_bytes()).decode()}"

    faltando = []
    # <p><img ...></p> followed by <p><strong>Figura N.</strong> ...</p>
    padrao = re.compile(
        r'<p><img alt="(?P<alt>[^"]*)" src="(?P<src>[^"]+)"\s*/?></p>\s*'
        r'(?:<p>(?P<leg>\s*<strong>Figura[^<]*</strong>.*?)</p>)?',
        re.S)

    def troca(m):
        uri = dados(m.group("src"))
        if uri is None:
            faltando.append(m.group("src"))
            return m.group(0)
        leg = m.group("leg")
        cap = f"<figcaption>{leg.strip()}</figcaption>" if leg else ""
        return f'<figure><img alt="{m.group("alt")}" src="{uri}">{cap}</figure>'

    html = padrao.sub(troca, html)
    if faltando:
        sys.exit("image not found: " + ", ".join(faltando))
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada", type=Path)
    ap.add_argument("-o", "--saida", type=Path)
    args = ap.parse_args()
    saida = args.saida or args.entrada.with_suffix(".pdf")

    chrome = next((c for c in ("google-chrome", "google-chrome-stable", "chromium",
                               "chromium-browser") if shutil.which(c)), None)
    if not chrome:
        sys.exit("no Chrome or Chromium on PATH")

    corpo = markdown.markdown(args.entrada.read_text(encoding="utf-8"),
                              extensions=["tables", "sane_lists"])
    corpo = alinhar_colunas_numericas(corpo)
    corpo = embute_figuras(corpo, args.entrada.resolve().parent)
    html = (f"<!doctype html><html lang=pt-BR><head><meta charset=utf-8>"
            f"<title>{args.entrada.stem}</title><style>{CSS}</style></head>"
            f"<body>{corpo}</body></html>")

    with tempfile.TemporaryDirectory(prefix="md2pdf-") as tmp:
        fonte = Path(tmp) / "doc.html"
        fonte.write_text(html, encoding="utf-8")
        cmd = [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
               f"--user-data-dir={Path(tmp) / 'perfil'}",
               "--no-pdf-header-footer",
               f"--print-to-pdf={saida.resolve()}", fonte.resolve().as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if not saida.exists() or saida.stat().st_size == 0:
            sys.exit(f"Chrome did not produce the PDF\n{r.stderr[-2000:]}")

    print(f"{saida}  {saida.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
