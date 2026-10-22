"""Save a matplotlib figure as <base>_final.pdf only.

<base> is the output path given to a plotting script's --out, minus its
extension (so --out foo.png and --out foo.pdf both write foo_final.pdf).

The figure is first saved as a vector PDF with embedded TrueType (Type 42)
fonts to a temporary file, which is then converted by Ghostscript
(gs -sDEVICE=pdfwrite -dNoOutputFonts) so all text becomes outlines and the
final file contains no fonts at all. The temporary PDF is deleted; no PNG or
intermediate PDF is kept.

Raises if the intermediate PDF contains a Type 3 font (not allowed; set
pdf.fonttype = 42 in the calling script), or if Ghostscript (gs) is not on
PATH (without it the final PDF can't be made).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile


def _type3_fonts(pdf_path):
    """Names of Type 3 fonts in pdf_path. matplotlib writes font
    dictionaries as plain (uncompressed) PDF objects, so a byte search
    finds them."""
    with open(pdf_path, "rb") as f:
        data = f.read()
    names = []
    for obj in re.findall(rb"<<(?:(?!>>).)*?/Subtype\s*/Type3(?:(?!>>).)*?>>", data, re.S):
        m = re.search(rb"/BaseFont\s*/([^\s/<>\[\]]+)", obj)
        names.append(m.group(1).decode(errors="replace") if m else "?")
    return names


def _find_gs():
    """Path to Ghostscript: $GS, then PATH, then the running Python's env
    and the base miniconda install (gs lives there even when that bin dir
    isn't on PATH, e.g. under another env or on a compute node)."""
    candidates = [os.environ.get("GS"), shutil.which("gs"),
                  os.path.join(os.path.dirname(sys.executable), "gs"),
                  os.path.expanduser("~/miniconda3/bin/gs")]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def save_figure(fig, out, dpi=300, **savefig_kwargs):
    """Save fig as <base>_final.pdf (see module docstring). Extra keyword
    arguments go to fig.savefig. Returns the list of paths written."""
    base = os.path.splitext(out)[0]
    final = f"{base}_final.pdf"

    gs = _find_gs()
    if gs is None:
        raise RuntimeError(f"Ghostscript (gs) not found on PATH; can't write {final}")

    fd, pdf = tempfile.mkstemp(suffix=".pdf", dir=os.path.dirname(os.path.abspath(final)))
    os.close(fd)
    try:
        fig.savefig(pdf, dpi=dpi, **savefig_kwargs)
        type3 = _type3_fonts(pdf)
        if type3:
            raise RuntimeError(f"figure for {final} contains Type 3 font(s) {type3}; set "
                               'matplotlib.rcParams["pdf.fonttype"] = 42')
        subprocess.run([gs, "-q", "-o", final, "-sDEVICE=pdfwrite", "-dNoOutputFonts", pdf],
                       check=True)
    finally:
        os.remove(pdf)

    print(f"Saved {final}")
    return [final]
