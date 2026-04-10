from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # PyMuPDF

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FOLDER = SCRIPT_DIR
DEFAULT_OUTPUT_FOLDER = SCRIPT_DIR / "pdf_pages_png"

def pdf_to_png_pages(
    pdf_path: Path,
    output_root: Path,
    dpi: float = 150,
    prefix_pages: bool = True,
) -> int:
    """
    Render each page of pdf_path to PNG under output_root / <pdf_stem>/.

    Returns the number of pages written.
    """
    pdf_path = pdf_path.resolve()
    stem = pdf_path.stem
    out_dir = output_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    doc = fitz.open(pdf_path)
    try:
        n = doc.page_count
        for i in range(n):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            if prefix_pages:
                name = f"{stem}_page_{i + 1:04d}.png"
            else:
                name = f"page_{i + 1:04d}.png"
            pix.save(out_dir / name)
        return n
    finally:
        doc.close()


def iter_pdfs(folder: Path, recursive: bool) -> list[Path]:
    folder = folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(p for p in folder.glob(pattern) if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export each PDF page as a PNG image.")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT_FOLDER,
        help=f"Folder containing PDFs (default: {DEFAULT_INPUT_FOLDER})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Root folder for PNG output (default: {DEFAULT_OUTPUT_FOLDER})",
    )
    parser.add_argument(
        "--dpi",
        type=float,
        default=150,
        help="Render resolution in DPI (default: 150). Higher = larger files, sharper text.",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Also search subfolders for PDFs.",
    )
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Name files page_0001.png instead of <stem>_page_0001.png.",
    )
    args = parser.parse_args()

    pdfs = iter_pdfs(args.input, args.recursive)
    if not pdfs:
        print(f"No PDF files found under: {args.input.resolve()}")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    prefix_pages = not args.no_prefix

    for pdf in pdfs:
        n = pdf_to_png_pages(pdf, args.output, dpi=args.dpi, prefix_pages=prefix_pages)
        print(f"{pdf.name}: {n} page(s) -> {args.output / pdf.stem}")


if __name__ == "__main__":
    main()
