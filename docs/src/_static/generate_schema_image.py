"""Generate the README schema example image."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCALE = 2
OUT = Path(__file__).with_name("various_ome_arrow_schema.png")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a readable system font with a Pillow fallback."""
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size * SCALE)
    return ImageFont.load_default(size * SCALE)


def _text_center(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    fill: str = "#2b2f33",
) -> None:
    """Draw centered text inside a box."""
    x0, y0, x1, y1 = (v * SCALE for v in box)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=7 * SCALE)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.multiline_text(
        (x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2),
        text,
        font=font,
        fill=fill,
        align="center",
        spacing=7 * SCALE,
    )


def _rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str,
    width: int = 2,
    radius: int = 8,
) -> None:
    """Draw a scaled rounded rectangle."""
    draw.rounded_rectangle(
        tuple(v * SCALE for v in box),
        radius=radius * SCALE,
        fill=fill,
        outline=outline,
        width=width * SCALE,
    )


def _table(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    col_widths: tuple[int, ...],
    rows: int,
    *,
    headers: tuple[str, ...],
    row_h: int = 76,
    header_h: int = 48,
) -> list[list[tuple[int, int, int, int]]]:
    """Draw a light table and return body cell boxes."""
    border = "#cfd5db"
    total_w = sum(col_widths)
    total_h = header_h + rows * row_h
    draw.rectangle(
        (x * SCALE, y * SCALE, (x + total_w) * SCALE, (y + total_h) * SCALE),
        fill="#ffffff",
        outline=border,
        width=6 * SCALE,
    )

    cx = x
    header_font = _font(17)
    for idx, width in enumerate(col_widths):
        if idx:
            draw.line(
                (
                    cx * SCALE,
                    y * SCALE,
                    cx * SCALE,
                    (y + total_h) * SCALE,
                ),
                fill=border,
                width=6 * SCALE,
            )
        _text_center(draw, (cx, y, cx + width, y + header_h), headers[idx], header_font)
        cx += width

    draw.line(
        (
            x * SCALE,
            (y + header_h) * SCALE,
            (x + total_w) * SCALE,
            (y + header_h) * SCALE,
        ),
        fill=border,
        width=6 * SCALE,
    )
    for row in range(1, rows):
        yy = y + header_h + row * row_h
        draw.line(
            (x * SCALE, yy * SCALE, (x + total_w) * SCALE, yy * SCALE),
            fill=border,
            width=6 * SCALE,
        )

    cells: list[list[tuple[int, int, int, int]]] = []
    for row in range(rows):
        row_cells = []
        cx = x
        for width in col_widths:
            row_cells.append(
                (
                    cx,
                    y + header_h + row * row_h,
                    cx + width,
                    y + header_h + (row + 1) * row_h,
                )
            )
            cx += width
        cells.append(row_cells)
    return cells


def _draw_measurement(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
) -> None:
    """Draw a compact measurement icon."""
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    _rounded_rect(
        draw,
        (cx - 20, cy - 18, cx + 20, cy + 18),
        fill="#7ea2c5",
        outline="#607f9f",
        radius=7,
    )
    font = _font(15, bold=True)
    _text_center(
        draw,
        (cx - 20, cy - 19, cx + 20, cy + 18),
        "12\n34",
        font,
        fill="#fff",
    )


def _draw_image_tile(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
) -> None:
    """Draw a small microscopy image tile."""
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    tile = (cx - 23, cy - 23, cx + 23, cy + 23)
    draw.rectangle(
        tuple(v * SCALE for v in tile),
        fill="#f6c23e",
        outline="#d98a1a",
        width=3 * SCALE,
    )
    draw.rectangle(
        tuple((v + 5 if i < 2 else v - 5) * SCALE for i, v in enumerate(tile)),
        fill="#fdf5c9",
        outline="#f0ad2e",
        width=2 * SCALE,
    )
    points = []
    for step in range(26):
        t = step / 25
        px = cx - 14 + int(t * 28)
        py = cy + int(math.sin(t * math.tau * 1.5) * 9)
        points.append((px * SCALE, py * SCALE))
    draw.line(points, fill="#1fba6d", width=4 * SCALE)
    for px, py in ((cx - 9, cy - 5), (cx + 1, cy + 5), (cx + 10, cy - 3)):
        draw.ellipse(
            ((px - 2) * SCALE, (py - 2) * SCALE, (px + 2) * SCALE, (py + 2) * SCALE),
            fill="#229e5a",
        )


def _draw_label_tile(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
) -> None:
    """Draw a compact label raster icon."""
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    tile = (cx - 23, cy - 23, cx + 23, cy + 23)
    draw.rectangle(
        tuple(v * SCALE for v in tile),
        fill="#f6c23e",
        outline="#d98a1a",
        width=3 * SCALE,
    )
    draw.rectangle(
        tuple((v + 5 if i < 2 else v - 5) * SCALE for i, v in enumerate(tile)),
        fill="#fdf5c9",
        outline="#f0ad2e",
        width=2 * SCALE,
    )
    for dx, dy, color in ((-8, -4, "#56c2a4"), (4, 8, "#7bd389"), (9, -7, "#a5d76e")):
        draw.ellipse(
            (
                (cx + dx - 10) * SCALE,
                (cy + dy - 8) * SCALE,
                (cx + dx + 10) * SCALE,
                (cy + dy + 8) * SCALE,
            ),
            fill=color,
            outline="#258a5a",
            width=2 * SCALE,
        )


def _draw_shape_icon(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
) -> None:
    """Draw a vector shape icon."""
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    polygon = [
        ((cx - 20) * SCALE, (cy - 3) * SCALE),
        ((cx - 8) * SCALE, (cy - 19) * SCALE),
        ((cx + 16) * SCALE, (cy - 13) * SCALE),
        ((cx + 21) * SCALE, (cy + 11) * SCALE),
        ((cx - 4) * SCALE, (cy + 19) * SCALE),
    ]
    draw.polygon(polygon, fill="#d8efff", outline="#2c7fb8")
    draw.line([*polygon, polygon[0]], fill="#2c7fb8", width=3 * SCALE)
    for px, py in (
        (cx - 20, cy - 3),
        (cx - 8, cy - 19),
        (cx + 16, cy - 13),
        (cx + 21, cy + 11),
        (cx - 4, cy + 19),
    ):
        draw.ellipse(
            ((px - 3) * SCALE, (py - 3) * SCALE, (px + 3) * SCALE, (py + 3) * SCALE),
            fill="#0f5c99",
        )


def main() -> None:
    """Generate the image asset."""
    width, height = 1000, 252
    image = Image.new("RGBA", (width * SCALE, height * SCALE), "#ffffff")
    draw = ImageDraw.Draw(image)

    left = _table(
        draw,
        0,
        6,
        (132, 118),
        2,
        headers=("Measurements", "Images"),
        row_h=80,
        header_h=50,
    )
    _draw_measurement(draw, left[0][0])
    _draw_image_tile(draw, left[0][1])
    _draw_measurement(draw, left[1][0])
    _draw_image_tile(draw, left[1][1])

    _text_center(draw, (270, 96, 312, 142), "or", _font(18))

    center = _table(
        draw,
        335,
        6,
        (105, 105, 105),
        2,
        headers=("Images", "Labels", "Shapes"),
        row_h=80,
        header_h=50,
    )
    for row in center:
        _draw_image_tile(draw, row[0])
        _draw_label_tile(draw, row[1])
        _draw_shape_icon(draw, row[2])

    _text_center(draw, (670, 96, 712, 142), "or", _font(18))

    draw.rectangle(
        (745 * SCALE, 6 * SCALE, 995 * SCALE, 246 * SCALE),
        fill="#ffffff",
        outline="#cfd5db",
        width=6 * SCALE,
    )
    _text_center(draw, (768, 44, 972, 126), "Your\nschema\nhere!", _font(23))
    chip_font = _font(12, bold=True)
    chips = [
        (770, 155, 828, 181, "images", "#dff2ff", "#2c7fb8"),
        (836, 155, 892, 181, "labels", "#eaf8dc", "#4e9a33"),
        (900, 155, 962, 181, "shapes", "#e8efff", "#4569b2"),
        (804, 192, 930, 218, "measurements", "#edf1f5", "#61707f"),
    ]
    for x0, y0, x1, y1, label, fill, outline in chips:
        _rounded_rect(draw, (x0, y0, x1, y1), fill=fill, outline=outline, radius=6)
        _text_center(draw, (x0, y0, x1, y1), label, chip_font, fill="#25313b")

    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image.save(OUT)


if __name__ == "__main__":
    main()
