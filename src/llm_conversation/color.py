"""Color utilities for ANSI color palettes and distinct color generation."""

import functools

from coloraide import Color

RGB = tuple[int, int, int]


def _color_distance(c1: RGB, c2: RGB) -> float:
    """Calculate perceptually accurate color distance between two RGB colors.

    Args:
        c1, c2: RGB color values (0-255)

    Returns:
        Color difference value (lower = more similar)
    """
    # Convert RGB 0-255 to 0-1 range for ColorAide
    color1 = Color("srgb", [c1[0] / 255, c1[1] / 255, c1[2] / 255])
    color2 = Color("srgb", [c2[0] / 255, c2[1] / 255, c2[2] / 255])
    return color1.delta_e(color2, method="2000")


@functools.cache
def get_ansi16_palette() -> dict[str, RGB]:
    """Generate the ANSI 16-color palette.

    The 16 color palette consists of:
    - 8 standard colors
    - 8 bright colors

    Returns:
        Dictionary mapping ANSI color names to RGB values.
    """
    return {
        "black": (0, 0, 0),
        "red": (128, 0, 0),
        "green": (0, 128, 0),
        "yellow": (128, 128, 0),
        "blue": (0, 0, 128),
        "magenta": (128, 0, 128),
        "cyan": (0, 128, 128),
        "white": (192, 192, 192),
        "bright_black": (128, 128, 128),
        "bright_red": (255, 0, 0),
        "bright_green": (0, 255, 0),
        "bright_yellow": (255, 255, 0),
        "bright_blue": (0, 0, 255),
        "bright_magenta": (255, 0, 255),
        "bright_cyan": (0, 255, 255),
        "bright_white": (255, 255, 255),
    }


@functools.cache
def get_ansi256_palette() -> list[RGB]:
    """Generate the ANSI 256-color palette.

    The 256 color palette consists of:
    - Colors 0-15: Standard 16 colors
    - Colors 16-231: 216-color RGB cube (6x6x6)
    - Colors 232-255: 24 grayscale colors

    Returns:
        Tuple containing RGB values corresponding to the index in the ANSI 256-color palette.
    """
    palette: list[RGB] = [
        (0, 0, 0),
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        (0, 128, 128),
        (192, 192, 192),
        (128, 128, 128),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ]

    for r6 in range(6):
        for g6 in range(6):
            for b6 in range(6):
                # Convert the 0-5 range to 0-255 RGB values based on the color levels
                r, g, b = (0 if c == 0 else 55 + c * 40 for c in (r6, g6, b6))
                palette.append((r, g, b))

    for i in range(24):
        v = 8 + i * 10
        palette.append((v, v, v))

    return palette


@functools.cache
def rgb_to_ansi16(c: RGB) -> str:
    """Convert RGB values (0-255) to ANSI 16-color name.

    Args:
        r, g, b: RGB color values (0-255)

    Returns:
        ANSI 16-color name
    """
    palette = get_ansi16_palette()
    # Find the nearest color in the color palette using the color distance function.
    nearest_color = min(palette.keys(), key=lambda color: _color_distance(palette[color], c))
    return nearest_color


@functools.cache
def rgb_to_ansi256(c: RGB) -> int:
    """Convert RGB values (0-255) to ANSI 256-color code (0-255).

    Args:
        r, g, b: RGB color values (0-255)

    Returns:
        ANSI 256-color code (0-255)
    """
    palette = get_ansi256_palette()
    # Find the nearest color in the color palette using the color distance function.
    nearest_color = min(range(len(palette)), key=lambda i: _color_distance(palette[i], c))
    return nearest_color


def generate_distinct_colors(n_colors: int, lightness: float = 0.85, chroma: float = 0.1) -> list[RGB]:
    """Generate visually distinct colors.

    Args:
        n_colors: Number of colors to generate
        pastel_factor: 0.0 = vivid colors, 1.0 = very pastel colors

    Returns:
        List of RGB tuples with values 0-255
    """
    colors: list[RGB] = []

    for i in range(n_colors):
        # Distribute hues evenly around the color wheel
        hue = (i * 360) // n_colors

        # Create color in OkLCh space for perceptual uniformity
        color = Color("oklch", [lightness, chroma, hue])

        # Convert to sRGB and clamp to valid range
        rgb_color = color.convert("srgb").clip()
        rgb_coords = rgb_color.coords()

        rgb256 = RGB(round(c * 255) for c in rgb_coords)
        colors.append(rgb256)

    return colors
