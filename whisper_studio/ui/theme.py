from functools import cache
from tkinter import font as tkfont

import customtkinter as ctk

BG = "#0E0F1A"
SURFACE = "#161827"
SURFACE_2 = "#1E2135"
BORDER = "#2A2E48"
TEXT = "#E8EAF6"
MUTED = "#8B90B3"
ACCENT = "#7C5CFF"
ACCENT_HOVER = "#6A4BEB"
BLUE = "#3B82F6"
BLUE_HOVER = "#2F6FDB"
SUCCESS = "#22C55E"
WARNING = "#F59E0B"
ERROR = "#EF4444"
DANGER_HOVER = "#C23B3B"
DISABLED = "#3A3E5C"

RADIUS_CARD = 14
RADIUS_BUTTON = 10
PAD = 16
PAD_MID = 12
PAD_SMALL = 8

FAMILY = "Segoe UI"
FAMILY_SEMIBOLD = "Segoe UI Semibold"
MONO = "Cascadia Mono"

_ICON_FAMILIES = ("Segoe Fluent Icons", "Segoe MDL2 Assets")


def tint(color: str, amount: float = 0.18, base: str = SURFACE_2) -> str:
    """Blend `color` over `base` so pills get a translucent look without alpha support."""
    def parts(value: str) -> tuple[int, int, int]:
        return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)

    fg, bg = parts(color), parts(base)
    mixed = tuple(round(b + (f - b) * amount) for f, b in zip(fg, bg))
    return "#{:02X}{:02X}{:02X}".format(*mixed)


@cache
def font(size: int, semibold: bool = False) -> ctk.CTkFont:
    return ctk.CTkFont(family=FAMILY_SEMIBOLD if semibold else FAMILY, size=size)


@cache
def mono_font(size: int = 13) -> ctk.CTkFont:
    families = set(tkfont.families())
    family = MONO if MONO in families else "Consolas"
    return ctk.CTkFont(family=family, size=size)


@cache
def icon_family() -> str:
    families = set(tkfont.families())
    for name in _ICON_FAMILIES:
        if name in families:
            return name
    return FAMILY


@cache
def icon_font(size: int = 15) -> ctk.CTkFont:
    return ctk.CTkFont(family=icon_family(), size=size)


class Glyph:
    PASTE = ""
    ADD = ""
    FOLDER = ""
    FOLDER_OPEN = ""
    DELETE = ""
    PLAY = ""
    STOP = ""
    COPY = ""
    SETTINGS = ""
    DIAGNOSTIC = ""
    VIEW = ""
    OPEN_FILE = ""
    VIDEO = ""
    FILE = ""
    DOWNLOAD = ""
    CLEAR = ""
    CHECK = ""
    LINK = ""
