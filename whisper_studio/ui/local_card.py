from pathlib import Path
from tkinter import filedialog
from typing import Callable

import customtkinter as ctk

from whisper_studio.media import MEDIA_EXTENSIONS
from whisper_studio.ui import theme
from whisper_studio.ui.theme import Glyph
from whisper_studio.ui.widgets import Card, secondary_button


def media_filetypes() -> list[tuple[str, str]]:
    patterns = " ".join(f"*{ext}" for ext in sorted(MEDIA_EXTENSIONS))
    return [("Audio y video", patterns), ("Todos los archivos", "*.*")]


class LocalFilesCard(Card):
    def __init__(
        self,
        master: ctk.CTkBaseClass,
        on_paths: Callable[[list[Path]], None],
    ) -> None:
        super().__init__(master, "Tus videos", Glyph.FILE)
        self._on_paths = on_paths
        body = self.body
        body.grid_columnconfigure((0, 1), weight=1)
        secondary_button(body, "Elegir videos", self._pick_files).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        secondary_button(body, "Elegir carpeta", self._pick_folder).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.drop_zone = ctk.CTkLabel(
            body, text="⬇   Arrastra tus videos, audios o carpetas aquí", font=theme.font(13),
            text_color=theme.MUTED, height=64, corner_radius=theme.RADIUS_BUTTON, fg_color=theme.SURFACE_2,
        )
        self.drop_zone.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(theme.PAD_MID, 0))
        self.set_drop_highlight(False)

    def set_drop_highlight(self, active: bool) -> None:
        self.drop_zone.configure(
            fg_color=theme.tint(theme.ACCENT, 0.28) if active else theme.SURFACE_2,
            text_color=theme.TEXT if active else theme.MUTED,
        )

    def _pick_files(self) -> None:
        chosen = filedialog.askopenfilenames(
            parent=self.winfo_toplevel(), title="Seleccionar archivos de audio o video", filetypes=media_filetypes(),
        )
        if chosen:
            self._on_paths([Path(p) for p in chosen])

    def _pick_folder(self) -> None:
        chosen = filedialog.askdirectory(parent=self.winfo_toplevel(), title="Seleccionar carpeta con audio o video")
        if chosen:
            self._on_paths([Path(chosen)])
