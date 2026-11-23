from typing import Any, Callable

import customtkinter as ctk

from whisper_studio.config import FORMATS, MODELS, AppConfig
from whisper_studio.ui import theme
from whisper_studio.ui.helpers import ellipsize
from whisper_studio.ui.theme import Glyph
from whisper_studio.ui.widgets import Card, Tooltip, icon_button, secondary_button

LANGUAGE_LABELS = {"es": "Español", "en": "Inglés", "auto": "Automático"}
DEVICE_LABELS = {"auto": "Automático", "cuda": "GPU (CUDA)", "cpu": "CPU"}
POLICY_LABELS = {"skip": "Omitir", "regenerate": "Regenerar"}
FORMAT_LABELS = {"txt": "TXT", "srt": "SRT", "vtt": "VTT", "md": "Markdown", "json": "JSON"}

ConfigChange = Callable[[str, Any], None]


def reverse(mapping: dict[str, str]) -> dict[str, str]:
    return {label: key for key, label in mapping.items()}


def menu(master: Any, values: list[str], command: Callable[[str], None], width: int = 150) -> ctk.CTkOptionMenu:
    return ctk.CTkOptionMenu(
        master, values=values, command=command, width=width, height=32, font=theme.font(12),
        dropdown_font=theme.font(12), corner_radius=theme.RADIUS_BUTTON - 2, fg_color=theme.SURFACE_2,
        button_color=theme.BORDER, button_hover_color=theme.ACCENT, text_color=theme.TEXT,
        dropdown_fg_color=theme.SURFACE_2, dropdown_hover_color=theme.ACCENT, dropdown_text_color=theme.TEXT,
    )


def menu_cell(
    master: Any, text: str, values: list[str], command: Callable[[str], None], row: int, column: int
) -> ctk.CTkOptionMenu:
    cell = ctk.CTkFrame(master, fg_color="transparent")
    cell.grid(row=row, column=column, sticky="ew", padx=(0 if column == 0 else 6, 6 if column == 0 else 0), pady=(0, 8))
    cell.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(cell, text=text, font=theme.font(11), text_color=theme.MUTED, anchor="w").grid(row=0, column=0, sticky="w")
    option = menu(cell, values, command)
    option.grid(row=1, column=0, sticky="ew")
    return option


def check(master: Any, text: str, command: Callable[[], None]) -> ctk.CTkCheckBox:
    return ctk.CTkCheckBox(
        master, text=text, command=command, font=theme.font(12), text_color=theme.TEXT, checkbox_width=18,
        checkbox_height=18, corner_radius=5, border_width=2, border_color=theme.BORDER, fg_color=theme.ACCENT,
        hover_color=theme.ACCENT_HOVER, width=20,
    )


def switch(master: Any, text: str, command: Callable[[], None]) -> ctk.CTkSwitch:
    return ctk.CTkSwitch(
        master, text=text, command=command, font=theme.font(12), text_color=theme.TEXT, progress_color=theme.ACCENT,
        button_color=theme.TEXT, button_hover_color=theme.TEXT, fg_color=theme.BORDER,
    )


def segmented(master: Any, values: list[str], command: Callable[[str], None]) -> ctk.CTkSegmentedButton:
    return ctk.CTkSegmentedButton(
        master, values=values, command=command, font=theme.font(12), corner_radius=theme.RADIUS_BUTTON,
        fg_color=theme.SURFACE_2, selected_color=theme.ACCENT, selected_hover_color=theme.ACCENT_HOVER,
        unselected_color=theme.SURFACE_2, unselected_hover_color=theme.BORDER, text_color=theme.TEXT, height=32,
    )


class QuickSettingsCard(Card):
    def __init__(
        self,
        master: ctk.CTkBaseClass,
        on_change: ConfigChange,
        on_pick_output: Callable[[], None],
        on_open_output: Callable[[], None],
    ) -> None:
        super().__init__(master, "Configuración rápida", Glyph.SETTINGS)
        self._on_change = on_change
        self._language_by_label = reverse(LANGUAGE_LABELS)
        self._device_by_label = reverse(DEVICE_LABELS)
        self._policy_by_label = reverse(POLICY_LABELS)
        body = self.body
        body.grid_columnconfigure((0, 1), weight=1)

        self.model_menu = menu_cell(body, "Modelo", list(MODELS), lambda v: self._on_change("model", v), 0, 0)
        self.language_menu = menu_cell(
            body, "Idioma", list(LANGUAGE_LABELS.values()),
            lambda v: self._on_change("language", self._language_by_label[v]), 0, 1,
        )
        self.device_menu = menu_cell(
            body, "Dispositivo", list(DEVICE_LABELS.values()),
            lambda v: self._on_change("device", self._device_by_label[v]), 1, 0,
        )

        formats = ctk.CTkFrame(body, fg_color="transparent")
        formats.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 8))
        ctk.CTkLabel(formats, text="Formatos de salida", font=theme.font(11), text_color=theme.MUTED).grid(
            row=0, column=0, columnspan=5, sticky="w"
        )
        self._format_boxes: dict[str, ctk.CTkCheckBox] = {}
        for index, fmt in enumerate(FORMATS):
            box = check(formats, FORMAT_LABELS[fmt], self._formats_changed)
            box.grid(row=1, column=index, padx=(0, 10), pady=(4, 0), sticky="w")
            self._format_boxes[fmt] = box

        output = ctk.CTkFrame(body, fg_color="transparent")
        output.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        output.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(output, text="Carpeta de salida", font=theme.font(11), text_color=theme.MUTED).grid(
            row=0, column=0, sticky="w", columnspan=3
        )
        self.output_label = ctk.CTkLabel(
            output, text="", font=theme.font(12), text_color=theme.TEXT, anchor="w", fg_color=theme.SURFACE_2,
            corner_radius=theme.RADIUS_BUTTON - 2, height=32,
        )
        self.output_label.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        self._output_tooltip = Tooltip(self.output_label, "")
        secondary_button(output, "Cambiar…", on_pick_output, width=92, height=32).grid(row=1, column=1)
        icon_button(output, Glyph.FOLDER_OPEN, on_open_output, "Abrir carpeta de salida").grid(row=1, column=2, padx=(4, 0))

        self.keep_media = switch(body, "Conservar archivos multimedia", lambda: self._bool_changed("keep_media", self.keep_media))
        self.word_ts = switch(body, "Marcas de tiempo por palabra", lambda: self._bool_changed("word_timestamps", self.word_ts))
        self.vad = switch(body, "Filtro VAD (omite silencios)", lambda: self._bool_changed("vad_filter", self.vad))
        for row, widget in enumerate((self.keep_media, self.word_ts, self.vad), start=4):
            widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6))

        ctk.CTkLabel(body, text="Si ya existe una transcripción", font=theme.font(11), text_color=theme.MUTED).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        self.policy = segmented(
            body, list(POLICY_LABELS.values()), lambda v: self._on_change("existing_policy", self._policy_by_label[v])
        )
        self.policy.grid(row=8, column=0, columnspan=2, sticky="ew")

    def _formats_changed(self) -> None:
        chosen = [f for f in FORMATS if self._format_boxes[f].get()]
        if not chosen:
            chosen = ["txt"]
            self._format_boxes["txt"].select()
        self._on_change("formats", chosen)

    def _bool_changed(self, name: str, widget: ctk.CTkSwitch) -> None:
        self._on_change(name, bool(widget.get()))

    def apply_config(self, cfg: AppConfig) -> None:
        self.model_menu.set(cfg.model)
        self.language_menu.set(LANGUAGE_LABELS[cfg.language])
        self.device_menu.set(DEVICE_LABELS[cfg.device])
        self.policy.set(POLICY_LABELS[cfg.existing_policy])
        for fmt, box in self._format_boxes.items():
            box.select() if fmt in cfg.formats else box.deselect()
        for widget, value in ((self.keep_media, cfg.keep_media), (self.word_ts, cfg.word_timestamps), (self.vad, cfg.vad_filter)):
            widget.select() if value else widget.deselect()
        self.output_label.configure(text="  " + ellipsize(cfg.output_dir, 42))
        self._output_tooltip.set_text(cfg.output_dir)
