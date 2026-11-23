import tkinter as tk
from typing import Callable

import customtkinter as ctk

from whisper_studio.models import ACTIVE_STATUSES, Job, JobStatus
from whisper_studio.ui import theme
from whisper_studio.ui.theme import Glyph
from whisper_studio.ui.widgets import Card, JobRow, secondary_button

TICK_MS = 500


class JobsPanel(Card):
    def __init__(
        self,
        master: tk.Misc,
        on_select: Callable[[str], None],
        on_remove: Callable[[str], None],
        on_open_folder: Callable[[str], None],
        on_view: Callable[[str], None],
        on_clear_finished: Callable[[], None],
    ) -> None:
        super().__init__(master, None)
        self._rows: dict[str, JobRow] = {}
        self._selected: str | None = None
        self._handlers = (on_select, on_remove, on_open_folder, on_view)
        self._pulse = False
        self.body.grid_configure(padx=theme.PAD_SMALL, pady=(theme.PAD_SMALL, theme.PAD_SMALL))
        self.body.grid_columnconfigure(0, weight=1)
        self.body.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.body, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=theme.PAD_SMALL, pady=(4, 6))
        header.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(header, text=Glyph.CHECK, font=theme.icon_font(16), text_color=theme.ACCENT).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkLabel(header, text="Trabajos", font=theme.font(15, True), text_color=theme.TEXT).grid(row=0, column=1, sticky="w")
        self.count_label = ctk.CTkLabel(header, text="", font=theme.font(11), text_color=theme.MUTED)
        self.count_label.grid(row=0, column=2, padx=(0, 12))
        secondary_button(header, "Limpiar terminados", on_clear_finished, height=30, width=140).grid(row=0, column=3)

        self.list_frame = ctk.CTkScrollableFrame(
            self.body, fg_color="transparent", scrollbar_button_color=theme.BORDER, height=120
        )
        self.list_frame.grid(row=1, column=0, sticky="nsew")
        self.list_frame.grid_columnconfigure(0, weight=1)
        self.empty_label = ctk.CTkLabel(
            self.list_frame, text="Aún no hay trabajos.\nArrastra tus videos o pega un enlace de YouTube.",
            font=theme.font(13), text_color=theme.MUTED, justify="center",
        )
        self.empty_label.grid(row=0, column=0, pady=60)
        self._refresh_count()
        self.after(TICK_MS, self._tick)

    def _refresh_count(self) -> None:
        total = len(self._rows)
        pending = sum(1 for r in self._rows.values() if r.job.status == JobStatus.PENDING)
        self.count_label.configure(text=f"{total} en total · {pending} pendientes" if total else "")
        if total:
            self.empty_label.grid_remove()
        else:
            self.empty_label.grid()

    def add_job(self, job: Job) -> None:
        if job.id in self._rows:
            self.update_job(job)
            return
        row = JobRow(self.list_frame, job, *self._handlers)
        row.grid(row=len(self._rows) + 1, column=0, sticky="ew", pady=(0, theme.PAD_SMALL), padx=(0, 4))
        self._rows[job.id] = row
        self._refresh_count()
        self.after(60, lambda: self.list_frame._parent_canvas.yview_moveto(1.0))

    def update_job(self, job: Job) -> None:
        row = self._rows.get(job.id)
        if row is not None:
            row.update_job(job)
            self._refresh_count()

    def remove_job(self, job_id: str) -> None:
        row = self._rows.pop(job_id, None)
        if row is None:
            return
        row.destroy()
        if self._selected == job_id:
            self._selected = None
        for index, remaining in enumerate(self._rows.values(), start=1):
            remaining.grid_configure(row=index)
        self._refresh_count()

    def select(self, job_id: str | None) -> None:
        self._selected = job_id
        for current_id, row in self._rows.items():
            row.set_selected(current_id == job_id)

    def jobs(self) -> list[Job]:
        return [row.job for row in self._rows.values()]

    def job(self, job_id: str) -> Job | None:
        row = self._rows.get(job_id)
        return row.job if row else None

    def _tick(self) -> None:
        self._pulse = not self._pulse
        for row in self._rows.values():
            if row.job.status in ACTIVE_STATUSES:
                row.tick(self._pulse)
        self.after(TICK_MS, self._tick)


class TranscriptPanel(Card):
    def __init__(
        self,
        master: tk.Misc,
        on_copy: Callable[[], None],
        on_open_folder: Callable[[], None],
        on_open_txt: Callable[[], None],
    ) -> None:
        super().__init__(master, None)
        self.body.grid_configure(padx=theme.PAD, pady=theme.PAD_MID)
        self.body.grid_columnconfigure(0, weight=1)
        self.body.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.body, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, theme.PAD_SMALL))
        header.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(header, text=Glyph.VIEW, font=theme.icon_font(16), text_color=theme.ACCENT).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkLabel(header, text="Transcripción", font=theme.font(15, True), text_color=theme.TEXT).grid(row=0, column=1, sticky="w")
        self.subtitle = ctk.CTkLabel(header, text="", font=theme.font(11), text_color=theme.MUTED)
        self.subtitle.grid(row=0, column=2, padx=(0, 12))
        self.copy_btn = secondary_button(header, "Copiar", on_copy, height=30, width=80)
        self.folder_btn = secondary_button(header, "Abrir carpeta", on_open_folder, height=30, width=110)
        self.txt_btn = secondary_button(header, "Abrir TXT", on_open_txt, height=30, width=96)
        for column, button in enumerate((self.copy_btn, self.folder_btn, self.txt_btn), start=3):
            button.grid(row=0, column=column, padx=(4, 0))

        self.textbox = ctk.CTkTextbox(
            self.body, font=theme.font(13), fg_color=theme.SURFACE_2, text_color=theme.TEXT,
            corner_radius=theme.RADIUS_BUTTON, border_width=0, wrap="word", scrollbar_button_color=theme.BORDER, height=100,
        )
        self.textbox.grid(row=1, column=0, sticky="nsew")
        self.textbox.configure(state="disabled")
        self._placeholder = False
        self.set_text(
            "", "Selecciona un trabajo o inicia el procesamiento para ver la transcripción aquí.", placeholder=True
        )

    def set_actions_enabled(self, has_text: bool, has_folder: bool) -> None:
        self.copy_btn.configure(state="normal" if has_text else "disabled")
        self.folder_btn.configure(state="normal" if has_folder else "disabled")
        self.txt_btn.configure(state="normal" if has_folder else "disabled")

    def set_text(self, subtitle: str, text: str, placeholder: bool = False) -> None:
        self._placeholder = placeholder
        self.subtitle.configure(text=subtitle)
        self.textbox.configure(state="normal")
        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", text)
        self.textbox.configure(state="disabled")

    def append(self, line: str) -> None:
        at_bottom = self.textbox.yview()[1] >= 0.98
        self.textbox.configure(state="normal")
        self.textbox.insert("end", line + "\n")
        self.textbox.configure(state="disabled")
        if at_bottom:
            self.textbox.see("end")

    def get_text(self) -> str:
        if self._placeholder:
            return ""
        return self.textbox.get("1.0", "end").strip()
