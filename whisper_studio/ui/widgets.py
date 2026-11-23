import tkinter as tk
from typing import Callable

import customtkinter as ctk

from whisper_studio.models import ACTIVE_STATUSES, Job, JobKind, JobStatus
from whisper_studio.ui import theme
from whisper_studio.ui.helpers import ellipsize, format_elapsed, status_color
from whisper_studio.ui.theme import Glyph

TITLE_LIMIT = 58


def primary_button(master: tk.Misc, text: str, command: Callable[[], None], **kwargs) -> ctk.CTkButton:
    kwargs.setdefault("height", 38)
    return ctk.CTkButton(
        master, text=text, command=command, font=theme.font(13, True),
        corner_radius=theme.RADIUS_BUTTON, fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
        text_color=theme.TEXT, text_color_disabled=theme.MUTED, **kwargs,
    )


def secondary_button(master: tk.Misc, text: str, command: Callable[[], None], **kwargs) -> ctk.CTkButton:
    kwargs.setdefault("height", 38)
    return ctk.CTkButton(
        master, text=text, command=command, font=theme.font(13),
        corner_radius=theme.RADIUS_BUTTON, fg_color=theme.SURFACE_2, hover_color=theme.BORDER,
        border_width=1, border_color=theme.BORDER, text_color=theme.TEXT,
        text_color_disabled=theme.MUTED, **kwargs,
    )


def danger_button(master: tk.Misc, text: str, command: Callable[[], None], **kwargs) -> ctk.CTkButton:
    kwargs.setdefault("height", 38)
    return ctk.CTkButton(
        master, text=text, command=command, font=theme.font(13),
        corner_radius=theme.RADIUS_BUTTON, fg_color=theme.SURFACE_2, hover_color=theme.DANGER_HOVER,
        border_width=1, border_color=theme.ERROR, text_color=theme.ERROR,
        text_color_disabled=theme.MUTED, **kwargs,
    )


def icon_button(
    master: tk.Misc, glyph: str, command: Callable[[], None], tooltip: str = "", size: int = 32,
    color: str = theme.TEXT, hover: str = theme.BORDER,
) -> ctk.CTkButton:
    button = ctk.CTkButton(
        master, text=glyph, command=command, width=size, height=size, font=theme.icon_font(15),
        corner_radius=theme.RADIUS_BUTTON - 2, fg_color="transparent", hover_color=hover,
        text_color=color, text_color_disabled=theme.DISABLED,
    )
    if tooltip:
        Tooltip(button, tooltip)
    return button


def section_title(master: tk.Misc, text: str, glyph: str | None = None) -> ctk.CTkFrame:
    row = ctk.CTkFrame(master, fg_color="transparent")
    column = 0
    if glyph:
        ctk.CTkLabel(row, text=glyph, font=theme.icon_font(16), text_color=theme.ACCENT).grid(
            row=0, column=0, padx=(0, theme.PAD_SMALL)
        )
        column = 1
    ctk.CTkLabel(row, text=text, font=theme.font(15, True), text_color=theme.TEXT).grid(
        row=0, column=column, sticky="w"
    )
    return row


class Card(ctk.CTkFrame):
    """Rounded surface with an optional title; children go in `self.body`."""

    def __init__(self, master: tk.Misc, title: str | None = None, glyph: str | None = None) -> None:
        super().__init__(
            master, fg_color=theme.SURFACE, corner_radius=theme.RADIUS_CARD,
            border_width=1, border_color=theme.BORDER,
        )
        self.grid_columnconfigure(0, weight=1)
        row = 0
        if title:
            section_title(self, title, glyph).grid(
                row=0, column=0, sticky="w", padx=theme.PAD, pady=(theme.PAD_MID, 0)
            )
            row = 1
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=row, column=0, sticky="nsew", padx=theme.PAD, pady=(theme.PAD_MID, theme.PAD))
        self.grid_rowconfigure(row, weight=1)


class Pill(ctk.CTkLabel):
    def __init__(self, master: tk.Misc, text: str = "", color: str = theme.MUTED, base: str = theme.SURFACE_2) -> None:
        super().__init__(
            master, text=text, font=theme.font(11, True), corner_radius=10, height=22,
            padx=10, text_color=color, fg_color=theme.tint(color, base=base),
        )
        self._base = base
        self._color = color
        self._text = text

    def set(self, text: str, color: str) -> None:
        if (text, color) == (self._text, self._color):
            return
        self._text, self._color = text, color
        self.configure(text=text, text_color=color, fg_color=theme.tint(color, base=self._base))

    def set_text_color(self, color: str) -> None:
        self.configure(text_color=color)


class Tooltip:
    def __init__(self, widget: tk.Misc, text: str | Callable[[], str]) -> None:
        self._widget = widget
        self._text = text
        self._window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def set_text(self, text: str | Callable[[], str]) -> None:
        self._text = text

    def _show(self, _event: tk.Event | None = None) -> None:
        text = self._text() if callable(self._text) else self._text
        if not text or self._window is not None:
            return
        window = tk.Toplevel(self._widget)
        window.wm_overrideredirect(True)
        window.attributes("-topmost", True)
        window.configure(bg=theme.BORDER)
        label = tk.Label(
            window, text=text, bg=theme.SURFACE_2, fg=theme.TEXT, font=(theme.FAMILY, 10),
            padx=10, pady=6, justify="left", wraplength=360,
        )
        label.pack(padx=1, pady=1)
        x = self._widget.winfo_rootx()
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 6
        window.geometry(f"+{x}+{y}")
        self._window = window

    def _hide(self, _event: tk.Event | None = None) -> None:
        if self._window is not None:
            self._window.destroy()
            self._window = None


class ToastHost:
    """Bottom-right overlay that stacks short-lived messages on top of the window."""

    _LEVEL_COLORS = {"info": theme.BLUE, "success": theme.SUCCESS, "warning": theme.WARNING, "error": theme.ERROR}
    _MAX_VISIBLE = 4

    def __init__(self, root: tk.Misc, bottom_offset: int = 84, duration_ms: int = 4000) -> None:
        self._root = root
        self._bottom = bottom_offset
        self._duration = duration_ms
        self._toasts: list[ctk.CTkFrame] = []

    def show(self, message: str, level: str = "info", duration_ms: int | None = None) -> None:
        color = self._LEVEL_COLORS.get(level, theme.BLUE)
        while len(self._toasts) >= self._MAX_VISIBLE:
            self._dismiss(self._toasts[0])
        toast = ctk.CTkFrame(
            self._root, fg_color=theme.SURFACE_2, corner_radius=theme.RADIUS_BUTTON,
            border_width=1, border_color=color,
        )
        ctk.CTkFrame(toast, width=4, height=1, fg_color=color, corner_radius=2).pack(side="left", fill="y", padx=(8, 0), pady=8)
        ctk.CTkLabel(
            toast, text=message, font=theme.font(13), text_color=theme.TEXT, wraplength=340, justify="left",
        ).pack(side="left", padx=(10, 14), pady=10)
        self._toasts.append(toast)
        self._layout()
        toast.after(duration_ms or self._duration, lambda: self._dismiss(toast))

    def _layout(self) -> None:
        self._root.update_idletasks()
        offset = self._bottom
        for toast in reversed(self._toasts):
            toast.place(relx=1.0, rely=1.0, anchor="se", x=-theme.PAD, y=-offset)
            toast.lift()
            offset += toast.winfo_reqheight() + theme.PAD_SMALL

    def _dismiss(self, toast: ctk.CTkFrame) -> None:
        if toast not in self._toasts:
            return
        self._toasts.remove(toast)
        toast.destroy()
        self._layout()


class JobRow(ctk.CTkFrame):
    """One queue entry; `update_job` mutates widgets in place and skips unchanged values."""

    def __init__(
        self,
        master: tk.Misc,
        job: Job,
        on_select: Callable[[str], None],
        on_remove: Callable[[str], None],
        on_open_folder: Callable[[str], None],
        on_view: Callable[[str], None],
    ) -> None:
        super().__init__(
            master, fg_color=theme.SURFACE_2, corner_radius=theme.RADIUS_CARD - 2,
            border_width=2, border_color=theme.SURFACE_2,
        )
        self.job_id = job.id
        self.job = job
        self._selected = False
        self._signature: tuple | None = None
        self._bar_mode = "determinate"
        self._on_select = on_select
        self.grid_columnconfigure(1, weight=1)

        glyph = Glyph.VIDEO if job.kind == JobKind.YOUTUBE else Glyph.FILE
        self._icon = ctk.CTkLabel(
            self, text=glyph, font=theme.icon_font(18), text_color=theme.ACCENT, width=36, height=36,
            fg_color=theme.tint(theme.ACCENT, 0.14), corner_radius=10,
        )
        self._icon.grid(row=0, column=0, rowspan=3, padx=(12, 10), pady=12, sticky="n")

        self._title = ctk.CTkLabel(self, text="", font=theme.font(13, True), text_color=theme.TEXT, anchor="w")
        self._title.grid(row=0, column=1, sticky="w", pady=(10, 0))
        self._pill = Pill(self)
        self._pill.grid(row=0, column=2, sticky="e", padx=(8, 8), pady=(10, 0))

        self._stage = ctk.CTkLabel(self, text="", font=theme.font(11), text_color=theme.MUTED, anchor="w")
        self._stage.grid(row=1, column=1, sticky="w")
        self._elapsed = ctk.CTkLabel(self, text="", font=theme.font(11), text_color=theme.MUTED, anchor="e")
        self._elapsed.grid(row=1, column=2, sticky="e", padx=(8, 8))

        self._bar_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._bar_frame.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(0, 12), pady=(4, 0))
        self._bar_frame.grid_columnconfigure(0, weight=1)
        self._bar = ctk.CTkProgressBar(
            self._bar_frame, height=6, corner_radius=3, fg_color=theme.BORDER, progress_color=theme.ACCENT,
        )
        self._bar.set(0)
        self._bar.grid(row=0, column=0, sticky="ew")
        self._pct = ctk.CTkLabel(self._bar_frame, text="", font=theme.font(11), text_color=theme.MUTED, width=56, anchor="e")
        self._pct.grid(row=0, column=1, padx=(8, 0))

        self._error = ctk.CTkLabel(
            self, text="", font=theme.font(11), text_color=theme.ERROR, anchor="w", justify="left", wraplength=520,
        )
        self._error.grid(row=3, column=1, columnspan=2, sticky="w", pady=(2, 0), padx=(0, 12))

        actions = ctk.CTkFrame(self, fg_color="transparent", height=1)
        actions.grid(row=4, column=1, columnspan=2, sticky="e", padx=(0, 6), pady=(2, 6))
        self._view_btn = icon_button(actions, Glyph.VIEW, lambda: on_view(self.job_id), "Ver transcripción", 28)
        self._folder_btn = icon_button(actions, Glyph.FOLDER_OPEN, lambda: on_open_folder(self.job_id), "Abrir carpeta", 28)
        self._remove_btn = icon_button(
            actions, Glyph.DELETE, lambda: on_remove(self.job_id), "Quitar de la lista", 28,
            color=theme.ERROR, hover=theme.tint(theme.ERROR, 0.25),
        )
        self._buttons = (self._view_btn, self._folder_btn, self._remove_btn)

        self._bind_select(self)
        self.update_job(job)

    def _bind_select(self, widget: tk.Misc) -> None:
        if isinstance(widget, ctk.CTkButton):
            return
        widget.bind("<Button-1>", lambda _e: self._on_select(self.job_id), add="+")
        for child in widget.winfo_children():
            self._bind_select(child)

    def set_selected(self, selected: bool) -> None:
        if selected == self._selected:
            return
        self._selected = selected
        self.configure(border_color=theme.ACCENT if selected else theme.SURFACE_2)

    def update_job(self, job: Job) -> None:
        self.job = job
        status = job.status
        active = status in ACTIVE_STATUSES
        has_output = job.output_dir is not None and job.output_dir.exists()
        determinate_pct = None if job.progress is None else int(job.progress * 100)
        signature = (
            job.title, status, job.stage_text, job.speed_text, job.error, job.progress_is_estimate,
            determinate_pct, has_output, bool(job.transcript_text),
        )
        if signature == self._signature:
            return
        self._signature = signature

        self._title.configure(text=ellipsize(job.title, TITLE_LIMIT))
        color = status_color(status)
        self._pill.set(str(status), color)
        stage = job.stage_text
        if job.speed_text and active:
            stage = f"{stage} · {job.speed_text}" if stage else job.speed_text
        self._stage.configure(text=ellipsize(stage, 90))

        self._update_bar(job, color, active)
        self._error.configure(text=job.error if status == JobStatus.ERROR else "")
        if status == JobStatus.ERROR and job.error:
            self._error.grid()
        else:
            self._error.grid_remove()
        self._update_actions(job, active, has_output)
        self.tick(False)

    def _update_bar(self, job: Job, color: str, active: bool) -> None:
        show_bar = active or job.status == JobStatus.COMPLETED
        if not show_bar:
            self._stop_indeterminate()
            self._bar_frame.grid_remove()
            return
        self._bar_frame.grid()
        self._bar.configure(progress_color=color)
        if job.progress is None and active:
            self._pct.configure(text="")
            if self._bar_mode != "indeterminate":
                self._bar.configure(mode="indeterminate")
                self._bar_mode = "indeterminate"
                self._bar.start()
            return
        self._stop_indeterminate()
        value = 1.0 if job.progress is None else job.progress
        self._bar.set(value)
        pct = int(value * 100)
        self._pct.configure(text=f"≈ {pct}%" if job.progress_is_estimate else f"{pct}%")

    def _stop_indeterminate(self) -> None:
        if self._bar_mode == "indeterminate":
            self._bar.stop()
            self._bar.configure(mode="determinate")
            self._bar_mode = "determinate"

    def _update_actions(self, job: Job, active: bool, has_output: bool) -> None:
        for button in self._buttons:
            button.pack_forget()
        if job.status == JobStatus.COMPLETED and (job.transcript_text or has_output) and job.mode != "download_only":
            self._view_btn.pack(side="left", padx=2)
        if has_output:
            self._folder_btn.pack(side="left", padx=2)
        if not active:
            self._remove_btn.pack(side="left", padx=2)

    def tick(self, pulse: bool) -> None:
        """Called by the panel's timer: refresh the live elapsed time and pulse the active pill."""
        job = self.job
        if job.status in ACTIVE_STATUSES or job.started_at is not None:
            text = format_elapsed(job.elapsed)
            if text != self._elapsed.cget("text"):
                self._elapsed.configure(text=text)
        else:
            self._elapsed.configure(text="")
        if job.status == JobStatus.TRANSCRIBING:
            self._pill.set_text_color(theme.TEXT if pulse else theme.ACCENT)
            self._pill.configure(text="● Transcribiendo" if pulse else "○ Transcribiendo")

    def destroy(self) -> None:
        self._stop_indeterminate()
        super().destroy()
