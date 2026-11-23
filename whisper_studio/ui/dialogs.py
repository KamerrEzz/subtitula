from typing import Any, Callable

import customtkinter as ctk

from whisper_studio import APP_NAME, __version__ as app_version
from whisper_studio.config import VIDEO_QUALITIES, AppConfig
from whisper_studio.diagnostics import CheckResult, run_diagnostics
from whisper_studio.paths import data_dir
from whisper_studio.transcriber import WhisperEngine
from whisper_studio.ui import theme
from whisper_studio.ui.download_card import MODE_LABELS, QUALITY_LABELS
from whisper_studio.ui.quick_settings import POLICY_LABELS, menu, reverse, segmented, switch
from whisper_studio.ui.services import UiBridge, open_path
from whisper_studio.ui.widgets import primary_button, secondary_button



class _Dialog(ctk.CTkToplevel):
    def __init__(self, master: Any, title: str, width: int, height: int) -> None:
        super().__init__(master, fg_color=theme.BG)
        self.title(title)
        self.geometry(f"{width}x{height}")
        self.minsize(width - 100, min(height, 420))
        self.transient(master)
        self.after(150, self._modalize)

    def _modalize(self) -> None:
        try:
            self.grab_set()
            self.focus_set()
        except Exception:
            pass


class SettingsDialog(_Dialog):
    def __init__(
        self,
        master: Any,
        cfg: AppConfig,
        on_change: Callable[[str, Any], None],
        on_pick_output: Callable[[], None],
    ) -> None:
        super().__init__(master, "Configuración", 560, 640)
        self._on_change = on_change
        self._mode_by_label = reverse(MODE_LABELS)
        self._quality_by_label = reverse(QUALITY_LABELS)
        self._policy_by_label = reverse(POLICY_LABELS)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        body = ctk.CTkScrollableFrame(self, fg_color="transparent", scrollbar_button_color=theme.BORDER)
        body.grid(row=0, column=0, sticky="nsew", padx=theme.PAD, pady=(theme.PAD, 0))
        body.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(body, text="Configuración", font=theme.font(22, True), text_color=theme.TEXT).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, theme.PAD)
        )

        self.output_label = ctk.CTkLabel(body, text="", font=theme.font(12), text_color=theme.TEXT, anchor="w", justify="left", wraplength=240)
        self._row(body, 1, "Carpeta de salida", self.output_label)
        secondary_button(body, "Cambiar…", on_pick_output, width=100, height=32).grid(row=1, column=2, padx=(8, 0))

        self.policy = segmented(body, list(POLICY_LABELS.values()), lambda v: on_change("existing_policy", self._policy_by_label[v]))
        self._row(body, 2, "Si ya existe una transcripción", self.policy)
        self.quality = menu(body, [QUALITY_LABELS[q] for q in VIDEO_QUALITIES], lambda v: on_change("video_quality", self._quality_by_label[v]), 140)
        self._row(body, 3, "Calidad de video", self.quality)
        self.mode = menu(body, list(MODE_LABELS.values()), lambda v: on_change("download_mode", self._mode_by_label[v]), 240)
        self._row(body, 4, "Modo de descarga predeterminado", self.mode)

        self.keep_media = switch(body, "Conservar archivos multimedia", lambda: on_change("keep_media", bool(self.keep_media.get())))
        self.word_ts = switch(body, "Marcas de tiempo por palabra", lambda: on_change("word_timestamps", bool(self.word_ts.get())))
        self.vad = switch(body, "Filtro VAD (omite silencios)", lambda: on_change("vad_filter", bool(self.vad.get())))
        for row, widget in enumerate((self.keep_media, self.word_ts, self.vad), start=5):
            widget.grid(row=row, column=0, columnspan=3, sticky="w", pady=(theme.PAD_SMALL, 0))

        folders = ctk.CTkFrame(body, fg_color="transparent")
        folders.grid(row=8, column=0, columnspan=3, sticky="w", pady=(theme.PAD, 0))
        secondary_button(folders, "Abrir carpeta de logs", lambda: self._open(data_dir() / "logs")).pack(side="left", padx=(0, 8))
        secondary_button(folders, "Abrir carpeta de modelos", lambda: self._open(data_dir() / "models")).pack(side="left")

        ctk.CTkLabel(body, text=f"{APP_NAME} · versión {app_version}", font=theme.font(11), text_color=theme.MUTED).grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(theme.PAD, 0)
        )
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=1, column=0, sticky="e", padx=theme.PAD, pady=theme.PAD)
        primary_button(footer, "Cerrar", self.destroy, width=100).pack()
        self.apply_config(cfg)

    @staticmethod
    def _row(body: ctk.CTkFrame, row: int, label: str, widget: Any) -> None:
        ctk.CTkLabel(body, text=label, font=theme.font(12), text_color=theme.MUTED, anchor="w").grid(
            row=row, column=0, sticky="w", pady=(theme.PAD_SMALL, 0), padx=(0, theme.PAD)
        )
        widget.grid(row=row, column=1, sticky="w", pady=(theme.PAD_SMALL, 0))

    @staticmethod
    def _open(path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        open_path(path)

    def apply_config(self, cfg: AppConfig) -> None:
        self.output_label.configure(text=cfg.output_dir)
        self.policy.set(POLICY_LABELS[cfg.existing_policy])
        self.quality.set(QUALITY_LABELS[cfg.video_quality])
        self.mode.set(MODE_LABELS[cfg.download_mode])
        for widget, value in ((self.keep_media, cfg.keep_media), (self.word_ts, cfg.word_timestamps), (self.vad, cfg.vad_filter)):
            widget.select() if value else widget.deselect()


_ICONS = {True: ("✔", theme.SUCCESS), None: ("⚠", theme.WARNING), False: ("✖", theme.ERROR)}


def format_report(results: list[CheckResult]) -> str:
    lines = [f"{APP_NAME} {app_version} - diagnóstico"]
    for r in results:
        mark = {True: "OK", None: "AVISO", False: "ERROR"}[r.ok]
        lines.append(f"[{mark}] {r.name}: {r.detail}")
        if r.fix and r.ok is not True:
            lines.append(f"      Solución: {r.fix}")
    return "\n".join(lines)


class DiagnosticsDialog(_Dialog):
    def __init__(
        self,
        master: Any,
        bridge: UiBridge,
        get_config: Callable[[], AppConfig],
        queue_is_running: Callable[[], bool],
        shared_engine: WhisperEngine,
        on_engine_replaced: Callable[[], None],
        notify: Callable[[str, str], None],
    ) -> None:
        super().__init__(master, "Diagnóstico", 720, 640)
        self._bridge = bridge
        self._get_config = get_config
        self._queue_is_running = queue_is_running
        self._engine = shared_engine
        self._on_engine_replaced = on_engine_replaced
        self._notify = notify
        self._results: list[CheckResult] = []
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=theme.PAD, pady=(theme.PAD, 0))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="Diagnóstico del sistema", font=theme.font(22, True), text_color=theme.TEXT).grid(row=0, column=0, sticky="w")
        self.status = ctk.CTkLabel(header, text="", font=theme.font(12), text_color=theme.MUTED)
        self.status.grid(row=1, column=0, sticky="w")
        self.spinner = ctk.CTkProgressBar(header, mode="indeterminate", height=4, progress_color=theme.ACCENT, fg_color=theme.BORDER)
        self.spinner.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self.list_frame = ctk.CTkScrollableFrame(self, fg_color="transparent", scrollbar_button_color=theme.BORDER)
        self.list_frame.grid(row=1, column=0, sticky="nsew", padx=theme.PAD, pady=theme.PAD_MID)
        self.list_frame.grid_columnconfigure(0, weight=1)

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=theme.PAD, pady=(0, theme.PAD))
        self.deep_btn = secondary_button(footer, "Prueba profunda de CUDA", lambda: self._run(True))
        self.deep_btn.pack(side="left")
        self.copy_btn = secondary_button(footer, "Copiar informe", self._copy)
        self.copy_btn.pack(side="left", padx=theme.PAD_SMALL)
        primary_button(footer, "Cerrar", self.destroy, width=100).pack(side="right")
        self._run(False)

    def _run(self, deep: bool) -> None:
        if deep and self._queue_is_running():
            self._notify("La prueba profunda de CUDA necesita la cola inactiva: espera a que termine el procesamiento.", "warning")
            return
        cfg = self._get_config()
        self.deep_btn.configure(state="disabled")
        self.status.configure(
            text="Ejecutando prueba profunda (puede descargar el modelo tiny la primera vez)…" if deep else "Ejecutando comprobaciones…"
        )
        self.spinner.grid()
        self.spinner.start()
        # With the queue idle the shared engine can be reused so two models never sit in 4 GB of VRAM;
        # the test replaces its loaded model, so the main window is told to refresh its engine state.
        engine = self._engine if deep else None
        self._bridge.run_async(
            lambda: run_diagnostics(cfg, deep_cuda=deep, engine=engine),
            lambda results, error: self._done(deep, results, error),
        )

    def _done(self, deep: bool, results: list[CheckResult] | None, error: BaseException | None) -> None:
        if deep:
            self._on_engine_replaced()
        if not self.winfo_exists():
            return
        self.spinner.stop()
        self.spinner.grid_remove()
        self.deep_btn.configure(state="normal")
        if error is not None or results is None:
            self.status.configure(text="No se pudo completar el diagnóstico.", text_color=theme.ERROR)
            return
        self._results = results
        failed = sum(1 for r in results if r.ok is False)
        warned = sum(1 for r in results if r.ok is None)
        summary = "Todo en orden" if not failed and not warned else f"{failed} error(es) · {warned} aviso(s)"
        self.status.configure(text=summary, text_color=theme.ERROR if failed else theme.WARNING if warned else theme.SUCCESS)
        self._render(results)
        self.after(60, lambda: self.list_frame._parent_canvas.yview_moveto(0))

    def _render(self, results: list[CheckResult]) -> None:
        for child in self.list_frame.winfo_children():
            child.destroy()
        for index, result in enumerate(results):
            mark, color = _ICONS[result.ok]
            row = ctk.CTkFrame(self.list_frame, fg_color=theme.SURFACE, corner_radius=theme.RADIUS_BUTTON, border_width=1, border_color=theme.BORDER)
            row.grid(row=index, column=0, sticky="ew", pady=(0, theme.PAD_SMALL))
            row.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(row, text=mark, font=theme.font(16, True), text_color=color, width=32).grid(row=0, column=0, rowspan=3, padx=(8, 4), pady=8, sticky="n")
            ctk.CTkLabel(row, text=result.name, font=theme.font(13, True), text_color=theme.TEXT, anchor="w").grid(row=0, column=1, sticky="w", pady=(8, 0))
            ctk.CTkLabel(row, text=result.detail, font=theme.font(11), text_color=theme.MUTED, anchor="w", justify="left", wraplength=560).grid(row=1, column=1, sticky="w", pady=(0, 8 if not result.fix or result.ok else 0))
            if result.fix and result.ok is not True:
                ctk.CTkLabel(row, text=f"Solución: {result.fix}", font=theme.font(11), text_color=color, anchor="w", justify="left", wraplength=560).grid(row=2, column=1, sticky="w", pady=(0, 8))

    def _copy(self) -> None:
        if not self._results:
            return
        self.clipboard_clear()
        self.clipboard_append(format_report(self._results))
        self._notify("Informe copiado al portapapeles.", "success")
