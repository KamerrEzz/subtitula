import copy
import queue
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk
from PIL import ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD

from whisper_studio.config import AppConfig, default_model_for, load_config, save_config
from whisper_studio.downloader import YtDlpDownloader
from whisper_studio.job_queue import (
    EngineStatus,
    JobAdded,
    JobFinished,
    JobQueue,
    JobUpdated,
    Notice,
    QueueEvent,
    QueueFinished,
    SegmentEvent,
)
from whisper_studio.logging_setup import get_logger
from whisper_studio.media import discover_media, folder_parts, is_media_file
from whisper_studio.models import FINISHED_STATUSES, Job, JobKind, JobStatus, PlaylistSlot
from whisper_studio.paths import data_dir
from whisper_studio.transcriber import EngineInfo, WhisperEngine
from whisper_studio.ui import theme
from whisper_studio import APP_NAME
from whisper_studio.ui.dialogs import DiagnosticsDialog, SettingsDialog
from whisper_studio.ui.download_card import DownloadCard
from whisper_studio.ui.helpers import (
    DeviceState,
    device_badge,
    engine_badge,
    ellipsize,
    format_clock,
    parse_drop_paths,
    restore_geometry,
    summarize_jobs,
)
from whisper_studio.ui.local_card import LocalFilesCard
from whisper_studio.ui.quick_settings import QuickSettingsCard
from whisper_studio.ui.right_panels import JobsPanel, TranscriptPanel
from whisper_studio.ui.services import (
    UiBridge,
    cancel_pending_callbacks,
    detect_device,
    make_app_icon,
    open_path,
    read_transcript_file,
)
from whisper_studio.ui.widgets import Pill, ToastHost, Tooltip, danger_button, primary_button, secondary_button

log = get_logger("ui")

DEFAULT_GEOMETRY = "1320x860"
MIN_SIZE = (1100, 720)
POLL_MS = 100
EVENTS_PER_TICK = 200
SHUTDOWN_TIMEOUT_S = 10
_DUPLICATE_SUFFIX = "ya está en la lista."


class App(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, config_path: Path | None = None) -> None:
        ctk.set_appearance_mode("dark")
        super().__init__(fg_color=theme.BG)
        self._dnd_ok = self._init_dnd()
        self._config_path = config_path
        self.cfg: AppConfig = load_config(config_path)
        self.engine = WhisperEngine()
        self.queue = JobQueue(self.engine, YtDlpDownloader())
        self.bridge = UiBridge()
        self.device = DeviceState()
        self._engine_info: EngineInfo | None = None
        self._loading_model: str | None = None
        self._engine_failed = False
        self._engine_status_text = ""
        self._running = False
        self._closing = False
        self._follow_running = True
        self._selected_id: str | None = None
        self._displayed_id: str | None = None
        self._running_id: str | None = None
        self._live_lines: dict[str, list[str]] = {}
        self._poll_id: str | None = None
        self._settings_dialog: SettingsDialog | None = None
        self._diagnostics_dialog: DiagnosticsDialog | None = None

        self.title(APP_NAME)
        self._apply_icon()
        self.minsize(*MIN_SIZE)
        self.geometry(self._initial_geometry())
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_main()
        self._build_footer()
        self.toasts = ToastHost(self)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind_all("<Control-v>", self._on_ctrl_v)
        self.bind_all("<Control-V>", self._on_ctrl_v)
        self._setup_drop()

        self._apply_config_to_views()
        self._refresh_badges()
        self._refresh_footer()
        self.bridge.run_async(detect_device, self._device_detected)
        self._poll_id = self.after(POLL_MS, self._poll)

    def _init_dnd(self) -> bool:
        try:
            self.TkdndVersion = TkinterDnD._require(self)
            return True
        except Exception:
            log.warning("Drag and drop is unavailable", exc_info=True)
            return False

    def _apply_icon(self) -> None:
        try:
            image = make_app_icon(256)
            path = Path(tempfile.gettempdir()) / "ytws_icon.ico"
            image.save(path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
            self.iconbitmap(str(path))
            self._icon_photo = ImageTk.PhotoImage(image.resize((64, 64)))
            self.iconphoto(True, self._icon_photo)
        except Exception:
            log.warning("Could not set the window icon", exc_info=True)

    def _initial_geometry(self) -> str:
        return restore_geometry(
            self.cfg.window_geometry,
            self.winfo_screenwidth(),
            self.winfo_screenheight(),
            DEFAULT_GEOMETRY,
            MIN_SIZE,
        )

    def report_callback_exception(self, exc: type[BaseException], val: BaseException, tb: Any) -> None:
        log.error("Unhandled error in a UI callback", exc_info=(exc, val, tb))
        try:
            self.toasts.show("Ocurrió un error inesperado. Los detalles quedaron en el registro.", "error")
        except Exception:
            pass

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=theme.PAD * 1.5, pady=(theme.PAD, theme.PAD_SMALL))
        header.grid_columnconfigure(1, weight=1)
        logo = ctk.CTkImage(light_image=make_app_icon(96), dark_image=make_app_icon(96), size=(44, 44))
        ctk.CTkLabel(header, image=logo, text="").grid(row=0, column=0, rowspan=2, padx=(0, theme.PAD_MID))
        ctk.CTkLabel(header, text=APP_NAME, font=theme.font(22, True), text_color=theme.TEXT).grid(row=0, column=1, sticky="sw")
        ctk.CTkLabel(
            header, text="Transcribe y subtitula tus videos y YouTube, todo en tu equipo.",
            font=theme.font(11), text_color=theme.MUTED,
        ).grid(row=1, column=1, sticky="nw")

        right = ctk.CTkFrame(header, fg_color="transparent")
        right.grid(row=0, column=2, rowspan=2, sticky="e")
        self.device_pill = Pill(right, "Detectando…", theme.MUTED, base=theme.BG)
        self.engine_pill = Pill(right, "Motor sin cargar", theme.MUTED, base=theme.BG)
        self.device_pill.pack(side="left", padx=(0, theme.PAD_SMALL))
        self.engine_pill.pack(side="left", padx=(0, theme.PAD_MID))
        self._device_tip = Tooltip(self.device_pill, lambda: self._device_tooltip())
        self._engine_tip = Tooltip(self.engine_pill, lambda: self._engine_status_text)
        self.device_pill.bind("<Button-1>", lambda _e: self._explain_fallback())
        secondary_button(right, "Diagnóstico", self._open_diagnostics, height=34, width=130).pack(side="left", padx=(0, theme.PAD_SMALL))
        secondary_button(right, "Configuración", self._open_settings, height=34, width=140).pack(side="left")
        self._header_logo = logo

    def _build_main(self) -> None:
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=1, column=0, sticky="nsew", padx=theme.PAD * 1.5, pady=theme.PAD_SMALL)
        main.grid_columnconfigure(0, weight=0)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(main, width=470, height=100, fg_color="transparent", scrollbar_button_color=theme.BORDER)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, theme.PAD))
        left.grid_columnconfigure(0, weight=1)
        self.local_card = LocalFilesCard(left, self.add_local_paths)
        self.download_card = DownloadCard(
            left, self.bridge, self._add_youtube, self.notify, lambda name, value: self.set_config(name, value)
        )
        self.quick_card = QuickSettingsCard(left, self.set_config, self._pick_output_dir, self._open_output_dir)
        for row, card in enumerate((self.local_card, self.download_card, self.quick_card)):
            card.grid(row=row, column=0, sticky="ew", pady=(0, theme.PAD_MID))

        right = ctk.CTkFrame(main, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=3)
        right.grid_rowconfigure(1, weight=2)
        self.jobs_panel = JobsPanel(
            right, self._select_job, self._remove_job, self._open_job_folder, self._view_job, self._clear_finished,
            self._retry_failed,
        )
        self.jobs_panel.grid(row=0, column=0, sticky="nsew", pady=(0, theme.PAD_MID))
        self.transcript_panel = TranscriptPanel(right, self._copy_transcript, self._open_displayed_folder, self._open_displayed_txt)
        self.transcript_panel.grid(row=1, column=0, sticky="nsew")
        self.transcript_panel.set_actions_enabled(False, False)

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color=theme.SURFACE, corner_radius=theme.RADIUS_CARD, border_width=1, border_color=theme.BORDER)
        footer.grid(row=2, column=0, sticky="ew", padx=theme.PAD * 1.5, pady=(theme.PAD_SMALL, theme.PAD))
        footer.grid_columnconfigure(5, weight=1)
        self.start_btn = primary_button(footer, "▶  Iniciar procesamiento", self.start_processing, width=210)
        self.cancel_btn = secondary_button(footer, "Cancelar actual", self.queue.cancel_current, width=130)
        self.cancel_all_btn = danger_button(footer, "Cancelar todo", self._confirm_cancel_all, width=120)
        self.results_btn = secondary_button(footer, "Abrir carpeta de resultados", self._open_output_dir, width=230)
        self.log_btn = secondary_button(footer, "Abrir registro", self._open_log, width=130)
        buttons = (self.start_btn, self.cancel_btn, self.cancel_all_btn, self.results_btn, self.log_btn)
        for column, button in enumerate(buttons):
            button.grid(row=0, column=column, padx=(theme.PAD if column == 0 else theme.PAD_SMALL, 0), pady=theme.PAD_MID)
        self.start_btn.configure(font=theme.font(13, True))
        progress = ctk.CTkFrame(footer, fg_color="transparent")
        progress.grid(row=0, column=5, sticky="ew", padx=theme.PAD)
        progress.grid_columnconfigure(0, weight=1)
        self.summary_label = ctk.CTkLabel(progress, text="", font=theme.font(12), text_color=theme.TEXT, anchor="e")
        self.summary_label.grid(row=0, column=0, sticky="e")
        self.overall_bar = ctk.CTkProgressBar(progress, height=8, corner_radius=4, fg_color=theme.BORDER, progress_color=theme.ACCENT)
        self.overall_bar.set(0)
        self.overall_bar.grid(row=1, column=0, sticky="ew", pady=(4, 0))

    # configuration -------------------------------------------------------------------------

    def set_config(self, name: str, value: Any) -> None:
        setattr(self.cfg, name, value)
        if name == "model":
            self.cfg.model_user_set = True
        self._save_config()
        self._apply_config_to_views()
        if self.queue.is_running:
            self.queue.start(self.cfg)

    def _save_config(self) -> None:
        try:
            save_config(self.cfg, self._config_path)
        except OSError:
            log.exception("Could not save settings")
            self.notify("No se pudo guardar la configuración.", "error")

    def _apply_config_to_views(self) -> None:
        self.quick_card.apply_config(self.cfg)
        self.download_card.apply_config(self.cfg.download_mode, self.cfg.video_quality)
        if self._settings_dialog is not None and self._settings_dialog.winfo_exists():
            self._settings_dialog.apply_config(self.cfg)

    def _pick_output_dir(self) -> None:
        chosen = filedialog.askdirectory(parent=self, title="Carpeta de salida", initialdir=self.cfg.output_dir)
        if chosen:
            self.set_config("output_dir", str(Path(chosen)))

    def _device_detected(self, state: DeviceState | None, error: BaseException | None) -> None:
        self.device = state or DeviceState(checked=True)
        if not self.cfg.model_user_set:
            self.cfg.model = default_model_for(self.device.cuda_detected)
            self._apply_config_to_views()
        self._refresh_badges()

    # badges --------------------------------------------------------------------------------

    def _refresh_badges(self) -> None:
        self.device_pill.set(*device_badge(self.device, self._engine_info))
        self.engine_pill.set(*engine_badge(self._engine_info, self._loading_model, self._engine_failed))

    def _device_tooltip(self) -> str:
        info = self._engine_info
        if info is not None and info.fallback_reason:
            return f"{info.fallback_reason}\n(haz clic para ver el detalle)"
        return ""

    def _explain_fallback(self) -> None:
        info = self._engine_info
        if info is None or not info.fallback_reason:
            return
        detail = f"\n\nDetalle técnico:\n{info.cuda_error_detail}" if info.cuda_error_detail else ""
        messagebox.showwarning("Se usa la CPU", info.fallback_reason + detail, parent=self)

    def _engine_replaced(self) -> None:
        self._engine_info = self.engine.info
        self._refresh_badges()

    # notifications -------------------------------------------------------------------------

    def notify(self, message: str, level: str = "info") -> None:
        self.toasts.show(message, level)

    # events --------------------------------------------------------------------------------

    def _poll(self) -> None:
        self._poll_id = None
        if self._closing:
            return
        try:
            self.bridge.drain()
            self._drain_queue_events()
        except Exception:
            log.exception("Event polling failed")
        finally:
            if not self._closing:
                self._poll_id = self.after(POLL_MS, self._poll)

    def _drain_queue_events(self) -> None:
        notices: list[Notice] = []
        for _ in range(EVENTS_PER_TICK):
            try:
                event = self.queue.events.get_nowait()
            except queue.Empty:
                break
            try:
                if isinstance(event, Notice):
                    notices.append(event)
                else:
                    self._handle_event(event)
            except Exception:
                log.exception("Failed to handle %s", type(event).__name__)
        self._show_notices(notices)

    def _show_notices(self, notices: list[Notice]) -> None:
        duplicates = [n for n in notices if n.message.endswith(_DUPLICATE_SUFFIX)]
        others = [n for n in notices if n not in duplicates]
        if len(duplicates) == 1:
            others.append(duplicates[0])
        elif duplicates:
            others.append(Notice("warning", f"{len(duplicates)} elementos ya estaban en la lista."))
        for notice in others[:3]:
            self.notify(notice.message, notice.level)

    def _handle_event(self, event: QueueEvent) -> None:
        if isinstance(event, JobAdded):
            self.jobs_panel.add_job(event.job)
        elif isinstance(event, JobUpdated):
            self._on_job_updated(event.job)
        elif isinstance(event, SegmentEvent):
            self._on_segment(event)
        elif isinstance(event, EngineStatus):
            self._on_engine_status(event)
        elif isinstance(event, JobFinished):
            self._on_job_finished(event.job)
        elif isinstance(event, QueueFinished):
            self._on_queue_finished()
        self._refresh_footer()

    def _on_job_updated(self, job: Job) -> None:
        self.jobs_panel.update_job(job)
        if job.status == JobStatus.TRANSCRIBING and job.id != self._running_id:
            self._running_id = job.id
            self._live_lines.setdefault(job.id, [])
            if self._follow_running:
                self._select_job(job.id, user=False)
        if job.id == self._displayed_id and job.status in FINISHED_STATUSES:
            self._show_job(job.id)

    def _on_segment(self, event: SegmentEvent) -> None:
        line = f"{format_clock(event.segment.start)} {event.segment.text.strip()}"
        self._live_lines.setdefault(event.job_id, []).append(line)
        if event.job_id == self._displayed_id:
            self.transcript_panel.append(line)
            self.transcript_panel.set_actions_enabled(True, self._has_folder(event.job_id))

    def _on_engine_status(self, event: EngineStatus) -> None:
        if event.info is None:
            self._loading_model = self.cfg.model
            self._engine_failed = False
            self._engine_status_text = event.text
        else:
            self._engine_info = event.info
            self._loading_model = None
            self._engine_failed = False
            self._engine_status_text = event.text
        self._refresh_badges()

    def _on_job_finished(self, job: Job) -> None:
        self.jobs_panel.update_job(job)
        if job.status == JobStatus.ERROR and self._loading_model is not None:
            self._loading_model = None
            self._engine_failed = True
            self._refresh_badges()
        if job.id == self._running_id:
            self._running_id = None
        if job.status == JobStatus.COMPLETED:
            verb = "Descarga completada" if job.mode == "download_only" and job.kind == JobKind.YOUTUBE else "Transcripción completada"
            self.notify(f"{verb}: {ellipsize(job.title, 60)}", "success")
        elif job.status == JobStatus.ERROR:
            self.notify(f"Error en «{ellipsize(job.title, 40)}»: {job.error}", "error")
        if job.id == self._displayed_id:
            self._show_job(job.id)

    def _on_queue_finished(self) -> None:
        self._running = False
        self.bell()
        self.notify("Procesamiento terminado.", "success")

    # footer --------------------------------------------------------------------------------

    def _refresh_footer(self) -> None:
        summary = summarize_jobs(self.jobs_panel.jobs())
        self.summary_label.configure(text=summary.text)
        self.overall_bar.set(summary.fraction)
        self.start_btn.configure(state="normal" if summary.pending and not self._running else "disabled")
        self.cancel_btn.configure(state="normal" if summary.active else "disabled")
        self.cancel_all_btn.configure(state="normal" if summary.active or summary.pending else "disabled")

    def start_processing(self) -> None:
        if self._running:
            return
        self._follow_running = True
        self._running = True
        self.queue.start(self.cfg)
        self._refresh_footer()

    def _confirm_cancel_all(self) -> None:
        if messagebox.askyesno("Cancelar todo", "¿Cancelar el trabajo actual y todos los pendientes?", parent=self):
            self.queue.cancel_all()

    # adding jobs ---------------------------------------------------------------------------

    def _add_youtube(
        self, url: str, video_id: str, title: str, mode: str, duration: float | None, thumb: str | None,
        playlist: PlaylistSlot | None,
    ) -> bool:
        return self.queue.add_youtube(url, video_id, title, mode, duration, thumb, playlist) is not None

    def add_local_paths(self, paths: list[Path]) -> None:
        self.notify("Buscando archivos de audio y video…", "info")

        def scan() -> tuple[list[tuple[Path, tuple[str, ...]]], int]:
            found: list[tuple[Path, tuple[str, ...]]] = []
            unsupported = 0
            for path in paths:
                if path.is_dir():
                    found.extend((file, folder_parts(file, path)) for file in discover_media(path))
                elif path.is_file() and is_media_file(path):
                    found.append((path, ()))
                else:
                    unsupported += 1
            return found, unsupported

        self.bridge.run_async(scan, self._local_scanned)

    def _local_scanned(self, result: tuple[list[tuple[Path, tuple[str, ...]]], int] | None, error: BaseException | None) -> None:
        if error is not None or result is None:
            self.notify("No se pudieron leer los archivos seleccionados.", "error")
            return
        found, unsupported = result
        added = sum(1 for path, folder in found if self.queue.add_local(path, folder) is not None)
        duplicates = len(found) - added
        parts = [f"{added} añadido(s)"]
        if duplicates:
            parts.append(f"{duplicates} duplicado(s)")
        if unsupported:
            parts.append(f"{unsupported} no compatible(s)")
        if not found and not unsupported:
            parts = ["No se encontraron archivos de audio o video"]
        self.notify(" · ".join(parts), "success" if added else "warning")

    def _setup_drop(self) -> None:
        if not self._dnd_ok:
            self.local_card.drop_zone.configure(text="Arrastrar y soltar no está disponible; usa los botones.")
            return
        try:
            for target in (self, self.local_card.drop_zone):
                target.drop_target_register(DND_FILES)
                target.dnd_bind("<<DropEnter>>", self._drop_enter)
                target.dnd_bind("<<DropLeave>>", self._drop_leave)
                target.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            log.warning("Could not register drop targets", exc_info=True)
            self._dnd_ok = False

    def _drop_enter(self, event: Any) -> Any:
        self.local_card.set_drop_highlight(True)
        return event.action

    def _drop_leave(self, event: Any) -> Any:
        self.local_card.set_drop_highlight(False)
        return event.action

    def _on_drop(self, event: Any) -> Any:
        self.local_card.set_drop_highlight(False)
        paths = parse_drop_paths(event.data)
        if paths:
            self.add_local_paths(paths)
        return event.action

    def _on_ctrl_v(self, _event: tk.Event) -> None:
        try:
            focus = self.focus_get()
        except KeyError:
            focus = None
        if isinstance(focus, (tk.Entry, tk.Text)):
            return
        self.download_card.paste_from_clipboard()

    # job list actions ----------------------------------------------------------------------

    def _select_job(self, job_id: str, user: bool = True) -> None:
        if user:
            self._follow_running = job_id == self._running_id
        self._selected_id = job_id
        self.jobs_panel.select(job_id)
        self._show_job(job_id)

    def _view_job(self, job_id: str) -> None:
        self._select_job(job_id)

    def _has_folder(self, job_id: str) -> bool:
        job = self.jobs_panel.job(job_id)
        return job is not None and job.output_dir is not None and job.output_dir.exists()

    def _show_job(self, job_id: str | None) -> None:
        self._displayed_id = job_id
        job = self.jobs_panel.job(job_id) if job_id else None
        if job is None:
            self.transcript_panel.set_text("", "Selecciona un trabajo para ver su transcripción.", placeholder=True)
            self.transcript_panel.set_actions_enabled(False, False)
            return
        title = ellipsize(job.title, 50)
        live = self._live_lines.get(job.id, [])
        if job.transcript_text:
            self.transcript_panel.set_text(title, job.transcript_text)
        elif job.status == JobStatus.TRANSCRIBING or (live and not job.is_finished):
            subtitle = f"{title} · en vivo"
            if live:
                self.transcript_panel.set_text(subtitle, "\n".join(live) + "\n")
            else:
                self.transcript_panel.set_text(subtitle, "Esperando los primeros segmentos…", placeholder=True)
        elif job.status in (JobStatus.COMPLETED, JobStatus.SKIPPED) and job.output_dir is not None:
            self.transcript_panel.set_text(title, "Cargando transcripción…", placeholder=True)
            folder = job.output_dir
            self.bridge.run_async(lambda: read_transcript_file(folder), lambda text, e: self._file_loaded(job.id, title, text))
        else:
            self.transcript_panel.set_text(title, self._empty_reason(job), placeholder=True)
        self.transcript_panel.set_actions_enabled(bool(job.transcript_text or live), self._has_folder(job.id))

    @staticmethod
    def _empty_reason(job: Job) -> str:
        if job.status == JobStatus.PENDING:
            return "Este trabajo está en cola; la transcripción aparecerá cuando se procese."
        if job.status == JobStatus.DOWNLOADING:
            return "Descargando; la transcripción empezará al terminar la descarga."
        if job.status == JobStatus.PREPARING:
            return "Preparando el motor y el audio…"
        if job.status == JobStatus.ERROR:
            return "El trabajo terminó con error y no hay transcripción."
        if job.mode == "download_only" and job.kind == JobKind.YOUTUBE:
            return "Este trabajo solo descarga el archivo; no genera transcripción."
        return "No hay transcripción disponible para este trabajo."

    def _file_loaded(self, job_id: str, title: str, text: str | None) -> None:
        if job_id != self._displayed_id:
            return
        if text is None:
            self.transcript_panel.set_text(title, "No se encontró el archivo transcripcion.txt de este trabajo.", placeholder=True)
        elif not text.strip():
            self.transcript_panel.set_text(title, "La transcripción está vacía: no se detectó voz en este archivo.", placeholder=True)
            self.transcript_panel.set_actions_enabled(False, self._has_folder(job_id))
        else:
            self.transcript_panel.set_text(title, text)
            self.transcript_panel.set_actions_enabled(True, self._has_folder(job_id))

    def _remove_job(self, job_id: str) -> None:
        if not self.queue.remove(job_id):
            self.notify("No se puede quitar un trabajo en curso; cancélalo primero.", "warning")
            return
        self.jobs_panel.remove_job(job_id)
        self._live_lines.pop(job_id, None)
        if job_id == self._displayed_id:
            self._show_job(None)
        self._refresh_footer()

    def _retry_failed(self) -> None:
        retried = self.queue.retry_failed()
        if not retried:
            self.notify("No hay trabajos con error para reintentar.", "info")
            return
        self.notify(f"{retried} trabajo(s) vuelto(s) a la cola.", "info")
        if self._running:
            self.queue.start(self.cfg)
        else:
            self.start_processing()

    def _open_log(self) -> None:
        log_file = data_dir() / "logs" / "app.log"
        self._open(log_file if log_file.exists() else log_file.parent)

    def _clear_finished(self) -> None:
        removed = self.queue.clear_finished()
        alive = {job.id for job in self.queue.jobs()}
        for job in self.jobs_panel.jobs():
            if job.id not in alive:
                self.jobs_panel.remove_job(job.id)
                self._live_lines.pop(job.id, None)
        if self._displayed_id is not None and self.jobs_panel.job(self._displayed_id) is None:
            self._show_job(None)
        self.notify(f"{removed} trabajo(s) terminado(s) quitado(s)." if removed else "No hay trabajos terminados.", "info")
        self._refresh_footer()

    def _open(self, path: Path) -> None:
        try:
            open_path(path)
        except OSError:
            log.exception("Could not open %s", path)
            self.notify("No se pudo abrir la ubicación.", "error")

    def _open_job_folder(self, job_id: str) -> None:
        job = self.jobs_panel.job(job_id)
        if job is not None and job.output_dir is not None and job.output_dir.exists():
            self._open(job.output_dir)
        else:
            self.notify("La carpeta de este trabajo ya no existe.", "warning")

    def _open_displayed_folder(self) -> None:
        if self._displayed_id:
            self._open_job_folder(self._displayed_id)

    def _open_displayed_txt(self) -> None:
        job = self.jobs_panel.job(self._displayed_id) if self._displayed_id else None
        txt = job.output_dir / "transcripcion.txt" if job and job.output_dir else None
        if txt is not None and txt.exists():
            self._open(txt)
        else:
            self.notify("Todavía no existe el archivo TXT de este trabajo.", "warning")

    def _open_output_dir(self) -> None:
        folder = Path(self.cfg.output_dir)
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.notify("No se pudo crear la carpeta de resultados.", "error")
            return
        self._open(folder)

    def _copy_transcript(self) -> None:
        text = self.transcript_panel.get_text()
        if not text:
            self.notify("No hay texto para copiar.", "warning")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.notify("Copiado", "success")

    # dialogs -------------------------------------------------------------------------------

    def _open_settings(self) -> None:
        if self._settings_dialog is not None and self._settings_dialog.winfo_exists():
            self._settings_dialog.lift()
            return
        self._settings_dialog = SettingsDialog(self, self.cfg, self.set_config, self._pick_output_dir)

    def _open_diagnostics(self) -> None:
        if self._diagnostics_dialog is not None and self._diagnostics_dialog.winfo_exists():
            self._diagnostics_dialog.lift()
            return
        self._diagnostics_dialog = DiagnosticsDialog(
            self, self.bridge, lambda: copy.deepcopy(self.cfg), lambda: self._running or self.queue.is_running,
            self.engine, self._engine_replaced, self.notify,
        )

    # shutdown ------------------------------------------------------------------------------

    def _on_close(self) -> None:
        if self._closing:
            return
        if self._running or self.queue.is_running:
            if not messagebox.askyesno("Salir", "Hay trabajos en curso. ¿Cancelar y salir?", parent=self):
                return
            self._closing = True
            self.summary_label.configure(text="Cerrando…")
            for button in (self.start_btn, self.cancel_btn, self.cancel_all_btn, self.results_btn):
                button.configure(state="disabled")
            worker = threading.Thread(target=lambda: self.queue.shutdown(timeout=SHUTDOWN_TIMEOUT_S), daemon=True)
            worker.start()
            self._wait_for_shutdown(worker)
            return
        self._closing = True
        self._finalize()

    def _wait_for_shutdown(self, worker: threading.Thread) -> None:
        if worker.is_alive():
            self.after(100, lambda: self._wait_for_shutdown(worker))
        else:
            self._finalize()

    def _finalize(self) -> None:
        try:
            self.cfg.window_geometry = self.geometry()
            self._save_config()
        finally:
            cancel_pending_callbacks(self)
            self.destroy()
