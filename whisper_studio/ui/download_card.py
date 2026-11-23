from typing import Callable

import customtkinter as ctk

from whisper_studio.downloader import ProbeResult, is_supported_url, probe
from whisper_studio.errors import DownloadFailed
from whisper_studio.logging_setup import get_logger
from whisper_studio.ui import theme
from whisper_studio.ui.helpers import format_elapsed, parse_urls
from whisper_studio.ui.services import THUMB_SIZE, UiBridge, fetch_thumbnail
from whisper_studio.ui.theme import Glyph
from whisper_studio.ui.widgets import Card, primary_button, secondary_button

log = get_logger("ui.download")

DEBOUNCE_MS = 600
MODE_LABELS = {
    "audio_transcribe": "Solo audio + transcripción",
    "video_transcribe": "Video + transcripción",
    "download_only": "Solo descargar",
}
QUALITY_LABELS = {"best": "Mejor", "1080": "1080p", "720": "720p", "480": "480p", "360": "360p"}
_MODE_BY_LABEL = {label: mode for mode, label in MODE_LABELS.items()}
_QUALITY_BY_LABEL = {label: q for q, label in QUALITY_LABELS.items()}

AddYoutube = Callable[[str, str, str, str, float | None, str | None], bool]


class DownloadCard(Card):
    """URL entry with debounced preview. Adding never blocks: probing runs in a thread."""

    def __init__(
        self,
        master: ctk.CTkBaseClass,
        bridge: UiBridge,
        add_youtube: AddYoutube,
        notify: Callable[[str, str], None],
        on_option_changed: Callable[[str, str], None],
    ) -> None:
        super().__init__(master, "Descargar de YouTube", Glyph.DOWNLOAD)
        self._bridge = bridge
        self._add_youtube = add_youtube
        self._notify = notify
        self._on_option_changed = on_option_changed
        self._debounce_id: str | None = None
        self._token = 0
        self._probing = False
        self._pending_add = False
        self._result: ProbeResult | None = None
        self._result_url = ""
        self._thumb_image: ctk.CTkImage | None = None
        self._last_text = ""
        self._build()

    def _build(self) -> None:
        body = self.body
        body.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(
            body, height=44, corner_radius=theme.RADIUS_BUTTON, font=theme.font(13),
            placeholder_text="Pega aquí un enlace de YouTube…", fg_color=theme.SURFACE_2,
            border_color=theme.BORDER, text_color=theme.TEXT, placeholder_text_color=theme.MUTED,
        )
        self.entry.grid(row=0, column=0, sticky="ew")
        self.entry.bind("<KeyRelease>", lambda _e: self._schedule_probe(), add="+")
        self.entry.bind("<Return>", lambda _e: self.submit(), add="+")

        buttons = ctk.CTkFrame(body, fg_color="transparent")
        buttons.grid(row=0, column=1, padx=(theme.PAD_SMALL, 0))
        secondary_button(buttons, "Pegar", self.paste_from_clipboard, width=84).pack(side="left", padx=(0, theme.PAD_SMALL))
        primary_button(buttons, "Añadir", self.submit, width=96).pack(side="left")

        self.error_label = ctk.CTkLabel(
            body, text="", font=theme.font(11), text_color=theme.ERROR, anchor="w", justify="left", wraplength=420,
        )
        self.error_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.error_label.grid_remove()

        self.mode_menu = ctk.CTkSegmentedButton(
            body, values=list(MODE_LABELS.values()), command=self._mode_chosen, font=theme.font(12),
            corner_radius=theme.RADIUS_BUTTON, fg_color=theme.SURFACE_2, selected_color=theme.ACCENT,
            selected_hover_color=theme.ACCENT_HOVER, unselected_color=theme.SURFACE_2,
            unselected_hover_color=theme.BORDER, text_color=theme.TEXT, height=34,
        )
        self.mode_menu.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(theme.PAD_MID, 0))

        quality = ctk.CTkFrame(body, fg_color="transparent")
        quality.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(theme.PAD_SMALL, 0))
        ctk.CTkLabel(quality, text="Calidad de video", font=theme.font(12), text_color=theme.MUTED).pack(side="left")
        self.quality_menu = ctk.CTkOptionMenu(
            quality, values=list(QUALITY_LABELS.values()), command=self._quality_chosen, width=110, height=30,
            font=theme.font(12), dropdown_font=theme.font(12), corner_radius=theme.RADIUS_BUTTON - 2,
            fg_color=theme.SURFACE_2, button_color=theme.BORDER, button_hover_color=theme.ACCENT,
            text_color=theme.TEXT, dropdown_fg_color=theme.SURFACE_2, dropdown_hover_color=theme.ACCENT,
            dropdown_text_color=theme.TEXT, text_color_disabled=theme.MUTED,
        )
        self.quality_menu.pack(side="left", padx=(theme.PAD_SMALL, 0))

        self._build_preview(body)

    def _build_preview(self, body: ctk.CTkFrame) -> None:
        self.preview = ctk.CTkFrame(body, fg_color=theme.SURFACE_2, corner_radius=theme.RADIUS_BUTTON)
        self.preview.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(theme.PAD_MID, 0))
        self.preview.grid_columnconfigure(1, weight=1)
        self.thumb_label = ctk.CTkLabel(
            self.preview, text="", width=THUMB_SIZE[0], height=THUMB_SIZE[1], fg_color=theme.BORDER, corner_radius=12,
        )
        self.thumb_label.grid(row=0, column=0, rowspan=3, padx=10, pady=10)
        self.title_label = ctk.CTkLabel(
            self.preview, text="", font=theme.font(13, True), text_color=theme.TEXT, anchor="w",
            justify="left", wraplength=220,
        )
        self.title_label.grid(row=0, column=1, sticky="sw", padx=(0, 10), pady=(12, 0))
        self.meta_label = ctk.CTkLabel(
            self.preview, text="", font=theme.font(11), text_color=theme.MUTED, anchor="w", justify="left",
            wraplength=220,
        )
        self.meta_label.grid(row=1, column=1, sticky="nw", padx=(0, 10))
        self.spinner = ctk.CTkProgressBar(
            self.preview, mode="indeterminate", height=4, progress_color=theme.ACCENT, fg_color=theme.BORDER,
        )
        self.spinner.grid(row=2, column=1, sticky="ew", padx=(0, 14), pady=(0, 12))
        self.preview.grid_remove()

    def apply_config(self, mode: str, quality: str) -> None:
        self.mode_menu.set(MODE_LABELS[mode])
        self.quality_menu.set(QUALITY_LABELS[quality])
        self._update_quality_state()

    def _mode_chosen(self, label: str) -> None:
        self._on_option_changed("download_mode", _MODE_BY_LABEL[label])
        self._update_quality_state()

    def _quality_chosen(self, label: str) -> None:
        self._on_option_changed("video_quality", _QUALITY_BY_LABEL[label])

    def _update_quality_state(self) -> None:
        is_video_mode = _MODE_BY_LABEL[self.mode_menu.get()] != "audio_transcribe"
        self.quality_menu.configure(state="normal" if is_video_mode else "disabled")

    @property
    def mode(self) -> str:
        return _MODE_BY_LABEL[self.mode_menu.get()]

    def paste_from_clipboard(self) -> None:
        try:
            text = self.winfo_toplevel().clipboard_get()
        except Exception:
            self._show_error("El portapapeles no contiene texto.")
            return
        self.set_text(text)
        self.entry.focus_set()

    def set_text(self, text: str) -> None:
        self.entry.delete(0, "end")
        self.entry.insert(0, text.strip().replace("\r", " ").replace("\n", " "))
        self._schedule_probe()

    def _show_error(self, message: str) -> None:
        self.error_label.configure(text=message)
        self.error_label.grid()

    def _clear_error(self) -> None:
        self.error_label.grid_remove()

    def _stop_spinner(self) -> None:
        self.spinner.stop()
        self.spinner.grid_remove()

    def _reset_preview(self) -> None:
        self._token += 1
        self._probing = False
        self._pending_add = False
        self._result = None
        self._result_url = ""
        self._stop_spinner()
        self.preview.grid_remove()
        self._clear_error()

    def _schedule_probe(self) -> None:
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
            self._debounce_id = None
        text = self.entry.get().strip()
        if text == self._last_text:
            return
        self._last_text = text
        self._reset_preview()
        if text:
            self._debounce_id = self.after(DEBOUNCE_MS, self._debounced)

    def _debounced(self) -> None:
        self._debounce_id = None
        urls = parse_urls(self.entry.get())
        if not urls:
            self._show_error("Eso no parece un enlace. Pega un enlace de YouTube (video o lista de reproducción).")
        elif len(urls) > 1:
            self._show_preview_text(f"{len(urls)} enlaces detectados", "Se añadirán todos al pulsar «Añadir».")
        elif not is_supported_url(urls[0]):
            self._show_error("Solo se admiten enlaces de YouTube (youtube.com o youtu.be).")
        else:
            self._start_probe(urls[0])

    def _show_preview_text(self, title: str, meta: str, probing: bool = False) -> None:
        self.title_label.configure(text=title)
        self.meta_label.configure(text=meta)
        self.thumb_label.configure(image=None, text="")
        self.preview.grid()
        if probing:
            self.spinner.grid()
            self.spinner.start()
        else:
            self._stop_spinner()

    def _start_probe(self, url: str) -> None:
        self._token += 1
        token = self._token
        self._probing = True
        self._show_preview_text("Obteniendo información…", url, probing=True)
        self._bridge.run_async(lambda: probe(url), lambda r, e: self._probe_done(token, url, r, e))

    def _probe_done(self, token: int, url: str, result: ProbeResult | None, error: BaseException | None) -> None:
        if token != self._token:
            return
        self._probing = False
        self._stop_spinner()
        if error is not None or result is None:
            message = error.user_message if isinstance(error, DownloadFailed) else "No se pudo obtener información del enlace."
            self.preview.grid_remove()
            self._pending_add = False
            self._show_error(message)
            return
        self._result, self._result_url = result, self.entry.get().strip()
        self._render_result(result, token)
        if self._pending_add:
            self._pending_add = False
            self._commit(result, url)

    def _render_result(self, result: ProbeResult, token: int) -> None:
        if result.kind == "playlist":
            meta = f"Lista de reproducción · {len(result.entries)} videos"
            if result.unavailable_count:
                meta += f" ({result.unavailable_count} no disponibles)"
            self._show_preview_text(result.title, meta)
            return
        parts = [format_elapsed(result.duration)] if result.duration else []
        if result.playlist_hint:
            parts.append("el enlace pertenece a una lista; se añadirá solo este video")
        self._show_preview_text(result.title, " · ".join(parts))
        if result.thumbnail:
            thumb_url = result.thumbnail
            self._bridge.run_async(lambda: fetch_thumbnail(thumb_url), lambda img, e: self._thumb_done(token, img, e))

    def _thumb_done(self, token: int, image, error: BaseException | None) -> None:
        if token != self._token or error is not None or image is None:
            return
        self._thumb_image = ctk.CTkImage(light_image=image, dark_image=image, size=THUMB_SIZE)
        self.thumb_label.configure(image=self._thumb_image)

    def submit(self) -> None:
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
            self._debounce_id = None
        urls = parse_urls(self.entry.get())
        if not urls:
            self._show_error("Pega primero un enlace de YouTube.")
            return
        invalid = [u for u in urls if not is_supported_url(u)]
        if invalid:
            self._show_error("Solo se admiten enlaces de YouTube (youtube.com o youtu.be).")
            return
        self._clear_error()
        if len(urls) > 1:
            self._add_many(urls)
        elif self._result is not None and self._result_url == self.entry.get().strip():
            self._commit(self._result, urls[0])
        elif self._probing:
            self._pending_add = True
        else:
            self._pending_add = True
            self._start_probe(urls[0])

    def _add_many(self, urls: list[str]) -> None:
        self._token += 1
        token = self._token
        self._show_preview_text(f"Obteniendo información de {len(urls)} enlaces…", "", probing=True)

        def work() -> list[tuple[str, ProbeResult | None, str]]:
            out: list[tuple[str, ProbeResult | None, str]] = []
            for url in urls:
                try:
                    out.append((url, probe(url), ""))
                except DownloadFailed as exc:
                    out.append((url, None, exc.user_message))
            return out

        self._bridge.run_async(work, lambda r, e: self._many_done(token, r, e))

    def _many_done(self, token: int, results: list | None, error: BaseException | None) -> None:
        if token != self._token:
            return
        self._stop_spinner()
        self.preview.grid_remove()
        if error is not None or results is None:
            self._show_error("No se pudo obtener información de los enlaces.")
            return
        added, failures = 0, []
        for url, result, message in results:
            if result is None:
                failures.append(message)
            else:
                added += self._enqueue(result, url)
        self._finish_add(added, failures)

    def _commit(self, result: ProbeResult, url: str) -> None:
        added = self._enqueue(result, url)
        self._finish_add(added, [])

    def _enqueue(self, result: ProbeResult, url: str) -> int:
        mode = self.mode
        if result.kind == "playlist":
            return sum(
                1 for e in result.entries if self._add_youtube(e.url, e.id, e.title, mode, e.duration, None)
            )
        thumb = result.thumbnail
        return int(self._add_youtube(result.webpage_url or url, result.id, result.title, mode, result.duration, thumb))

    def _finish_add(self, added: int, failures: list[str]) -> None:
        if added:
            self._notify(f"{added} {'video añadido' if added == 1 else 'videos añadidos'} a la lista.", "success")
            self.entry.delete(0, "end")
            self._last_text = ""
            self._reset_preview()
        if failures:
            self._show_error(failures[0] if len(failures) == 1 else f"{len(failures)} enlaces no se pudieron procesar. {failures[0]}")
        elif not added:
            self._show_error("No se añadió nada (el video ya estaba en la lista).")
