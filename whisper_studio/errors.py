class CancelledError(Exception):
    """Raised inside long operations when the user cancels the current job."""


class EngineError(Exception):
    """Transcription engine failure. `user_message` is Spanish and safe to display."""

    def __init__(self, user_message: str, detail: str = "") -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class DownloadFailed(Exception):
    """Download or probe failure. `user_message` is Spanish and safe to display."""

    def __init__(self, user_message: str, detail: str = "") -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail
