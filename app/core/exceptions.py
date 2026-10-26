class EchoFlowError(Exception):
    """Excepción base del dominio backend de EchoFlow."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class InvalidYouTubeUrlError(EchoFlowError):
    def __init__(self, raw_input: str) -> None:
        super().__init__(
            f"No se pudo extraer un ID de video de YouTube válido desde '{raw_input}'.",
            status_code=422,
        )


class TranscriptUnavailableError(EchoFlowError):
    def __init__(self, video_id: str, reason: str = "") -> None:
        detail = (
            f"No hay subtítulos en inglés disponibles para el video '{video_id}'."
        )
        if reason:
            detail = f"{detail} Detalle: {reason}"
        super().__init__(detail, status_code=404)


class EmptyTranscriptError(EchoFlowError):
    def __init__(self, video_id: str) -> None:
        super().__init__(
            f"Los subtítulos del video '{video_id}' solo contienen pistas no verbales o están vacíos.",
            status_code=422,
        )

