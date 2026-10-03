"""Errors the HTTP layer maps to 404 / 400."""


class NotFound(Exception):
    pass


class Invalid(ValueError):
    pass


class WorkerError(Exception):
    """A model call could not produce a usable answer. `retries` tells how many extra attempts were made."""

    def __init__(self, message: str, retries: int = 0, usage: dict | None = None):
        super().__init__(message)
        self.retries = retries
        self.usage = usage or {}
