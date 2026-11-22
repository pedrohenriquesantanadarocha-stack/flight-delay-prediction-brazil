"""Google Colab glue: the only module that imports `google.colab`."""

from __future__ import annotations


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def setup(mount_drive: bool = True, authenticate: bool = True) -> None:
    """Mount Google Drive and authenticate the user for BigQuery (ADC)."""
    if not in_colab():
        return
    if mount_drive:
        from google.colab import drive

        drive.mount("/content/drive")
    if authenticate:
        from google.colab import auth

        auth.authenticate_user()
