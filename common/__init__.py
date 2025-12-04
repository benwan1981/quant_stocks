# -*- coding: utf-8 -*-
"""
Common helpers shared across scripts.
"""

from __future__ import annotations

from pathlib import PurePath
import unicodedata
from typing import Union


PathInput = Union[str, PurePath, None]


def ensure_utf8_filename(value: PathInput) -> str:
    """
    Normalize any filename/path fragment to UTF-8 encoded text.
    This helps when downstream scripts拼接中文文件名时
    在不同终端保持一致的字符编码。
    """
    if value is None:
        return ""

    if isinstance(value, PurePath):
        text = value.as_posix()
    else:
        text = str(value)

    normalized = unicodedata.normalize("NFC", text)
    # Encode→decode ensures字符串按照 UTF-8 存储，遇到无法编码的字符会被忽略。
    return normalized.encode("utf-8", errors="ignore").decode("utf-8")
