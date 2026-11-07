"""Parse the usetts command without changing stored settings on invalid input."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Literal

from .settings import VoicePresetSchema

USAGE = (
    "用法：\n"
    "/usetts 音色名 字数 概率，例如 /usetts 花火 80 50\n"
    "/usetts on：开启\n"
    "/usetts off：关闭\n"
    "/usetts list：列出可用音色\n"
    "/usetts：查看状态\n"
    "字数必须为正整数，概率必须为 0–100 的整数。音色名包含空格时请加引号。"
)


@dataclass(frozen=True)
class TtsCommand:
    action: Literal["status", "on", "off", "list", "configure"]
    preset: VoicePresetSchema | None = None


def parse_command(message: str) -> TtsCommand:
    """Accept status, list, one switch, or three configuration arguments."""
    try:
        parts = shlex.split(message.strip())
    except ValueError as error:
        raise ValueError(USAGE) from error
    arguments = parts[1:]
    if not arguments:
        return TtsCommand(action="status")
    if len(arguments) == 1:
        action = arguments[0].lower()
        if action == "on":
            return TtsCommand(action="on")
        if action == "off":
            return TtsCommand(action="off")
        if action == "list":
            return TtsCommand(action="list")
    if len(arguments) == 3:
        try:
            max_chars = int(arguments[1])
            probability = int(arguments[2])
            preset = VoicePresetSchema(
                voice=arguments[0], max_chars=max_chars, probability=probability
            )
            return TtsCommand(action="configure", preset=preset)
        except ValueError as error:
            raise ValueError(USAGE) from error
    raise ValueError(USAGE)
