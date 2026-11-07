"""Identify LLM-originated sends without guessing from text or destination."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal
from weakref import WeakKeyDictionary

from astrbot.api.event import AstrMessageEvent
from astrbot.core.message.message_event_result import MessageChain, ResultContentType

Origin = Literal["llm", "cron", "background_task"]
CURRENT_ORIGIN: ContextVar[Origin | None] = ContextVar("gsvi_llm_origin", default=None)
REQUEST_ORIGIN_KEY = "astrbot_plugin_gsvi_short_tts.request_origin"


@dataclass(frozen=True)
class RequestOrigin:
    tasks: WeakKeyDictionary[asyncio.Task[object], Origin] = field(
        default_factory=WeakKeyDictionary
    )


def agent_origin(event: AstrMessageEvent | None) -> Origin:
    if getattr(event, "event_kind", "message") == "agent_wakeup":
        source = getattr(event, "source", None)
        if source == "cron":
            return "cron"
        if source == "background_task":
            return "background_task"
    return "llm"


def mark_request_origin(event: AstrMessageEvent) -> None:
    task = asyncio.current_task()
    if task is None:
        return
    request = event.get_extra(REQUEST_ORIGIN_KEY)
    if not isinstance(request, RequestOrigin):
        request = RequestOrigin()
        event.set_extra(REQUEST_ORIGIN_KEY, request)
    request.tasks[task] = agent_origin(event)


@contextmanager
def origin_scope(source: Origin | None) -> Iterator[None]:
    token = CURRENT_ORIGIN.set(source)
    try:
        yield
    finally:
        CURRENT_ORIGIN.reset(token)


def event_origin(
    event: AstrMessageEvent, message: MessageChain | None = None
) -> Origin | None:
    inherited = CURRENT_ORIGIN.get()
    if inherited is not None:
        return inherited
    request = event.get_extra(REQUEST_ORIGIN_KEY)
    task = asyncio.current_task()
    if not isinstance(request, RequestOrigin) or task is None:
        return None
    source = request.tasks.get(task)
    if source is None:
        return None
    if message is not None and message.type == "tool_direct_result":
        return source
    result = event.get_result()
    if result is None:
        return None
    if message is None:
        if (
            result.result_content_type == ResultContentType.STREAMING_RESULT
            and result.async_stream is not None
        ):
            return source
        return None
    if (
        result.is_model_result()
        and message.chain
        and all(
            any(component is original for original in result.chain)
            for component in message.chain
        )
    ):
        return source
    return None
