"""Convert delivery copies after AstrBot has decided whether to send."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import TypeAlias, cast

from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.message_session import MessageSession

from .channels import Channel, resolve_channel, resolve_destination
from .origins import CURRENT_ORIGIN, event_origin, origin_scope
from .streaming import convert_stream

SKIP_TTS_KEY = "astrbot_plugin_gsvi_short_tts.skip_delivery"
Convert: TypeAlias = Callable[
    [Channel, MessageChain], Awaitable[tuple[MessageChain, Path | None]]
]
EventSend: TypeAlias = Callable[
    [AstrMessageEvent, MessageChain | None], Awaitable[None]
]
StreamSend: TypeAlias = Callable[
    [AstrMessageEvent, AsyncGenerator[MessageChain, None], bool], Awaitable[None]
]
SessionSend: TypeAlias = Callable[
    [object, MessageSession, MessageChain], Awaitable[None]
]
ContextSend: TypeAlias = Callable[
    [Context, str | MessageSession, MessageChain], Awaitable[bool]
]


@dataclass(frozen=True)
class SendScope:
    task: asyncio.Task[object] | None
    destination: str


@dataclass(frozen=True)
class Patch:
    owner: type
    name: str
    original: object
    installed: object
    local: bool


class OutgoingTts:
    def __init__(
        self,
        context: Context,
        convert: Convert,
        enabled: Callable[[Channel], Awaitable[bool]],
    ) -> None:
        self.context = context
        self.convert = convert
        self.enabled = enabled
        self.patches: dict[tuple[type, str], Patch] = {}
        self.scope: ContextVar[SendScope | None] = ContextVar(
            "gsvi_delivery_scope", default=None
        )

    def remember(self, owner: type, name: str, original: object, installed: object):
        self.patches[owner, name] = Patch(
            owner, name, original, installed, name in owner.__dict__
        )
        setattr(owner, name, installed)

    def active(self, owner: type, name: str, installed: object) -> bool:
        patch = self.patches.get((owner, name))
        return patch is not None and patch.installed is installed

    def nested(self, channel: Channel) -> bool:
        scope = self.scope.get()
        return (
            scope is not None
            and scope.task is asyncio.current_task()
            and scope.destination == channel.key
        )

    def install(self) -> None:
        if self.patches:
            return
        self.bind_context()
        for platform in self.context.platform_manager.platform_insts:
            self.bind_platform(platform)
        pending = list(AstrMessageEvent.__subclasses__())
        while pending:
            cls = pending.pop()
            self.bind_event_type(cls)
            pending.extend(cls.__subclasses__())

    def uninstall(self) -> None:
        for patch in reversed(list(self.patches.values())):
            if getattr(patch.owner, patch.name) is patch.installed:
                if patch.local:
                    setattr(patch.owner, patch.name, patch.original)
                else:
                    delattr(patch.owner, patch.name)
        self.patches.clear()

    def bind_context(self) -> None:
        original = cast(ContextSend, Context.send_message)

        @wraps(original)
        async def send(context, session, message_chain):
            if self.active(Context, "send_message", send) and context is self.context:
                target = session
                if isinstance(target, str):
                    try:
                        target = MessageSession.from_str(target)
                    except ValueError:
                        return await original(context, session, message_chain)
                platform = context.get_platform_inst(target.platform_id)
                if platform is not None:
                    self.bind_platform(platform)
            # Routing only: history and tool bookkeeping retain the original text.
            return await original(context, session, message_chain)

        self.remember(Context, "send_message", original, send)

    def bind_platform(self, platform) -> None:
        cls = type(platform)
        if (cls, "send_by_session") in self.patches:
            return
        original = cast(SessionSend, cls.send_by_session)

        @wraps(original)
        async def send(instance, session, message_chain):
            if (
                CURRENT_ORIGIN.get() is None
                or not self.active(cls, "send_by_session", send)
                or not any(
                    instance is item
                    for item in self.context.platform_manager.platform_insts
                )
            ):
                return await original(instance, session, message_chain)
            try:
                channel = resolve_destination(session, instance.meta().name)
            except ValueError:
                return await original(instance, session, message_chain)
            if self.nested(channel):
                return await original(instance, session, message_chain)
            token = self.scope.set(SendScope(asyncio.current_task(), channel.key))
            path = None
            try:
                converted, path = await self.convert(channel, message_chain)
                return await original(instance, session, converted)
            finally:
                self.scope.reset(token)
                self.cleanup(path)

        self.remember(cls, "send_by_session", original, send)

    def bind_event(self, event: AstrMessageEvent) -> None:
        platform = self.context.get_platform_inst(event.get_platform_id())
        if platform is not None:
            self.bind_platform(platform)
        for cls in type(event).__mro__:
            if cls is AstrMessageEvent:
                break
            self.bind_event_type(cls)

    def event_channel(self, event: AstrMessageEvent) -> Channel | None:
        if self.context.get_platform_inst(event.get_platform_id()) is None:
            return None
        try:
            channel = resolve_channel(event)
        except ValueError:
            return None
        if self.nested(channel):
            return None
        return channel

    def bind_event_type(self, cls: type) -> None:
        if "send" in cls.__dict__ and (cls, "send") not in self.patches:
            self.bind_event_send(cls)
        if (
            "send_streaming" in cls.__dict__
            and (cls, "send_streaming") not in self.patches
        ):
            self.bind_event_stream(cls)

    def bind_event_send(self, cls: type) -> None:
        original = cast(EventSend, cls.send)

        @wraps(original)
        async def send(event, message):
            if not self.active(cls, "send", send):
                return await original(event, message)
            if self.skip_event(event):
                with origin_scope(None):
                    return await original(event, message)
            source = event_origin(event, message)
            if source is None:
                return await original(event, message)
            if getattr(event, "event_kind", "message") == "agent_wakeup":
                # Keep the original text through Context's history bookkeeping.
                # Its adapter send is the delivery boundary for wakeups.
                with origin_scope(source):
                    return await original(event, message)
            channel = None
            if self.active(cls, "send", send) and message is not None:
                channel = self.event_channel(event)
            if channel is None:
                return await original(event, message)
            token = self.scope.set(SendScope(asyncio.current_task(), channel.key))
            path = None
            try:
                with origin_scope(source):
                    converted, path = await self.convert(channel, message)
                    return await original(event, converted)
            finally:
                self.scope.reset(token)
                self.cleanup(path)

        self.remember(cls, "send", original, send)

    def bind_event_stream(self, cls: type) -> None:
        original = cast(StreamSend, cls.send_streaming)

        @wraps(original)
        async def send(event, generator, use_fallback=False):
            if not self.active(cls, "send_streaming", send):
                return await original(event, generator, use_fallback)
            if self.skip_event(event):
                with origin_scope(None):
                    return await original(event, generator, use_fallback)
            source = event_origin(event)
            if source is None:
                return await original(event, generator, use_fallback)
            if getattr(event, "event_kind", "message") == "agent_wakeup":
                with origin_scope(source):
                    return await original(event, generator, use_fallback)
            channel = None
            if self.active(cls, "send_streaming", send):
                channel = self.event_channel(event)
            if (
                channel is None
                or self.skip_event(event)
                or not await self.enabled(channel)
            ):
                return await original(event, generator, use_fallback)
            token = self.scope.set(SendScope(asyncio.current_task(), channel.key))
            paths: list[Path] = []
            deliver_voice = event.send
            if event.get_platform_name() == "webchat":
                # WebChat records streamed attachments in its completion frame.
                deliver_voice = None
            transformed = convert_stream(
                generator, channel, self.convert, self.scope, paths, deliver_voice
            )
            try:
                with origin_scope(source):
                    async with aclosing(transformed):
                        return await original(event, transformed, use_fallback)
            finally:
                self.scope.reset(token)
                for path in paths:
                    self.cleanup(path)

        self.remember(cls, "send_streaming", original, send)

    @staticmethod
    def skip_event(event: AstrMessageEvent) -> bool:
        return bool(
            event.get_extra(SKIP_TTS_KEY) or event.get_extra("action_type") == "live"
        )

    @staticmethod
    def cleanup(path: Path | None) -> None:
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).warning("Failed to remove GSVI audio.")
