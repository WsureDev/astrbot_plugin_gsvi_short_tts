"""Apply destination settings to proactive sends without an incoming event."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from pathlib import Path
from typing import TypeAlias

from astrbot.api.star import Context
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.message_session import MessageSession

Send: TypeAlias = Callable[
    [Context, str | MessageSession, MessageChain], Awaitable[bool]
]
Convert: TypeAlias = Callable[
    [MessageSession, MessageChain], Awaitable[tuple[MessageChain, Path | None]]
]


class OutgoingTts:
    """Reversibly wrap the common plugin/tool proactive delivery entry point."""

    def __init__(self, context: Context, convert: Convert) -> None:
        self.context = context
        self.convert = convert
        self.original: Send | None = None
        self.installed: Send | None = None

    def install(self) -> None:
        if self.installed is not None:
            return
        original = Context.send_message
        self.original = original

        @wraps(original)
        async def send(
            context: Context,
            session: str | MessageSession,
            message_chain: MessageChain,
        ) -> bool:
            if self.installed is not send or context is not self.context:
                return await original(context, session, message_chain)
            target = session
            if isinstance(target, str):
                try:
                    target = MessageSession.from_str(target)
                except ValueError:
                    # Preserve the core's validation and error behavior.
                    return await original(context, session, message_chain)
            path = None
            try:
                converted, path = await self.convert(target, message_chain)
                # Preserve the caller's chain for tool bookkeeping and history.
                return await original(context, session, converted)
            finally:
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        logging.getLogger(__name__).warning(
                            "Failed to remove a proactive GSVI audio file."
                        )

        self.installed = send
        Context.send_message = send

    def uninstall(self) -> None:
        if self.installed is Context.send_message and self.original is not None:
            Context.send_message = self.original
        self.installed = None
        self.original = None
