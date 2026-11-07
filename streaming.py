"""Convert complete presentation messages without changing model streaming."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from contextvars import ContextVar
from pathlib import Path
from typing import TypeVar

from astrbot.api.message_components import Plain
from astrbot.core.message.message_event_result import MessageChain

from .channels import Channel
from .origins import origin_scope

Scope = TypeVar("Scope")


async def convert_stream(
    generator: AsyncGenerator[MessageChain, None],
    channel: Channel,
    convert: Callable[
        [Channel, MessageChain], Awaitable[tuple[MessageChain, Path | None]]
    ],
    scope: ContextVar[Scope | None],
    paths: list[Path],
) -> AsyncGenerator[MessageChain, None]:
    frames: list[MessageChain] = []

    async def flush() -> AsyncGenerator[MessageChain, None]:
        if not frames:
            return
        pending = list(frames)
        frames.clear()
        if all(isinstance(c, Plain) for frame in pending for c in frame.chain):
            message = pending[0].derive(
                chain=[component for frame in pending for component in frame.chain]
            )
            converted, path = await convert(channel, message)
            if path is not None:
                paths.append(path)
                yield converted
                return
        for frame in pending:
            yield frame

    async with aclosing(generator):
        while True:
            # Input iteration runs the agent and tools. Do not lend the
            # presentation send's recursion guard to those independent sends.
            token = scope.set(None)
            try:
                with origin_scope(None):
                    frame = await anext(generator)
            except StopAsyncIteration:
                break
            finally:
                scope.reset(token)
            if frame.type is not None:
                async for message in flush():
                    yield message
                yield frame
            else:
                frames.append(frame)
        async for message in flush():
            yield message
