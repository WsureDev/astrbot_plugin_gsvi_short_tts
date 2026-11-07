"""Deliver marked TTS requests only after their final-response hooks complete."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from functools import wraps
from typing import TypeAlias, TypeVar
from weakref import WeakKeyDictionary

from astrbot.api import logger
from astrbot.core.agent.response import AgentResponse, AgentResponseData
from astrbot.core.agent.runners.base import AgentState
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import ProviderRequest

from .origins import (
    CURRENT_TOOL,
    ToolDelivery,
    agent_origin,
    event_skip_reason,
    origin_scope,
)

REQUEST_KEY = "astrbot_plugin_gsvi_short_tts.request"
Runner: TypeAlias = ToolLoopAgentRunner[AstrAgentContext]
Step: TypeAlias = Callable[[Runner], AsyncGenerator[AgentResponse, None]]
ToolResult = TypeVar("ToolResult")


class FinalReplyGate:
    """Avoid early delivery in skills_like re-query and merged-reply branches."""

    def __init__(self) -> None:
        self.original_tools = None
        self.installed_tools = None
        self.original: Step | None = None
        self.installed: Step | None = None
        self.requests: WeakKeyDictionary[Runner, ProviderRequest] = WeakKeyDictionary()

    def install(self) -> None:
        if self.installed is not None:
            return
        original = ToolLoopAgentRunner.step
        self.original = original
        self.install_tool_scope()

        @wraps(original)
        async def gated_step(runner: Runner) -> AsyncGenerator[AgentResponse, None]:
            event = getattr(runner.run_context.context, "event", None)
            marked = event is not None and event.get_extra(REQUEST_KEY) is runner.req
            enabled = self.installed is gated_step and (
                marked or self.requests.get(runner) is runner.req
            )
            if enabled:
                self.requests[runner] = runner.req
            terminal_chain = None
            async with aclosing(original(runner)) as responses:
                while True:
                    try:
                        # Agent response/done hooks are not model output. Only
                        # the actual tool executor receives proactive provenance.
                        with origin_scope(None):
                            response = await anext(responses)
                    except StopAsyncIteration:
                        break
                    if (
                        response.type == "tool_call_result"
                        and response.data["chain"].type == "tool_direct_result"
                    ):
                        # Explicit user-facing tool output is not tool telemetry.
                        yield response
                        continue
                    if enabled and response.type == "llm_result":
                        candidate = response.data["chain"]
                        if candidate.type is None and candidate.chain:
                            terminal_chain = candidate
                    if enabled and response.type in {
                        "llm_result",
                        "streaming_delta",
                        "tool_call",
                        "tool_call_result",
                    }:
                        continue
                    yield response
            if not enabled or not runner.done() or runner.was_aborted():
                return
            final = runner.get_final_llm_resp()
            if final is None:
                if runner._state != AgentState.DONE or terminal_chain is None:
                    return
                chain = terminal_chain
            else:
                if final.role != "assistant":
                    return
                chain = final.result_chain
                if chain is None and final.completion_text:
                    chain = MessageChain().message(final.completion_text)
            if chain is None or not chain.chain:
                return
            chain = chain.derive(chain=list(chain.chain))
            event.set_extra("_llm_reasoning_content", None)
            yield AgentResponse(type="llm_result", data=AgentResponseData(chain=chain))
            if runner.streaming:
                yield AgentResponse(
                    type="streaming_delta", data=AgentResponseData(chain=chain)
                )

        self.installed = gated_step
        ToolLoopAgentRunner.step = gated_step

    def install_tool_scope(self) -> None:
        original = ToolLoopAgentRunner._iter_tool_executor_results
        self.original_tools = original

        @wraps(original)
        async def tool_results(
            runner: Runner, executor: AsyncGenerator[ToolResult, None]
        ) -> AsyncGenerator[ToolResult, None]:
            event = runner.run_context.context.event
            timeout = runner.run_context.tool_call_timeout
            # Leave room for the platform send and tool bookkeeping. This is
            # an upper bound, not permission to swallow external cancellation.
            deadline = asyncio.get_running_loop().time() + timeout * 0.75
            delivery = ToolDelivery(event, deadline)
            try:
                async with aclosing(original(runner, executor)) as results:
                    while True:
                        try:
                            source = None
                            reason = event_skip_reason(event)
                            if self.installed_tools is tool_results and reason is None:
                                source = agent_origin(event)
                            if reason is not None:
                                logger.debug(
                                    "[GSVI TTS] entry=tool "
                                    f"target={event.unified_msg_origin} "
                                    f"reason={reason}"
                                )
                            token = CURRENT_TOOL.set(delivery)
                            try:
                                with origin_scope(source):
                                    result = await anext(results)
                            finally:
                                CURRENT_TOOL.reset(token)
                        except StopAsyncIteration:
                            break
                        yield result
            finally:
                # Detached background sends retain provenance, but must not
                # inherit an expired foreground tool deadline indefinitely.
                delivery.active = False

        self.installed_tools = tool_results
        ToolLoopAgentRunner._iter_tool_executor_results = tool_results

    def uninstall(self) -> None:
        if (
            self.installed_tools is ToolLoopAgentRunner._iter_tool_executor_results
            and self.original_tools is not None
        ):
            ToolLoopAgentRunner._iter_tool_executor_results = self.original_tools
        self.original_tools = None
        self.installed_tools = None
        if self.installed is ToolLoopAgentRunner.step and self.original is not None:
            ToolLoopAgentRunner.step = self.original
        self.installed = None
        self.original = None
        self.requests.clear()
