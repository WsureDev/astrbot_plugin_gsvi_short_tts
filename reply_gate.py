"""Deliver marked TTS requests only after their final-response hooks complete."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from functools import wraps
from typing import TypeAlias
from weakref import WeakKeyDictionary

from astrbot.api.event import AstrMessageEvent
from astrbot.core.agent.response import AgentResponse, AgentResponseData
from astrbot.core.agent.runners.base import AgentState
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import ProviderRequest

REQUEST_KEY = "astrbot_plugin_gsvi_short_tts.request"
Runner: TypeAlias = ToolLoopAgentRunner[AstrAgentContext]
Step: TypeAlias = Callable[[Runner], AsyncGenerator[AgentResponse, None]]
Finish: TypeAlias = Callable[[AstrMessageEvent, MessageChain], Awaitable[MessageChain]]


class FinalReplyGate:
    """Avoid early delivery in skills_like re-query and merged-reply branches."""

    def __init__(self, finish: Finish) -> None:
        self.finish = finish
        self.original: Step | None = None
        self.installed: Step | None = None
        self.requests: WeakKeyDictionary[Runner, ProviderRequest] = WeakKeyDictionary()

    def install(self) -> None:
        if self.installed is not None:
            return
        original = ToolLoopAgentRunner.step
        self.original = original

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
                async for response in responses:
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
                # No native final-response hook runs on a tool-ended turn.
                chain = await self.finish(event, terminal_chain)
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

    def uninstall(self) -> None:
        if self.installed is ToolLoopAgentRunner.step and self.original is not None:
            ToolLoopAgentRunner.step = self.original
        self.installed = None
        self.original = None
        self.requests.clear()
