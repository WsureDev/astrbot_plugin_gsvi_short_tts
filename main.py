"""Independent, per-channel short-text TTS through the GSVI HTTP protocol."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import AsyncGenerator
from contextlib import suppress
from pathlib import Path
from typing import cast

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.message_components import Plain, Record
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
from pydantic import ValidationError

from .channels import Channel, resolve_channel
from .commands import parse_command
from .gsvi import GsviClient, GsviError
from .reply_gate import REQUEST_KEY, FinalReplyGate
from .settings import ChannelSettingsSchema, GsviSettingsSchema, VoiceChoicesCacheSchema

POLICY_KEY = "astrbot_plugin_gsvi_short_tts.policy"
PROCESSED_KEY = "astrbot_plugin_gsvi_short_tts.processed_response"


class GsviShortTtsPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config
        try:
            self.settings = GsviSettingsSchema.model_validate(config)
        except ValidationError as error:
            fields = {
                ".".join(str(part) for part in item["loc"]) for item in error.errors()
            }
            raise ValueError(
                f"GSVI 插件配置不正确，请检查：{'、'.join(sorted(fields))}"
            ) from None
        self.client = GsviClient(self.settings)
        self.directory = Path(get_astrbot_temp_path()) / "gsvi_short_tts"
        self.gate = FinalReplyGate()
        self.voice_refresh_task: asyncio.Task[None] | None = None
        self.voice_choices: list[str] = []

    async def initialize(self) -> None:
        self.gate.install()
        source = f"{self.client.base}/models/{self.settings.version}"
        cached = await self.get_kv_data("voice_choices", None)
        if cached is not None:
            try:
                entry = VoiceChoicesCacheSchema.model_validate(cached)
                if entry.source == source:
                    self.voice_choices = entry.voices
            except ValidationError:
                self.logger.warning("音色列表缓存格式不正确，将重新拉取。")
        self.show_voice_choices("正在刷新音色列表，稍后重新打开配置查看。")
        if self.config.get("refresh_voices", False):
            # Reset only here, before starting the background request. An old
            # refresh task must never overwrite configuration saved during reload.
            self.config["refresh_voices"] = False
            await self.config.save_config_async()
        self.voice_refresh_task = asyncio.create_task(self.refresh_voice_choices())

    async def terminate(self) -> None:
        self.gate.uninstall()
        if self.voice_refresh_task is not None:
            self.voice_refresh_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.voice_refresh_task
            self.voice_refresh_task = None
        await self.client.close()

    def show_voice_choices(self, status: str) -> None:
        """Update only runtime schema metadata consumed by AstrBot's config UI."""
        metadata = self.config.schema["preset"]["items"]["voice"]
        if self.voice_choices:
            options = list(self.voice_choices)
            labels = list(options)
            selected = self.settings.preset.voice
            if selected not in options:
                options.append(selected)
                labels.append(f"{selected}（当前选择，服务列表未列出）")
            metadata["options"] = options
            metadata["labels"] = labels
        else:
            metadata.pop("options", None)
            metadata.pop("labels", None)
        metadata["hint"] = status

    async def refresh_voice_choices(self) -> None:
        """Refresh in the background and preserve cached options on failure."""
        try:
            async with asyncio.timeout(min(self.settings.timeout_ms, 10000) / 1000):
                models = await self.client.list_models()
            voices = sorted(models, key=str.casefold)
            if not voices:
                raise GsviError("GSVI 返回的音色列表为空。")
            entry = VoiceChoicesCacheSchema(
                source=f"{self.client.base}/models/{self.settings.version}",
                voices=voices,
            )
            await self.put_kv_data("voice_choices", entry.model_dump())
            self.voice_choices = voices
            self.show_voice_choices(
                f"已加载 {len(voices)} 个音色。重新打开配置可查看最新列表。"
            )
            self.logger.info("已刷新 GSVI 音色列表，共 %s 个。", len(voices))
        except Exception as error:
            reason = "音色列表刷新超时。"
            if isinstance(error, GsviError):
                reason = str(error)
            elif not isinstance(error, TimeoutError):
                reason = "音色列表刷新或缓存保存失败。"
            self.show_voice_choices(
                f"{reason}保留已有列表及当前选择；没有缓存时可手动填写。"
            )
            self.logger.warning("%s", reason)

    async def load_channel(
        self, channel: Channel
    ) -> tuple[ChannelSettingsSchema, bool]:
        stored = await self.get_kv_data(f"channel:{channel.key}", None)
        if stored is None:
            return ChannelSettingsSchema(**self.settings.preset.model_dump()), False
        return ChannelSettingsSchema.model_validate(stored), True

    @filter.command("usetts")
    async def usetts(
        self, event: AstrMessageEvent
    ) -> AsyncGenerator[MessageEventResult, None]:
        """Configure this group, private chat, or WebChat conversation."""
        # AstrBot's call_llm flag suppresses its automatic LLM path when true.
        event.should_call_llm(True)
        try:
            command = parse_command(event.message_str)
            if command.action == "list":
                models = await self.client.list_models()
                voices = sorted(models, key=str.casefold)
                message = "当前 GSVI 服务没有可用音色。"
                if voices:
                    message = (
                        f"可用音色（{len(voices)} 个，{self.settings.version}）：\n"
                        + "\n".join(voices)
                        + "\n\n使用：/usetts 音色名 字数 概率"
                    )
                yield event.plain_result(message)
                event.stop_event()
                return
            channel = resolve_channel(event)
            current, initialized = await self.load_channel(channel)
            if command.action != "status":
                if command.action == "configure":
                    if command.preset is None:
                        raise ValueError("缺少音色设置。")
                    await self.client.validate_voice(command.preset.voice)
                    current = ChannelSettingsSchema(
                        **command.preset.model_dump(), enabled=True
                    )
                else:
                    if command.action == "on" and not initialized:
                        await self.client.validate_voice(current.voice)
                    current = current.model_copy(
                        update={"enabled": command.action == "on"}
                    )
                await self.put_kv_data(f"channel:{channel.key}", current.model_dump())
                initialized = True
            state = "关闭"
            if current.enabled:
                state = "开启"
            source = "尚未初始化，展示全局预设"
            if initialized:
                source = "已保存的独立配置"
            message = (
                f"{channel.label} 的 TTS：{state}\n"
                f"音色：{current.voice}\n"
                f"字数界限：{current.max_chars}\n"
                f"触发概率：{current.probability}%\n"
                f"配置来源：{source}"
            )
        except ValidationError:
            message = "当前渠道配置不正确，请检查插件设置。"
        except (ValueError, GsviError) as error:
            message = str(error)
        yield event.plain_result(message)
        event.stop_event()

    @filter.event_message_type(filter.EventMessageType.ALL, priority=1000)
    async def prepare_reply(self, event: AstrMessageEvent) -> None:
        event.set_extra(POLICY_KEY, None)
        if event.get_extra("action_type") == "live":
            return
        try:
            policy, _ = await self.load_channel(resolve_channel(event))
        except (ValueError, ValidationError):
            return
        if not policy.enabled or policy.probability == 0:
            return
        event.set_extra(POLICY_KEY, policy)
        event.set_extra("enable_streaming", False)
        event.set_extra("enable_reasoning", False)

    @filter.on_llm_request()
    async def mark_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        if event.get_extra(POLICY_KEY) is not None:
            event.set_extra(REQUEST_KEY, req)

    @filter.on_llm_response(priority=-1000)
    async def convert_final_reply(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> None:
        policy = cast(ChannelSettingsSchema | None, event.get_extra(POLICY_KEY))
        if policy is None or resp.role != "assistant" or resp.tools_call_name:
            return
        if event.is_stopped() or event.get_extra("agent_stop_requested"):
            return
        stop_signal = event.get_extra("_background_stop_signal")
        if isinstance(stop_signal, asyncio.Event) and stop_signal.is_set():
            return
        if event.get_extra(PROCESSED_KEY) is resp:
            return
        event.set_extra(PROCESSED_KEY, resp)
        chain = resp.result_chain
        if chain is None:
            chain = MessageChain().message(resp.completion_text or "")
        if any(isinstance(component, Record) for component in chain.chain):
            return
        text = "".join(
            component.text for component in chain.chain if isinstance(component, Plain)
        ).strip()
        if not text or len(text) > policy.max_chars:
            return
        if secrets.randbelow(100) >= policy.probability:
            return
        try:
            path = await self.client.synthesize(text, policy.voice, self.directory)
            event.track_temporary_local_file(str(path))
            record = Record(file=str(path), url=str(path), text=text)
            components = []
            inserted = False
            for component in chain.chain:
                if isinstance(component, Plain):
                    if not inserted:
                        components.append(record)
                        inserted = True
                else:
                    components.append(component)
            resp.result_chain = chain.derive(chain=components)
        except (GsviError, OSError) as error:
            self.logger.warning("GSVI TTS 失败，保留原文字：%s", error)
