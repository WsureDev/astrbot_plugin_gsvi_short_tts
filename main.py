"""Independent, per-channel short-text TTS through the GSVI HTTP protocol."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import AsyncGenerator
from contextlib import suppress
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.message_components import Plain, Record
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
from pydantic import ValidationError

from .channels import Channel, resolve_channel
from .commands import parse_command
from .gsvi import GsviClient, GsviError
from .origins import CURRENT_TOOL, mark_request_origin
from .outgoing import SKIP_TTS_KEY, OutgoingTts
from .reply_gate import REQUEST_KEY, FinalReplyGate
from .settings import ChannelSettingsSchema, GsviSettingsSchema, VoiceChoicesCacheSchema

POLICY_KEY = "astrbot_plugin_gsvi_short_tts.policy"


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
        self.outgoing = OutgoingTts(
            context, self.convert_outgoing, self.delivery_enabled
        )
        self.voice_refresh_task: asyncio.Task[None] | None = None
        self.voice_choices: list[str] = []

    async def initialize(self) -> None:
        self.gate.install()
        self.outgoing.install()
        source = f"{self.client.base}/models/{self.settings.version}"
        cached = await self.get_kv_data("voice_choices", None)
        if cached is not None:
            try:
                entry = VoiceChoicesCacheSchema.model_validate(cached)
                if entry.source == source:
                    self.voice_choices = entry.voices
            except ValidationError:
                logger.warning("音色列表缓存格式不正确，将重新拉取。")
        self.show_voice_choices("正在刷新音色列表，稍后重新打开配置查看。")
        if self.config.get("refresh_voices", False):
            # Reset only here, before starting the background request. An old
            # refresh task must never overwrite configuration saved during reload.
            self.config["refresh_voices"] = False
            await self.config.save_config_async()
        self.voice_refresh_task = asyncio.create_task(self.refresh_voice_choices())

    async def terminate(self) -> None:
        self.gate.uninstall()
        self.outgoing.uninstall()
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
            logger.info(f"已刷新 GSVI 音色列表，共 {len(voices)} 个。")
        except Exception as error:
            reason = "音色列表刷新超时。"
            if isinstance(error, GsviError):
                reason = str(error)
            elif not isinstance(error, TimeoutError):
                reason = "音色列表刷新或缓存保存失败。"
            self.show_voice_choices(
                f"{reason}保留已有列表及当前选择；没有缓存时可手动填写。"
            )
            logger.warning(reason)

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
        event.set_extra(SKIP_TTS_KEY, True)
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
        self.outgoing.bind_event(event)
        event.set_extra(POLICY_KEY, None)
        if event.get_extra("action_type") == "live":
            logger.debug(
                f"[GSVI TTS] entry=prepare target={event.unified_msg_origin} "
                "reason=live_mode"
            )
            return
        try:
            policy, initialized = await self.load_channel(resolve_channel(event))
        except (ValueError, ValidationError):
            logger.debug(
                f"[GSVI TTS] entry=prepare target={event.unified_msg_origin} "
                "reason=invalid_channel_or_policy"
            )
            return
        logger.debug(
            f"[GSVI TTS] entry=prepare target={event.unified_msg_origin} "
            f"initialized={initialized} enabled={policy.enabled} "
            f"max_chars={policy.max_chars} probability={policy.probability}"
        )
        if not policy.enabled or policy.probability == 0:
            return
        event.set_extra(POLICY_KEY, policy)

    @filter.on_llm_request()
    async def mark_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        mark_request_origin(event)
        logger.debug(
            f"[GSVI TTS] entry=request target={event.unified_msg_origin} "
            f"request={id(req)} task={id(asyncio.current_task())} reason=request_marked"
        )
        if event.get_extra(POLICY_KEY) is not None:
            event.set_extra(REQUEST_KEY, req)

    async def convert_outgoing(
        self, channel: Channel, chain: MessageChain
    ) -> tuple[MessageChain, Path | None]:
        if chain.type in {
            "tool_call",
            "tool_call_result",
            "agent_stats",
            "reasoning",
            "break",
        }:
            logger.debug(f"[GSVI TTS] target={channel.key} reason=control_frame")
            return chain, None
        try:
            policy, initialized = await self.load_channel(channel)
        except (ValueError, ValidationError):
            logger.warning(f"[GSVI TTS] target={channel.key} reason=invalid_policy")
            return chain, None
        logger.debug(
            f"[GSVI TTS] target={channel.key} initialized={initialized} "
            f"enabled={policy.enabled} max_chars={policy.max_chars} "
            f"probability={policy.probability}"
        )
        if not policy.enabled or policy.probability == 0:
            logger.debug(
                f"[GSVI TTS] target={channel.key} reason=disabled_or_zero_probability"
            )
            return chain, None
        return await self.convert_chain(chain, policy)

    async def delivery_enabled(self, channel: Channel) -> bool:
        try:
            policy, _ = await self.load_channel(channel)
        except (ValueError, ValidationError):
            logger.warning(f"[GSVI TTS] target={channel.key} reason=invalid_policy")
            return False
        enabled = policy.enabled and policy.probability > 0
        if not enabled:
            logger.debug(
                f"[GSVI TTS] target={channel.key} reason=disabled_or_zero_probability"
            )
        return enabled

    async def convert_chain(
        self, chain: MessageChain, policy: ChannelSettingsSchema
    ) -> tuple[MessageChain, Path | None]:
        if not all(isinstance(component, Plain) for component in chain.chain):
            logger.debug("[GSVI TTS] reason=mixed_content")
            return chain, None
        text = "".join(component.text for component in chain.chain).strip()
        if not text or len(text) > policy.max_chars:
            logger.debug(
                f"[GSVI TTS] reason=empty_or_over_limit chars={len(text)} "
                f"max_chars={policy.max_chars}"
            )
            return chain, None
        if secrets.randbelow(100) >= policy.probability:
            logger.debug("[GSVI TTS] reason=probability_miss")
            return chain, None
        timeout = self.settings.timeout_ms / 1000
        tool = CURRENT_TOOL.get()
        if tool is not None:
            timeout = min(timeout, 10)
            if tool.active:
                timeout = min(
                    timeout, tool.deadline - asyncio.get_running_loop().time()
                )
        if timeout <= 0:
            logger.debug("[GSVI TTS] reason=tool_budget_exhausted fallback=text")
            return chain, None
        try:
            # One budget covers synthesis, download and writing the WAV.
            async with asyncio.timeout(timeout):
                path = await self.client.synthesize(text, policy.voice, self.directory)
            logger.debug(f"[GSVI TTS] reason=converted chars={len(text)}")
            record = Record(file=str(path), url=str(path), text=text)
            return chain.derive(chain=[record]), path
        except (GsviError, OSError, TimeoutError) as error:
            detail = "音频文件写入失败。"
            if isinstance(error, GsviError):
                detail = str(error)
            elif isinstance(error, TimeoutError):
                detail = "合成、下载或文件写入的总预算耗尽。"
            logger.warning(
                f"[GSVI TTS] reason=synthesis_failed error={type(error).__name__} "
                f"fallback=text detail={detail}"
            )
            return chain, None
