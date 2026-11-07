"""Resolve stable channel identities without coupling groups to LLM isolation."""

from __future__ import annotations

from dataclasses import dataclass

from astrbot.api.event import AstrMessageEvent
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.message_type import MessageType


@dataclass(frozen=True)
class Channel:
    key: str
    label: str


def resolve_channel(event: AstrMessageEvent) -> Channel:
    """Build a separate key for each platform instance and chat target."""
    if event.get_platform_name() == "webchat":
        return Channel(key=event.unified_msg_origin, label="WebChat 会话")

    group_id = event.get_group_id()
    if group_id:
        session = MessageSession(
            platform_name=event.get_platform_id(),
            message_type=MessageType.GROUP_MESSAGE,
            session_id=str(group_id),
        )
        return Channel(key=str(session), label=f"群聊 {group_id}")

    if event.is_private_chat():
        sender_id = event.get_sender_id()
        session = MessageSession(
            platform_name=event.get_platform_id(),
            message_type=MessageType.FRIEND_MESSAGE,
            session_id=str(sender_id),
        )
        return Channel(key=str(session), label=f"私聊 {sender_id}")

    raise ValueError("该消息类型不支持渠道 TTS 设置。")
