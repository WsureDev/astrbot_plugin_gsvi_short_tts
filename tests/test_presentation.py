"""Offline whole-reply delivery regressions, optionally using real RespondStage."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import logging
import os
import random
import secrets
import sys
import tempfile
import unittest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("tts-presentation-test")


class ComponentType(str, Enum):
    Plain = "Plain"
    Record = "Record"
    Reply = "Reply"
    At = "At"


class Plain:
    type = ComponentType.Plain

    def __init__(self, text):
        self.text = text


class Record:
    type = ComponentType.Record

    def __init__(self, file, url=None, text=""):
        self.file, self.url, self.text = file, url, text


class At:
    type = ComponentType.At

    def __init__(self, qq):
        self.qq = qq


class MessageType(str, Enum):
    GROUP_MESSAGE = "GroupMessage"
    FRIEND_MESSAGE = "FriendMessage"


@dataclass
class MessageSession:
    platform_name: str
    message_type: MessageType
    session_id: str

    @property
    def platform_id(self):
        return self.platform_name

    def __str__(self):
        return f"{self.platform_id}:{self.message_type.value}:{self.session_id}"

    @classmethod
    def from_str(cls, value):
        platform, kind, target = value.split(":", 2)
        return cls(platform, MessageType(kind), target)


class ResultContentType(Enum):
    LLM_RESULT = 1
    STREAMING_RESULT = 2
    STREAMING_FINISH = 3


class MessageChain:
    def __init__(self, chain=None, type=None):
        self.chain = list(chain or [])
        self.type = type
        self.use_t2i_ = None
        self.use_markdown_ = None
        self.result_content_type = ResultContentType.LLM_RESULT
        self.async_stream = None

    def derive(self, chain=None):
        result = MessageChain(chain, self.type)
        result.use_t2i_, result.use_markdown_ = self.use_t2i_, self.use_markdown_
        return result

    def get_plain_text(self):
        return "".join(item.text for item in self.chain if isinstance(item, Plain))

    def is_model_result(self):
        return self.result_content_type == ResultContentType.LLM_RESULT


class Event:
    def __init__(self, context, result):
        self.context = context
        self.result = result
        self.extra = {}
        self.plugins_name = None
        self.session = MessageSession("test", MessageType.GROUP_MESSAGE, "123")
        self.unified_msg_origin = str(self.session)
        self._has_send_oper = False
        self.child = None

    def get_result(self):
        return self.result

    def clear_result(self):
        self.result = None

    def get_extra(self, key, default=None):
        return self.extra.get(key, default)

    def set_extra(self, key, value):
        self.extra[key] = value

    def get_platform_id(self):
        return "test"

    def get_platform_name(self):
        return "aiocqhttp"

    def get_group_id(self):
        return "123"

    def get_sender_id(self):
        return "42"

    def get_sender_name(self):
        return "offline"

    def _outline_chain(self, chain):
        return "offline"


class PlatformEvent(Event):
    async def send(self, message):
        if self.child is not None:
            callback = self.child
            self.child = None
            await asyncio.create_task(callback())
        await self.context.send_message(self.session, message)
        self._has_send_oper = True


class Platform:
    def __init__(self):
        self.sent = []

    def meta(self):
        return SimpleNamespace(name="aiocqhttp", id="test")

    async def send_by_session(self, session, message):
        for component in message.chain:
            if isinstance(component, Record):
                assert Path(component.file).is_file()
        self.sent.append(message)


class Context:
    def __init__(self):
        self.platform = Platform()
        self.platform_manager = SimpleNamespace(platform_insts=[self.platform])

    def get_platform_inst(self, platform_id):
        return self.platform

    def get_config(self, destination):
        return {"plugin_set": ["*"]}

    async def send_message(self, session, message):
        await self.platform.send_by_session(session, message)
        return True


class RespondStage:
    async def process(self, event):
        result = event.get_result()
        if result.get_plain_text() in event.get_extra(
            "_send_message_to_user_current_session_plain_texts", []
        ):
            return
        for component in result.chain:
            await event.send(result.derive([component]))
        event.clear_result()


async def call_event_hook(*args):
    return False


def framework_respond_stage():
    source = os.environ.get("ASTRBOT_SOURCE")
    if not source:
        return RespondStage
    root = Path(source)
    if (root / "astrbot").is_dir():
        root = root / "astrbot"
    path = root / "core/pipeline/respond/stage.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(
        node for node in tree.body if getattr(node, "name", None) == "RespondStage"
    )
    node.bases = []
    node.decorator_list = []
    components = {"Plain": Plain, "Record": Record, "At": At}
    for name in (
        "Reply",
        "Face",
        "Video",
        "Image",
        "Poke",
        "Node",
        "Nodes",
        "File",
        "Json",
        "Share",
        "Music",
        "Forward",
        "Location",
        "Contact",
        "Shake",
        "Dice",
        "RPS",
        "Unknown",
    ):
        components[name] = type(name, (), {})
    namespace = {
        "asyncio": asyncio,
        "Comp": SimpleNamespace(**components),
        "random": random,
        "ComponentType": ComponentType,
        "MessageChain": MessageChain,
        "ResultContentType": ResultContentType,
        "logger": LOGGER,
        "call_event_hook": call_event_hook,
        "EventType": SimpleNamespace(OnAfterMessageSentEvent=1),
    }
    tree.body = ast.parse("from __future__ import annotations").body + [node]
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace["RespondStage"]


Stage = framework_respond_stage()


def load_outgoing():
    exports = {
        "astrbot.api": {"logger": LOGGER},
        "astrbot.api.event": {"AstrMessageEvent": Event},
        "astrbot.api.message_components": {"Plain": Plain},
        "astrbot.api.star": {"Context": Context},
        "astrbot.core.message.message_event_result": {
            "MessageChain": MessageChain,
            "ResultContentType": ResultContentType,
        },
        "astrbot.core.pipeline.respond.stage": {"RespondStage": Stage},
        "astrbot.core.platform.message_session": {"MessageSession": MessageSession},
        "astrbot.core.platform.message_type": {"MessageType": MessageType},
    }
    modules = {}
    for name, values in exports.items():
        parts = name.split(".")
        for index in range(1, len(parts) + 1):
            key = ".".join(parts[:index])
            modules.setdefault(key, ModuleType(key))
        modules[name].__dict__.update(values)
    package = ModuleType("tts_presentation_under_test")
    package.__path__ = [str(ROOT)]
    sys.modules[package.__name__] = package
    name = package.__name__ + ".outgoing"
    spec = importlib.util.spec_from_file_location(name, ROOT / "outgoing.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
        origins = sys.modules[package.__name__ + ".origins"]
    return module, origins


OUTGOING, ORIGINS = load_outgoing()


class GsviError(Exception):
    pass


def converter_type():
    path = ROOT / "main.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(
        node
        for node in tree.body
        if getattr(node, "name", None) == "GsviShortTtsPlugin"
    )
    cls.bases = []
    cls.body = [
        node for node in cls.body if getattr(node, "name", None) == "convert_chain"
    ]
    tree.body = ast.parse("from __future__ import annotations").body + [cls]
    namespace = {
        "asyncio": asyncio,
        "secrets": secrets,
        "logger": LOGGER,
        "Plain": Plain,
        "Record": Record,
        "GsviError": GsviError,
        "CURRENT_TOOL": ORIGINS.CURRENT_TOOL,
    }
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace["GsviShortTtsPlugin"]


class PresentationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.context = Context()
        self.model = converter_type()()
        self.model.settings = SimpleNamespace(timeout_ms=1000)
        self.model.directory = Path(self.temp.name)
        self.model.client = SimpleNamespace(synthesize=self.synthesize)
        self.policy = SimpleNamespace(voice="offline", max_chars=120, probability=100)
        self.decisions = []
        self.syntheses = []
        self.synthesis_fails = False
        self.outgoing = OUTGOING.OutgoingTts(self.context, self.convert, self.enabled)
        self.outgoing.install()

    async def asyncTearDown(self):
        self.outgoing.uninstall()
        self.assertEqual(list(Path(self.temp.name).glob("*.wav")), [])
        self.temp.cleanup()

    async def synthesize(self, text, voice, directory):
        self.syntheses.append(text)
        if self.synthesis_fails:
            raise GsviError("offline synthesis failure")
        path = directory / (uuid4().hex + ".wav")
        path.write_bytes(b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 32)
        return path

    async def enabled(self, channel):
        return True

    async def convert(self, channel, chain):
        self.decisions.append(chain.get_plain_text())
        return await self.model.convert_chain(chain, self.policy)

    def event(self, texts):
        result = MessageChain([Plain(text) for text in texts])
        result.use_markdown_ = True
        event = PlatformEvent(self.context, result)
        ORIGINS.mark_request_origin(event)
        return event, result

    async def respond(self, event):
        stage = Stage()
        stage.platform_settings = {}
        stage.enable_seg = True
        stage.only_llm_result = True
        stage.interval_method = "random"
        stage.interval = [0, 0]
        await stage.process(event)

    async def test_short_reply_is_one_audio_and_keeps_original_model_text(self):
        event, result = self.event(["第一句。", "第二句。", "第三句。"])
        await self.respond(event)
        self.assertEqual(self.syntheses, ["第一句。第二句。第三句。"])
        self.assertEqual(len(self.context.platform.sent), 1)
        sent = self.context.platform.sent[0]
        self.assertIsInstance(sent.chain[0], Record)
        self.assertTrue(sent.use_markdown_)
        self.assertEqual(result.get_plain_text(), "第一句。第二句。第三句。")

    async def test_over_limit_reply_keeps_native_text_segments_without_tts(self):
        event, _ = self.event(["长" * 70, "文" * 70])
        await self.respond(event)
        self.assertEqual(self.syntheses, [])
        self.assertEqual(len(self.decisions), 1)
        self.assertEqual(len(self.context.platform.sent), 2)
        self.assertTrue(
            all(
                isinstance(chain.chain[0], Plain)
                for chain in self.context.platform.sent
            )
        )

    async def test_probability_is_sampled_once_for_the_whole_reply(self):
        self.policy.probability = 50
        event, _ = self.event(["第一句。", "第二句。"])
        with patch.object(secrets, "randbelow", return_value=99) as draw:
            await self.respond(event)
        self.assertEqual(draw.call_count, 1)
        self.assertEqual(self.syntheses, [])
        self.assertEqual(len(self.context.platform.sent), 2)

    async def test_synthesis_failure_keeps_text_without_retrying_each_fragment(self):
        self.synthesis_fails = True
        event, _ = self.event(["第一句。", "第二句。"])
        with self.assertLogs(LOGGER, "WARNING"):
            await self.respond(event)
        self.assertEqual(len(self.syntheses), 1)
        self.assertEqual(len(self.context.platform.sent), 2)

    async def test_fallback_does_not_reenter_adapter_conversion_under_tool_origin(self):
        event, _ = self.event(["长" * 70, "文" * 70])
        with ORIGINS.origin_scope("llm"):
            await self.respond(event)
        self.assertEqual(self.syntheses, [])
        self.assertEqual(len(self.decisions), 1)

    async def test_native_dedup_does_not_synthesize(self):
        event, result = self.event(["第一句。", "第二句。"])
        event.set_extra(
            "_send_message_to_user_current_session_plain_texts",
            [result.get_plain_text()],
        )
        await self.respond(event)
        self.assertEqual(self.decisions, [])
        self.assertEqual(self.context.platform.sent, [])

    async def test_mixed_reply_remains_ineligible_after_native_header_extraction(self):
        event, result = self.event(["第一句。", "第二句。"])
        result.chain.insert(0, At("42"))
        await self.respond(event)
        self.assertEqual(self.syntheses, [])
        components = [
            item for message in self.context.platform.sent for item in message.chain
        ]
        self.assertTrue(any(isinstance(item, At) for item in components))
        self.assertEqual(
            "".join(item.text for item in components if isinstance(item, Plain)),
            "第一句。第二句。",
        )

    async def test_explicit_separate_sends_are_still_independent(self):
        event, _ = self.event(["无关的模型结果"])
        with ORIGINS.origin_scope("llm"):
            await event.send(MessageChain([Plain("第一次主动发送")]))
            await event.send(MessageChain([Plain("第二次主动发送")]))
        self.assertEqual(len(self.syntheses), 2)
        self.assertEqual(len(self.context.platform.sent), 2)

    async def test_child_task_does_not_inherit_parent_presentation_decision(self):
        event, result = self.event(["第一句。", "第二句。"])

        async def independent():
            await event.send(result.derive([result.chain[1]]))

        event.child = independent
        await self.respond(event)
        self.assertEqual(self.syntheses, ["第一句。第二句。", "第二句。"])
        self.assertEqual(len(self.context.platform.sent), 2)


if __name__ == "__main__":
    unittest.main()
