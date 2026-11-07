"""Call only GSVI endpoints and download their returned audio URLs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import aiohttp
from astrbot.api import logger
from multidict import CIMultiDict
from pydantic import ValidationError
from yarl import URL

from .settings import GsviSettingsSchema, ModelsResponseSchema, SynthesisResponseSchema


class GsviError(Exception):
    """Report request failures without exposing credentials or response bodies."""


class GsviClient:
    def __init__(self, settings: GsviSettingsSchema) -> None:
        self.settings = settings
        self.base = URL(str(settings.api_base).rstrip("/"))
        self.session: aiohttp.ClientSession | None = None
        self.writes: set[asyncio.Task[int]] = set()

    async def close(self) -> None:
        if self.writes:
            await asyncio.shield(asyncio.gather(*self.writes, return_exceptions=True))
        if self.session is not None:
            await self.session.close()
            self.session = None

    def service_headers(self) -> CIMultiDict[str]:
        """Merge header names case-insensitively; explicit values take priority."""
        headers: CIMultiDict[str] = CIMultiDict()
        api_key = self.settings.api_key.get_secret_value()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(self.settings.request_headers)
        return headers

    def headers_for(self, url: URL, download_origin: URL | None) -> CIMultiDict[str]:
        headers: CIMultiDict[str] = CIMultiDict()
        if url.origin() == self.base.origin():
            headers.update(self.service_headers())
        if download_origin is not None and url.origin() == download_origin:
            headers.update(self.settings.download_headers)
        return headers

    async def request(
        self,
        method: str,
        url: URL,
        *,
        download: bool = False,
        payload: dict[str, str] | None = None,
    ) -> bytes:
        """Follow redirects while recomputing the headers for each destination."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.settings.timeout_ms / 1000)
            )
        download_origin = None
        if download:
            download_origin = url.origin()
        try:
            async with asyncio.timeout(self.settings.timeout_ms / 1000):
                for _ in range(6):
                    headers = self.headers_for(url, download_origin)
                    if payload is not None:
                        headers["Content-Type"] = "application/json"
                    async with self.session.request(
                        method,
                        url,
                        json=payload,
                        headers=headers,
                        allow_redirects=False,
                    ) as response:
                        if response.status in {301, 302, 303, 307, 308}:
                            location = response.headers.get("Location")
                            if not location:
                                raise GsviError("GSVI 重定向缺少 Location。")
                            url = url.join(URL(location))
                            if url.scheme not in {"http", "https"}:
                                raise GsviError("GSVI 返回了不支持的下载地址。")
                            if response.status == 303 or (
                                method == "POST" and response.status in {301, 302}
                            ):
                                method = "GET"
                                payload = None
                            continue
                        if response.status < 200 or response.status >= 300:
                            raise GsviError(f"GSVI 请求失败，HTTP {response.status}。")
                        return await response.read()
            raise GsviError("GSVI 重定向次数过多。")
        except (aiohttp.ClientError, TimeoutError, ValueError) as error:
            raise GsviError("GSVI 连接失败或请求超时。") from error

    async def list_models(self) -> dict[str, dict[str, list[str]]]:
        try:
            content = await self.request(
                "GET", URL(f"{self.base}/models/{self.settings.version}")
            )
            return ModelsResponseSchema.model_validate_json(content).models
        except ValidationError as error:
            raise GsviError("GSVI 音色列表格式不正确。") from error

    async def validate_voice(self, voice: str) -> None:
        models = await self.list_models()
        if voice not in models:
            sample = "、".join(list(models)[:10])
            raise GsviError(f"音色不存在：{voice}。可用角色示例：{sample}")
        if not any(
            self.settings.emotion in emotions for emotions in models[voice].values()
        ):
            raise GsviError(f"音色 {voice} 不支持情绪预设 {self.settings.emotion}。")

    async def synthesize(self, text: str, voice: str, directory: Path) -> Path:
        """Download a WAV before returning its event-scoped temporary filename."""
        payload = {
            "dl_url": str(self.base),
            "version": self.settings.version,
            "model_name": voice,
            "text": text,
            "text_lang": self.settings.text_lang,
            "emotion": self.settings.emotion,
            "media_type": "wav",
        }
        if self.settings.prompt_text_lang:
            payload["prompt_text_lang"] = self.settings.prompt_text_lang
        content = await self.request(
            "POST", URL(f"{self.base}/infer_single"), payload=payload
        )
        try:
            result = SynthesisResponseSchema.model_validate_json(content)
        except ValidationError as error:
            raise GsviError("GSVI 合成响应格式不正确。") from error
        if result.msg != "合成成功":
            raise GsviError("GSVI 未返回合成成功状态。")
        try:
            audio_url = URL(f"{self.base}/").join(URL(result.audio_url))
            if audio_url.scheme not in {"http", "https"} or not audio_url.host:
                raise ValueError("Unsupported audio URL")
            # Validate the authority/port before entering the download request.
            audio_url.origin()
            _ = audio_url.port
        except ValueError as error:
            raise GsviError("GSVI 返回了无效的下载地址。") from error
        audio = await self.request("GET", audio_url, download=True)
        if len(audio) < 12 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
            raise GsviError("GSVI 未返回有效的 WAV 音频。")
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        path = directory / f"gsvi_short_tts_{uuid4().hex}.wav"
        write = asyncio.create_task(asyncio.to_thread(path.write_bytes, audio))
        self.writes.add(write)
        write.add_done_callback(self.writes.discard)
        try:
            # Cancelling the waiter must not cancel the task tracking the thread.
            await asyncio.shield(write)
        except asyncio.CancelledError:
            # The thread can still create the file after this coroutine returns.
            # Delete only after it has really finished, retaining its task above.
            write.add_done_callback(lambda task: self.cleanup_write(task, path))
            raise
        except OSError:
            path.unlink(missing_ok=True)
            raise
        return path

    @staticmethod
    def cleanup_write(write: asyncio.Task[int], path: Path) -> None:
        try:
            write.result()
        except (OSError, asyncio.CancelledError):
            logger.warning("[GSVI TTS] reason=cancelled_write_failed")
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("[GSVI TTS] reason=audio_cleanup_failed")
