"""Validate native plugin settings and per-channel speech policies."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, Json, SecretStr, StrictInt


class VoicePresetSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    voice: str = Field(default="花火", min_length=1)
    max_chars: StrictInt = Field(default=100, gt=0)
    probability: StrictInt = Field(default=100, ge=0, le=100)


class ChannelSettingsSchema(VoicePresetSchema):
    enabled: bool = False


class GsviSettingsSchema(BaseModel):
    model_config = ConfigDict(frozen=True, validate_default=True)

    api_base: HttpUrl = Field(default="http://127.0.0.1:8000")
    version: Literal["v1", "v2", "v4"] = "v4"
    api_key: SecretStr = Field(default="", repr=False)
    request_headers: Json[dict[str, str]] = Field(default="{}", repr=False)
    download_headers: Json[dict[str, str]] = Field(default="{}", repr=False)
    timeout_ms: StrictInt = Field(default=180000, ge=1000)
    text_lang: str = Field(default="中文", min_length=1)
    emotion: str = Field(default="默认", min_length=1)
    prompt_text_lang: str = ""
    refresh_voices: bool = False
    preset: VoicePresetSchema = Field(default_factory=VoicePresetSchema)


class VoiceChoicesCacheSchema(BaseModel):
    source: str
    voices: list[str]


class ModelsResponseSchema(BaseModel):
    models: dict[str, dict[str, list[str]]]


class SynthesisResponseSchema(BaseModel):
    msg: str
    audio_url: str = Field(min_length=1)
