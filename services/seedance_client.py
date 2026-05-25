"""Seedance/Ark video generation client."""

from __future__ import annotations

import json
import mimetypes
import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union
from urllib.parse import urljoin

from dotenv import dotenv_values, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

try:
    import requests
except ImportError as import_error:  # pragma: no cover - depends on local env
    requests = None
    _REQUESTS_IMPORT_ERROR = import_error
else:
    _REQUESTS_IMPORT_ERROR = None


JsonDict = Dict[str, Any]
SEEDANCE_MODELS = {
    "avatar_fast": "doubao-seedance-1-0-pro-fast-251015",
    "t2v_2_0": "doubao-seedance-2-0-260128",
    "t2v_2_0_fast": "doubao-seedance-2-0-fast-260128",
    "i2v_2_0": "doubao-seedance-2-0-260128",
    "i2v_2_0_fast": "doubao-seedance-2-0-fast-260128",
}
DEFAULT_SEEDANCE_MODEL = SEEDANCE_MODELS["avatar_fast"]
DEFAULT_ARK_API_BASE = "https://ark.cn-beijing.volces.com"
SEEDANCE_PRIVACY_ERROR_CODE = "InputImageSensitiveContentDetected.PrivacyInformation"
SEEDANCE_PRIVACY_ERROR_TYPE = "privacy_person_image"
SEEDANCE_PRIVACY_SUGGESTED_ACTION = "use_avatar_fast_or_virtual_t2v_or_authorized_asset"
SEEDANCE_PRIVACY_USER_MESSAGE = (
    "Seedance 2.0 image_to_video 已触发真人图片隐私风控。当前图片可能包含真人头像。"
    "建议使用 Seedance fast 保留头像模式，或改用虚拟讲解人像模式，或使用官方授权素材后再试。"
)


def is_seedance_privacy_error_code(error_code: Optional[str]) -> bool:
    return str(error_code or "") == SEEDANCE_PRIVACY_ERROR_CODE


def seedance_error_metadata(error_code: Optional[str]) -> JsonDict:
    if is_seedance_privacy_error_code(error_code):
        return {
            "seedance_error_type": SEEDANCE_PRIVACY_ERROR_TYPE,
            "suggested_action": SEEDANCE_PRIVACY_SUGGESTED_ACTION,
            "privacy_guard_triggered": True,
            "user_message": SEEDANCE_PRIVACY_USER_MESSAGE,
        }
    return {
        "seedance_error_type": None,
        "suggested_action": None,
        "privacy_guard_triggered": False,
        "user_message": None,
    }


def seedance_display_message(error_code: Optional[str], fallback_message: str) -> str:
    if is_seedance_privacy_error_code(error_code):
        return SEEDANCE_PRIVACY_USER_MESSAGE
    return fallback_message


def get_seedance_env_status() -> JsonDict:
    file_values = dotenv_values(PROJECT_ROOT / ".env")
    values = {
        "SEEDANCE_API_KEY": (file_values.get("SEEDANCE_API_KEY") or os.getenv("SEEDANCE_API_KEY") or "").strip(),
        "ARK_API_KEY": (file_values.get("ARK_API_KEY") or os.getenv("ARK_API_KEY") or "").strip(),
        "VOLCENGINE_API_KEY": (file_values.get("VOLCENGINE_API_KEY") or "").strip(),
    }
    selected_source = None
    selected_key = ""
    for name in ("SEEDANCE_API_KEY", "ARK_API_KEY", "VOLCENGINE_API_KEY"):
        if values[name]:
            selected_source = name
            selected_key = values[name]
            break
    return {
        "has_seedance_api_key": bool(values["SEEDANCE_API_KEY"]),
        "has_ark_api_key": bool(values["ARK_API_KEY"]),
        "has_volcengine_api_key": bool(values["VOLCENGINE_API_KEY"]),
        "selected_key_source": selected_source,
        "key_length": len(selected_key),
        "api_base": (file_values.get("SEEDANCE_API_BASE") or os.getenv("SEEDANCE_API_BASE") or DEFAULT_ARK_API_BASE).strip(),
        "model": (file_values.get("SEEDANCE_MODEL") or os.getenv("SEEDANCE_MODEL") or DEFAULT_SEEDANCE_MODEL).strip(),
    }


def get_seedance_api_key() -> str:
    for name in ("SEEDANCE_API_KEY", "ARK_API_KEY", "VOLCENGINE_API_KEY"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


class SeedanceClientError(Exception):
    """Base error that can be serialized into the probe result JSON."""

    def __init__(
        self,
        message: str,
        *,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
        response_json: Optional[JsonDict] = None,
        request_payload: Optional[JsonDict] = None,
        request_headers: Optional[JsonDict] = None,
        stage: Optional[str] = None,
        model: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_text = response_text
        self.response_json = response_json
        self.request_payload = request_payload
        self.request_headers = request_headers
        self.stage = stage
        self.model = model
        self.error_code = error_code
        self.error_message = error_message or message

    def to_result_error(self) -> JsonDict:
        metadata = seedance_error_metadata(self.error_code)
        return {
            "success": False,
            "stage": self.stage,
            "status_code": self.status_code,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "display_message": metadata.get("user_message") or self.error_message,
            "seedance_error_type": metadata.get("seedance_error_type"),
            "suggested_action": metadata.get("suggested_action"),
            "privacy_guard_triggered": metadata.get("privacy_guard_triggered"),
            "response_text": self.response_text,
            "response_json": self.response_json,
            "request_payload_sanitized": self.request_payload,
            "request_headers_sanitized": self.request_headers,
            "endpoint": self.endpoint,
            "model": self.model,
            "message": self.message,
        }


class SeedanceConfigurationError(SeedanceClientError):
    """Raised when required Seedance configuration is missing."""


class SeedanceHTTPError(SeedanceClientError):
    """Raised when the Seedance API returns a non-success HTTP response."""


class SeedanceClient:
    def __init__(
        self,
        api_key: str = "",
        api_base: str = "",
        *,
        model: str = DEFAULT_SEEDANCE_MODEL,
        endpoint_generate: str = "",
        endpoint_status: str = "",
        endpoint_upload: str = "",
        timeout: int = 120,
        poll_interval_seconds: int = 5,
        max_poll_seconds: int = 300,
    ) -> None:
        self.api_key = (api_key or get_seedance_api_key()).strip()
        self.api_base = (api_base or os.getenv("SEEDANCE_API_BASE") or DEFAULT_ARK_API_BASE).strip().rstrip("/")
        self.model = (model or DEFAULT_SEEDANCE_MODEL).strip()
        self.endpoint_generate = (endpoint_generate or "").strip()
        self.endpoint_status = (endpoint_status or "").strip()
        self.endpoint_upload = (endpoint_upload or "").strip()
        self.timeout = timeout
        self.poll_interval_seconds = poll_interval_seconds
        self.max_poll_seconds = max_poll_seconds

        if not self.api_key:
            raise SeedanceConfigurationError(
                "Missing Seedance/Ark API key. Checked: SEEDANCE_API_KEY, ARK_API_KEY, VOLCENGINE_API_KEY.",
                request_payload=get_seedance_env_status(),
            )
        if not self.api_base:
            raise SeedanceConfigurationError("Missing SEEDANCE_API_BASE")

    def upload_file(self, file_path: Union[Path, str], *, purpose: str) -> JsonDict:
        """Prepare a local file for Seedance.

        If SEEDANCE_ENDPOINT_UPLOAD is configured, this calls that upload API.
        Otherwise, the file is represented as a data URL. Ark's video generation
        payload accepts URL-style media references; using data URLs lets this
        probe exercise local test assets without introducing a storage service.
        """

        path = Path(file_path)
        if not path.exists():
            raise SeedanceConfigurationError(
                f"File does not exist: {path}",
                request_payload={"file_path": str(path), "purpose": purpose},
            )

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        payload_summary = {
            "file_name": path.name,
            "file_size": path.stat().st_size,
            "content_type": content_type,
            "purpose": purpose,
        }

        if not self.endpoint_upload:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return {
                **payload_summary,
                "source": "data_url",
                "url": f"data:{content_type};base64,{encoded}",
            }

        endpoint = self.endpoint_upload

        with path.open("rb") as fh:
            files = {"file": (path.name, fh, content_type)}
            data = {"purpose": purpose}
            response = self._requests().post(
                self._url(endpoint),
                headers=self._auth_headers(),
                data=data,
                files=files,
                timeout=self.timeout,
            )

        return self._parse_response(
            response,
            endpoint=self._url(endpoint),
            request_payload=payload_summary,
            request_headers=self._auth_headers(),
            stage="upload_file",
        )

    def create_task(
        self,
        capability: str,
        *,
        prompt: Optional[str] = None,
        uploaded_assets: Optional[JsonDict] = None,
        duration: Optional[Union[int, float]] = None,
        resolution: Optional[str] = None,
        ratio: Optional[str] = None,
        fps: Optional[Union[int, float]] = None,
    ) -> JsonDict:
        endpoint = self._require_endpoint(
            self.endpoint_generate,
            "SEEDANCE_ENDPOINT_GENERATE",
        )
        payload = self.build_task_payload(
            capability,
            prompt=prompt,
            uploaded_assets=uploaded_assets or {},
            duration=duration,
            resolution=resolution,
            ratio=ratio,
            fps=fps,
        )
        return self._post_json(endpoint, payload, stage="create_task")

    def poll_task(self, task_id: str) -> JsonDict:
        endpoint = self._require_endpoint(self.endpoint_status, "SEEDANCE_ENDPOINT_STATUS")
        last_response: JsonDict = {}
        started = time.time()

        while True:
            status_endpoint = self._status_endpoint(endpoint, task_id)
            last_response = self._get_json(
                status_endpoint,
                request_payload={"task_id": task_id},
                stage="poll_task",
            )
            status = str(
                self._first_deep_get(
                    last_response,
                    (
                        ("status",),
                        ("state",),
                        ("task_status",),
                        ("data", "status"),
                        ("data", "state"),
                        ("data", "task_status"),
                    ),
                )
                or ""
            ).lower()
            if status in {"succeeded", "success", "completed", "done", "failed", "error", "cancelled"}:
                return last_response
            if time.time() - started >= self.max_poll_seconds:
                return {
                    "status": "timeout",
                    "task_id": task_id,
                    "last_response": last_response,
                }
            time.sleep(self.poll_interval_seconds)

    def download_result(self, result: JsonDict, output_path: Union[Path, str]) -> Optional[str]:
        output_url = self.extract_output_video_url(result)
        if not output_url:
            return None

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(self._download_bytes(output_url))
        return str(output)

    def build_task_payload(
        self,
        capability: str,
        *,
        prompt: Optional[str],
        uploaded_assets: JsonDict,
        duration: Optional[Union[int, float]] = None,
        resolution: Optional[str] = None,
        ratio: Optional[str] = None,
        fps: Optional[Union[int, float]] = None,
    ) -> JsonDict:
        content = []
        if capability == "text_to_video":
            content.append({"type": "text", "text": prompt or ""})
        elif capability == "image_to_video":
            content.append({"type": "text", "text": prompt or ""})
            content.append(self._media_content("image", uploaded_assets, "avatar", "first_frame"))
        elif capability == "image_audio_to_talking_head":
            content.append({
                "type": "text",
                "text": (
                    "Use image 1 as a presenter reference and audio 1 as the speech track. "
                    "Generate a talking-head video with natural lip sync."
                ),
            })
            content.append(self._media_content("image", uploaded_assets, "avatar", "reference_image"))
            content.append(self._media_content("audio", uploaded_assets, "audio", "reference_audio"))
        elif capability == "video_audio_lipsync":
            content.append({
                "type": "text",
                "text": (
                    "Use video 1 as the source performance and audio 1 as the target speech. "
                    "Adjust the mouth movement to match the audio while preserving the person and scene."
                ),
            })
            content.append(self._media_content("video", uploaded_assets, "video", "reference_video"))
            content.append(self._media_content("audio", uploaded_assets, "audio", "reference_audio"))
        else:
            raise SeedanceConfigurationError(
                f"Unknown Seedance capability: {capability}",
                endpoint=self.endpoint_generate,
                request_payload={"capability": capability},
            )

        payload = {"model": self.model, "content": content}
        for key, value in {
            "duration": duration,
            "resolution": resolution,
            "ratio": ratio,
            "fps": fps,
        }.items():
            if value is not None:
                payload[key] = value
        return payload

    def extract_task_id(self, response_json: JsonDict) -> Optional[str]:
        for key in ("task_id", "id"):
            value = response_json.get(key)
            if value:
                return str(value)
        for path in (
            ("data", "task_id"),
            ("data", "id"),
            ("result", "task_id"),
            ("result", "id"),
        ):
            value = self._deep_get(response_json, path)
            if value:
                return str(value)
        return None

    def extract_output_video_url(self, response_json: JsonDict) -> Optional[str]:
        for path in (
            ("output_video",),
            ("video_url",),
            ("url",),
            ("content", "video_url"),
            ("data", "output_video"),
            ("data", "video_url"),
            ("data", "url"),
            ("data", "content", "video_url"),
            ("result", "output_video"),
            ("result", "video_url"),
            ("result", "url"),
            ("result", "content", "video_url"),
        ):
            value = self._deep_get(response_json, path)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value
        return None

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def _require_endpoint(self, endpoint: str, env_name: str) -> str:
        if not endpoint:
            raise SeedanceConfigurationError(f"Missing {env_name}", endpoint=endpoint)
        return endpoint

    def _url(self, endpoint: str) -> str:
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        base = self.api_base
        if "ark.cn-beijing.volces.com" in base and not base.rstrip("/").endswith("/api/v3"):
            base = f"{base.rstrip('/')}/api/v3"
        return urljoin(f"{base}/", endpoint.lstrip("/"))

    def _status_endpoint(self, endpoint: str, task_id: str) -> str:
        if "{task_id}" in endpoint:
            return endpoint.format(task_id=task_id)
        return f"{endpoint.rstrip('/')}/{task_id}"

    def _post_json(self, endpoint: str, payload: JsonDict, *, stage: str = "create_task") -> JsonDict:
        headers = {**self._auth_headers(), "Content-Type": "application/json"}
        if requests is not None:
            try:
                response = requests.post(
                    self._url(endpoint),
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as error:
                raise SeedanceHTTPError(
                    "Seedance API request failed",
                    endpoint=self._url(endpoint),
                    response_text=str(error),
                    request_payload=self.sanitize_payload(payload),
                    request_headers=self.sanitize_headers(headers),
                    stage=stage,
                    model=self.model,
                    error_message=str(error),
                )
            return self._parse_response(
                response,
                endpoint=self._url(endpoint),
                request_payload=payload,
                request_headers=headers,
                stage=stage,
            )
        data = json.dumps(payload).encode("utf-8")
        Request, _ = self._urllib()
        request = Request(
            self._url(endpoint),
            data=data,
            headers=headers,
            method="POST",
        )
        return self._urlopen_json(
            request,
            endpoint=self._url(endpoint),
            request_payload=payload,
            request_headers=headers,
            stage=stage,
        )

    def _get_json(self, endpoint: str, *, request_payload: JsonDict, stage: str = "poll_task") -> JsonDict:
        headers = self._auth_headers()
        if requests is not None:
            try:
                response = requests.get(
                    self._url(endpoint),
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as error:
                raise SeedanceHTTPError(
                    "Seedance API request failed",
                    endpoint=self._url(endpoint),
                    response_text=str(error),
                    request_payload=self.sanitize_payload(request_payload),
                    request_headers=self.sanitize_headers(headers),
                    stage=stage,
                    model=self.model,
                    error_message=str(error),
                )
            return self._parse_response(
                response,
                endpoint=self._url(endpoint),
                request_payload=request_payload,
                request_headers=headers,
                stage=stage,
            )
        Request, _ = self._urllib()
        request = Request(
            self._url(endpoint),
            headers=headers,
            method="GET",
        )
        return self._urlopen_json(
            request,
            endpoint=self._url(endpoint),
            request_payload=request_payload,
            request_headers=headers,
            stage=stage,
        )

    def _download_bytes(self, url: str) -> bytes:
        headers = {"Accept": "*/*"}
        if requests is not None:
            try:
                response = requests.get(url, timeout=self.timeout)
            except requests.RequestException as error:
                raise SeedanceHTTPError(
                    "Failed to download Seedance result",
                    endpoint=url,
                    response_text=str(error),
                    request_payload={"output_url": url},
                    request_headers=headers,
                    stage="download_result",
                    model=self.model,
                    error_message=str(error),
                )
            if response.status_code >= 400:
                parsed, error_code, error_message = self._extract_error(response.text)
                raise SeedanceHTTPError(
                    "Failed to download Seedance result",
                    endpoint=url,
                    status_code=response.status_code,
                    response_text=response.text,
                    response_json=parsed,
                    request_payload={"output_url": url},
                    request_headers=headers,
                    stage="download_result",
                    model=self.model,
                    error_code=error_code,
                    error_message=error_message or "Failed to download Seedance result",
                )
            return response.content
        Request, urlopen = self._urllib()
        HTTPError, URLError = self._urllib_errors()
        request = Request(url, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except HTTPError as error:
            response_text = error.read().decode("utf-8", errors="replace")
            raise SeedanceHTTPError(
                "Failed to download Seedance result",
                endpoint=url,
                status_code=error.code,
                response_text=response_text,
                response_json=self._parse_json_text(response_text),
                request_payload={"output_url": url},
                request_headers=headers,
                stage="download_result",
                model=self.model,
            )
        except URLError as error:
            raise SeedanceHTTPError(
                "Failed to download Seedance result",
                endpoint=url,
                response_text=str(error.reason),
                request_payload={"output_url": url},
                request_headers=headers,
                stage="download_result",
                model=self.model,
                error_message=str(error.reason),
            )

    def _urlopen_json(self, request: Any, *, endpoint: str, request_payload: JsonDict,
                      request_headers: JsonDict, stage: str) -> JsonDict:
        _, urlopen = self._urllib()
        HTTPError, URLError = self._urllib_errors()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_text = response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            response_text = error.read().decode("utf-8", errors="replace")
            parsed, error_code, error_message = self._extract_error(response_text)
            raise SeedanceHTTPError(
                seedance_display_message(error_code, "Seedance API returned an error"),
                endpoint=endpoint,
                status_code=error.code,
                response_text=response_text,
                response_json=parsed,
                request_payload=self.sanitize_payload(request_payload),
                request_headers=self.sanitize_headers(request_headers),
                stage=stage,
                model=self.model,
                error_code=error_code,
                error_message=error_message or seedance_display_message(error_code, "Seedance API returned an error"),
            )
        except URLError as error:
            raise SeedanceHTTPError(
                "Seedance API request failed",
                endpoint=endpoint,
                response_text=str(error.reason),
                request_payload=self.sanitize_payload(request_payload),
                request_headers=self.sanitize_headers(request_headers),
                stage=stage,
                model=self.model,
                error_message=str(error.reason),
            )

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            parsed = {"raw_response": response_text}
        if not isinstance(parsed, dict):
            parsed = {"raw_response": parsed}
        return parsed

    def _urllib(self) -> Any:
        try:
            from urllib.request import Request, urlopen
        except ImportError as error:
            raise SeedanceConfigurationError(
                "Python runtime is missing urllib.request; use a complete project Python environment.",
                request_payload={"import_error": str(error)},
            )
        return Request, urlopen

    def _urllib_errors(self) -> Any:
        try:
            from urllib.error import HTTPError, URLError
        except ImportError as error:
            raise SeedanceConfigurationError(
                "Python runtime is missing urllib.error; use a complete project Python environment.",
                request_payload={"import_error": str(error)},
            )
        return HTTPError, URLError

    def _parse_response(
        self,
        response: Any,
        *,
        endpoint: str,
        request_payload: JsonDict,
        request_headers: JsonDict,
        stage: str,
    ) -> JsonDict:
        response_text = response.text
        parsed = self._parse_json_text(response_text)
        if response.status_code >= 400:
            _, error_code, error_message = self._extract_error(response_text)
            raise SeedanceHTTPError(
                seedance_display_message(error_code, "Seedance API returned an error"),
                endpoint=endpoint,
                status_code=response.status_code,
                response_text=response_text,
                response_json=parsed,
                request_payload=self.sanitize_payload(request_payload),
                request_headers=self.sanitize_headers(request_headers),
                stage=stage,
                model=self.model,
                error_code=error_code,
                error_message=error_message or seedance_display_message(error_code, "Seedance API returned an error"),
            )

        if not isinstance(parsed, dict):
            parsed = {"raw_response": parsed}
        return parsed

    def _media_content(self, kind: str, uploaded_assets: JsonDict, asset_key: str, role: str) -> JsonDict:
        asset = uploaded_assets.get(asset_key)
        url = self._asset_url(asset)
        if not url:
            raise SeedanceConfigurationError(
                f"Uploaded asset does not include a URL: {asset_key}",
                endpoint=self.endpoint_generate,
                request_payload={asset_key: self.sanitize_payload(asset or {})},
            )

        field_by_kind = {
            "image": "image_url",
            "audio": "audio_url",
            "video": "video_url",
        }
        field = field_by_kind[kind]
        return {
            "type": field,
            field: {"url": url},
            "role": role,
        }

    def _asset_url(self, asset: Any) -> Optional[str]:
        if not isinstance(asset, dict):
            return None
        for path in (
            ("url",),
            ("uri",),
            ("data", "url"),
            ("data", "uri"),
            ("result", "url"),
            ("result", "uri"),
        ):
            value = self._deep_get(asset, path)
            if isinstance(value, str) and value:
                return value
        asset_id = self._first_deep_get(asset, (("asset_id",), ("data", "asset_id"), ("result", "asset_id")))
        if asset_id:
            return f"asset://{asset_id}"
        return None

    def sanitize_headers(self, headers: Any) -> Any:
        if not isinstance(headers, dict):
            return {}
        sanitized = {}
        for key, value in headers.items():
            if key.lower() == "authorization":
                sanitized[key] = "Bearer ***" if value else value
            else:
                sanitized[key] = value
        return sanitized

    def sanitize_payload(self, value: Any) -> Any:
        if isinstance(value, dict):
            sanitized = {}
            for key, item in value.items():
                if key in {"image_url", "audio_url", "video_url"} and isinstance(item, dict):
                    url = item.get("url", "")
                    sanitized[key] = {
                        "has_url": bool(url),
                        "is_public_url": isinstance(url, str) and url.startswith(("http://", "https://")),
                        "is_data_url": isinstance(url, str) and url.startswith("data:"),
                        "url_summary": self._summarize_url(url),
                    }
                elif key == "content" and isinstance(item, list):
                    sanitized[key] = [self.sanitize_payload(content_item) for content_item in item]
                elif key in {"model", "type", "role", "duration", "resolution", "ratio", "fps", "text"}:
                    sanitized[key] = self.sanitize_payload(item)
                else:
                    sanitized[key] = self.sanitize_payload(item)
            return sanitized
        if isinstance(value, list):
            return [self.sanitize_payload(item) for item in value]
        if isinstance(value, str):
            if value.startswith("data:") and ";base64," in value:
                prefix = value.split(";base64,", 1)[0]
                return f"{prefix};base64,<omitted {len(value)} chars>"
            if len(value) > 1000:
                return f"<omitted {len(value)} chars>"
        return value

    def _summarize_url(self, url: Any) -> Any:
        if not isinstance(url, str):
            return None
        if url.startswith("data:") and ";base64," in url:
            prefix = url.split(";base64,", 1)[0]
            return f"{prefix};base64,<omitted {len(url)} chars>"
        if len(url) > 180:
            return f"{url[:120]}...<omitted {len(url) - 120} chars>"
        return url

    def _parse_json_text(self, text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return {"raw_response": text}

    def _extract_error(self, response_text: str):
        parsed = self._parse_json_text(response_text)
        error_code = None
        error_message = None
        if isinstance(parsed, dict):
            for path in (
                ("error", "code"),
                ("error_code",),
                ("code",),
                ("data", "error", "code"),
                ("data", "error_code"),
            ):
                value = self._deep_get(parsed, path)
                if value:
                    error_code = str(value)
                    break
            for path in (
                ("error", "message"),
                ("error", "msg"),
                ("message",),
                ("msg",),
                ("error_message",),
                ("data", "error", "message"),
                ("data", "message"),
            ):
                value = self._deep_get(parsed, path)
                if value:
                    error_message = str(value)
                    break
        return parsed, error_code, error_message

    def _requests(self) -> Any:
        if requests is None:
            raise SeedanceConfigurationError(
                "Missing dependency: requests. Install project requirements before calling Seedance APIs.",
                request_payload={"missing_dependency": "requests"},
            )
        return requests

    def _deep_get(self, data: JsonDict, path: Iterable[str]) -> Any:
        current: Any = data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _first_deep_get(self, data: JsonDict, paths: Iterable[Iterable[str]]) -> Any:
        for path in paths:
            value = self._deep_get(data, path)
            if value is not None:
                return value
        return None
