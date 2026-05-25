"""LLM JSON client with safe diagnostics.

This module never prints real API keys. It supports OpenAI-compatible chat
completion endpoints such as Volcengine Ark.
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

_client_cache = {}


def _env_value(name):
    return (os.getenv(name) or "").strip()


def _select_api_key():
    for name in ("LLM_API_KEY", "ARK_API_KEY", "VOLCENGINE_API_KEY"):
        value = _env_value(name)
        if value:
            return value, name
    return "", "none"


def _looks_like_seedance_model(model):
    return "seedance" in (model or "").lower()


def get_llm_env_status():
    api_key, source = _select_api_key()
    base_url = _env_value("LLM_BASE_URL")
    model = _env_value("LLM_MODEL")
    return {
        "has_llm_base_url": bool(base_url),
        "llm_base_url_exists": bool(base_url),
        "has_llm_api_key": bool(_env_value("LLM_API_KEY")),
        "has_ark_api_key": bool(_env_value("ARK_API_KEY")),
        "has_volcengine_api_key": bool(_env_value("VOLCENGINE_API_KEY")),
        "llm_api_key_exists": bool(api_key),
        "has_llm_model": bool(model),
        "llm_model_exists": bool(model),
        "selected_key_source": source,
        "key_length": len(api_key) if api_key else 0,
        "llm_model": model,
        "llm_model_looks_like_seedance": _looks_like_seedance_model(model),
    }


def _request_payload_sanitized(model, prompt):
    return {
        "model": model,
        "messages": [
            {"role": "system", "content_chars": 32},
            {"role": "user", "content_chars": len(prompt or "")},
        ],
        "temperature": 0.3,
    }


def _request_headers_sanitized():
    return {"Authorization": "Bearer ***"}


def _diagnostic(
    *,
    success,
    stage,
    error_type=None,
    error_message=None,
    model=None,
    prompt="",
    status_code=None,
    response_text="",
    raw_response="",
    exception_type=None,
):
    env_status = get_llm_env_status()
    payload = _request_payload_sanitized(model or env_status.get("llm_model") or "", prompt)
    data = {
        "success": bool(success),
        "stage": stage,
        "error_type": error_type,
        "error_message": error_message,
        "llm_base_url_exists": env_status["llm_base_url_exists"],
        "llm_api_key_exists": env_status["llm_api_key_exists"],
        "llm_model_exists": env_status["llm_model_exists"],
        "selected_key_source": env_status["selected_key_source"],
        "status_code": status_code,
        "response_text": response_text or "",
        "llm_raw_response": raw_response or "",
        "request_payload_sanitized": payload,
        "request_headers_sanitized": _request_headers_sanitized(),
        "model": model or env_status.get("llm_model") or "",
    }
    if exception_type:
        data["exception_type"] = exception_type
    return data


def _missing_env_diagnostic(stage, prompt):
    env_status = get_llm_env_status()
    missing = []
    if not env_status["llm_base_url_exists"]:
        missing.append("LLM_BASE_URL")
    if not env_status["llm_api_key_exists"]:
        missing.append("LLM_API_KEY/ARK_API_KEY/VOLCENGINE_API_KEY")
    if not env_status["llm_model_exists"]:
        missing.append("LLM_MODEL")
    if env_status["llm_model_looks_like_seedance"]:
        missing.append("LLM_MODEL must be a text/chat model, not a Seedance video model")
    message = "LLM 配置不完整：" + ", ".join(missing)
    return _diagnostic(
        success=False,
        stage=stage,
        error_type="missing_env",
        error_message=message,
        model=env_status.get("llm_model") or "",
        prompt=prompt,
    )


def _get_client(base_url, api_key):
    cache_key = (base_url, api_key[:8])
    if cache_key not in _client_cache:
        _client_cache[cache_key] = OpenAI(base_url=base_url, api_key=api_key)
    return _client_cache[cache_key]


# Robust JSON extraction
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _balanced_object(s, start_index):
    depth = 0
    in_str = False
    escape = False
    for i in range(start_index, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_json_object(text):
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    for fence_match in _FENCE_RE.finditer(s):
        inner = fence_match.group(1).strip()
        if not inner:
            continue
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            first = inner.find("{")
            if first != -1:
                last = _balanced_object(inner, first)
                if last != -1:
                    try:
                        obj = json.loads(inner[first:last + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
    i = 0
    while True:
        first = s.find("{", i)
        if first == -1:
            break
        last = _balanced_object(s, first)
        if last == -1:
            break
        try:
            obj = json.loads(s[first:last + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        i = first + 1
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last > first:
        try:
            obj = json.loads(s[first:last + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def call_llm_json(prompt, stage="llm"):
    """Call the configured OpenAI-compatible LLM and parse a JSON object."""
    env_status = get_llm_env_status()
    model = env_status.get("llm_model") or ""
    raw_response = ""
    if (
        not env_status["llm_base_url_exists"]
        or not env_status["llm_api_key_exists"]
        or not env_status["llm_model_exists"]
        or env_status["llm_model_looks_like_seedance"]
    ):
        diagnostic = _missing_env_diagnostic(stage, prompt)
        return {
            "result": None,
            "raw_response": "",
            "error": diagnostic["error_message"],
            "model": model,
            "success": False,
            "diagnostic": diagnostic,
            **diagnostic,
        }

    try:
        api_key, _source = _select_api_key()
        client = _get_client(_env_value("LLM_BASE_URL"), api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你只输出合法 JSON，不要输出任何其他内容。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        raw_response = response.choices[0].message.content or ""
    except Exception as error:
        status_code = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        response_text = ""
        if response is not None:
            response_text = getattr(response, "text", "") or str(response)
        if not response_text:
            response_text = getattr(error, "body", "") or ""
            if isinstance(response_text, (dict, list)):
                response_text = json.dumps(response_text, ensure_ascii=False)
        diagnostic = _diagnostic(
            success=False,
            stage=stage,
            error_type="request_error",
            error_message=f"{error.__class__.__name__}: {error}",
            model=model,
            prompt=prompt,
            status_code=status_code,
            response_text=response_text,
            exception_type=error.__class__.__name__,
        )
        return {
            "result": None,
            "raw_response": raw_response,
            "error": diagnostic["error_message"],
            "model": model,
            "success": False,
            "diagnostic": diagnostic,
            **diagnostic,
        }

    parsed = extract_json_object(raw_response)
    if isinstance(parsed, dict):
        diagnostic = _diagnostic(
            success=True,
            stage=stage,
            model=model,
            prompt=prompt,
            raw_response=raw_response,
        )
        return {
            "result": parsed,
            "raw_response": raw_response,
            "error": None,
            "model": model,
            "success": True,
            "diagnostic": diagnostic,
            **diagnostic,
        }

    diagnostic = _diagnostic(
        success=False,
        stage=stage,
        error_type="parse_error",
        error_message="LLM 未返回合法 JSON（解析失败）",
        model=model,
        prompt=prompt,
        raw_response=raw_response,
    )
    return {
        "result": None,
        "raw_response": raw_response,
        "error": diagnostic["error_message"],
        "model": model,
        "success": False,
        "diagnostic": diagnostic,
        **diagnostic,
    }
