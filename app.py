from pathlib import Path
import json
import shutil
import subprocess
import sys

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

from pipeline import (
    generate_clips_plan,
    generate_videos_from_plan,
    generate_videos_from_render_jobs,
    save_all_clips_changes,
    save_selected_render_plan,
)
from services.layout_engine import LayoutConfig, to_dict
from services.layout_preview import create_layout_preview
from services.subtitle_utils import format_seconds
from services.streamlit_compat import st_image_compat
from services.voice_style import list_voice_styles, DEFAULT_VOICE_STYLE
from services import knowledge_planner as kp
from services.llm_client import get_llm_env_status
from services.digital_human import (
    DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC,
    DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_2_0_EXPERIMENTAL,
    DIGITAL_HUMAN_MODE_SEEDANCE_T2V_VIRTUAL_2_0,
    DIGITAL_HUMAN_MODE_SEEDANCE_T2V_VIRTUAL_2_0_FAST,
    DIGITAL_HUMAN_MODE_STATIC,
)

st.set_page_config(page_title="数字人短视频生成器", layout="wide")
st.title("数字人录屏短视频自动生成")
st.caption(f"Streamlit version: {st.__version__}")

# ── session state ────────────────────────────────────────────
for key, default in [
    ("current_project_dir", None),
    ("current_clips_path", None),
    ("current_video_name", None),
    ("current_source_video_path", None),
    ("current_video_input_mode", None),
    ("current_source_video_size_bytes", None),
    ("current_avatar_name", None),
    ("plan_result", None),
    ("per_clip_previews", {}),
    ("video_result", None),
    ("selected_render_plan_path", None),  # set when user saves a render plan
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════
# 第 1 步 · 输入素材（文字来源）
# ══════════════════════════════════════════════════════════════
st.markdown("## 1️⃣ 输入素材")
st.caption("先选定转写文字来源；下面再上传视频、头像和（可选）声音样本。")

input_mode_label = st.radio("选择文字来源", [
    "自动从视频提取音频并转文字",
    "手动粘贴转写文字",
    "上传转写文本文件",
])

# ══════════════════════════════════════════════════════════════
# 第 2 步 · 切片策略（决定内容计划怎么生成）
# ══════════════════════════════════════════════════════════════
st.markdown("## 2️⃣ 切片策略")
clip_plan_mode = st.radio(
    "切片策略",
    ["知识点模块切片", "时间线连续切片"],
    index=0,
    help=(
        "知识点模块切片：先清洗 + 打标签，再按知识点聚类成 module，"
        "每个 module 可包含多个不连续 fragment，生成时分别裁剪并 concat。\n\n"
        "时间线连续切片：保留旧逻辑，整段按时间切 2-3 个连续短视频。"
    ),
)

# ══════════════════════════════════════════════════════════════
# 第 4 步 · 声音 / 克隆音色 / 数字人形象（默认折叠，按需展开调）
# ══════════════════════════════════════════════════════════════
# Variable defaults (in case the user never expands the section).
voice_sample_paths: list[str] = []
voice_name = "demo_voice_clone"
voice_sample_files = None
remove_background_noise = False
voice_consent = False
source_audio_clone_target_seconds = 90
audio_mode_label = "保留原视频原声"
force_regenerate_seedance_avatar = False
seedance_quality_mode = "fast"
digital_human_window_style = "card"
digital_human_video_mode = "preview_loop"
fallback_experimental_i2v_to_fast = False

st.markdown("## 4️⃣ 声音 · 数字人 · 字幕")
st.caption("可在生成视频前再调；默认折叠不打扰主流程。")

with st.expander("🎙️ 声音模式（保留原声 / 替换 / 克隆）", expanded=False):
    audio_mode_label = st.radio("声音模式", [
        "保留原视频原声",
        "使用已有 ElevenLabs voice_id 替换原声",
        "上传声音样本，通过 ElevenLabs API 云端克隆后替换原声",
        "从原视频原声自动提取样本并克隆后替换原声",
    ])

    if audio_mode_label.startswith("上传声音样本"):
        st.info(
            "请仅上传你本人或已获得授权的声音样本。"
            "未经授权克隆他人声音可能涉及侵权。本 Demo 仅用于授权范围内的内部测试。"
        )
        voice_name = st.text_input("克隆音色名称", value="demo_voice_clone")
        voice_sample_files = st.file_uploader(
            "上传声音样本（mp3/wav/m4a/aac/ogg），建议清晰单人声音，30秒~3分钟",
            type=["mp3", "wav", "m4a", "aac", "ogg"],
            accept_multiple_files=True,
        )
        remove_background_noise = st.checkbox("创建克隆音色时尝试去除背景噪音", value=False)
        voice_consent = st.checkbox(
            "我确认已获得该声音所有者授权，并同意仅用于本次 Demo 视频生成",
            value=False,
        )
    elif audio_mode_label.startswith("从原视频原声"):
        st.info(
            "系统会从原视频中自动抽取清晰单人讲解声音片段，用于创建 ElevenLabs 克隆音色。"
            "生成视频时，克隆音将朗读清洗后的 voice_script，而不是原始口头转写。"
        )
        voice_name = st.text_input("克隆音色名称", value="source_video_voice_clone")
        source_audio_clone_target_seconds = st.slider("样本目标总时长（秒）", 45, 180, 90, 5)
        remove_background_noise = st.checkbox("创建克隆音色时尝试去除背景噪音", value=True)
        voice_consent = st.checkbox(
            "我确认该视频声音来自本人或已获得授权，并同意用于本次 Demo 的声音克隆。",
            value=False,
        )

voice_style = st.selectbox(
    "克隆声音讲述风格",
    list_voice_styles(),
    index=list_voice_styles().index(DEFAULT_VOICE_STYLE) if DEFAULT_VOICE_STYLE in list_voice_styles() else 0,
    help=(
        "影响 TTS voice_settings (stability / similarity_boost / style / speed)，"
        "也会作为提示传给 LLM 用于 voice_script 写作风格调整。"
    ),
)

digital_human_mode_options = {
    "静态头像兜底": {
        "key": DIGITAL_HUMAN_MODE_STATIC,
        "description": "不调用 Seedance，使用上传头像作为右下角静态头像兜底。",
    },
    "保留头像动态小窗（Seedance fast）": {
        "key": DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC,
        "description": "使用上传头像生成动态讲解人像，目前是最稳定的头像保留方案。",
    },
    "高质量虚拟讲解人像（Seedance 2.0）": {
        "key": DIGITAL_HUMAN_MODE_SEEDANCE_T2V_VIRTUAL_2_0,
        "description": "使用 Seedance 2.0 文生视频生成虚拟讲解人像，画质更高，但不保证保留上传头像身份。",
    },
    "高质量虚拟讲解人像 fast（Seedance 2.0 Fast）": {
        "key": DIGITAL_HUMAN_MODE_SEEDANCE_T2V_VIRTUAL_2_0_FAST,
        "description": "使用 Seedance 2.0 Fast 文生视频生成虚拟讲解人像，速度更快，但不保证保留上传头像身份。",
    },
    "2.0 保留真人头像实验模式": {
        "key": DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_2_0_EXPERIMENTAL,
        "description": "当前真人头像可能触发隐私风控，需要官方授权素材或平台允许的人像素材。",
    },
}
with st.expander("🤖 数字人形象（静态头像 / Seedance 动态）", expanded=False):
    selected_digital_human_label = st.selectbox(
        "数字人形象模式",
        list(digital_human_mode_options.keys()),
        index=0,
    )
    selected_digital_human_mode = digital_human_mode_options[selected_digital_human_label]
    digital_human_provider = selected_digital_human_mode["key"]

    digital_human_style_options = {
        "直接叠加": "direct",
        "卡片小窗（默认）": "card",
    }
    digital_human_video_mode_options = {
        "预览短循环": "preview_loop",
        "正式完整时长": "full_length",
    }

    st.caption(f"当前数字人模式 key: `{digital_human_provider}`")
    st.info(selected_digital_human_mode["description"])
    st.caption(
        "模式说明：fast 保留头像；Seedance 2.0 质量更高但走 text_to_video，不保证本人头像；"
        "Seedance 2.0 真人头像 i2v 是实验模式，可能触发隐私风控；当前不是精准 lip-sync。"
    )

    if digital_human_provider == DIGITAL_HUMAN_MODE_SEEDANCE_DYNAMIC:
        st.info(
            "Seedance fast 保留头像模式会走 image_to_video，尽量保留上传头像身份，"
            "叠加现有 TTS 音频；当前不是精准 lip-sync。"
        )
        digital_human_style_label = st.selectbox(
            "数字人样式",
            list(digital_human_style_options.keys()),
            index=1,
            help="卡片小窗会增加半透明容器、细边框和阴影，减少贴图感。",
        )
        digital_human_window_style = digital_human_style_options[digital_human_style_label]
        digital_human_video_mode_label = st.selectbox(
            "数字人视频模式",
            list(digital_human_video_mode_options.keys()),
            index=0,
            help="正式完整时长会尽量生成接近视频长度的数字人视频，避免 5 秒短循环的重复感。",
        )
        digital_human_video_mode = digital_human_video_mode_options[digital_human_video_mode_label]
        st.caption(
            "当前 fast 保留头像；卡片小窗样式可减少贴图感。"
            "正式完整时长会尽量生成接近视频长度的数字人视频。当前不是精准 lip-sync。"
        )
        force_regenerate_seedance_avatar = st.checkbox(
            "强制重新生成 Seedance 动态人像",
            value=False,
        )
        if st.button("清理 Seedance 动态人像缓存", key="clean_seedance_cache_btn"):
            cache_dir = Path("outputs") / "seedance_cache"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            st.success("已清理 Seedance 动态人像缓存")
    elif digital_human_provider in {
        DIGITAL_HUMAN_MODE_SEEDANCE_T2V_VIRTUAL_2_0,
        DIGITAL_HUMAN_MODE_SEEDANCE_T2V_VIRTUAL_2_0_FAST,
    }:
        st.info(
            "Seedance 2.0 虚拟讲解人像会走 text_to_video 生成虚拟中文讲解人，"
            "不使用上传头像，也不保证本人头像身份；当前不是精准 lip-sync。"
        )
    elif digital_human_provider == DIGITAL_HUMAN_MODE_SEEDANCE_I2V_AVATAR_2_0_EXPERIMENTAL:
        st.warning(
            "Seedance 2.0 image_to_video 会尝试使用上传头像生成保留真人头像的小窗视频。"
            "当前真人头像可能触发隐私风控，需要官方授权素材或平台允许的人像素材；失败时不会静默回退。"
        )
        fallback_experimental_i2v_to_fast = st.checkbox(
            "实验失败后自动回退到 Seedance fast 保留头像模式",
            value=False,
        )
        _legacy_experimental_notice = (
            "Seedance 2.0 真人头像 image_to_video 目前是实验模式，真人头像可能触发隐私风控；"
            "本轮不会接入或调用该实验链路，生成时会沿用当前静态头像兜底。"
        )

# ══════════════════════════════════════════════════════════════
# 第 1 步（续）· 视频 / 头像 / Transcript 文件
# ══════════════════════════════════════════════════════════════
def _llm_env_missing(status):
    return not (
        status.get("has_llm_base_url")
        and status.get("llm_api_key_exists")
        and status.get("has_llm_model")
    )


def _llm_status_message(plan_result, env_status):
    if _llm_env_missing(env_status):
        return "warning", "当前未配置 LLM，系统已使用 deterministic 兜底。"
    if plan_result.get("used_chunked_planning"):
        return "success", f"长文本模式已启用：系统已将内容分成 {plan_result.get('chunk_count', 0)} 个语义块进行分析。"
    clean_status = plan_result.get("llm_clean_status")
    plan_status = plan_result.get("llm_plan_status")
    error_text = " ".join(str(plan_result.get(k) or "") for k in (
        "llm_error_message", "llm_plan_error", "llm_fallback_reason"
    ))
    if clean_status == "success" and plan_status == "fallback":
        if "duration" in error_text and "too large" in error_text:
            return "info", "LLM 已返回知识点计划，但部分知识点过长，已进入自动拆分/修复流程。"
        return "warning", "LLM 文本清洗成功，但知识点规划结果未通过校验，系统已尝试修复或回退。"
    if clean_status == "success" and plan_status == "repaired_success":
        return "success", "LLM 文本清洗成功，知识点规划已通过自动拆分/修复。"
    return None, None


llm_env_status = get_llm_env_status()
with st.expander("LLM 配置状态", expanded=not (
    llm_env_status.get("has_llm_base_url")
    and llm_env_status.get("llm_api_key_exists")
    and llm_env_status.get("has_llm_model")
)):
    st.write("LLM_BASE_URL", "已配置" if llm_env_status.get("has_llm_base_url") else "未配置")
    st.write("LLM_MODEL", "已配置" if llm_env_status.get("has_llm_model") else "未配置")
    st.write("LLM_API_KEY", "已配置" if llm_env_status.get("has_llm_api_key") else "未配置")
    st.write("ARK_API_KEY", "已配置" if llm_env_status.get("has_ark_api_key") else "未配置")
    st.write("VOLCENGINE_API_KEY", "已配置" if llm_env_status.get("has_volcengine_api_key") else "未配置")
    st.write("实际使用 key 来源", llm_env_status.get("selected_key_source"))
    plan_state = st.session_state.get("plan_result") or {}
    st.write("当前 clean 状态", plan_state.get("llm_clean_status"))
    st.write("当前 plan 状态", plan_state.get("llm_plan_status"))
    st.write("fallback reason", plan_state.get("llm_fallback_reason"))
    if llm_env_status.get("llm_model_looks_like_seedance"):
        st.error("LLM_MODEL 当前看起来是 Seedance 视频模型。LLM_MODEL 应该填写文本/chat 模型或方舟文本模型 endpoint id。")
    msg_level, msg_text = _llm_status_message(plan_state, llm_env_status)
    if msg_text:
        getattr(st, msg_level)(msg_text)
    if _llm_env_missing(llm_env_status):
        st.warning(
            "若要获得更准确的知识点模块切片，请配置 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL。"
        )

# ══════════════════════════════════════════════════════════════
# 复用历史内容计划（跳过 STT / 不重复 LLM 切片）
# ══════════════════════════════════════════════════════════════
def _list_reusable_plans(limit=20):
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return []
    candidates = []
    for project_dir in outputs_dir.iterdir():
        if not project_dir.is_dir():
            continue
        render_jobs_path = project_dir / "render_jobs.json"
        if not render_jobs_path.exists():
            continue
        try:
            mtime = render_jobs_path.stat().st_mtime
        except Exception:
            continue
        try:
            payload = json.loads(render_jobs_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if not payload.get("render_jobs"):
            continue
        candidates.append({
            "project_dir": project_dir,
            "mtime": mtime,
            "render_jobs_count": len(payload.get("render_jobs") or []),
            "video_topics_count": payload.get("video_topics_count") or 0,
        })
    candidates.sort(key=lambda x: x["mtime"], reverse=True)
    return candidates[:limit]


def _load_existing_plan(project_dir):
    project_dir = Path(project_dir)
    knowledge_path = project_dir / "knowledge_modules.json"
    clips_path = project_dir / "clips.json"
    if not knowledge_path.exists() or not clips_path.exists():
        raise FileNotFoundError("缺少 knowledge_modules.json 或 clips.json")
    knowledge_modules = json.loads(knowledge_path.read_text(encoding="utf-8"))
    clips_data = json.loads(clips_path.read_text(encoding="utf-8"))
    source_video_meta_path = project_dir / "source_video.json"
    source_meta = {}
    if source_video_meta_path.exists():
        try:
            source_meta = json.loads(source_video_meta_path.read_text(encoding="utf-8"))
        except Exception:
            source_meta = {}
    return {
        "project_id": project_dir.name,
        "project_dir": str(project_dir),
        "knowledge_modules": knowledge_modules,
        "clips": clips_data.get("clips") or [],
        "cleaned_text": clips_data.get("cleaned_text") or "",
        "clip_plan_mode": clips_data.get("clip_plan_mode") or "知识点模块切片",
        "video_topics_count": len(knowledge_modules.get("video_topics") or []),
        "plan_reusable": True,
        "source_video_path": source_meta.get("source_video_path"),
        "source_video_name": source_meta.get("source_video_name"),
        "video_input_mode": source_meta.get("video_input_mode") or "upload",
        "source_video_size_bytes": source_meta.get("source_video_size_bytes"),
        "warnings": [],
        "errors": [],
    }


_reusable_plans = _list_reusable_plans()
with st.expander(
    f"复用历史内容计划（{len(_reusable_plans)} 个可用）",
    expanded=False,
):
    if not _reusable_plans:
        st.caption("还没有可复用的历史内容计划。先在下方生成一份计划即可。")
    else:
        st.caption(
            "选择已有的内容计划，跳过 STT 与 LLM 切片，直接进入"
            "“选择短视频方案 → 生成视频”阶段。"
        )
        plan_labels = {}
        for entry in _reusable_plans:
            label = (
                f"{entry['project_dir'].name}  ·  "
                f"video_topics={entry['video_topics_count']}  ·  "
                f"render_jobs={entry['render_jobs_count']}"
            )
            plan_labels[label] = entry["project_dir"]
        selected_plan_label = st.selectbox(
            "历史内容计划",
            list(plan_labels.keys()),
            key="reusable_plan_select",
        )
        if st.button("载入这个内容计划", key="reuse_plan_btn"):
            target_dir = plan_labels.get(selected_plan_label)
            try:
                plan_state = _load_existing_plan(target_dir)
                st.session_state["current_project_dir"] = plan_state["project_dir"]
                st.session_state["current_video_name"] = plan_state.get("source_video_name")
                st.session_state["current_source_video_path"] = plan_state.get("source_video_path")
                st.session_state["current_video_input_mode"] = plan_state.get("video_input_mode")
                st.session_state["current_source_video_size_bytes"] = plan_state.get("source_video_size_bytes")
                st.session_state["plan_result"] = plan_state
                st.session_state["per_clip_previews"] = {}
                st.session_state["video_result"] = None
                # Refresh render-plan / render-jobs pointers from disk.
                rp_path = Path(plan_state["project_dir"]) / "selected_render_plan.json"
                rj_path = Path(plan_state["project_dir"]) / "selected_render_jobs.json"
                st.session_state["selected_render_plan_path"] = str(rp_path) if rp_path.exists() else None
                st.session_state["selected_render_jobs_path"] = str(rj_path) if rj_path.exists() else None
                st.success(
                    f"已载入内容计划 {plan_state['project_id']}，"
                    f"共 {plan_state.get('video_topics_count', 0)} 个推荐短视频方案。"
                )
                st.rerun()
            except Exception as e:
                st.error(f"载入失败：{e}")

# 如果当前 session 已经有内容计划，主动提示用户可直接进入第二步。
if st.session_state.get("plan_result") and st.session_state.get("current_project_dir"):
    _cur_plan = st.session_state["plan_result"]
    _topics_total = len((_cur_plan.get("knowledge_modules") or {}).get("video_topics") or []) if isinstance(_cur_plan.get("knowledge_modules"), dict) else 0
    if _topics_total > 0 and _cur_plan.get("plan_reusable"):
        st.info(
            f"已存在内容计划（project={Path(_cur_plan['project_dir']).name}，"
            f"{_topics_total} 个推荐短视频方案）。"
            f"可直接在下方选择短视频方案并生成视频，不会重新 STT、不会重新 LLM 切片。"
            f"如需重新生成，请用页面下方的“生成切片计划”按钮。"
        )


ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def _human_file_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "未知"
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def _validate_external_video_path(path_text: str):
    raw = (path_text or "").strip().strip('"')
    if not raw:
        return None, None, "请填写视频路径"
    path = Path(raw)
    if not path.exists():
        return raw, None, "文件不存在"
    if not path.is_file():
        return raw, None, "路径不是文件"
    if path.suffix.lower() not in ALLOWED_VIDEO_EXTS:
        return raw, None, "扩展名必须是 mp4 / mov / mkv / avi / m4v"
    try:
        with path.open("rb") as f:
            f.read(1)
        size = path.stat().st_size
    except Exception as e:
        return raw, None, f"文件不可读: {e}"
    return str(path.resolve()), size, None


st.divider()
st.markdown("## 1️⃣ 输入素材（视频 / 头像 / Transcript）")
st.caption("左侧：录屏视频；右侧：数字人头像。如果上面选了「手动 / 上传转写文字」，下方还会出现 transcript 输入。")

col_left, col_right = st.columns([2, 1])
with col_left:
    video_input_mode_label = st.radio(
        "视频来源",
        [
            "上传小视频文件",
            "填写本地视频路径",
            "填写 NAS / 共享盘视频路径",
        ],
    )
    video_file = None
    local_video_path = ""
    nas_video_path = ""
    source_video_path = None
    source_video_size_bytes = None
    source_video_error = None
    video_input_mode_key = "upload"
    if video_input_mode_label == "上传小视频文件":
        st.caption("适合小体积视频。大于几百 MB 的视频建议使用本地路径或 NAS 路径。")
        video_file = st.file_uploader("上传录屏讲解视频", type=["mp4", "mov", "mkv", "avi", "m4v"])
        if video_file is not None:
            source_video_size_bytes = int(getattr(video_file, "size", 0) or 0)
            source_video_path = None
            st.caption(f"上传文件大小: {_human_file_size(source_video_size_bytes)}")
    elif video_input_mode_label == "填写本地视频路径":
        video_input_mode_key = "local_path"
        local_video_path = st.text_input(
            "本地视频路径",
            placeholder=r"C:\Users\xxx\Videos\demo.mp4",
        )
        if local_video_path:
            source_video_path, source_video_size_bytes, source_video_error = _validate_external_video_path(local_video_path)
            if source_video_error:
                st.error(source_video_error)
            else:
                st.success(f"视频文件可用，大小: {_human_file_size(source_video_size_bytes)}。不会复制原文件，ffmpeg 将直接读取该路径。")
    else:
        video_input_mode_key = "nas_path"
        nas_video_path = st.text_input(
            "NAS / 共享盘视频路径",
            placeholder=r"\\NAS\share\video\demo.mp4 或 Z:\video\demo.mp4",
        )
        if nas_video_path:
            source_video_path, source_video_size_bytes, source_video_error = _validate_external_video_path(nas_video_path)
            if source_video_error:
                st.error(source_video_error)
            else:
                st.success(f"共享盘视频文件可读，大小: {_human_file_size(source_video_size_bytes)}。不会复制原文件，ffmpeg 将直接读取该路径。")
    manual_text = None
    transcript_file = None
    if input_mode_label == "手动粘贴转写文字":
        manual_text = st.text_area(
            "粘贴语音转写文本",
            height=200,
            placeholder="在此粘贴语音转文字结果...",
        )
    elif input_mode_label == "上传转写文本文件":
        transcript_file = st.file_uploader("上传转写文本文件", type=["txt", "md"])
with col_right:
    avatar_file = st.file_uploader("上传头像图片", type=["png", "jpg", "jpeg"])

# ══════════════════════════════════════════════════════════════
# 第 4 步（续）· 字幕排版（默认 bottom-center 安全区，自动避让数字人）
# ══════════════════════════════════════════════════════════════
st.divider()
st.markdown("## 4️⃣ 字幕 / 排版")
st.caption("默认字幕底部居中并自动避让右下角数字人；最多 2 行。下面字号、描边、背景都可调，并提供「预览当前排版」实时验证。")

pc1, pc2, pc3 = st.columns(3)
with pc1:
    avatar_scale = st.slider("数字人缩放比例", 0.08, 0.24, 0.14, 0.01,
                             help="数字人宽度 = 视频宽度 × 此值")
    avatar_margin_right = st.slider("数字人右边距(px)", 8, 80, 24, 2)
    avatar_margin_bottom = st.slider("数字人底部距离(px)", 30, 220, 100, 2)
with pc2:
    subtitle_size = st.slider(
        "字幕字号(px)",
        16, 48, 28, 1,
        help="1080p 推荐 28–34；720p 推荐 20–24；480p 推荐 16–18。",
    )
    subtitle_margin_bottom = st.slider("字幕底部距离(px)", 20, 200, 80, 2)
    subtitle_max_width_ratio = st.slider(
        "字幕最大宽度占比", 0.35, 0.80, 0.62, 0.01,
        help="字幕宽度不超过视频宽度的此比例；超过会自动换行至 2 行。",
    )
with pc3:
    title_size = st.slider("标题字号(px)", 18, 42, 28, 1)
    subtitle_align = st.radio(
        "字幕对齐",
        ["center", "left", "right"],
        index=0,
        horizontal=True,
        help="默认底部居中并自动避让右下角数字人小窗。",
    )
    subtitle_margin_left = st.slider(
        "字幕水平安全边距(px)", 16, 120, 36, 2,
        help="左右两侧各预留这么多像素的安全边距，避免贴边。",
    )

with st.expander("字幕样式（描边 / 阴影 / 背景）", expanded=False):
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        subtitle_outline_width = st.slider(
            "描边粗细", 0.0, 6.0, 2.5, 0.5,
            help="0 表示无描边；横屏录屏建议 2–3。",
        )
    with sc2:
        subtitle_shadow_depth = st.slider(
            "阴影深度", 0.0, 4.0, 1.0, 0.5,
            help="0 表示无阴影；纯黑描边可读性最高时可调小。",
        )
    with sc3:
        subtitle_show_box = st.toggle(
            "显示半透明背景条",
            value=False,
            help="开启后字幕背后加深色背景条，强对比但更显厚重。",
        )
        subtitle_box_opacity = st.slider(
            "背景条不透明度",
            0, 255, 140, 5,
            help="开启背景条时生效；数值越大背景越显眼。",
            disabled=not subtitle_show_box,
        )


def _build_layout_config() -> LayoutConfig:
    return LayoutConfig(
        avatar_scale=avatar_scale,
        avatar_margin_right=avatar_margin_right,
        avatar_margin_bottom=avatar_margin_bottom,
        subtitle_size=subtitle_size,
        subtitle_margin_left=subtitle_margin_left,
        subtitle_margin_bottom=subtitle_margin_bottom,
        subtitle_max_width_ratio=subtitle_max_width_ratio,
        subtitle_align=subtitle_align,
        subtitle_outline_width=subtitle_outline_width,
        subtitle_shadow_depth=subtitle_shadow_depth,
        subtitle_show_box=subtitle_show_box,
        subtitle_box_opacity=subtitle_box_opacity,
        title_size=title_size,
    )


def _save_uploads_to_tmp():
    tmp_dir = Path("outputs") / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if video_input_mode_key == "upload":
        v_path = tmp_dir / video_file.name
        v_path.write_bytes(video_file.getbuffer())
    else:
        v_path = Path(source_video_path)
    a_path = tmp_dir / avatar_file.name
    a_path.write_bytes(avatar_file.getbuffer())
    return v_path, a_path


def _render_preview_block(preview_result, expander_label="filter_complex（与最终视频相同）"):
    computed = preview_result.get("computed_layout") or {}
    st_image_compat(
        preview_result["preview_path"],
        caption="排版预览（与最终视频共享同一布局引擎）",
        use_container_width=True,
    )
    st.caption(
        f"预览字幕来源: {preview_result.get('subtitle_text_source', 'default')}  "
        f"| 字数: {len(preview_result.get('subtitle_text_used', ''))}"
    )
    st.text_area(
        "预览使用的字幕文本",
        value=preview_result.get("subtitle_text_used", ""),
        height=80,
        disabled=True,
        key=f"preview_subtitle_used_{id(preview_result)}",
    )
    st.json({
        "video_width": computed.get("video_width"),
        "video_height": computed.get("video_height"),
        "avatar_x": computed.get("avatar_x"),
        "avatar_y": computed.get("avatar_y"),
        "avatar_width": computed.get("avatar_width"),
        "subtitle_x": computed.get("subtitle_x"),
        "subtitle_width": computed.get("subtitle_width"),
        "subtitle_right_limit": computed.get("subtitle_right_limit"),
        "title_x": computed.get("title_x"),
        "title_y": computed.get("title_y"),
    })
    for w in preview_result.get("layout_warnings", []):
        st.warning(w)
    if preview_result.get("ass_path"):
        st.caption(f"ASS 文件: `{preview_result['ass_path']}`")
    if preview_result.get("filter_complex"):
        with st.expander(expander_label):
            st.code(preview_result["filter_complex"])


if st.button("预览当前排版"):
    if video_input_mode_key == "upload" and video_file is None:
        st.error("请先上传录屏视频，或改用本地/NAS 路径")
    elif video_input_mode_key != "upload" and (not source_video_path or source_video_error):
        st.error("请先填写一个可读的视频路径")
    elif avatar_file is None:
        st.error("请先上传头像图片")
    else:
        v_path, a_path = _save_uploads_to_tmp()
        layout_cfg = _build_layout_config()
        preview_path = str(Path("outputs") / "uploads" / "layout_preview.jpg")
        with st.spinner("正在生成预览..."):
            try:
                clips_json_path = None
                cur_proj = st.session_state.get("current_project_dir")
                if cur_proj:
                    candidate = Path(cur_proj) / "clips.json"
                    if candidate.exists():
                        clips_json_path = str(candidate)
                preview_result = create_layout_preview(
                    video_path=str(v_path),
                    avatar_image_path=str(a_path),
                    title="预览：当前排版",
                    output_path=preview_path,
                    layout_config=layout_cfg,
                    clips_json_path=clips_json_path,
                )
                _render_preview_block(preview_result)
            except Exception as e:
                st.error(f"预览失败: {e}")

# ══════════════════════════════════════════════════════════════
# 第 2 步 · 生成内容计划（CTA 1：只跑 STT / 清洗 / 切片，不生成视频）
# ══════════════════════════════════════════════════════════════
st.divider()
st.markdown("## 2️⃣ 生成内容计划")
st.caption("只跑 STT、文本清洗、知识点切片和短视频方案规划，**不会**生成最终 MP4。")

if input_mode_label.startswith("手动"):
    st.warning("手动文字没有真实时间戳，切片时间为估算；精准切片请使用自动转写模式。")

if st.button("生成切片计划", type="primary", use_container_width=True):
    input_mode = "auto" if "自动" in input_mode_label else ("manual" if "手动" in input_mode_label else "file")
    errors = []
    if video_input_mode_key == "upload" and video_file is None:
        errors.append("请上传录屏视频，或改用本地/NAS 路径")
    if video_input_mode_key != "upload" and (not source_video_path or source_video_error):
        errors.append("请填写一个存在且可读的视频路径")
    if avatar_file is None:
        errors.append("请上传头像图片")
    if input_mode == "manual" and (not manual_text or not manual_text.strip()):
        errors.append("请粘贴转写文本")
    if input_mode == "file" and transcript_file is None:
        errors.append("请上传转写文本文件")

    if errors:
        for e in errors:
            st.error(e)
    else:
        transcript_file_text = None
        if input_mode == "file" and transcript_file is not None:
            transcript_file_text = transcript_file.read().decode("utf-8")

        # Timeline mode uses clean_and_plan.md; knowledge mode loads its own
        # prompts inside pipeline.
        prompt_template = (Path("prompts") / "clean_and_plan.md").read_text(encoding="utf-8")

        progress_box = st.empty()
        progress_box.info("正在清洗文本")

        def _plan_progress(event):
            message = (event or {}).get("message") or (event or {}).get("stage") or "正在处理"
            current = (event or {}).get("current_chunk")
            total = (event or {}).get("total_chunks")
            if current and total:
                progress_box.info(f"{message}（{current}/{total}）")
            else:
                progress_box.info(message)

        plan_video_name = video_file.name if video_input_mode_key == "upload" else Path(source_video_path).name
        plan_video_bytes = video_file.getvalue() if video_input_mode_key == "upload" else None
        result = generate_clips_plan(
            video_bytes=plan_video_bytes,
            video_name=plan_video_name,
            avatar_bytes=avatar_file.getvalue(),
            avatar_name=avatar_file.name,
            input_mode=input_mode,
            prompt_template=prompt_template,
            manual_text=manual_text,
            transcript_file_text=transcript_file_text,
            clip_plan_mode=clip_plan_mode,
            voice_style=voice_style,
            prompts_dir="prompts",
            source_video_path=source_video_path if video_input_mode_key != "upload" else None,
            video_input_mode=video_input_mode_key,
            source_video_size_bytes=source_video_size_bytes,
            progress_callback=_plan_progress,
        )
        if result.get("errors"):
            progress_box.error("切片计划生成遇到错误")
        else:
            progress_box.success("切片计划生成完成")

        for err in result.get("errors", []):
            st.error(err)

        if result.get("clips"):
            st.session_state["current_project_dir"] = result["project_dir"]
            st.session_state["current_video_name"] = plan_video_name
            st.session_state["current_source_video_path"] = result.get("source_video_path")
            st.session_state["current_video_input_mode"] = result.get("video_input_mode")
            st.session_state["current_source_video_size_bytes"] = result.get("source_video_size_bytes")
            st.session_state["current_avatar_name"] = avatar_file.name
            st.session_state["plan_result"] = result
            st.session_state["per_clip_previews"] = {}
            st.session_state["video_result"] = None
            # Default render plan = everything; the user can later save a subset.
            st.session_state["selected_render_plan_path"] = str(
                Path(result["project_dir"]) / "selected_render_plan.json"
            ) if (Path(result["project_dir"]) / "selected_render_plan.json").exists() else None
            km = result.get("knowledge_modules") or {}
            if isinstance(km, dict):
                topic_count = len(km.get("video_topics") or [])
                visible_kp_count = len(km.get("user_visible_kp_ids") or km.get("knowledge_points") or [])
                st.success(
                    f"切片计划生成成功 ({clip_plan_mode})，"
                    f"共 {topic_count} 个推荐短视频方案 / 用户可选 {visible_kp_count} 个知识点。"
                )
            else:
                st.success(
                    f"切片计划生成成功 ({clip_plan_mode})，"
                    f"共 {len(result['clips'])} 个 clip / {len(km)} 个旧版 module（建议重新生成）"
                )
        elif not result.get("errors"):
            st.error("切片计划生成失败，未返回任何片段")


# ══════════════════════════════════════════════════════════════
# 8. plan_result 展示
# ══════════════════════════════════════════════════════════════
plan_result = st.session_state.get("plan_result")
if plan_result and plan_result.get("clips"):
    st.divider()
    st.markdown("## 3️⃣ 选择要生成的短视频")
    st.caption("已生成内容计划。下面只展示推荐短视频方案；semantic_units / fragments 等内部结构在「高级调试信息」里。")

    project_id = plan_result.get("project_id", "plan")
    project_dir = plan_result.get("project_dir")
    plan_mode = plan_result.get("clip_plan_mode", "时间线连续切片")
    st.caption(f"模式: **{plan_mode}**  |  project: `{Path(project_dir).name if project_dir else '?'}`")

    # 原始转写文本
    transcript_full_text = (
        plan_result.get("transcript_full_text")
        or plan_result.get("full_text")
        or ""
    )
    if not transcript_full_text and project_dir:
        tp = Path(project_dir) / "transcript.txt"
        if tp.exists():
            try:
                transcript_full_text = tp.read_text(encoding="utf-8")
            except Exception:
                transcript_full_text = ""
    if transcript_full_text:
        with st.expander("原始转写文本（来自 STT 或用户输入）"):
            st.text(transcript_full_text)
    cleaned_all = plan_result.get("cleaned_text", "") or ""
    if cleaned_all:
        with st.expander("查看清洗后全文"):
            st.text(cleaned_all)
    else:
        st.caption("清洗文本为空")

    # LLM 状态横幅
    llm_status = plan_result.get("llm_status", "unknown")
    if llm_status == "ok":
        st.success(f"LLM 状态：成功  (model={plan_result.get('llm_model','?')})")
    elif llm_status == "chunked_success":
        st.success(
            f"长文本模式已启用：系统已将内容分成 {plan_result.get('chunk_count', 0)} 个语义块进行分析。"
        )
        st.caption(
            f"chunk 成功 {plan_result.get('successful_chunk_count', 0)} 个，"
            f"失败 {plan_result.get('failed_chunk_count', 0)} 个；"
            f"global_merge={plan_result.get('global_merge_status')}，"
            f"hook_plan={plan_result.get('hook_plan_status')}"
        )
    elif llm_status == "repaired_success":
        st.success(
            f"LLM 状态：清洗成功，规划已自动修复  (model={plan_result.get('llm_model','?')})"
        )
    elif llm_status == "retry_success":
        st.info(f"LLM 状态：clean=ok, plan=ok 或 retry 成功")
    elif llm_status == "fallback":
        msg_level, msg_text = _llm_status_message(plan_result, llm_env_status)
        if msg_text:
            getattr(st, msg_level)(msg_text)
        st.warning(
            "本次使用了 deterministic 兜底（部分或全部 LLM 调用失败）。"
            "可手工编辑下方内容。  reason: "
            + (plan_result.get("llm_fallback_reason") or "")
        )
        if _llm_env_missing(llm_env_status):
            st.warning("若要获得更准确的知识点模块切片，请配置 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL。")

    if (plan_result.get("llm_first_error") or plan_result.get("llm_retry_error")
            or plan_result.get("llm_fallback_reason") or plan_result.get("llm_error_type")
            or plan_result.get("llm_plan_repaired") or plan_result.get("used_chunked_planning")):
        with st.expander("查看 LLM 调试信息", expanded=(llm_status == "fallback")):
            st.markdown(f"**llm_clean_status**: {plan_result.get('llm_clean_status')}")
            st.markdown(f"**llm_plan_status**: {plan_result.get('llm_plan_status')}")
            st.markdown(f"**selected_key_source**: {plan_result.get('llm_selected_key_source')}")
            st.markdown(f"**llm_model**: {plan_result.get('llm_model')}")
            st.markdown(f"**llm_error_type**: {plan_result.get('llm_error_type')}")
            st.markdown(f"**llm_error_message**: {plan_result.get('llm_error_message')}")
            st.markdown(f"**used_deterministic_fallback**: {plan_result.get('used_deterministic_fallback')}")
            st.markdown(f"**llm_plan_repaired**: {plan_result.get('llm_plan_repaired')}")
            st.markdown(f"**llm_plan_repair_reason**: {plan_result.get('llm_plan_repair_reason')}")
            st.markdown(f"**used_chunked_planning**: {plan_result.get('used_chunked_planning')}")
            st.markdown(f"**chunk_count**: {plan_result.get('chunk_count')}")
            st.markdown(f"**successful_chunk_count**: {plan_result.get('successful_chunk_count')}")
            st.markdown(f"**failed_chunk_count**: {plan_result.get('failed_chunk_count')}")
            st.markdown(f"**cached_chunk_count**: {plan_result.get('cached_chunk_count')}")
            st.markdown(f"**chunk_fallback_used**: {plan_result.get('chunk_fallback_used')}")
            st.markdown(f"**failed_chunks**: {plan_result.get('failed_chunks')}")
            st.markdown(f"**global_merge_status**: {plan_result.get('global_merge_status')}")
            st.markdown(f"**hook_plan_status**: {plan_result.get('hook_plan_status')}")
            if plan_result.get("llm_first_error"):
                st.code(plan_result["llm_first_error"], language="text")
            if plan_result.get("llm_first_response_preview"):
                st.caption("first response preview")
                st.code(plan_result["llm_first_response_preview"], language="text")
            if plan_result.get("llm_retry_error"):
                st.code(plan_result["llm_retry_error"], language="text")
            if plan_result.get("llm_retry_response_preview"):
                st.caption("retry response preview")
                st.code(plan_result["llm_retry_response_preview"], language="text")
            if project_dir:
                debug_path = Path(project_dir) / "debug"
                st.caption(f"debug 目录: `{debug_path}`")

    # ── Branch A: 知识点模块切片 → 推荐短视频方案 (video_topics) ──────────
    if (
        isinstance(plan_result.get("knowledge_modules"), dict)
        and plan_result["knowledge_modules"].get("knowledge_points")
    ):
        semantic_plan = plan_result["knowledge_modules"]
        topics = semantic_plan.get("video_topics") or []
        if not topics:
            # 兜底：老缓存计划没有 video_topics，现场补一份（仅在内存，不写回 disk）。
            try:
                kp.build_video_topics_from_plan(semantic_plan)
                topics = semantic_plan.get("video_topics") or []
            except Exception as e:
                st.warning(f"自动补齐 video_topics 失败: {e}")

        points = semantic_plan.get("knowledge_points") or []
        points_by_id = {kp_item.get("kp_id"): kp_item for kp_item in points}
        user_visible_ids = set(semantic_plan.get("user_visible_kp_ids") or [kp_item.get("kp_id") for kp_item in points])

        # 1. 内容总结（普通用户可见）
        st.markdown("## 内容总结")
        summary = semantic_plan.get("source_summary") or {}
        st.info(
            f"主题：{summary.get('main_topic', '(未识别)')}  ·  "
            f"内容类型：{summary.get('content_type', '(未知)')}  ·  "
            f"用户可选知识点：{len(user_visible_ids)} / 全部 {len(points)}  ·  "
            f"推荐方案：{len(topics)} 个"
        )
        if summary.get("speaker_intent"):
            st.caption(f"讲者意图：{summary.get('speaker_intent')}")
        if summary.get("overall_quality_notes"):
            st.caption(f"质量说明：{summary.get('overall_quality_notes')}")

        # 2. 推荐短视频方案（多选，每条选中的方案最终输出一条独立短视频）
        st.markdown("## 推荐短视频方案（可勾选多个，每个生成一条完整短视频）")
        st.caption("内容计划已生成并保存，后续生成视频会复用现有计划，不会重新提取转写或重新切片。")
        selected_topic_ids = []
        if not topics:
            st.error("当前计划没有可用的短视频方案，请重新生成内容计划。")
        else:
            # 读取本项目的 render_jobs.json；旧项目可能没有，现场补一份。
            render_jobs_path = Path(project_dir) / "render_jobs.json" if project_dir else None
            render_jobs_data = {}
            if render_jobs_path and render_jobs_path.exists():
                try:
                    render_jobs_data = json.loads(render_jobs_path.read_text(encoding="utf-8"))
                except Exception as e:
                    st.warning(f"读取 render_jobs.json 失败：{e}")
            if not render_jobs_data.get("render_jobs"):
                try:
                    render_jobs_data = kp.build_render_jobs_from_topics(semantic_plan, voice_style=voice_style)
                    render_jobs_data["project_id"] = project_id
                    render_jobs_data["plan_reusable"] = bool(render_jobs_data.get("render_jobs"))
                    if project_dir:
                        (Path(project_dir) / "render_jobs.json").write_text(
                            json.dumps(render_jobs_data, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                except Exception as e:
                    st.error(f"无法构造 render_jobs：{e}")
            jobs_by_topic_id = {j.get("topic_id"): j for j in (render_jobs_data.get("render_jobs") or [])}

            # 多选 UI：每个 topic 一个 checkbox
            sel_col1, sel_col2 = st.columns([1, 1])
            with sel_col1:
                if st.button("全选所有方案", key=f"topic_all_{project_id}"):
                    for t in topics:
                        st.session_state[f"topic_pick_{project_id}_{t.get('topic_id')}"] = True
            with sel_col2:
                if st.button("全部取消", key=f"topic_none_{project_id}"):
                    for t in topics:
                        st.session_state[f"topic_pick_{project_id}_{t.get('topic_id')}"] = False

            for idx, t in enumerate(topics):
                tid = t.get("topic_id") or f"topic_{idx+1:03d}"
                key = f"topic_pick_{project_id}_{tid}"
                default_checked = st.session_state.get(key, idx == 0)
                kp_ids_for_topic = [kid for kid in (t.get("recommended_kp_ids") or []) if kid in points_by_id]
                label = (
                    f"{t.get('topic_title') or tid}"
                    f"  ·  {t.get('video_type', 'short_video')}"
                    f"  ·  ≈ {t.get('estimated_duration', '?')}s"
                    f"  ·  {len(kp_ids_for_topic)} 个知识点"
                )
                picked = st.checkbox(label, value=default_checked, key=key)
                if t.get("topic_hook"):
                    st.caption(f"开场：{t.get('topic_hook')}")
                if t.get("topic_summary"):
                    st.caption(f"内容：{t.get('topic_summary')}")
                if t.get("is_auto_supplemented"):
                    st.caption("⚙ 自动补齐方案")
                with st.expander(
                    f"查看 {tid} 包含的 {len(kp_ids_for_topic)} 个知识点",
                    expanded=False,
                ):
                    for kid in kp_ids_for_topic:
                        kp_item = points_by_id.get(kid) or {}
                        st.markdown(f"- **{kp_item.get('kp_title') or kid}**  (`{kid}`, {kp_item.get('kp_type', '?')})")
                        if kp_item.get("kp_summary"):
                            st.caption(kp_item.get("kp_summary"))
                if picked and kp_ids_for_topic:
                    selected_topic_ids.append(tid)

            if not selected_topic_ids:
                st.info("请至少勾选一个短视频方案。")
            else:
                st.success(f"已选 {len(selected_topic_ids)} 个方案，将各生成一条完整短视频。")

        # 3. 把当前勾选写成 selected_render_jobs.json
        selected_jobs_path = None
        if selected_topic_ids:
            selected_jobs_payload = {
                "project_id": project_id,
                "plan_reusable": True,
                "voice_style": voice_style,
                "selected_topic_ids": selected_topic_ids,
                "selected_job_ids": [],
                "render_jobs": [],
                "render_output_mode": "single_complete_video",
            }
            for tid in selected_topic_ids:
                job = jobs_by_topic_id.get(tid)
                if not job:
                    continue
                selected_jobs_payload["render_jobs"].append(job)
                if job.get("job_id"):
                    selected_jobs_payload["selected_job_ids"].append(job.get("job_id"))
            try:
                target_path = Path(project_dir) / "selected_render_jobs.json"
                target_path.write_text(
                    json.dumps(selected_jobs_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                selected_jobs_path = str(target_path)
                st.session_state["selected_render_jobs_path"] = selected_jobs_path
                # 保持向后兼容：第一份单 plan 仍然写一份（生成按钮内会优先用 jobs 路径）。
                if selected_jobs_payload["render_jobs"]:
                    first_job = selected_jobs_payload["render_jobs"][0]
                    first_plan = {
                        "selected_topic_id": first_job.get("topic_id"),
                        "selected_hook_id": first_job.get("selected_hook_id"),
                        "selected_hook_ids": [first_job.get("selected_hook_id")] if first_job.get("selected_hook_id") else [],
                        "selected_kp_ids": first_job.get("selected_kp_ids") or [],
                        "render_units": first_job.get("render_units") or [],
                        "render_output_mode": "single_complete_video",
                        "fragment_order": first_job.get("fragment_order") or "user_order",
                        "voice_style": voice_style,
                        "title": first_job.get("topic_title"),
                        "final_video_title": first_job.get("topic_title"),
                        "final_video_opening_hook": first_job.get("topic_hook") or "",
                        "final_video_structure": "按所选知识点顺序讲解",
                        "video_type": first_job.get("video_type") or "short_video",
                        "visible_output_count": 1,
                    }
                    save_selected_render_plan(project_dir, first_plan)
                    st.session_state["selected_render_plan_path"] = str(Path(project_dir) / "selected_render_plan.json")
            except Exception as e:
                st.error(f"保存 selected_render_jobs.json 失败：{e}")
        else:
            st.session_state["selected_render_jobs_path"] = None
        if selected_jobs_path:
            st.caption(f"selected_render_jobs.json：`{selected_jobs_path}`")

        # 4. 高级调试信息（开发者用，默认折叠；普通用户看不到 fragments / units）
        with st.expander("高级调试信息（开发者用）", expanded=False):
            st.markdown("### kp_cap_summary")
            st.json(semantic_plan.get("kp_cap_summary") or {})
            st.markdown("### user_visible_kp_ids / advanced_kp_ids")
            st.write({
                "user_visible_kp_ids": semantic_plan.get("user_visible_kp_ids") or [],
                "advanced_kp_ids": semantic_plan.get("advanced_kp_ids") or [],
            })
            st.markdown("### 全部 knowledge_points（含 visibility）")
            for kp_item in points:
                marker = "[advanced]" if kp_item.get("visibility") == "advanced" else "[user]"
                kp_id_str = kp_item.get("kp_id") or "?"
                title_str = kp_item.get("kp_title") or "(无标题)"
                st.markdown(f"- {marker} `{kp_id_str}` — {title_str}")
                try:
                    duration_str = f"{float(kp_item.get('duration', 0) or 0):.1f}s"
                except Exception:
                    duration_str = "?s"
                st.caption(
                    f"type={kp_item.get('kp_type')} | duration={duration_str} | "
                    f"importance={(kp_item.get('scores') or {}).get('importance', '?')} | "
                    f"reason={kp_item.get('advanced_reason') or '-'}"
                )
            st.markdown("### big_hooks（后台推荐排序，不直接暴露给用户）")
            st.json(semantic_plan.get("big_hooks") or [])
            st.markdown("### assembly_paths（后台叙事结构，不直接暴露给用户）")
            st.json(semantic_plan.get("assembly_paths") or [])
            st.markdown("### semantic_units / fragments（仅调试，普通模式隐藏）")
            st.json(semantic_plan.get("semantic_units") or [])
            st.markdown("### 当前 selected_render_plan.json")
            try:
                if rp_path and Path(rp_path).exists():
                    st.json(json.loads(Path(rp_path).read_text(encoding="utf-8")))
            except Exception as e:
                st.warning(f"读取 selected_render_plan 失败：{e}")

    elif isinstance(plan_result.get("knowledge_modules"), list) and plan_result.get("knowledge_modules"):
        modules = plan_result["knowledge_modules"]
        total_frags = sum(len(m.get("fragments") or []) for m in modules)
        st.warning(
            "这是旧版计划结构（knowledge_modules 是 list），请重新生成内容计划以获得新的短视频方案视图。"
        )
        st.caption(
            f"统计：{len(modules)} 个旧版 module / {total_frags} 个 fragment（未直接展示）"
        )
        with st.expander("高级调试信息：旧版 modules / fragments", expanded=False):
            for m in modules:
                st.markdown(
                    f"- **{m.get('module_title', m.get('module_id', '?'))}** "
                    f"(`{m.get('module_id', '?')}`, topic_key=`{m.get('topic_key', '')}`)"
                )
                st.caption(
                    f"fragments={len(m.get('fragments') or [])} | hook={m.get('hook', '')}"
                )
                if m.get("knowledge_points"):
                    st.caption("旧版知识点：" + "；".join(m["knowledge_points"]))
    # ── Branch B: 时间线连续切片 → 旧的 clip 编辑 UI ─────────────────────
    else:
        st.markdown("## 切片清单（可编辑）")
        st.caption("修改后必须保存，再点'根据当前选择生成视频'。")
        for idx, clip in enumerate(plan_result["clips"]):
            cid = clip.get("clip_id", f"clip_{idx + 1:03d}")
            kpx = f"{project_id}_{cid}"
            st.markdown(f"### {cid} — {clip.get('title', '(无标题)')}")
            try:
                s0 = float(clip.get("start", 0) or 0)
                e0 = float(clip.get("end", 0) or 0)
            except Exception:
                s0, e0 = 0.0, 0.0
            st.caption(f"时间戳: {format_seconds(s0)} - {format_seconds(e0)}  时长: {e0 - s0:.1f}s")
            c1, c2 = st.columns(2)
            with c1:
                new_start = st.number_input("start (秒)", min_value=0.0, value=s0, step=0.1, key=f"{kpx}_start")
            with c2:
                new_end = st.number_input("end (秒)", min_value=0.0, value=e0, step=0.1, key=f"{kpx}_end")
            new_title = st.text_input("标题", value=clip.get("title", "") or "", key=f"{kpx}_title")
            new_hook = st.text_input("钩子", value=clip.get("hook", "") or "", key=f"{kpx}_hook")
            new_summary = st.text_input("摘要", value=clip.get("summary", "") or "", key=f"{kpx}_summary")
            new_source = st.text_area("原始片段文本", value=clip.get("source_text", "") or "", height=120, key=f"{kpx}_source")
            new_cleaned = st.text_area("清洗后片段文本", value=clip.get("cleaned_clip_text", "") or "", height=120, key=f"{kpx}_cleaned")
            new_voice = st.text_area("克隆音朗读稿 (voice_script)", value=clip.get("voice_script", "") or "", height=160, key=f"{kpx}_voice")
            if st.button("保存该片段修改", key=f"{kpx}_save"):
                try:
                    save_all_clips_changes(project_dir, [{
                        "clip_id": cid,
                        "start": float(new_start), "end": float(new_end),
                        "title": new_title, "hook": new_hook, "summary": new_summary,
                        "source_text": new_source, "cleaned_clip_text": new_cleaned,
                        "voice_script": new_voice,
                    }])
                    st.success(f"{cid} 已保存")
                except Exception as e:
                    st.error(f"保存失败: {e}")
            st.divider()


# ══════════════════════════════════════════════════════════════
# 9. 按钮二：根据当前选择生成视频
# ══════════════════════════════════════════════════════════════
st.divider()
st.markdown("## 5️⃣ 生成所选短视频")
st.caption(
    "复用现有内容计划，**不会重新 STT、不会重新 LLM 切片**。"
    "勾选了几个 video_topic 就生成几条独立 MP4，每条命名按 topic_id 隔离。"
)

proj_dir_for_gate = st.session_state.get("current_project_dir")
render_plan_path_for_gate = st.session_state.get("selected_render_plan_path")
quality_gate_ok = True
quality_gate_reasons = []
if proj_dir_for_gate:
    try:
        km_path = Path(proj_dir_for_gate) / "knowledge_modules.json"
        rp_path = Path(render_plan_path_for_gate) if render_plan_path_for_gate else Path(proj_dir_for_gate) / "selected_render_plan.json"
        if km_path.exists():
            semantic_for_gate = json.loads(km_path.read_text(encoding="utf-8"))
            render_for_gate = json.loads(rp_path.read_text(encoding="utf-8")) if rp_path.exists() else None
            if isinstance(semantic_for_gate, dict) and semantic_for_gate.get("knowledge_points"):
                quality_gate_ok, quality_gate_reasons, quality_gate_metrics = kp.validate_render_quality(
                    semantic_for_gate,
                    selected_render_plan=render_for_gate,
                    require_assembly=False,
                )
                if render_for_gate is not None and not (
                    render_for_gate.get("selected_topic_id")
                    or render_for_gate.get("selected_hook_id")
                    or render_for_gate.get("selected_kp_ids")
                ):
                    quality_gate_ok = False
                    quality_gate_reasons.append("必须选择一个短视频方案")
                with st.expander("知识点切片质量门禁", expanded=not quality_gate_ok):
                    st.write({
                        "can_enter_video_generation": quality_gate_ok,
                        **quality_gate_metrics,
                    })
                    if quality_gate_reasons:
                        for reason in quality_gate_reasons:
                            st.error(reason)
                        st.info("需要重新生成或修复计划后再进入视频生成。")
                    else:
                        st.success("当前知识点计划通过视频生成前门禁。")
    except Exception as e:
        quality_gate_ok = False
        quality_gate_reasons = [f"质量门禁检查失败: {e}"]
        st.error(quality_gate_reasons[0])

if proj_dir_for_gate and not quality_gate_ok:
    if st.button("自动修复切片计划", type="secondary", use_container_width=True, key="repair_knowledge_plan_btn"):
        try:
            cmd = [sys.executable, "scripts/repair_knowledge_plan.py", str(proj_dir_for_gate), "--apply"]
            repair_run = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8")
            if repair_run.returncode != 0:
                st.error("自动修复失败")
                st.code((repair_run.stderr or repair_run.stdout or "")[-4000:], language="text")
            else:
                audit_run = subprocess.run(
                    [sys.executable, "scripts/audit_knowledge_plan_quality.py", str(proj_dir_for_gate)],
                    cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8",
                )
                st.success("已自动修复切片计划，并重新运行质量审计。")
                st.code(repair_run.stdout, language="json")
                if audit_run.stdout:
                    st.code(audit_run.stdout, language="json")
                km_path = Path(proj_dir_for_gate) / "knowledge_modules.json"
                clips_path = Path(proj_dir_for_gate) / "clips.json"
                if st.session_state.get("plan_result") and km_path.exists():
                    st.session_state["plan_result"]["knowledge_modules"] = json.loads(km_path.read_text(encoding="utf-8"))
                if st.session_state.get("plan_result") and clips_path.exists():
                    clips_data = json.loads(clips_path.read_text(encoding="utf-8"))
                    st.session_state["plan_result"]["clips"] = clips_data.get("clips", [])
                st.session_state["selected_render_plan_path"] = str(Path(proj_dir_for_gate) / "selected_render_plan.json")
                st.rerun()
        except Exception as e:
            st.error(f"自动修复切片计划失败: {e}")

if st.button("生成所选短视频", type="primary", use_container_width=True, key="gen_video_btn", disabled=not quality_gate_ok):
    proj_dir = st.session_state.get("current_project_dir")
    v_name = st.session_state.get("current_video_name")
    a_name = st.session_state.get("current_avatar_name")
    clips_exists = proj_dir and (Path(proj_dir) / "clips.json").exists()

    if not clips_exists:
        st.error("请先生成切片计划。")
    else:
        if audio_mode_label.startswith("保留"):
            audio_mode = "keep_original_audio"
        elif audio_mode_label.startswith("使用已有"):
            audio_mode = "existing_elevenlabs_voice_id"
        elif audio_mode_label.startswith("上传声音样本"):
            audio_mode = "upload_voice_sample_clone"
        else:
            audio_mode = "clone_from_source_video_audio"

        if audio_mode == "upload_voice_sample_clone" and voice_sample_files:
            sample_dir = Path("outputs") / "uploads" / "voice_samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for f in voice_sample_files:
                p = sample_dir / f.name
                p.write_bytes(f.getbuffer())
                voice_sample_paths.append(str(p))
            for p in voice_sample_paths:
                pp = Path(p)
                if pp.exists() and pp.stat().st_size > 0:
                    st.success(f"声音样本已保存: {pp.name}  size={pp.stat().st_size} bytes")
                else:
                    st.error(f"声音样本保存失败或为空: {pp.name}")

        layout_cfg = _build_layout_config()
        selected_jobs_path = st.session_state.get("selected_render_jobs_path")
        if selected_jobs_path and not Path(selected_jobs_path).exists():
            selected_jobs_path = None
        render_plan_path = st.session_state.get("selected_render_plan_path")
        if render_plan_path and not Path(render_plan_path).exists():
            render_plan_path = None

        common_kwargs = dict(
            audio_mode=audio_mode,
            layout_config=layout_cfg,
            voice_sample_paths=voice_sample_paths,
            voice_name=voice_name,
            remove_background_noise=remove_background_noise,
            voice_consent=voice_consent,
            voice_style=voice_style,
            digital_human_provider=digital_human_provider,
            force_regenerate_seedance_avatar=force_regenerate_seedance_avatar,
            seedance_quality_mode=seedance_quality_mode,
            digital_human_window_style=digital_human_window_style,
            digital_human_video_mode=digital_human_video_mode,
            fallback_experimental_i2v_to_fast=fallback_experimental_i2v_to_fast,
            source_video_path=st.session_state.get("current_source_video_path"),
            video_input_mode=st.session_state.get("current_video_input_mode"),
            source_video_size_bytes=st.session_state.get("current_source_video_size_bytes"),
            source_audio_clone_target_seconds=source_audio_clone_target_seconds,
        )

        if selected_jobs_path:
            # 前置 schema 校验：缺字段时直接报错，不进入渲染。
            try:
                preview_payload = json.loads(Path(selected_jobs_path).read_text(encoding="utf-8"))
            except Exception as e:
                preview_payload = {}
                st.error(f"读取 selected_render_jobs.json 失败：{e}")
            ok_schema, schema_reasons, schema_metrics = kp.validate_selected_render_jobs(preview_payload)
            if not ok_schema:
                for reason in schema_reasons:
                    st.error(reason)
                st.stop()
            batch_status = st.empty()
            batch_progress = st.progress(0.0, text="准备批量生成...")
            total_jobs = schema_metrics.get("render_jobs_count") or 1

            def _batch_progress(event):
                stage = (event or {}).get("stage") or ""
                total = (event or {}).get("total_jobs") or total_jobs
                current = (event or {}).get("current_job") or 0
                title = (event or {}).get("topic_title") or (event or {}).get("topic_id") or ""
                successes = (event or {}).get("successful_jobs", 0)
                failures = (event or {}).get("failed_jobs", 0)
                if stage == "batch_start":
                    batch_status.info(f"准备渲染 {total} 条短视频...")
                elif stage == "job_start":
                    batch_progress.progress(min(1.0, (current - 1) / max(1, total)),
                                            text=f"正在生成第 {current}/{total} 条：{title}")
                    batch_status.info(
                        f"正在生成第 {current}/{total} 条：**{title}**  "
                        f"·  已成功 {successes}  ·  已失败 {failures}"
                    )
                elif stage == "job_done":
                    batch_progress.progress(min(1.0, current / max(1, total)),
                                            text=f"第 {current}/{total} 条完成（{title}）")
                    batch_status.info(
                        f"已完成 {current}/{total}  ·  成功 {successes}  ·  失败 {failures}"
                    )
                elif stage == "batch_done":
                    batch_progress.progress(1.0, text="全部完成")
                    batch_status.success(
                        f"全部完成：共 {total} 条，成功 {successes}，失败 {failures}。"
                    )

            with st.spinner("复用现有内容计划，正在按所选方案批量生成短视频..."):
                result = generate_videos_from_render_jobs(
                    project_dir=proj_dir,
                    video_name=v_name,
                    avatar_name=a_name,
                    selected_render_jobs_path=selected_jobs_path,
                    progress_callback=_batch_progress,
                    **common_kwargs,
                )
        else:
            with st.spinner("正在生成视频..."):
                result = generate_videos_from_plan(
                    project_dir=proj_dir,
                    video_name=v_name,
                    avatar_name=a_name,
                    render_plan_path=render_plan_path,
                    **common_kwargs,
                )
        st.session_state["video_result"] = result

# 视频生成结果展示
video_result = st.session_state.get("video_result")
if video_result and ("jobs" in video_result and "selected_render_jobs_count" in video_result):
    # 批量渲染结果（一次内容计划，多次选择生成视频）
    st.divider()
    st.subheader("视频生成结果（批量）")
    if video_result.get("plan_reusable"):
        st.success(
            f"复用现有内容计划成功，"
            f"共渲染 {video_result.get('successful_jobs', 0)} / {video_result.get('selected_render_jobs_count', 0)} 条短视频。"
        )
    for warn in video_result.get("warnings", []):
        st.warning(warn)
    for err in video_result.get("errors", []):
        st.error(err)

    st.caption(
        f"plan_reusable={video_result.get('plan_reusable')}  ·  "
        f"selected_render_jobs.json：`{video_result.get('selected_render_jobs_path') or '(in-memory)'}`  ·  "
        f"report.json 已写入 `{Path(video_result.get('project_dir', '.')) / 'report.json'}`"
    )

    for job_entry in video_result.get("jobs") or []:
        st.markdown(f"### {job_entry.get('topic_title') or job_entry.get('topic_id')}")
        st.caption(
            f"topic_id={job_entry.get('topic_id')}  ·  "
            f"video_type={job_entry.get('video_type')}  ·  "
            f"≈ {job_entry.get('estimated_duration', '?')}s  ·  "
            f"knowledge_points={len(job_entry.get('selected_kp_ids') or [])}"
        )
        for w in job_entry.get("warnings") or []:
            st.warning(w)
        for e in job_entry.get("errors") or []:
            st.error(e)
        final_path = job_entry.get("final_complete_video_path")
        if final_path and Path(final_path).exists():
            st.video(final_path)
            try:
                with open(final_path, "rb") as f:
                    st.download_button(
                        f"下载 {Path(final_path).name}",
                        data=f,
                        file_name=Path(final_path).name,
                        mime="video/mp4",
                        key=f"dl_topic_{job_entry.get('topic_id')}",
                    )
            except Exception as e:
                st.warning(f"读取最终视频失败：{e}")
            st.caption(f"final_complete_video_path：`{final_path}`")
        else:
            st.error("本方案未生成完整短视频。")
        st.divider()

    st.caption(f"项目目录：`{video_result.get('project_dir', '')}`")
elif video_result:
    st.divider()
    st.subheader("视频生成结果")

    for warn in video_result.get("warnings", []):
        st.warning(warn)
    for err in video_result.get("errors", []):
        st.error(err)

    st.info(
        f"声音模式: {video_result.get('audio_mode', '')} | "
        f"TTS 状态: {video_result.get('tts_status', '')} | "
        f"voice_style: {video_result.get('voice_style', '')} | "
        f"请求数字人模式: {video_result.get('digital_human_requested_provider', video_result.get('digital_human_mode', ''))}"
    )
    st.info(
        f"请求模式: {video_result.get('digital_human_requested_provider', '')} | "
        f"实际模式: {video_result.get('digital_human_actual_provider', '')} | "
        f"使用模型: {video_result.get('seedance_model_used') or '(none)'} | "
        f"生成类型: {video_result.get('seedance_generation_type') or '(none)'} | "
        f"是否保留上传头像身份: {'是' if video_result.get('identity_preserved_from_avatar') else '否'} | "
        f"是否是精准口型同步: {'是' if video_result.get('is_lip_sync') else '否'}"
    )

    st.caption(
        f"Seedance model: {video_result.get('seedance_model', '')} | "
        f"requested: {video_result.get('seedance_model_requested', '')} | "
        f"used: {video_result.get('seedance_model_used', '')} | "
        f"quality: {video_result.get('quality_mode', '')} | "
        f"video_mode: {video_result.get('digital_human_video_mode', '')} | "
        f"style: {video_result.get('digital_human_window_style', '')} | "
        f"supported: {', '.join(video_result.get('seedance_supported_modes') or [])} | "
        f"lip-sync supported: {video_result.get('seedance_lipsync_supported')}"
    )
    if video_result.get("seedance_fallback_reason"):
        st.warning(f"Seedance 模型回退: {video_result.get('seedance_fallback_reason')}")
    if video_result.get("fallback_from_2_0_i2v_to_fast"):
        st.info("实验模式已按勾选项自动回退到 Seedance fast 保留头像模式。")

    if video_result.get("digital_human_requested_provider") in {
        DIGITAL_HUMAN_MODE_SEEDANCE_T2V_VIRTUAL_2_0,
        DIGITAL_HUMAN_MODE_SEEDANCE_T2V_VIRTUAL_2_0_FAST,
    }:
        st.info("该模式使用 Seedance 2.0 文生视频生成虚拟讲解人像，不代表上传头像本人。")

    privacy_guard_triggered = bool(video_result.get("privacy_guard_triggered"))
    if privacy_guard_triggered or video_result.get("seedance_error_code"):
        if privacy_guard_triggered:
            st.warning(
                "Seedance 2.0 image_to_video 已触发真人图片隐私风控。当前图片可能包含真人头像。"
                "建议使用 Seedance fast 保留头像模式，或改用虚拟讲解人像模式，或使用官方授权素材后再试。"
            )
        with st.expander("Seedance 隐私风控处理结果", expanded=privacy_guard_triggered):
            st.write("privacy_guard_triggered", privacy_guard_triggered)
            st.write("error_code", video_result.get("seedance_error_code"))
            st.write("error_message", video_result.get("seedance_error_message"))
            st.write("suggested_action", video_result.get("suggested_action"))
            st.write("fallback_from_2_0_i2v_to_fast", video_result.get("fallback_from_2_0_i2v_to_fast"))

    dh_result = video_result.get("digital_human_provider_result") or {}
    seedance_debug = dh_result.get("debug") or {}
    seedance_error = seedance_debug.get("seedance_error") or {}
    seedance_attempt_errors = seedance_debug.get("seedance_attempt_errors") or []
    st.caption(
        f"实际数字人模式: {dh_result.get('actual_provider') or dh_result.get('provider') or 'static'} | "
        f"是否使用 talking_head_video: {bool(dh_result.get('talking_head_video_path')) and bool(dh_result.get('success'))} | "
        f"talking_head_video_path: {dh_result.get('talking_head_video_path') or '(none)'}"
    )
    if seedance_error or seedance_attempt_errors or video_result.get("seedance_model_probe_summary"):
        with st.expander("查看 Seedance API 真实错误", expanded=bool(seedance_error)):
            st.write("requested_quality_mode", video_result.get("quality_mode"))
            st.write("requested_model_candidates", video_result.get("requested_model_candidates"))
            st.write("selected_model", video_result.get("selected_model"))
            st.write("fallback_model", video_result.get("fallback_model"))
            st.write("fallback_reason", video_result.get("seedance_fallback_reason"))
            if seedance_error:
                st.write("stage", seedance_error.get("stage"))
                st.write("endpoint", seedance_error.get("endpoint"))
                st.write("model", seedance_error.get("model"))
                st.write("status_code", seedance_error.get("status_code"))
                st.write("error_code", seedance_error.get("error_code"))
                st.write("error_message", seedance_error.get("error_message"))
                st.code(seedance_error.get("response_text") or "", language="json")
                st.json(seedance_error.get("request_payload_sanitized") or {})
                st.json(seedance_error.get("request_headers_sanitized") or {})
            if seedance_attempt_errors:
                st.markdown("**all_attempt_errors**")
                st.json(seedance_attempt_errors)

    if dh_result.get("fallback_to_static_avatar") or dh_result.get("error"):
        st.warning(f"数字人已回退静态头像: {dh_result.get('error') or '当前模式不可用'}")

    with st.expander("最终视频使用的布局参数 (layout_config + computed_layout)", expanded=False):
        st.markdown("**layout_config（输入）**")
        st.json(video_result.get("layout_config") or {})
        st.markdown("**computed_layout（实际渲染坐标）**")
        st.json(video_result.get("computed_layout") or {})
        warns = video_result.get("layout_warnings") or []
        if warns:
            st.markdown("**layout_warnings**")
            for w in warns:
                st.warning(w)

    with st.expander("TTS voice_settings (本次使用)", expanded=False):
        st.json(video_result.get("voice_settings_default") or {})

    clone_res = video_result.get("voice_clone_result")
    if clone_res:
        if clone_res.get("success"):
            st.success(f"克隆音色创建成功: voice_id={clone_res.get('voice_id')}")
        else:
            st.error(f"克隆音色创建失败: {clone_res.get('error', '')}")
            if clone_res.get("status_code") or clone_res.get("response_text"):
                st.caption(f"status_code: {clone_res.get('status_code')}")
                st.text_area("ElevenLabs response_text", value=clone_res.get("response_text") or "", height=120)

    sample_res = video_result.get("voice_sample_extraction_result")
    if sample_res:
        if sample_res.get("success"):
            st.success(
                f"已从原视频提取 {len(sample_res.get('sample_paths') or [])} 个声音样本片段，"
                f"总时长 {sample_res.get('total_duration')} 秒"
            )
            st.caption(f"merged_sample_path: `{sample_res.get('merged_sample_path')}`")
            st.caption(f"voice_sample_manifest: `{video_result.get('voice_sample_manifest_path')}`")
        else:
            st.warning(f"原视频声音样本提取失败: {sample_res.get('error')}")
        st.caption(
            f"TTS 是否使用 voice_script: {video_result.get('tts_text_source') == 'voice_script'} | "
            f"字幕是否同源: {video_result.get('subtitle_uses_same_text_as_tts')}"
        )

    clips = video_result.get("clips", [])
    if video_result.get("output_mode") == "single_complete_video":
        final_complete = video_result.get("final_complete_video_path")
        st.subheader("已生成 1 条完整讲解短视频")
        st.info(
            f"选题: {video_result.get('final_video_title') or video_result.get('selected_hook_id') or ''} | "
            f"包含知识点: {len(video_result.get('selected_kp_ids') or [])} 个 | "
            f"数字人: {'已启用' if video_result.get('final_video_has_digital_human') else '未启用'} | "
            f"配音: {video_result.get('tts_text_source')} | "
            f"字幕: {video_result.get('subtitle_text_source')} | "
            f"输出: {video_result.get('final_video_count_visible_to_user')} 条完整视频"
        )
        if final_complete and Path(final_complete).exists():
            st.video(final_complete)
            with open(final_complete, "rb") as f:
                st.download_button(
                    "下载完整讲解短视频",
                    data=f,
                    file_name=Path(final_complete).name,
                    mime="video/mp4",
                    key="dl_complete_video",
                )
            st.caption(f"final_complete_video_path: `{final_complete}`")
        else:
            st.error("完整视频生成失败")
        with st.expander("高级：查看内部 unit 视频", expanded=False):
            for idx, unit_path in enumerate(video_result.get("intermediate_unit_videos") or [], 1):
                st.caption(f"unit_{idx:03d}: `{unit_path}`")
            for clip in clips:
                st.json({
                    "clip_id": clip.get("clip_id"),
                    "kp_id": clip.get("kp_id"),
                    "final_video": clip.get("final_video"),
                    "tts_text_source": clip.get("tts_text_source"),
                    "subtitle_text_source": clip.get("subtitle_text_source"),
                    "subtitle_uses_same_text_as_tts": clip.get("subtitle_uses_same_text_as_tts"),
                })
    elif clips:
        st.subheader(f"共生成 {len(clips)} 个最终视频")
        for clip in clips:
            cid = clip.get("clip_id", "?")
            title = clip.get("title", "无标题")
            final_video = clip.get("final_video")
            try:
                start = float(clip.get("start", 0) or 0)
                end = float(clip.get("end", 0) or 0)
            except Exception:
                start, end = 0.0, 0.0
            dur = clip.get("duration", end - start)
            render_mode = clip.get("render_mode", "single")

            st.markdown(f"### {cid} — {title}")
            st.caption(
                f"render_mode: {render_mode}  "
                f"|  advisory time: {format_seconds(start)} - {format_seconds(end)}  "
                f"|  duration: {dur}s"
            )
            st.markdown(f"**钩子：** {clip.get('hook', '')}")
            st.markdown(f"**摘要：** {clip.get('summary', '')}")
            if clip.get("used_original_audio"):
                st.caption("音频: 原视频原声")
            else:
                st.caption(
                    f"音频: 克隆音 voice_id={clip.get('voice_id_used', '')} "
                    f"voice_style={clip.get('voice_style', '')}"
                )
            if clip.get("tts_error"):
                st.caption(f"TTS 错误: {clip['tts_error']}")

            st.caption(
                f"数字人请求: {clip.get('digital_human_requested_provider')} | "
                f"实际: {clip.get('digital_human_actual_provider')} | "
                f"used_talking_head: {clip.get('final_composer_used_talking_head')} | "
                f"style: {clip.get('final_composer_used_window_style')} | "
                f"full: {clip.get('used_full_length_avatar_video')} | "
                f"loop: {clip.get('used_loop_avatar_video')} | "
                f"generation: {clip.get('generation_mode')} | "
                f"fallback: {clip.get('fallback_to_static_avatar')} | "
                f"path: {clip.get('talking_head_video_path') or '(none)'}"
            )
            if clip.get("fallback_to_static_avatar") or clip.get("digital_human_error"):
                st.warning(f"数字人回退原因: {clip.get('digital_human_error') or '未使用动态 talking head'}")

            if render_mode == "concat_fragments" and clip.get("fragments_rendered"):
                with st.expander(f"{cid} 实际拼接的 fragments ({len(clip['fragments_rendered'])})"):
                    for fr in clip["fragments_rendered"]:
                        st.markdown(
                            f"- `{fr['fragment_id']}`  "
                            f"[{format_seconds(fr['start'])} - {format_seconds(fr['end'])}]  "
                            f"composed=`{Path(fr['composed_video']).name if fr.get('composed_video') else '(失败)'}`"
                        )
                        if fr.get("cleaned_text"):
                            st.caption(fr["cleaned_text"][:120])

            if final_video and Path(final_video).exists():
                st.video(final_video)
                with open(final_video, "rb") as f:
                    st.download_button(
                        f"下载 {cid}_final.mp4",
                        data=f,
                        file_name=f"{cid}_final.mp4",
                        mime="video/mp4",
                        key=f"dl_{cid}",
                    )
            else:
                st.error(clip.get("error", "视频生成失败"))

            st.text_area("原始片段文本", value=clip.get("source_text", "") or "",
                         height=100, disabled=True, key=f"final_{cid}_source")
            st.text_area("清洗后片段文本", value=clip.get("cleaned_clip_text", "") or "",
                         height=100, disabled=True, key=f"final_{cid}_cleaned")
            st.text_area("克隆音朗读稿", value=clip.get("voice_script", "") or "",
                         height=120, disabled=True, key=f"final_{cid}_voice")
            st.divider()

    st.caption(f"项目目录: `{video_result.get('project_dir', '')}`")

# ══════════════════════════════════════════════════════════════
# 10. 单独测试声音克隆
# ══════════════════════════════════════════════════════════════
st.divider()
with st.expander("单独测试声音克隆"):
    st.markdown("上传声音样本，创建克隆音色并生成测试音频。不需要上传视频。")
    t_name = st.text_input("音色名称", value="test_clone", key="t_name")
    t_files = st.file_uploader(
        "上传声音样本", type=["mp3", "wav", "m4a", "aac", "ogg"],
        accept_multiple_files=True, key="t_files",
    )
    t_text = st.text_input("测试文案", value="你好，这是一个克隆音色的测试。", key="t_text")
    t_consent = st.checkbox("我确认已获得该声音所有者授权", value=False, key="t_consent")

    if st.button("创建克隆音色并生成测试音频"):
        if not t_files:
            st.error("请上传声音样本")
        elif not t_consent:
            st.error("请勾选声音授权确认")
        else:
            sample_dir = Path("outputs") / "uploads" / "voice_samples" / "test"
            sample_dir.mkdir(parents=True, exist_ok=True)
            test_paths: list[str] = []
            for f in t_files:
                p = sample_dir / f.name
                p.write_bytes(f.getbuffer())
                test_paths.append(str(p))
            from services.voice_clone_elevenlabs import clone_voice_with_elevenlabs
            from services.tts_elevenlabs import generate_tts
            with st.spinner("正在创建克隆音色..."):
                cr = clone_voice_with_elevenlabs(sample_paths=test_paths, voice_name=t_name)
            if cr["success"]:
                st.success(f"克隆成功: voice_id={cr['voice_id']}")
                with st.spinner(f"正在用 {voice_style} 生成测试音频..."):
                    tts_r = generate_tts(
                        t_text, str(sample_dir / "test_voice.mp3"),
                        voice_id=cr["voice_id"], voice_style=voice_style,
                    )
                if tts_r["success"]:
                    st.audio(tts_r["audio_path"])
                    st.caption(
                        f"voice_style={tts_r.get('voice_style')}  "
                        f"text_length={tts_r.get('text_length')}"
                    )
                    st.json(tts_r.get("voice_settings") or {})
                else:
                    st.error(f"TTS 失败: {tts_r['error']}")
            else:
                st.error(f"克隆失败: {cr['error']}")
                st.json(cr.get("debug", {}))
