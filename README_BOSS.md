# README_BOSS

## 大视频输入说明

当前 Demo 支持三种视频来源：

- 上传小视频文件：适合几百 MB 以内的普通测试视频。
- 填写本地视频路径：适合 2GB 这类大视频，系统只记录路径，不复制原文件。
- 填写 NAS / 共享盘视频路径：适合团队共享素材，ffmpeg 会直接读取该路径。

`.streamlit/config.toml` 已将普通上传上限设置为 `1024MB`，但这只是为了提升普通上传的兼容性，不建议通过浏览器上传 2GB 大视频。大视频请优先使用本地路径或 NAS 路径。

## 原视频声音克隆说明

系统新增“从原视频原声自动提取样本并克隆后替换原声”模式：

1. 从原始录屏视频中自动挑选较清晰的单人讲解声音片段；
2. 合并为 `voice_samples/voice_clone_sample_merged.wav`；
3. 调用 ElevenLabs 创建克隆音色；
4. 用克隆音色朗读清洗后的 `voice_script`；
5. 字幕使用同源的 `subtitle_lines`，不直接使用带口头禅的原始转写文本。

该功能只适用于本人声音或已获得明确授权的声音。页面中必须勾选授权确认后，系统才会调用 ElevenLabs 声音克隆接口。

## 输出记录

生成目录中会记录：

- `source_video.json`：视频来源、原始路径、文件大小、是否复制到项目目录；
- `voice_samples/voice_sample_manifest.json`：选中的声音样本时间段、原因、合并样本路径、总时长；
- `report.json`：声音克隆结果、`voice_id`、TTS 文本来源、字幕来源、字幕是否和 TTS 同源、TTS 与画面时长调整方式。

正常情况下：

- `tts_text_source = voice_script`
- `raw_text_used_for_tts = false`
- `subtitle_text_source = subtitle_lines`
- `subtitle_uses_same_text_as_tts = true`
