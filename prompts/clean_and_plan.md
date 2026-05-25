只返回 JSON，不要返回 markdown，不要返回解释，不要返回代码块。

## 任务
把下面的录屏讲解转写稿拆成 2-3 个独立短视频片段，每段朗读 30-90 秒。

## 规则
- start / end 必须从下方 segments 选取，不要凭空编造时间戳。
- source_segment_indexes 是该 clip 覆盖的 segments 索引数组（0 起）。
- 如果 segments 全为 0 时间或为空，按文本长度估算每段时长（每秒 4 字），从 0 累加。
- voice_script 只删除"嗯/啊/那个/这个/就是说"等口语词，不改变讲解顺序，不新增知识点。
- cleaned_clip_text 可更正式，但不改变原意。
- cleaned_text 是整段去口语化后的完整文本。
- 输出必须是单个 JSON object，不要包代码块，不要写解释。

## 输出 schema
{
  "cleaned_text": "string",
  "clips": [
    {
      "clip_id": "clip_001",
      "start": 0,
      "end": 90,
      "title": "string",
      "hook": "string",
      "summary": "string",
      "source_segment_indexes": [0, 1],
      "source_text": "string",
      "cleaned_clip_text": "string",
      "voice_script": "string",
      "action_tags": []
    }
  ]
}

## 输入

full_text:
{{FULL_TEXT}}

segments:
{{SEGMENTS_JSON}}
