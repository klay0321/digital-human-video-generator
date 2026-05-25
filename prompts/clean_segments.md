只返回 JSON，不要返回 markdown，不要返回解释，不要返回代码块。

## 任务
对下方 STT segments 做"去口语化清洗"并标注"知识点"。不要重新切分时间，只对每个 segment 一对一打标。

## 规则
- 删除"嗯/啊/这个/那个/然后呢/就是说/那么/呃/哎/对吧/是吧"等口头禅；
- 不改变句子的知识含义；
- 保留每段原始 start / end；
- 给每段补一个 segment_id，从 "seg_0001" 开始顺序编号；
- topic_tags：3-6 个短关键词（例 ["MCP","工具调用"]）；
- knowledge_point：一句话标题（例 "MCP 基础概念"）；如不属于任何知识点（寒暄/转场）留空；
- is_filler：仅当该段几乎全是口头禅/寒暄/转场时为 true；
- filler_reason：is_filler=true 时简短说明，否则空字符串；
- importance：0.0 ~ 1.0，知识密度越高越接近 1.0；
- cleaned_text：整段去口语化后的完整文本（所有非 filler 段的 clean_text 自然拼接）；
- 输出必须是单个 JSON object。

## 输出 schema
{
  "cleaned_text": "string",
  "cleaned_segments": [
    {
      "segment_id": "seg_0001",
      "start": 0.0,
      "end": 8.5,
      "raw_text": "string",
      "clean_text": "string",
      "is_filler": false,
      "filler_reason": "",
      "topic_tags": ["string"],
      "knowledge_point": "string",
      "importance": 0.85
    }
  ]
}

## 输入

full_text:
{{FULL_TEXT}}

segments:
{{SEGMENTS_JSON}}
