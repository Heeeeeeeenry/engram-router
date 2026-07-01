"""Tests for MemoryStore._summarize, _clean_sentence, and _truncate_cjk."""

from engram_router.store import MemoryStore


# ---------------------------------------------------------------------------
# _clean_sentence
# ---------------------------------------------------------------------------

_clean = MemoryStore._clean_sentence


class TestCleanSentence:
    def test_removes_filler_phrases(self):
        assert _clean("就是说这个东西不错") == "东西不错"
        assert _clean("那个张三来了") == "张三来了"
        assert _clean("对吧这件事很重要") == "这件事很重要"
        assert _clean("怎么说呢我觉得还行") == "我觉得还行"

    def test_removes_multiple_fillers(self):
        result = _clean("嗯就是说那个张三对吧送了一把键盘")
        assert "张三" in result
        assert "送了一把键盘" in result
        assert "就是说" not in result
        assert "那个" not in result
        assert "对吧" not in result

    def test_preserves_entity_names(self):
        assert "张三" in _clean("那个张三前天来了")
        assert "腾讯" in _clean("就是说腾讯发布了新财报")
        assert "HHKB" in _clean("嗯那个 HHKB 键盘不错")
        assert "OpenAI" in _clean("就是说 OpenAI 那个发布了新模型")

    def test_preserves_numbers(self):
        assert "120" in _clean("就是说他花了 120 块钱")
        assert "2024" in _clean("那个 2024 年的事情")

    def test_short_text_preserved(self):
        assert _clean("你好") == "你好"
        assert _clean("OK") == "OK"

    def test_no_fillers_unchanged(self):
        original = "张三送了一把 HHKB 键盘给李四"
        assert _clean(original) == original

    def test_empty_after_cleaning(self):
        # If only fillers, returns empty string
        result = _clean("嗯呃啊嘛呢哈哦噢")
        # Should be empty or near-empty after cleaning
        assert len(result.strip()) <= 1


# ---------------------------------------------------------------------------
# _truncate_cjk
# ---------------------------------------------------------------------------

_trunc = MemoryStore._truncate_cjk


class TestTruncateCJK:
    def test_short_text_unchanged(self):
        assert _trunc("你好", 120) == "你好"
        assert _trunc("Hello world", 120) == "Hello world"

    def test_exact_boundary(self):
        text = "你好世界"
        assert _trunc(text, 4) == "你好世界"

    def test_truncates_long_text(self):
        long_text = "这是一个很长的句子用来测试截断功能是否正常工" + "作" * 50
        result = _trunc(long_text, 10)
        assert len(result) <= 10

    def test_no_cjk_char_cut_midway(self):
        """CJK characters in Python 3 str are single code points,
        so slicing naturally avoids cutting inside a character."""
        text = "你好世界测试截断"
        result = _trunc(text, 3)  # Should give exactly 3 chars
        assert len(result) == 3
        assert result == "你好世"

    def test_strips_trailing_whitespace(self):
        text = "你好世界    "
        result = _trunc(text, 6)
        assert result == "你好世界"
        assert not result.endswith(" ")

    def test_default_max_chars(self):
        long_text = "测" * 200
        result = _trunc(long_text)
        assert len(result) == 120


# ---------------------------------------------------------------------------
# _summarize
# ---------------------------------------------------------------------------

_summ = MemoryStore._summarize


class TestSummarize:
    def test_removes_filler_words(self):
        result = _summ("就是说那个张三送了我一把 HHKB 键盘。这件事让我很开心。")
        assert "就是说" not in result
        assert "那个" not in result
        assert "张三" in result
        assert "HHKB" in result

    def test_preserves_entity_names(self):
        result = _summ("嗯那个张三和腾讯那边谈了个合作。")
        assert "张三" in result
        assert "腾讯" in result

    def test_preserves_special_terms(self):
        result = _summ("就是说 OpenAI 那个发布了 GPT-5 模型。业界震动很大。")
        assert "OpenAI" in result
        assert "GPT-5" in result

    def test_no_cjk_char_cut(self):
        """Summary must never end inside a CJK code point."""
        text = "张三送了我一把HHKB键盘说是生日礼物然后我就很开心地收下并且每天使用。后续还有更多故事。"
        result = _summ(text)
        # Every character in result should be a valid Unicode scalar
        for ch in result:
            assert ord(ch) <= 0x10FFFF

    def test_max_120_chars(self):
        long_text = (
            "张三送了我一把HHKB键盘说是生日礼物然后我就很开心地" * 10 + "。结束。"
        )
        result = _summ(long_text)
        assert len(result) <= 120

    def test_short_text_returned_as_is(self):
        result = _summ("张三送了一把键盘。")
        assert "张三" in result
        assert "键盘" in result
        # Should be clean but preserve content
        assert len(result) <= len("张三送了一把键盘。")

    def test_no_sentence_boundary_uses_full_text(self):
        result = _summ("张三送了一把HHKB键盘说是生日礼物")
        # No sentence-ending punctuation, full text cleaned and truncated
        assert "张三" in result
        assert "HHKB" in result

    def test_multiline_input(self):
        result = _summ("第一句话。\n第二句话。\n第三句话。")
        # Should stop at first sentence boundary
        assert "第二句话" not in result

    def test_english_sentence_boundary(self):
        result = _summ("张三 said the HHKB is great! Then he left.")
        assert "张三" in result
        # Should stop at !
        assert "Then he left" not in result
