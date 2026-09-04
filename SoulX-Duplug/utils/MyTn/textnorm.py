import os, sys
import argparse
import string

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pypinyin import lazy_pinyin
from zhon.hanzi import punctuation
from cn_tn import TextNorm

# WeTextProcessing (tn) 依赖 pynini，需要现场编译 OpenFst，在 macOS/arm64 上
# 极易失败。它只被 process_text() 用到，而推理链路（zh_norm / zh_remove_punc）
# 并不需要它，因此改为惰性导入：缺失时不影响推理，仅在真正调用 process_text 时报错。
try:
    from tn.chinese.normalizer import Normalizer as Normalizer_zh
    from tn.english.normalizer import Normalizer as Normalizer_en

    _TN_AVAILABLE = True
except ImportError:  # pragma: no cover
    Normalizer_zh = Normalizer_en = None
    _TN_AVAILABLE = False


def process_text(text: str, lang: str) -> str:
    if not text.strip():
        raise ValueError("Input text cannot be empty")

    if not _TN_AVAILABLE:
        raise ImportError(
            "process_text() 需要 WeTextProcessing (tn)，当前环境未安装。"
            "请先安装 pynini + WeTextProcessing；推理链路无需此函数。"
        )

    if lang == "zh":
        try:
            normalizer = Normalizer_zh()
            tn_text = normalizer.normalize(text)
            py = lazy_pinyin(tn_text)
            return py
        except Exception as e:
            raise ValueError(f"Failed to normalize Chinese text: {e}")
    elif lang == "en":
        try:
            normalizer = Normalizer_en()
            tn_text = normalizer.normalize(text)
            return tn_text
        except Exception as e:
            raise ValueError(f"Failed to normalize English text: {e}")
    else:
        raise ValueError(
            "Unsupported language. Use 'zh' for Chinese or 'en' for English"
        )


def zh_norm(text):
    normalizer = TextNorm()
    return normalizer(text)


def zh_remove_punc(text):
    punctuation_all = punctuation + string.punctuation
    for x in punctuation_all:
        text = text.replace(x, "")

    text = text.replace("  ", " ")
    return text


def en_remove_punc(text):
    punctuation_all = punctuation + string.punctuation
    punctuation_all = [i for i in punctuation_all if (i != "'" and i != "-")]

    for x in punctuation_all:
        text = text.replace(x, "")

    text = text.replace("  ", " ")
    return text
