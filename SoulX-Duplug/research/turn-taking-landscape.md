# SoulX-Duplug 研究问答

> 问答式活文档。每次对话新增一个 Q,答案写在前面,思考过程和参考资料折叠在后面。
> 想快速了解结论 → 只看每个 Q 的**答**;想追细节 → 展开折叠区。

| 项目 | 内容 |
|---|---|
| 分支 | `RESEARCH` |
| 最后更新 | 2026-08-20 |
| 版本 | v1.7 |

**目录**

- [Q1:什么是 VAD?什么是 VAP?SoulX-Duplug 又是什么?](#q1什么是-vad什么是-vapsoulx-duplug-又是什么)
- [Q2:Duplug 模型是怎么设计的?输入输出和内部信息流?](#q2duplug-模型是怎么设计的输入输出和内部信息流)
- [Q3:不重新训练,能调哪些参数?](#q3不重新训练能调哪些参数)
- [Q4:有哪些和 SoulX 同类型的工作?(semantic VAD 专项)](#q4有哪些和-soulx-同类型的工作semantic-vad-专项)
- [Q5:有没有引入视觉模态的 Semantic VAD?](#q5有没有引入视觉模态的-semantic-vad)
- [附录 A:文献总表](#附录-a文献总表)
- [附录 B:代码锚点](#附录-b代码锚点)

---

## Q1:什么是 VAD?什么是 VAP?SoulX-Duplug 又是什么?

### 答

三个概念是**递进**关系,先逐个讲清楚,最后对比。

---

### 1. VAD — Voice Activity Detection(语音活动检测)

**定义**
> 对连续音频流**逐帧做二分类**,判断每一帧是**语音**还是**非语音**(静音、噪声、音乐)。

判断依据是**声学特征** —— 短时能量、零穿越率、频谱特性等。
这是语音处理里最基础的前端模块,1990 年代起就是电信编码、ASR 前处理的标配。

**形象例子:声控灯**

声音大到一定程度灯就亮,安静下来灯就灭。
它只认「响不响」,不认「说什么」。
所以你咳嗽一声灯也亮,你轻声说话灯反而不亮。

**用它做端点检测的困境:一个只看你嘴动没动的餐厅服务员**

你说「我要一份……」低头看菜单,他立刻转身走了 —— 因为你嘴不动了。
你只能把他叫回来。
那让他多等等?可你真说完了,他还得站那儿发呆三秒。

> **快和准无法兼得。** 这不是工程调参问题,是信息不足(见折叠区)。

---

### 2. VAP — Voice Activity Projection(语音活动投影)

**定义**(Ekstedt & Skantze, 2022 `[F2]`)
> 一个**自监督预测任务**:输入对话双方各一路音频,输出**未来约 2 秒内
> 双方说话/沉默模式**的概率分布。
> 做法是把未来窗口切成若干时间格,每格标记谁在说话,
> 组合成有限个离散状态,转化为分类问题。

**为什么不需要人工标注**
训练时把录音往后播 2 秒,**实际发生的 VAD 就是标准答案**。
数据自己给自己出题 —— 这是它最漂亮的地方。

**形象例子:老夫老妻**

结婚三十年的两口子。老公话说到一半,老婆已经开始接后半句。

她不是听懂内容才反应,而是几十年里对**节奏**太熟:
他这个语调、这个停顿长度、这个吸气声,后面就该我说了。

VAP 学的就是这种「节奏默契」。因为它同时听两路,
所以还能预判**谁会抢谁的话、谁该给谁让话**。

**局限**
老婆闭着眼也能接话;但如果老公讲的是她完全不懂的专业内容,
她只能接上**节奏**,接不上**意思**。

---

### 3. SoulX-Duplug

**定义**(依据 `[F13]` 摘要与本 repo 代码)
> 一个**即插即用的流式状态预测模块**。逐块(160ms)处理用户音频,
> **联合执行流式 ASR**,以增量文本为条件,输出当前对话状态:
> `idle`(无语义)/ `nonidle`(说话中)/ `speak`(可接话)/ `blank`(音频不足一块)。

官方定位:**semantic VAD(语义 VAD)**。

> ⚠️ **命名容易混**:模型内部预测的是 `<|user_complete|>` / `<|user_incomplete|>` /
> `<|user_backchannel|>` 等词表 token,但**对外 API 返回的 state 只有 4 个**:
> `idle` / `nonidle` / `speak` / `blank`。
> `incomplete` 和 `backchannel` 是内部状态,不直接暴露 —— 它们的作用是让模块
> **继续等待**(对外表现为 `idle`)。下文凡说「对外状态」用 `speak`,
> 说「模型预测」用 `<|user_complete|>`。

**形象例子:老练的相声捧哏**

逗哏说「我昨天去了一个地方……」——
捧哏绝不会接话,因为他**听懂了这句话没说完**,后面必须有个地名。

等对方说「我昨天去了天津」,他立刻「哦?天津卫!」

捧哏靠的不是等静音,是**听懂话说到哪儿了**。

**具体到三种停顿:**

| 你说 | 然后停住 | 模型内部预测 | 对外返回 |
|---|---|---|---|
| 「我想问一下……」 | 停 1 秒 | `<|user_incomplete|>` → 继续等 | `idle` |
| 「今天天气怎么样」 | 停 1 秒 | `<|user_complete|>` → 说完了 | **`speak`** |
| 「嗯嗯」「对对」 | — | `<|user_backchannel|>` → 只是附和 | `idle` |

第三行是 VAD 和 VAP 都容易搞错的:
你说「嗯嗯」表示在听,系统却以为你要插话,把自己说的话掐断了。

---

### 三者对比

**一句话版本**

| | 一句话 | 类比 |
|---|---|---|
| **VAD** | 现在有没有声? | 声控灯 |
| **VAP** | 接下来 2 秒谁会说? | 老夫老妻的节奏默契 |
| **SoulX** | 这句话说完了没? | 相声捧哏,听懂了才接 |

**技术对比**

| | VAD | VAP | SoulX-Duplug |
|---|---|---|---|
| 判断对象 | 当前帧 | **未来 2 秒** | 当前块(160ms) |
| 判断依据 | 声学能量 | 声学节奏 | **文本语义** |
| 输入 | 单路 | **双路(stereo)** | 单路(仅用户侧) |
| 输出 | 有声 / 无声 | 未来活动模式分布 | 4 个对外状态 |
| 标注成本 | 几乎零 | **零(自监督)** | 高(需人工标状态) |
| 懂内容吗 | ✗ | ✗ | **✓** |
| 建模双方互动 | ✗ | **✓** | ✗ |
| 算力 | 极低 | 低 | **高(带 LLM)** |

**关键:三者不是一条直线,是分叉**

```
                判断「当下」          预测「未来」
   靠声学  │  VAD(声控灯)      │  VAP(老夫老妻)
   靠语义  │  SoulX(捧哏)      │      (空白)
```

VAD 是共同起点:

- **往右** = 保持声学,改成预测未来 → **VAP**
- **往下** = 保持判断当下,升级到懂语义 → **SoulX**

所以 **SoulX 是 VAD 的语义升级版,不是 VAP 的改进版**。
它自称 semantic VA**D** 而不是 semantic VA**P**,用词是准确的。

> 右下角那个空白格(靠语义 + 预测未来)目前是空的。

<details>
<summary>思考过程 & 参考资料</summary>

#### 为什么说 VAD 的阈值矛盾"无解"

不是工程没调好,是**信息论层面的不足**:

纯声学信号里**根本不包含**「说完了没」这个答案。
同样一段 500ms 静音,可能是思考停顿,也可能是话轮结束 ——
声学上二者无法区分。

所以任何阈值都只是在「抢话」和「迟钝」两类错误之间**挪动**,
不可能同时减少。要突破,必须引入声学之外的信息。

这正是 `[F3]` FastTurn 和 `[F13]` SoulX 共同的出发点:**必须引入语义**。

#### 怎么确认 SoulX 是这么工作的

不是靠读摘要,而是先读代码。

**证据 1:输出的是词表里的状态 token,不是波形也不是文本**

```python
# config/config.py:29-33
user_complete_token_id: int = 151676
user_backchannel_token_id: int = 151677
user_incomplete_token_id: int = 151678
assistant_backchannel_token_id: int = 151679
user_idle_token_id: int = 151680
```

这解释了为什么它是「LLM 做判别任务」——
状态被编码成扩展词表里的特殊 token,复用了 LLM 的输出头。

**证据 2:判决方式是直接比 logits 大小**(`service/model.py:703-711`),
说明是**判别式分类**,不是生成任务。

**证据 3:状态预测以文本为条件**(`service/model.py:398-400`)

```python
delta_text = self._asr(audio_embeds)
state = self._state_predict(delta_text)
```

先出文字,再判状态 —— 这就是官方说的 "text-guided"。

**证据 4:单路输入**。`process(audio_chunk)` 只接一路音频,没有 assistant 侧通道。
而 VAP 的立身之本是 stereo 双路,因为它要建模**双方的交互动态**。

#### 补充:严格的技术对照(含实现细节)

| | VAD | VAP `[F2]` | SoulX `[F13]` |
|---|---|---|---|
| 任务类型 | 逐帧二分类 | 未来窗口离散状态分类 | 逐块多状态分类 |
| 监督方式 | 规则 / 轻量模型 | **自监督**(标签=未来 VAD) | 有监督(ASR + 状态联合) |
| 骨干 | 信号处理 / 小网络 | 专用 CPC/Transformer | GLM-4-Voice tokenizer → projector → Qwen3-0.6B + LoRA |
| 时间粒度 | 10~30ms 帧 | ~2s 预测窗 | 160ms 块(+0.96s 回看,+40ms 前瞻) |
| 输出转写 | 否 | 否 | 是(`speak` 时给整轮 `text`) |

#### 一个容易忽略的推论

**单路输入既是 SoulX「即插即用」的原因,也是它的代价。**

能挂到任何现有半双工系统上 —— 因为它只需要用户那一路音频。
但它不知道 assistant 在干什么,assistant 何时说话完全由外层系统决定。
所以它**做不了 VAP 那种双向重叠建模**(谁抢谁、谁让谁)。

#### 术语澄清:类比的边界

三个类比(声控灯 / 老夫老妻 / 捧哏)是为了直观,但有失真之处,需注意:

- **老夫老妻**:VAP 并不是「记住了这个人的习惯」,它学的是**人群统计规律**,
  对陌生说话人同样有效。类比夸大了个体适配。
- **捧哏**:SoulX 判断的是**语义完整性**(这句话结构上说完了没),
  不是**语用恰当性**(该不该由我接、接什么)。真捧哏还懂后者,SoulX 不管。

#### 参考资料

- `[F2]` Ekstedt & Skantze (2022) *Voice Activity Projection: Self-supervised Learning
  of Turn-taking Events*. arXiv:2205.09812;Interspeech 2023 (Show & Tell)
- `[F13]` *SoulX-Duplug*. arXiv:2603.14877(投稿 Interspeech 2026, under review)
  - 摘要原文自述:"By jointly performing streaming ASR, SoulX-Duplug explicitly
    leverages textual information to identify user intent,
    **effectively serving as a semantic VAD**."
- `[F1]` Skantze (2021) *Turn-taking in Conversational Systems and HRI: A Review*.
  Computer Speech & Language 67:101178 —— 传统 VAD 端点检测方案的系统梳理见该文
- `[F3]` *FastTurn*. arXiv:2604.01897 —— 同样批判纯声学 VAD 缺乏语义

⚠️ 关于 VAD 的定义与历史,属语音处理领域公认基础知识,本文档未逐条溯源到
原始文献(如 ITU-T G.729B、ETSI AMR VAD 等标准)。若需严格引用,应补查标准文本。

</details>

---

## Q2:Duplug 模型是怎么设计的?输入输出和内部信息流?

### 答

分两部分讲:

- **第一部分:模型本身** —— 输入什么、输出什么、怎么算
- **第二部分:整个系统** —— 模型外面还包了哪些东西

---

### 第一部分:模型本身

#### 1. 先认识 4 个词

| 词 | 大白话 |
|---|---|
| **采样点** | 麦克风每秒记 16000 个数字来描述声音,一个数字 = 一个采样点。2560 个 = 160 毫秒 |
| **块(chunk)** | 系统攒够 2560 个采样点(160 毫秒)算一「块」,一块一块地判 |
| **token** | 大模型不认识声音也不认识汉字,只认编号。一个编号 = 一个 token |
| **LLM** | 大语言模型,这里用 Qwen3-0.6B(6 亿参数的小模型) |

---

#### 2. 模型的输入输出(数据格式)

**大白话版:**

```
输入:160 毫秒的声音  +  这 160 毫秒里识别出的文字
输出:1 个词,说明这一块是什么情况
```

**实际数据格式(对照源码):**

最外层入口是 `TurnModel.process()`,`service/model.py:216`:

```python
def process(self, audio_chunk):        # audio_chunk: np.ndarray
    assert audio_chunk.dtype == np.float32
```

| | 类型 | shape | 说明 |
|---|---|---|---|
| **输入** | `np.ndarray` | `(N,)` | 任意长度 N,`float32`,16kHz,单声道,幅度 ±1.0 |
| **输出** | `dict` | — | 见下 |

输入**没有 batch 维**,就是一维数组。长度任意 —— 攒不够一块就先存着。

**输出的 dict 结构**(`service/model.py:223-227` 等处):

```python
{
    "state": str,          # "blank" | "idle" | "nonidle" | "speak"
    "asr_segment": str,    # 这一块新识别出的字,可能为 ""
    "asr_buffer": str,     # 最近 3.2 秒的识别结果
    "text": str,           # 仅当 state == "speak" 时存在:整轮完整转写
}
```

⚠️ `text` 这个 key **只在 `speak` 时才有**,其余情况不存在(不是空字符串)。
所以取值要用 `data["state"].get("text", "")`。

---

#### 2.1 内部张量流(逐步 shape)

从 `np.ndarray` 到最终输出,中间的张量形状变化:

**① 攒够并切片** — `get_chunk()`,`service/model.py:186-214`

```python
# self.buffer: np.ndarray (M,) float32，M 持续增长
# 必须 M >= 2560 + 15360 + 640 = 18560 (1.16s) 才动作
audio_back    = self.buffer[:15360]              # (15360,) float32  0.96s 垫料
process_chunk = self.buffer[15360:15360+2560]    # (2560,)  float32  0.16s 正菜
audio_ahead   = self.buffer[17920:18560]         # (640,)   float32  0.04s 垫料
self.buffer   = self.buffer[2560:]               # 滑动 160ms
```

**② 声音 → 离散 token** — `_audio_to_tokens()`,`service/model.py:406-459`

```python
audio_segment = np.concatenate([audio_back, process_chunk, audio_ahead])
#   → (18560,) float32

features = feature_extractor([audio_segment], ...)
#   features.input_features:  (1, 128, T)    Whisper mel 谱，T 随 padding 变
#   features.attention_mask:  (1, T)

outputs = glm_tokenizer(**features)             # WhisperVQEncoder
speech_tokens = outputs.quantized_token_ids     # (1, L) int64，L ≈ 15
#   L = 18560 / 1280 ≈ 15，每 1280 采样点(80ms)出 1 个 token

speech_token = speech_tokens[0][mask].tolist()  # List[int]，长度 L
audio_tokens = speech_token[12:14]              # List[int]，长度 2
#   12 = 15360 // 1280，只保留 chunk 对应的那 2 个
```

> **关键**:垫料对应的 token(索引 0~11 和 14)在这里被**丢掉**,
> 不进 LLM。它们只是让卷积编码器在边界上编得准。

**③ token → LLM 输入向量** — `_tokens_to_embeds()`,`service/model.py:461-477`

```python
tokens = torch.tensor(audio_tokens, device=self.device)   # (2,) int64
embeds = self.model.glm_tokenizer.codebook(tokens)        # (2, 1280) 查码本
embeds = self.model.audio_projector(embeds)               # (2, 1024) 转接头

# 不足 2 个时用 <|padding|> 补齐到 (2, 1024)
if embeds.shape[0] != self.chunk_token_len_small:         # chunk_token_len_small = 2
    embeds = torch.cat((embeds, self.audio_pad_embeds.expand(...)), dim=0)
```

`audio_projector` 就是 `EncoderProjector`(`model/model.py:17-35`):

```python
Linear(1280 → 2048) → ReLU → Linear(2048 → 2048) → ReLU → Linear(2048 → 1024)
#      audio_embed_dim                                            llm_dim
```

**④ 拼进序列** — `infer()`,`service/model.py:394-396`

```python
self.past_state["input_embeds"] = torch.cat(
    (self.past_state["input_embeds"], audio_embeds), dim=0
).unsqueeze(0)
#   → (1, S, 1024)   S = 上次残留的 token 数 + 2
```

首次调用时 `input_embeds` 是 prompt 的 embedding:

```python
# service/model.py:84-86, 107
self.input_text_tokens = tokenizer.encode("<|task_duplex_predict|><|punctuation_off|>")
self.text_embeds = embed_tokens_func(self.input_text_tokens)   # (2, 1024)
```

**⑤ 第一次 forward(问「有人说话吗」)** — `_asr()`,`service/model.py:483-490`

```python
outputs = self.model.llm(
    inputs_embeds=self.past_state["input_embeds"],      # (1, S, 1024)
    past_key_values=self.past_state["past_key_values"], # 累积的 KV cache
    use_cache=True,
)
logits = outputs.logits[0]                  # (S, 203566)  词表大小
pred   = torch.argmax(logits, -1)[-1]       # ()  标量，最后一个位置的预测

if pred != self.model.asr_eos_token_id:     # asr_eos_token_id = 151674
    # 有人说话 → 去跑级联 ASR
```

**⑥ 文本 → 向量,准备第二次 forward** — `service/model.py:666-678`

```python
input_embeds_next = self.audio_eos_embeds.unsqueeze(0)     # (1, 1, 1024)

if delta_text:
    ids = tokenizer.encode(delta_text, add_special_tokens=False)
    input_ids = torch.tensor(ids).to(self.device)          # (K,) int64
    embeds = embed_tokens_func(input_ids).unsqueeze(0)     # (1, K, 1024)
    input_embeds_next = torch.cat((embeds, input_embeds_next), dim=1)
    #   → (1, K+1, 1024)   文本 + 句尾标记
```

**⑦ 第二次 forward(问「什么状态」)** — `_state_predict()`,`service/model.py:686-693`

```python
outputs = self.model.llm(
    inputs_embeds=self.past_state["input_embeds"],   # (1, K+1, 1024)
    past_key_values=self.past_state["past_key_values"],
    use_cache=True,
)
logits = outputs.logits[0]                 # (K+1, 203566)
pred   = torch.argmax(logits, -1)[-1]      # ()  标量
state  = self.model.tokenizer.decode(pred) # str，如 "<|user_nonidle|>"
```

**⑧ 转折点的受限比较** — `service/model.py:703-711`

```python
# 不用 argmax，只比两个特定 token 的 logit
if logits[-1, 151676] > logits[-1, 151678]:   # complete vs incomplete
    state = "<|user_complete|>"
else:
    state = "<|user_incomplete|>"
```

**一张表总结所有形状:**

| 阶段 | 变量 | 类型 | shape | dtype |
|---|---|---|---|---|
| 入口 | `audio_chunk` | `np.ndarray` | `(N,)` | `float32` |
| 切片 | `process_chunk` | `np.ndarray` | `(2560,)` | `float32` |
| 拼接 | `audio_segment` | `np.ndarray` | `(18560,)` | `float32` |
| mel | `input_features` | `Tensor` | `(1, 128, T)` | `float32` |
| VQ | `quantized_token_ids` | `Tensor` | `(1, ~15)` | `int64` |
| 截取 | `audio_tokens` | `List[int]` | `len=2` | — |
| 码本 | `embeds` | `Tensor` | `(2, 1280)` | `bf16` |
| 转接头 | `embeds` | `Tensor` | `(2, 1024)` | `bf16` |
| 序列 | `input_embeds` | `Tensor` | `(1, S, 1024)` | `bf16` |
| logits | `logits` | `Tensor` | `(S, 203566)` | `float32` |
| 预测 | `pred` | `Tensor` | `()` 标量 | `int64` |
| 出口 | 返回值 | `dict` | — | — |

⚠️ dtype 中 `bf16` 来自 `config.yaml` 的 `precision: bf16`;
词表 203566 来自 `config/config.py:14` 的 `tokenizer_vocab_size`。
两者我**未在运行时实测**,是从配置推断的。

---

#### 3. 输出的 5 个词

模型是语言模型,它只会干一件事:**猜下一个词**。这 5 个是它可能猜出的:

| 模型输出 | 意思 |
|---|---|
| `<\|user_idle\|>` | 这块没内容(静音/噪音) |
| `<\|user_nonidle\|>` | 这块在说话 |
| `<\|user_backchannel\|>` | 这块是附和(「嗯嗯」「对对」) |
| `<\|user_complete\|>` | 说完了 |
| `<\|user_incomplete\|>` | 没说完 |

前三个是**常规输出**,平时就在这三个里挑。

后两个是**特殊输出**,只在一种情况下才出现 —— 见下面第 6 节。

---

#### 4. 声音怎么变成模型能读的东西(概念版)

上面 2.1 是逐行代码,这里是同一件事的概念版:

```
160 毫秒声音
   ---> 声音翻译器(GLM-4-Voice,现成工具,冻结不训练)
   ---> 2 个 token          每 80 毫秒出 1 个,所以 160 毫秒 = 2 个
   ---> 查码本变成向量
   ---> 2 个向量(1280 长)
   ---> 转接头(3 层 MLP)     因为声音那边是 1280 长,大模型只吃 1024 长
   ---> 2 个向量(1024 长)   ← 大模型能读了
```

**为什么要转接头?** 尺寸不匹配,加个转换层。就像插头不对要加转接器。

**哪些部分参与训练?**

| 组件 | 是否训练 | 依据 |
|---|---|---|
| GLM-4-Voice tokenizer | **冻结** | `model/model.py:56-58`,`requires_grad = False` + `eval()` |
| `audio_projector`(转接头) | 训练 | `enable_projector: true`,未设 `freeze_projector` |
| Qwen3-0.6B 主干 | **LoRA 微调** | `enable_lora: true`,`lora_r: 32`,`lora_alpha: 64` |
| 级联 ASR(Paraformer) | 冻结,外部工具 | 独立调用,不在计算图内 |

所以实际训练的只有**转接头 + LoRA 权重 + 扩展词表的 embedding**,
这也是 checkpoint 只有一个 `.pth`(`SoulX-Duplug-0.6B-Bilingual.pth`)的原因。

---

#### 5. 模型内部:一条不断变长的链子

这是理解整个模型的关键。

模型有「记忆」,它把每一块的信息**串成一条越来越长的链子**,
三种东西交错排列:

```
[任务说明] [声音][文字][状态] [声音][文字][状态] [声音][文字][状态] ...
           └── 第1块 ──┘  └── 第2块 ──┘  └── 第3块 ──┘
```

每来一块新声音,就往链子尾部接三样东西:**声音、识别出的文字、判定的状态**。

**这条链子就是「文本引导」的真身。**

模型判断第 10 块时,能回头看到前 9 块说了什么字。
所以它不是在听声音猜,而是在**读文字判断**——
「今天天气怎么样」后面该跟什么?一个语言模型太擅长这个了。

---

#### 6. 模型每块要回答两个问题

同一块声音,模型要跑**两次**,问两个不同的问题。

**问题一:这 160 毫秒里有人说话吗?**

```
喂进去:[上一块的状态] + [2 个声音向量]
模型答:下一个词猜是什么?
        猜「结束符」  ---> 没人说话
        猜别的       ---> 有人说话
```

**这一问是为了省算力。** 识别文字很慢,如果这块是静音就不用识别了。

**中间穿插:去识别文字**

只有上面判定「有人说话」,才调用**现成的语音识别工具**(Paraformer),
把**最近 3.2 秒**的声音转成文字,再算出**这次新增了哪几个字**:

```
上次识别:  「今天天气」
这次识别:  「今天天气怎么」
                  ↑ 新增 = 「怎么」
```

**问题二:这一块是什么状态?**

```
喂进去:[新增的字「怎么」] + [句子结束标记]
模型答:idle? nonidle? backchannel?
```

注意:**新增的字被喂进去了**。所以模型是看着文字判断的。

---

#### 7. 什么时候才判「说完没」

模型平时**不判断说完没**,只在 `idle`/`nonidle`/`backchannel` 里挑。

**只有一种情况例外:上一块在说话,这一块突然安静了。**

这时候代码不用模型的自由回答,而是强行让它二选一:

```
模型输出的一堆分数:
    idle        ████████████  8.2   ← 自由选会选这个
    complete    ███████       4.1   ┐ 代码只看这两个
    incomplete  █████         2.7   ┘
                              ↓
                   4.1 > 2.7  ---> 判定「说完了」
```

(分数是示意值,非实测)

**为什么要强行掐?**
因为这一刻模型自己选的最高分是 `idle`,它只是在说「现在没声了」,
**没回答「说完没」**。所以代码忽略它的自由回答,改问:
「在说完了和没说完之间,你倾向哪个?」

**四种组合,只有一种触发判决:**

| 上一块 \ 这一块 | 安静 | 在说话 |
|---|---|---|
| **安静** | 一直没开口,不判 | 刚开口,不判 |
| **在说话** | **转折点! 判!** | 继续说,不判 |

---

#### 8. 一个完整例子

你说「今天天气……怎么样」,中间停顿了一下。

每 160 毫秒一块,看模型每块输出什么:

| 时刻 | 你在干什么 | 识别出的文字 | 模型输出 |
|---|---|---|---|
| 0.0s | 还没开口 | — | `idle` |
| 0.5s | 说「今天」 | 今天 | `nonidle` |
| 1.0s | 说「天气」 | 天气 | `nonidle` |
| **1.2s** | **停顿想词** | — | **`incomplete`** ← 转折点,判了 |
| 1.4s | 还在停 | — | `idle` |
| 1.8s | 说「怎么样」 | 怎么样 | `nonidle` |
| **2.2s** | 说完了 | — | **`complete`** ← 转折点,判了 |

**看 1.2 秒那行**:这是第一个转折点(1.0s 在说话,1.2s 安静了)。
模型比较分数后判定「没说完」,所以系统继续等 —— 没有抢话。

**看 2.2 秒那行**:第二个转折点。这次判定「说完了」,系统接话。

**如果 1.2 秒之后你一直不说话呢?**
系统会数静音,数到 8 块(1.28 秒)就不等了,强行当你说完 ——
这是防止模型误判「没说完」导致永远等下去的兜底。

(此表按机制推演,非实测日志)

---

### 第二部分:整个系统

模型只管「一块声音 → 一个词」。外面还包了三层。

#### 9. 调用层次

```
浏览器/客户端
   ---> WebSocket /turn        收 base64 编码的音频
   ---> server.py              按 session_id 找到这个用户的会话
   ---> TurnSession            管超时(60 秒没动静就回收)
   ---> TurnTakingEngine       装/卸这个用户的「记忆」
   ---> TurnModel              真正干活的,全进程只有一个
```

#### 10. 系统做的第一件事:攒够料

声音是连续流进来的,但**不能来一点就算一次**。要攒够 **1.16 秒**:

```
        ┌─── 0.96 秒 ───┐┌─0.16秒─┐┌0.04秒┐
声音流  │    前面的      ││ 要判的  ││ 后面的│
        └───────────────┘└────────┘└──────┘
              垫料           正菜      垫料
```

**为什么要垫料?**

比方说你要认出一张照片中间那一小块是什么。
如果把周围全裁掉、只剩那一小块,可能就认不出来了。留着周围一圈才看得清。

声音也一样:中间那 0.16 秒单独切出来,切口处会失真。
前后各留一点,翻译成 token 才准。

> **垫料只是帮忙翻译,翻译完就丢掉,不进大模型。**
> 真正进大模型的只有中间 0.16 秒(2 个 token)。

**攒不够怎么办?** 返回 `blank`,意思是「我先存着,你继续发」。
这时候**模型压根没运行**。

---

#### 11. 系统做的第二件事:把模型的词翻译成指令

模型输出 5 个词,但对外只返回 4 个:

```
模型输出                       对外返回
──────────────────────────────────────────
(模型没运行)          --->     blank      音频不够,继续发
<|user_idle|>        --->     idle       没事发生,继续听
<|user_nonidle|>     --->     nonidle    他在说话,继续听
<|user_backchannel|> --->     idle       他在附和,继续听
<|user_incomplete|>  --->     idle       他没说完,继续听
<|user_complete|>    --->     speak      可以接话了!
```

**3 个不同的模型输出,对外全变成 `idle`。**

因为对调用方来说,「他在附和」和「他还没说完」**要做的事一模一样 ——
什么都别做,继续听**。所以不必区分。

对外只回答一个问题:**我现在该说话了吗?**
该 → `speak`,不该 → `idle` / `nonidle`。

**`speak` 时会额外附带一份整句话的文字**(重新跑一遍完整 ASR 得到)。

> 这就是为什么你会看到两套词。`complete`/`incomplete` 是**模型内部**的,
> `speak`/`idle` 是**对外**的。前面第 8 节例子里 1.2 秒那次 `incomplete` 判决,
> 从外面看只是个普通的 `idle`,完全看不出来它刚做过判决。

---

#### 12. 系统做的第三件事:多人共用一个模型

模型很占显存,不可能每个用户加载一份。做法是**共用模型、各自存记忆**:

```
用户 A 说话 ---> 装入 A 的记忆 ---> 算 ---> 把 A 的记忆存回去
用户 B 说话 ---> 装入 B 的记忆 ---> 算 ---> 把 B 的记忆存回去
```

代码里就是 `restore_runtime()`(装回去)/ `snapshot_runtime()`(存起来)。
用户 60 秒不说话,记忆被回收。

---

#### 13. 完整流程串起来

```
声音流进来
   ---> 攒够 1.16 秒?   不够 ---> 返回 blank
   ---> 切出中间 0.16 秒(前后当垫料,用完丢)
   ---> 翻译成 2 个 token ---> 转接头 ---> 2 个向量
   ---> 【模型问题一】有人说话吗?
   |        没人 ---> 新增文字 = 空
   |        有人 ---> 识别最近 3.2 秒 ---> 算出新增的字
   ---> 【模型问题二】这一块是什么状态?
   ---> 是转折点吗?(上一块在说话 + 这一块安静)
   |        不是 ---> 直接用模型的答案
   |        是   ---> 强行比较 complete / incomplete
   ---> 翻译成对外的词 ---> 返回 blank / idle / nonidle / speak
```

**一句话总结:**

```
声音 ---> token ---> 向量 ---> 模型(问2次) ---> 5 个词之一 ---> 翻译成 4 个词之一
                                  ↑
                            识别出的文字也喂进去
```

<details>
<summary>思考过程 & 代码依据</summary>

#### 关键数字的来源

| 参数 | 值 | 出处 | 换算 |
|---|---|---|---|
| `chunk_size` | 2560 | `config.yaml` | 160 ms |
| `audio_back_size` | 15360 | `config.yaml` | 0.96 s |
| `audio_ahead_size` | 640 | `config.yaml` | 40 ms |
| 触发阈值 | 18560 | 三者相加(`service/model.py:187-192`) | 1.16 s |
| `token_samples` | 1280 | `model/model.py:50`,`0.08 × 16000` | 80 ms/token |
| `chunk_token_len_small` | 2 | `config/config.py:189` | 160ms → 2 token |
| `llm_dim` | **1024** | `config.yaml` | 转接头输出维度 |
| `audio_embed_dim` | 1280 | `config/config.py:38` | 转接头输入维度 |
| `max_wait_num` | **8** | `config.yaml` 覆盖默认 10 | 8×160ms ≈ 1.28 s |
| `max_mistake_num` | **3** | `config.yaml` 覆盖默认 5 | — |
| `far_field_threshold` | 0.02 | `config.yaml` | RMS 阈值 |

⚠️ **`config/config.py` 的默认值会被 `config.yaml` 覆盖**
(`OmegaConf.merge(default, cfg)`,后者优先)。
已知三处不一致:`llm_dim`(2048→1024)、`max_wait_num`(10→8)、
`max_mistake_num`(5→3)。看代码时容易读错,**以 yaml 为准**。

#### 只取中间 2 个 token 的算法

`service/model.py:415-419`:

```python
start_index = len(audio_back) // self.model.token_samples      # 15360//1280 = 12
end_index = min(start_index + 2,
                math.ceil(audio_segment.shape[0] / self.model.token_samples))
                                                               # min(14, 15) = 14
```

所以 `audio_tokens = speech_token[12:14]` —— 正好 2 个,对应那 160ms。

#### 两次 forward 的确切位置

| | 函数 | 输入 | 预测 |
|---|---|---|---|
| 问题一 | `_asr()`,`service/model.py:483-490` | 上一状态 token + audio×2 | 是否 `asr_eos_token_id` |
| 问题二 | `_state_predict()`,`service/model.py:686-693` | delta_text + `EOS` | 状态 token |

#### 强制二选一判决的确切条件

`service/model.py:700-711`:

```python
if (self.past_state["state"] == "<|user_nonidle|>" and state == "<|user_idle|>") \
   or self.past_state["mistake_len"] >= self.config.infer_config.max_mistake_num:
    if logits[-1, user_complete_token_id] > logits[-1, user_incomplete_token_id]:
        state = "<|user_complete|>"
    else:
        state = "<|user_incomplete|>"
```

两个触发条件:
1. **状态转折**:上一块 `nonidle` → 这一块 `idle`
2. **连续误判兜底**:`mistake_len >= 3`

`mistake_len`(`service/model.py:695-698`):
模型说 `nonidle` 但 ASR 一个字都没出 → 计数 +1,否则清零。
防止「模型认为有语音、ASR 却出不来字」的死循环。

**注意:判决是纯比大小,没有阈值。** 4.1 vs 4.0 也算「说完了」。
所以**没法调灵敏度**,想调只能改代码加偏置:

```python
if logits[complete] > logits[incomplete] + bias:   # bias 越大越保守
```

这也解释了 1.28 秒兜底为什么必要 —— 判成 `incomplete` 后
没有任何置信度信息可用,只能用时间兜。

#### 级联 ASR 的「改口」问题与 KV 回滚

这是设计里最精巧、也最容易忽略的一段。

**问题**:级联 ASR 每次对**最近 3.2s 全部重跑**(`service/model.py:501`)。
后续音频到来后,ASR 可能**改写它先前的输出** ——
比如先识别成「是」,多听一会儿发现应该是「十」。

但模型那条链子是**因果的、已经写进 KV cache 的**,不能直接改。

**解法**:KV cache 回滚(`service/model.py:600-636`)

```python
if need_correction and self.past_state["checkpoint"] is not None:
    self.past_state["past_key_values"] = self.past_state["checkpoint"]   # 回滚
    # 重喂:修正后的上一块文本 + EOS + nonidle + 当前 chunk 音频
```

`checkpoint` 的语义(源码注释):**"KV after Audio, before Text"** ——
即「音频已喂入、文本还没喂入」的位置。回滚到这里就能重写那段链子。

**怎么判断需要修正**:`service/model.py:505-586`
把新的 3.2s 识别结果与历史文本都做归一化
(`zh_remove_punc` → `zh_norm` → `split_cn_en`),
两边 ≥5 token 时用 LCS(`get_lcs_substrings`)对齐,再分三种情况:

| 情况 | 处理 |
|---|---|
| `len(full) > len(history)` | 尾部一致 → 直接取增量;不一致 → 需修正 |
| `len(full) == len(history)` | 一致 → 增量为空;不一致 → 需修正 |
| `len(full) < len(history)` | ASR 缩短了 → 需修正 |

#### 状态机各分支的完整行为

| 模型输出 | 副作用 | 对外返回 |
|---|---|---|
| `<\|user_idle\|>` | 若在等待中:`wait_idle_cnt += 1`,达到 8 → 强制收尾;若 `history_len > 200`(≈32s)且无事发生 → `reset()` | `idle`,或 **`speak`** |
| `<\|user_nonidle\|>` | `speech_detected = True`;清等待计数;**回补**前面被误判为 idle/backchannel 的 chunk(最多 5 块) | `nonidle` |
| `<\|user_backchannel\|>` | 若还没检测到语音 → `reset()`;若在等待中 → `wait_idle_cnt = 1`(**重置而非清零**) | `idle` |
| `<\|user_complete\|>` | 若 `speech_detected`:跑整轮 ASR → `reset()` | **`speak`** + `text` |
| `<\|user_incomplete\|>` | `monitoring_wait_silence = True`;音频累进 `buffer_for_asr` | `idle` |

**回补机制**(`service/model.py:298-322`)值得单独说:
判定 `nonidle` 时会回头看最近的 history_chunks,
把紧邻的、之前被判成 `idle`/`backchannel` 的 chunk(最多 5 块)
一起补进 `buffer_for_asr`。

源码注释:
> `semantic vad, yield nonidle after the first word or character finished`
> `in case it's cut off in previous chunks`

因为语义 VAD 要等第一个字说完才敢报 `nonidle`,
那么这个字的前半段音频已经被当成 idle 丢了 —— 回补就是为了救回它。

#### 三个缓冲区的分工

| 缓冲区 | 长度 | 用途 |
|---|---|---|
| `buffer` | 滑动,≥18560 才处理 | 切 back/chunk/ahead |
| `cascade_buffer` | 裁剪到 3.2s | 喂级联 ASR 算**增量文本** |
| `buffer_for_asr` | 整轮累积 | `speak` 时出**整轮转写** |

⚠️ 三者初始化都是 `np.random.randn(...) * 0.0001` 量级的**微小噪声**而非零。
推测是避免全零输入让编码器/ASR 出现数值问题,但源码无注释,属**我的推断**。

#### 远场过滤(`service/model.py:243-255`)

在模型判定之后还有一道拦截:若 `RMS < far_field_threshold(0.02)`
且本轮尚未检测到语音、且模型判 `nonidle` → 直接 `reset()` 返回 `idle`。

这是防止远处别人说话被误当成用户在说。**纯能量阈值,很脆** ——
近处轻声说话可能被误杀,远处大声说话可能漏过。

#### `speak` 之后发生什么

`reset()` 把**所有**运行时状态清空:三个缓冲区、`past_state`
(含 KV cache 与累积文本)、`history_chunks`、等待计数。

所以每一轮对话在模型内部是**完全独立**的,不跨轮携带上下文。

#### 一个可能的挂起路径(推断,未实测)

转折点判断用的是 `past_state["state"] == "<|user_nonidle|>"`,
而它存的是**模型原始输出**,包括 `<|user_backchannel|>`。

所以若出现 `nonidle` → `backchannel` → `idle`:

1. 转折点条件不成立(上一块是 `backchannel`)→ 不判决
2. 永远进不了 `incomplete` 分支 → `monitoring_wait_silence` 保持 `False`
3. 而 `wait_idle_cnt` **只在 `monitoring_wait_silence == True` 时才累加**
   (`service/model.py:266-267`)→ 1.28 秒兜底不生效
4. `history_len > 200` 那条 reset 也不触发,因为它要求 `not speech_detected`,
   而此时已是 `True`(`service/model.py:283-288`)

结果:这一轮可能一直挂着,直到 session 60 秒 TTL 回收。

⚠️ **读代码推断的理论路径,未实测复现。**
实际中模型在用户说完后大概率预测 `idle` 而非 `backchannel`,
该路径可能很少发生。要确认需构造测试用例。

#### 未核验 / 存疑

- **训练时的序列构造**是否与推理一致(尤其 KV 回滚在训练中如何体现),
  本 repo 只含推理代码,需查训练分支或论文正文。
- `<\|assistant_backchannel\|>` 在 `config.py` 中有定义,
  但**推理代码里从未使用** —— 可能训练时才用,或为未来预留。
- 微噪声初始化的动机属推断,见上。
- 第 8 节的时间轴表是**按机制推演的示意**,非实测日志。

</details>

---

## Q3:不重新训练,能调哪些参数?

### 答

**先说结论:`config.yaml` 里真正安全可调的有 4 个。
其中 `complete_bias` 是本仓库在 2026-08-21 补上的 —— 原始实现里,
「更不容易触发 speak」这件事配置里根本调不了,必须改代码。现在能调了。**

---

#### 1. 原始实现为什么调不了 speak 灵敏度

上游版本里,`speak` 的主路径就是这一行:

```python
if logits[-1, user_complete_token_id] > logits[-1, user_incomplete_token_id]:
    state = "<|user_complete|>"
```

**纯比大小 —— 没有阈值、没有偏置、没有温度、没有任何可调参数。**

所以模型一旦判「说完了」,你无法要求它「再确信一点才算」。
`max_wait_num` 只管**兜底那条路**(判了 incomplete 之后数静音),管不到主路径。

这条限制在本仓库已经解除,见下一节。

---

#### 2. `complete_bias`:调控 speak 灵敏度(已实现)

2026-08-21 已并入本仓库,改动 3 个文件共 25 行:

| 文件 | 改动 |
|---|---|
| `config/config.py:178` | `InferConfig` 新增 `complete_bias: float = 0.0` |
| `config/config.yaml` | 暴露 `complete_bias: 0.0` |
| `service/model.py:703-720` | 判决改为带偏置比较,并在 developer_mode 下打印 logits |

判决处现在是(`service/model.py:720`):

```python
complete_logit = logits[-1, self.config.model_config.user_complete_token_id]
incomplete_logit = logits[-1, self.config.model_config.user_incomplete_token_id]
complete_bias = self.config.infer_config.complete_bias

if complete_logit > incomplete_logit + complete_bias:
    state = "<|user_complete|>"
```

`config.yaml` 里调:

```yaml
infer_config:
  complete_bias: 2.0     # 越大越保守，越不容易判「说完了」
```

| `complete_bias` | 效果 |
|---|---|
| `0.0` | **默认值**,与原始行为完全一致 |
| `+2.0` | 保守,倾向「还没说完」,更不容易抢话 |
| `-2.0` | 激进,倾向「说完了」,反应更快 |

⚠️ **必须加在 `InferConfig` 里**,不能只写进 yaml ——
`service/model.py:38` 是 `OmegaConf.merge(RunConfig(), cfg)`,
structured config 处于 struct 模式,yaml 出现 schema 里没有的 key 会直接
`ConfigKeyError`。这也是原方案里 `.get("complete_bias", 0.0)` 那种写法行不通的原因。

**该给多大?先实测。** 开 `developer_mode: true` 后每次判决会打印:

```
[Decision] complete=12.3125 incomplete=10.8750 diff=1.4375 bias=0.0
```

看几次「被抢话」时的 `diff` 有多大,`complete_bias` 设成略高于该值即可。
下面表里的 `2.0` 只是示意量级,logits 的绝对尺度仍未实测(见文末未核验)。

**注意偏置只作用于主判决路径。** 兜底路径(`max_wait_num` 数静音)不经过这个比较,
所以 bias 调再大也挡不住 1.28 秒后的强制接话 —— 那条得靠 `max_wait_num`。

---

#### 3. A 类:配置里能改,安全

改完重启服务即生效。

| 参数 | 当前值 | 管什么 | 调大 | 调小 |
|---|---|---|---|---|
| `complete_bias` | **0.0** | complete/incomplete 判决偏置(主路径) | 更保守,不易抢话 | 更激进,反应更快 |
| `max_wait_num` | **8** | 判了 incomplete 后,等多少块静音才兜底接话 | 等更久(更耐心) | 更快接话 |
| `max_mistake_num` | **3** | 容忍多少次「模型说有声但识别不出字」后强制判决 | 更宽容 | 更快强制判决 |
| `far_field_threshold` | **0.02** | 远场/环境音过滤的 RMS 阈值 | 过滤更严(抗噪) | 更灵敏(易误触发) |
| `asr.model_name` | paraformer | 识别引擎 | — | 换 `sensevoice` 支持英文/双语 |
| `asr.language` | — | 语言 | — | sensevoice 时设 `en` / `auto` |
| `developer_mode` | false | 打印每步耗时 | — | **调优必开** |

`max_wait_num` 换算:值 × 160ms。所以 8 → 1.28s,12 → 1.92s,15 → 2.4s。

---

#### 4. B 类:配置里有,但不要动

这些和**训练时的设定强耦合**,改了会与训练不一致,或破坏内部索引计算。

| 参数 | 值 | 为什么别动 |
|---|---|---|
| `chunk_size` | 2560 | 训练时就是 160ms,改了输入分布就变了 |
| `audio_back_size` | 15360 | 同上;且 token 索引 `15360//1280=12` 会跟着错 |
| `audio_ahead_size` | 640 | 同上 |
| `chunk_token_len_small` | 2 | 与 `chunk_size` 严格耦合(160ms ÷ 80ms) |
| `precision` | bf16 | 改 fp16 可能数值溢出 |
| `seed` | 42 | 推理阶段影响很小,但改了不利于复现 |

---

#### 5. C 类:硬编码在代码里

想调必须改源码。按「值不值得改」排序:

| 位置 | 值 | 管什么 | 值得改吗 |
|---|---|---|---|
| ~~`model.py:703`~~ | ~~(无)~~ | ~~complete/incomplete 判决偏置~~ | ✅ **已实现**,升为 A 类,见第 2 节 |
| `model.py:284` | 200 | 空转多少块后重置(≈32 秒) | 视场景 |
| `server.py:16` | 60 | session 空闲超时(秒) | 视并发 |
| `model.py:124,640` | 3.2s | 增量识别的回看窗口 | 谨慎 |
| `model.py:309` | 5 | 回补机制最多补几块 | 谨慎 |
| `model.py:260` | 5 | 保留最近几块历史 | 与回补耦合,别单独改 |
| `model.py:123` | 1.6s | 整轮 ASR 缓冲初始长度 | 不建议 |
| `model.py:512` | 5 | 触发 LCS 对齐的最小 token 数 | 不建议 |
| `server.py:17` | 10 | GC 轮询间隔(秒) | 无关紧要 |

---

#### 6. 按目标选参数(实操)

| 你的问题 | 怎么调 |
|---|---|
| **总抢我话** | `complete_bias: 2.0` + `max_wait_num: 12` |
| **反应太慢** | `max_wait_num: 5` + `complete_bias: -1.0` |
| **环境嘈杂,总被别人说话触发** | `far_field_threshold: 0.03~0.05` |
| **说英文/中英混说** | `asr.model_name: sensevoice` + `language: en` 或 `auto` |
| **想知道慢在哪** | `developer_mode: true` |
| **偶尔卡住不接话** | `max_mistake_num: 2`(更快强制判决) |

<details>
<summary>思考过程 & 代码依据</summary>

#### 完整配置项来源

`config/config.py:168-199` 定义 `InferConfig` 全部默认值,
`config/config.yaml` 覆盖其中一部分。
(`complete_bias` 于 2026-08-21 加在 `config/config.py:178`,py 与 yaml 默认值均为 `0.0`。)

⚠️ **已知三处 yaml 覆盖了 py 默认值**,以 yaml 为准:

| 参数 | py 默认 | yaml 实际 |
|---|---|---|
| `max_wait_num` | 10 | **8** |
| `max_mistake_num` | 5 | **3** |
| `llm_dim` | 2048 | **1024** |

另有两项在 py 中定义但**推理代码中未被使用**:
`single_round`、`asr.max_chunk_token_length`。

#### `max_wait_num` 的确切作用位置

`service/model.py:266-281`,只在 `<|user_idle|>` 分支内生效:

```python
if self.monitoring_wait_silence:          # 必须先判过 incomplete
    self.wait_idle_cnt += 1
    if self.wait_idle_cnt >= self.config.infer_config["max_wait_num"]:
        # 强制收尾，返回 speak
```

**注意两个前提**:
1. 必须 `monitoring_wait_silence == True`,即之前判过一次 `incomplete`
2. 必须 `speech_detected == True`,否则只 reset 不返回 speak

所以它**不是全局静音超时**,而是「判了没说完之后的耐心值」。

#### `max_mistake_num` 的确切作用

`service/model.py:695-702`:

```python
if state == "<|user_nonidle|>" and not delta_text:
    self.past_state["mistake_len"] += 1     # 模型说有声，但 ASR 出不来字
else:
    self.past_state["mistake_len"] = 0
```

达到阈值后**强制进入 complete/incomplete 判决**(与转折点条件是 `or` 关系)。
所以调小它 = 更快跳出「有声无字」的僵局。

典型触发场景:环境噪声让模型误判 `nonidle`,但 ASR 识别不出任何字。

#### `far_field_threshold` 的确切作用

`service/model.py:243-255`:

```python
if (self.get_rms(process_chunk) < self.config.infer_config.far_field_threshold
    and not self.speech_detected
    and state == "<|user_nonidle|>"):
    self.reset()
    return {"state": "idle", ...}
```

**三个条件同时成立才拦截**:音量低于阈值 + 本轮还没检测到语音 + 模型判了 nonidle。

关键是第二个条件 `not self.speech_detected` ——
**一旦本轮已经开始说话,这道过滤就完全失效**。
所以它只防「一开口就是远场噪声」,不防「说话中途混入远场噪声」。

`get_rms`(`service/model.py:142-152`):对 float32 输入算
`np.sqrt(np.mean(clip(x,-1,1)**2))`,标准 RMS。

#### 为什么不建议动 chunk 相关参数

token 索引是**硬算出来的**(`service/model.py:415-419`):

```python
start_index = len(audio_back) // self.model.token_samples   # 15360 // 1280 = 12
end_index = min(start_index + 2, ...)                       # 14
```

`token_samples = int(0.08 * 16000) = 1280` 写死在 `model/model.py:50`。
改了 `audio_back_size` 而不同步改这里,截取范围就错位。

而 `+2` 这个硬编码值与 `chunk_token_len_small` 是同一含义,
但**它是字面量,不读配置** —— 改配置不会改这里。这是个隐藏耦合。

#### 关于 `complete_bias` 的正确性

这个改动只涉及一处比较,不触碰模型权重和序列构造,
所以**不会引起训练/推理不一致** —— 等价于把决策边界平移。
默认 `0.0` 时表达式退化为原始的纯比大小,行为逐位一致。

两点注意:
1. logits 的绝对尺度未知,`2.0` 只是**示意量级**。
   实际合适值需实测,先开 `developer_mode` 看 `[Decision]` 那行的 `diff` 分布。
2. 这个偏置对**兜底路径无效** —— 兜底走 `max_wait_num` 计数,不经过该比较。

打印放在判决分支内部,而非每块都打 —— 该分支只在
「nonidle → idle 转折点」或 `mistake_len` 达阈值时进入,
所以格式化 tensor 带来的 GPU 同步开销可忽略。

#### 未核验

- `complete_bias` 已实现并通过语法/配置合并检查,但**未在带模型的机器上实际跑过**
  (核验环境无 conda,`torch`/`omegaconf` 不可导入)。
- 各参数调整后的实际效果**未做消融实验**,方向性判断来自代码逻辑推理。
- logits 的典型数值范围仍未实测,`complete_bias` 的合适量级待实测确定。

</details>

---

## Q4:有哪些和 SoulX 同类型的工作?(semantic VAD 专项)

### 答

**好消息:X2-Turn 把 SoulX 当基线做了实测对比,所以我们有了一张权威的横向对比表。**

**先看时间线 —— 这个方向 15 个月里出了 7 个方案:**

```
2020-10  TurnGPT              纯文本 LM 预测 turn-shift(学术源头)
   ...
2025-05  TEN Turn Detection   Qwen2.5-7B 纯文本三分类
2025-05  Smart Turn v2        wav2vec2 94.8M 纯音频
2025-09  EasyTurn             声学+语言双模态,已发 ICASSP 2026
2025-09  Phoenix-VAD          LLM + 滑窗(未开源)
2025-12  Smart Turn v3        换 Whisper Tiny,瘦身到 8M
2026-03  SoulX-Duplug  ←本repo 流式 160ms + 外挂 ASR
2026-03  JAL-Turn             冻结 SenseVoice+CPC
2026-04  FastTurn             流式 CTC + 声学,早决策
2026-08  X2-Turn              双头共享表征,帧同步 80ms
```

先看各方法,最后看总表。

---

#### 1. TEN Turn Detection(2025-05,工业开源,纯文本)

VAD 切段 → ASR 转写 → **文本模型**判断话轮。**级联三段式**,最传统的做法。
在 X2-Turn 的实验里作为 `Paraformer + TEN Turn`(中文)/
`SenseVoice En + TEN Turn`(英文)的下游判决器。

- **骨干**:**Qwen2.5-7B**(约 7B 参数 —— 全表最大)
- **输入**:**纯文本**,不看音频
- **三分类**:`finished`(说完了)/ `unfinished`(还会继续)/ `wait`(叫 AI 别说话)
- **语言**:中英双语,附带开源双语测试集
- **许可**:Apache-2.0 **附加额外限制**

**它的 `wait` 状态很特别** —— 专门识别「闭嘴」这类**明确要求 AI 沉默**的指令,
其他方案都没有这一类。

- **优点**:模块清晰,想换 ASR 就换
- **缺点**:必须等 VAD 切完段才能开始,延迟层层累加;7B 纯文本模型成本不低

---

#### 2. Smart Turn v2 (2025-05) / v3 (2025-12)(工业开源,纯音频)

pipecat 出的开源模型。**最反直觉的一点:它是纯音频模型,却自称 semantic VAD。**

**v2 与 v3 换了骨干,且大幅瘦身:**

| | v2 (2025-05) | v3 (2025-12) |
|---|---|---|
| 编码器 | wav2vec2 | **Whisper Tiny encoder** |
| 参数量 | 94.8M | **8M** |
| 权重体积 | 360MB | **8MB**(ONNX int8)/ 32MB(未量化) |
| 推理 | L40S GPU 12ms | **CPU 12ms** |

- **输入**:原始波形,**不读转写文本**
- **输出**:单个概率值,≥0.5 = 说完了
- **靠什么判断语义**:语调、填充词(「um…」「えーと…」)这类声学线索
- **消融结论**(v2):wav2vec2 + linear **优于** LSTM 和更深的 Transformer

**软肋很明显**:在 X2-Turn 的中文测试里,v3 的 `ACC_incomp` 只有 **60%** ——
纯声学判「没说完」确实吃力,因为「我想问一下」和「今天天气怎么样」
声学上都是一句话结束的样子。

> **但注意 v3 只有 8M 却拿到中文 `ACC_comp` 91.33%,高于 0.6B 的 SoulX(77.67%)。**
> 这说明「判说完了」这件事不需要很大的模型;难的是「判没说完」。

---

#### 3. EasyTurn(2025-09,西工大 ASLP + 华为,声学+语言双模态)

已发 **ICASSP 2026**(pp. 16957-16961)。开源模型 + 训练集 + 测试集(Apache-2.0)。

- **做法**:对 VAD 切好的整句,**联合预测转写和四类话轮状态**,
  用一个统一模型替掉「独立的下游话轮判决器」
- **四类状态**:`complete` / `incomplete` / `backchannel` / `wait`
- **测试集**:800 条,人工标注,真实:合成 = 1:1

**它是准确率上的王者** —— 中文 96.33 / 97.67 / 91.00,全面领先。
**但它不是流式的**,必须等 VAD 切完整句。

> 这个数据集正在成为该方向的事实标准 —— X2-Turn 用它评测,
> 连训练数据都用了它的中文子集。

---

#### 4. SoulX-Duplug(2026-03,本 repo)

- **做法**:把 chunk 级 ASR 和状态 token **交错进同一条自回归流**
- **流式**:✓ 每 160ms 出一个状态
- **仍依赖外挂 ASR**:X2-Turn 原文点明了这一点(见下)

---

#### 5. FastTurn(2026-04,同实验室,流式 CTC)

用**部分 CTC 假设** + 声学线索做早决策,不等整句说完。

X2-Turn 原文的描述:
> "FastTurn integrates partial CTC hypotheses with acoustic cues to make
> low-latency decisions as the transcript is incrementally updated,
> without waiting for utterance completion"

---

#### 6. JAL-Turn(2026-03,冻结特征)

冻结 SenseVoice + CPC 表征,在**候选边界点**分类 hold / shift 两态。
比上面几个粒度更粗(只判 hold/shift),但用冻结特征所以训练成本低。

---

#### 7. X2-Turn(2026-08,X Square Robot / 星海图,双头帧同步)

**这次调研发现的最新、也最激进的方案。**

- **做法**:在 Voxtral-Mini-4B-Realtime 上**加一个和 ASR 头并行的状态头**,
  两头**共享同一份因果解码器表征**,单次前向同时出 ASR token 和话轮状态
- **粒度**:**80ms 一帧**(比 SoulX 快一倍)
- **不需要外挂 ASR** —— ASR 就是自己的另一个头
- **五个状态**:`idle` / `noidle` / `incomplete` / `complete` / `backchannel`

**它对 SoulX 的评价(原文)**:
> "SoulX-Duplug further interleaves chunk-level ASR and state tokens within
> a single autoregressive stream, **achieving strong turn-taking performance**.
> However, it still relies on an external ASR model to guide state
> prediction during inference"

先肯定后指出局限 —— **不是全盘否定**。
而且它的 GitHub 致谢里把 SoulX-Duplug 列为「语义话轮 + 对话系统基座」,
说明对话系统那部分复用了 SoulX 的成果。

**一个关键的术语差异**:X2-Turn 原文特别声明,
它的 `noidle` 指「已开始说话但语义还不够」,**不是**静音或背景噪声 ——
明确说这与 SoulX 的定义不同。

---

### 横向对比表

#### 表 1:准确率与延迟(全景表)

**⚠️ 读表前必看:这张表的数字来自不同评测集,不能直接横向比较。**

「来源」列标明每行数字的出处:

- `X2` = X2-Turn 论文 Table 1,评测集 **EasyTurn-zh/en**(X2-Turn 配置 τ=480ms)
- `官方` = 该方法自己公布的数字,**评测集各不相同**
- `—` = 未找到公开数据

| | 方法 | 时间 | 流式 | ACC_comp ↑ | ACC_incomp ↑ | ACC_bc ↑ | 延迟(ms) ↓ | 来源 |
|---|---|---|---|---|---|---|---|---|
| **ZH** | Paraformer + TEN Turn | 2025-05 | ✗ | 86.67 | 89.30 | – | vad + 204 | X2 |
| | TEN Turn(纯文本,自测集) | 2025-05 | ✗ | 98.90 | 92.74 | – | | 官方 |
| | Smart Turn V3 | 2025-12 | ✗ | 91.33 | **60.00** | – | vad + 24 | X2 |
| | EasyTurn | 2025-09 | ✗ | **96.33** | **97.67** | 91.00 | vad + 263 | X2 |
| | **SoulX-Duplug** | **2026-03** | ✓ | **77.67** | 88.96 | – | 295 | X2 |
| | X2-Turn | 2026-08 | ✓ | 91.00 | 93.00 | **96.00** | 288 | X2 |
| | LiveKit 文本版(自测集) | 2025 | ✗ | 99.30※ | 86.60※ | – | 50~160 | 官方 |
| | Phoenix-VAD | 2025-09 | ✓ | | | | | — |
| | JAL-Turn | 2026-03 | ✓ | | | | | — |
| | FastTurn | 2026-04 | ✓ | | | | | — |
| **EN** | SenseVoice En + TEN Turn | 2025-05 | ✗ | 95.60 | 76.59 | – | vad + 57 | X2 |
| | TEN Turn(纯文本,自测集) | 2025-05 | ✗ | 90.64 | 98.44 | – | | 官方 |
| | Smart Turn V3 | 2025-12 | ✗ | 78.93 | 72.24 | – | vad + 21 | X2 |
| | EasyTurn | 2025-09 | ✗ | | | | | — |
| | **SoulX-Duplug** | **2026-03** | ✓ | 89.33 | 79.33 | – | 205 | X2 |
| | X2-Turn | 2026-08 | ✓ | **92.10** | **84.60** | – | 225 | X2 |
| | LiveKit 文本版(自测集) | 2025 | ✗ | 99.30※ | 87.00※ | – | 50~160 | 官方 |
| | Phoenix-VAD | 2025-09 | ✓ | | | | | — |
| | JAL-Turn | 2026-03 | ✓ | | | | | — |
| | FastTurn | 2026-04 | ✓ | | | | | — |

**指标定义**(X2-Turn 原文):`ACC_comp/incomp/bc` 是**整句级**准确率,
拿「最后一个非 idle 的预测状态」与真实标注比。
`latency_vad` 是级联方法必须付的前端 VAD 切段延迟(**论文未给具体数值**)。
`–` = 该方法不支持这一类;**空白 = 数据不可得**。

**※ LiveKit 那两行是我做的指标映射**:官方报的是 TPR / TNR,
我按 TPR ≈「判对说完了」→ `ACC_comp`、TNR ≈「判对还会继续」→ `ACC_incomp` 对应。
**语义相近但定义未必严格等价**,仅供参考。

**Smart Turn V2 未列入**:官方只给了「均衡测试集上的总体准确率」
(中文 87.2%、英文 94.3%),**没有分开报 comp / incomp**,
放进任何一列都会误导,故留空不填。

---

**同一方法两组数字差很远,为什么?**

以 TEN 中文为例:X2 测出 `ACC_comp` 86.67,官方自测 98.90 —— **差 12 个点**。

三个原因:

1. **评测集不同**:X2 用 EasyTurn,官方用自建测试集
2. **X2 测的是级联端到端**(Paraformer 转写 → TEN 判决),
   官方测的是**纯文本输入**(相当于给定完美转写)—— 前者含 ASR 错误传播
3. 指标定义可能不完全对齐

**所以「官方」那几行只能看量级,不能和 `X2` 行比大小。**

这也提示一件事:**级联方案的真实表现被 ASR 错误拖累**。
TEN 本身判得很准(98.90),但接上 Paraformer 后掉到 86.67 ——
掉的这 12 个点是 ASR 的错,不是 TEN 的错。
而 SoulX / X2-Turn 这类端到端方案,ASR 错误已经内含在分数里了。

---

**四个方法完全没有可比数据:**

| 方法 | 情况 |
|---|---|
| Phoenix-VAD | 摘要未给任何数值,未开源,无法复现 |
| JAL-Turn | 仅从 X2-Turn 引述得知存在,未读原文 |
| FastTurn | 摘要称「准确率更高、延迟更低」但**未给数值,也未点名基线** |
| EasyTurn(英文) | X2-Turn 只在中文行列了它 |

**FastTurn 这个缺口值得注意** —— 它和 SoulX 同实验室、同批作者,
按理最该直接对比,但两篇都没有对方的数字。

---

**怎么读这张表(只看 `X2` 行,因为只有它们同基准):**

1. **SoulX 中文 `ACC_comp` 77.67% 是全表最低** —— 最需改进处。
   而 `ACC_incomp` 88.96% 尚可 → **它偏保守,倾向判「没说完」**。
   这与 Q3 发现的「无阈值 + 1.28 秒硬兜底」表现一致
   (X2 评测用的是原始实现;本仓库现已加 `complete_bias`,见 Q3 第 2 节)。
2. **Smart Turn V3 中文 `ACC_incomp` 只有 60%** —— 纯声学判「没说完」确实难。
   但它 8M 的体量拿到 `ACC_comp` 91.33,**反超 0.6B 的 SoulX**。
3. **EasyTurn 准确率最高(96.33/97.67)但不流式** —— 延迟还要加 `latency_vad`。
4. **只有 SoulX / X2-Turn / Phoenix-VAD / JAL-Turn / FastTurn 是流式**,
   前两者延迟数字完整(不用加 VAD)。
5. **X2-Turn 在流式阵营全面超过 SoulX**,延迟相当(288 vs 295 / 225 vs 205)。
6. **`ACC_bc` 一栏几乎全空** —— 只有 EasyTurn(91.00)和 X2-Turn(96.00)
   报了附和识别。多数方案根本不区分「嗯嗯」和真的要说话。

> ⚠️ 这张表是 X2-Turn 作者跑的,**存在自评偏向的可能**。
> SoulX 的低分未必反映其最佳配置(比如是否调过 `max_wait_num`、
> 用哪个 ASR 后端,论文未说明)。**要坐实需要自己复现。**

#### 表 2:工程属性对比(时间、开源、体量、依赖)

按时间排序:

| 方法 | 提出时间 | 开源 | 权重 | 参数量 | 骨干 | 需外挂 ASR | 粒度 | 许可 |
|---|---|---|---|---|---|---|---|---|
| TurnGPT | 2020-10 | ✓ | ✓ | 未核实 | GPT-2 系 | 是(纯文本) | 词级 | 未核实 |
| TEN Turn Detection | 2025-05 | ✓ | ✓ HF | **~7B** | **Qwen2.5-7B** | **是** | 整句 | Apache-2.0 + 附加限制 |
| Smart Turn v2 | 2025-05 | ✓ | ✓ HF | **94.8M** | wav2vec2 | 否(不读文本) | 8s 窗口 | 未核实(疑 BSD-2) |
| EasyTurn | 2025-09 | ✓ | ✓ HF | 未核实 | 未核实 | 内含 | 整句 | Apache-2.0(数据集) |
| Phoenix-VAD | 2025-09 | **未见仓库** | 未见 | 未公开 | LLM(型号未公开) | 未公开 | 滑窗 | — |
| **Smart Turn v3** | **2025-12** | ✓ | ✓ HF | **8M** | **Whisper Tiny encoder** | 否(不读文本) | 窗口 | 未核实 |
| **SoulX-Duplug** | **2026-03** | ✓ | ✓ HF | **0.6B + LoRA** | Qwen3-0.6B | **是**(Paraformer) | **160ms** | Apache-2.0 |
| JAL-Turn | 2026-03 | 未核实 | 未核实 | 未核实 | 冻结 SenseVoice+CPC | 冻结内含 | 候选边界 | 未核实 |
| FastTurn | 2026-04 | 仓库存在 | 未核实 | 未核实 | CTC + 声学 | 内含 CTC | 增量 | 未核实 |
| **X2-Turn** | **2026-08** | ✓ | ✓ HF | **4B**(全参微调) | Voxtral-Mini-4B-Realtime | **否**(自带头) | **80ms** | Apache-2.0 |

**两个纠错**(上一版我写错了,已核验原始页面修正):

1. **TEN Turn Detection 是 Qwen2.5-7B,约 7B 参数** —— 不是我之前写的"文本模型/未核实"。
   它是**纯文本三分类**(`finished`/`unfinished`/`wait`),中英双语,
   许可是 Apache-2.0 **附加额外限制**。
2. **Smart Turn v3 换了骨干** —— 从 v2 的 wav2vec2(94.8M)
   改成 **Whisper Tiny encoder,只有 8M**,ONNX int8 量化后 **8MB**,
   CPU 推理 12ms。v2 → v3 是**大幅瘦身**,不是简单升级。

**体量跨度极大 —— 从 8M 到 7B,差了近 900 倍:**

```
8M    Smart Turn v3      (Whisper Tiny,纯音频)
94.8M Smart Turn v2      (wav2vec2,纯音频)
0.6B  SoulX-Duplug       (Qwen3-0.6B + 外挂 ASR)
4B    X2-Turn            (Voxtral-4B,双头)
~7B   TEN Turn Detection (Qwen2.5-7B,纯文本)
```

**有意思的是:表 1 里 8M 的 Smart Turn v3 中文 ACC_comp(91.33)
反而高于 0.6B 的 SoulX(77.67)。** 参数量不是决定因素 ——
架构与训练数据更关键。

**关键权衡:X2-Turn 用 4B 换掉了外挂 ASR。**

| | SoulX | X2-Turn |
|---|---|---|
| 主模型 | 0.6B(LoRA 微调) | **4B(全参微调)** |
| 外挂 ASR | 需要 Paraformer | 不需要 |
| 显存 | 较低 | **≥24GB** |
| 前向次数/块 | 2 次 LLM + 1 次 ASR | **1 次前向,双头输出** |

**所以"省掉一个 ASR"的代价是主模型大了近 7 倍。**
这是真实的权衡,不是单方面的进步 ——
SoulX 的 0.6B + 即插即用在资源受限场景仍然有意义。

---

### X2-Turn 项目链接(已开源)

| 资源 | 地址 |
|---|---|
| 代码 | https://github.com/X-Square-Robot/X2-Turn |
| 权重 | https://huggingface.co/x-square-robot/X2-Turn-4B-0812 |
| 论文 | https://arxiv.org/abs/2608.10878 |
| 基座 | https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602 |

许可 Apache-2.0,代码 + 权重都给了。含浏览器 demo(只要 GPU + 4B 权重,
不需要 LLM/TTS/vLLM)、全双工 demo、打过补丁的 vLLM 集成。

⚠️ 但仓库**非常新**:11 star / 4 commits / 0 fork(截至查询时),
社区验证几乎为零。

---

### 一个对 SoulX 有用的启示

X2-Turn 的状态设计里有个东西 SoulX 没有 —— **它的 API 返回逐帧置信度**:

```python
for f in result.turn_frames:
    print(f.start_ms, f.end_ms, f.label, f.confidence)
```

这正好补上 Q3 里指出的 SoulX 缺陷:**判决是纯比 logits 大小、没有置信度输出**,
所以只能靠固定 1.28 秒兜底。
(本仓库加的 `complete_bias` 只是平移了决策边界,仍不产出置信度 ——
两个 logit 的差值可以当粗糙的置信度代理,但没有校准过。)有了置信度就能做「很确定没说完就多等,
不太确定就少等」的自适应策略。

<details>
<summary>思考过程 & 参考资料</summary>

#### 时间信息的来源

| 方法 | 时间 | 依据 |
|---|---|---|
| TurnGPT | 2020-10 | arXiv:2010.10874 编号 |
| TEN Turn Detection | 2025-05 | ModelScope 模型页 2025-05-28;GitHub 无 Releases,引用标 2025 |
| Smart Turn v2 | 2025-05 | 搜索结果显示 v2 时间;⚠️ HF 页面未标日期 |
| EasyTurn | 2025-09 | arXiv:2509.23938 编号 |
| Phoenix-VAD | 2025-09 | arXiv:2509.20410 编号(v1 2025-09-24) |
| Smart Turn v3 | 2025-12 | HF 页面搜索结果显示 2025-12-02 |
| SoulX-Duplug | 2026-03 | arXiv:2603.14877 编号 |
| JAL-Turn | 2026-03 | arXiv:2603.26515 编号 |
| FastTurn | 2026-04 | arXiv:2604.01897 编号 |
| X2-Turn | 2026-08 | arXiv:2608.10878(v1 08-11,v2 08-19) |

⚠️ **arXiv 编号推断的是首次预印本时间,不是正式发表时间。**
如 EasyTurn 预印本 2025-09,正式发表在 ICASSP 2026。
工业模型(TEN / Smart Turn)无论文,时间取自模型页/仓库,精度较低。

#### 上一版的两处错误(已核验修正)

**错误 1:TEN Turn Detection 的规模**
我之前写「文本模型 / 参数量未核实」。核验 GitHub README 后确认:
**基座是 Qwen2.5-7B,约 7B 参数** —— 是本表中最大的模型。
且它有第三个状态 `wait`(用户明确要求 AI 别说话,如 "Shut up"),
这个类别其他方案都没有。
许可是 **Apache-2.0 附加额外限制**,不是纯 Apache-2.0。

其官方自测数据(TEN 自己的测试集,非 EasyTurn):

| 语言 | finished | unfinished | wait |
|---|---|---|---|
| 英文 | 90.64% | 98.44% | 91% |
| 中文 | 98.90% | 92.74% | 92% |

⚠️ 注意这与表 1 里 TEN 的分数(中文 86.67 / 89.30)差异很大 ——
**因为评测集不同**(TEN 自测集 vs EasyTurn),且表 1 是级联 Paraformer 后的端到端结果。
**不可直接比较。**

**错误 2:Smart Turn v3 的骨干**
我之前写「wav2vec2 系 / 未核实」。核验 HF 页面后确认:
**v3 换成 Whisper Tiny encoder,参数量只有 8M**(v2 是 wav2vec2 94.8M),
ONNX int8 量化后 **8MB**,**CPU 推理 12ms**。
所以 v2 → v3 是**换骨干 + 大幅瘦身**,不是简单版本升级。

⚠️ v3 的许可证、支持语言数、各语言准确率 HF 页面均未列出,仍未核实。

#### 一个值得注意的观察

**参数量与准确率不成正比:**

```
8M    Smart Turn v3   → ZH ACC_comp 91.33
0.6B  SoulX-Duplug    → ZH ACC_comp 77.67   ← 大 75 倍，反而更低
4B    X2-Turn         → ZH ACC_comp 91.00
~7B   TEN Turn (级联)  → ZH ACC_comp 86.67
```

「判说完了」这件事似乎不需要大模型;
真正难的是「判没说完」(Smart Turn v3 的 `ACC_incomp` 只有 60%)。
**SoulX 的问题不在容量,而在别处** —— 可能是训练数据、
判决机制(评测时无阈值纯比大小,见 Q3),或评测配置。

#### 表 1 各行数据的确切来源

| 行 | 来源 | 评测集 | 核验方式 |
|---|---|---|---|
| 标 `X2` 的 9 行 | X2-Turn 论文 Table 1 | EasyTurn-zh/en | **抓 HTML 全文,原文直引** |
| TEN 官方 2 行 | TEN GitHub README | TEN 自建双语测试集 | 抓 GitHub 页面 |
| LiveKit 2 行 | LiveKit 官方文档 | LiveKit 自建集 | 抓文档页面 |
| 空白单元格 | — | — | 检索后确认无公开数据 |

**未列入表 1 的数据及原因:**

- **Smart Turn v2**:官方只给「均衡测试集总体准确率」
  (ZH 87.2 / EN 94.3 / TR 96.8 / NL 96.7 / DE 95.8 ...),
  **未分开报 comp / incomp**。填进任一列都会误导,故不列。
- **Phoenix-VAD**:摘要与 arXiv 页面**未给任何数值**,无开源仓库,无法自测。
- **FastTurn**:摘要称「更高准确率、更低打断延迟」,
  但**未给具体数值,也未点名任何基线系统**。
- **JAL-Turn**:仅从 X2-Turn 参考文献引述得知,**未抓原文**。
- **EasyTurn 英文**:X2-Turn Table 1 只在 ZH 分区列了 EasyTurn,EN 分区没有。

#### LiveKit 指标映射的依据与风险

LiveKit 官方报的是 TPR / TNR,我做了如下映射:

| 官方指标 | 官方定义 | 我映射到 |
|---|---|---|
| TPR (True Positive Rate) | 正确识别「用户说完了」 | `ACC_comp` |
| TNR (True Negative Rate) | 正确识别「用户还会继续」 | `ACC_incomp` |

**语义相近,但严格性不足**:
`ACC_comp` 是「整句级、取最后一个非 idle 预测」与标注比对(X2-Turn 定义),
而 TPR 是标准二分类召回率。**两者计算方式不同**,只能看量级。

LiveKit 文本版全语言数据(供参考):

| 语言 | TPR | TNR |
|---|---|---|
| 印地语 | 99.4% | 96.3% |
| 韩语 | 99.3% | 94.5% |
| 英语 | 99.3% | 87.0% |
| 中文 | 99.3% | 86.6% |
| 意大利语 | 99.3% | 85.1% |

**规律很清楚:TPR 全部 99%+,TNR 只有 85~96%。**
即「判说完了」容易,「判还没说完」难 ——
与 Smart Turn V3 中文 `ACC_incomp` 只有 60% 是同一个病,只是没那么重。

#### 一个跨表推论:级联方案的分数被 ASR 拖累

```
TEN 纯文本输入(完美转写)      ZH ACC_comp = 98.90
TEN + Paraformer 端到端        ZH ACC_comp = 86.67
                                   ↓
              这 12 个点的落差是 ASR 错误传播造成的
```

**这对评价 SoulX 有意义**:SoulX 的 77.67 是**含 ASR 错误的端到端分数**,
而它内部也用 Paraformer。如果按同样逻辑,
它「假设完美转写」时的上限应当更高 —— 但无人测过这个上限。

⚠️ 这是**我的推论**,非文献结论。要验证需要给 SoulX 喂标注文本做 oracle 实验。

#### 检索与核验路径

1. 三簇检索:学术源头(TurnGPT)/ 工业方案(smart-turn, TEN)/ LLM 语义端点
2. 发现 X2-Turn 后**抓取其 HTML 全文**,拿到 Table 1 完整数值 + 指标定义
3. 抓取 GitHub 仓库核实开源状态、权重地址、许可
4. 抓取 HF 模型页核实 Smart Turn v2 架构与参数量
5. 抓取 ModelScope 核实 EasyTurn 测试集构成

**表 1 数值直接引自 X2-Turn 论文 Table 1 原文**,非二手转述。

#### 新增文献(编号从 F14 续)

| ID | 文献 | 出处 | 级 |
|---|---|---|---|
| **F14** | *Phoenix-VAD: Streaming Semantic Endpoint Detection for Full-Duplex Speech Interaction* | arXiv:2509.20410 (v4) | T2 |
| **F15** | Li et al. *Easy Turn* | arXiv:2509.23938;**ICASSP 2026, pp.16957-16961** | **T1** |
| **F16** | Fu et al. *X2-Turn* | arXiv:2608.10878 (v2, 2026-08-19) | T2 |
| **F17** | Smart Turn v2 (pipecat-ai) | HF + GitHub | **T3** |
| **F18** | Ekstedt & Skantze *TurnGPT* | arXiv:2010.10874;**EMNLP 2020 Findings** | **T1** |
| **F19** | Yang et al. *JAL-Turn: Joint Acoustic-Linguistic Modeling for Real-Time and Robust Turn-Taking Detection* | arXiv:2603.26515 | T2 |
| **F20** | Liao et al. *FlexDuo: A Pluggable System for Enabling Full-Duplex Capabilities* | arXiv:2502.13472 | T2 |
| **F21** | Liu et al. *Voxtral Realtime* | arXiv:2602.11298 | T2 |
| **F22** | Zeghidour et al. *Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling* | arXiv:2509.08753 | T2 |

F19/F20 是从 X2-Turn 参考文献中发现的,**我只读到其被引述的一句话描述,
未抓取原文核验**。F21/F22 是 X2-Turn 的方法基础。

#### X2-Turn 方法细节(全文核验)

**骨干**:Voxtral Realtime = 因果音频编码器 → 时序适配器(降采样到 12.5Hz)
→ decoder-only LM,每 80ms 出 1 个 token。

**双头**:保留原 ASR 头,新增并行状态头,共享隐状态 `h_i`。
联合损失 `L = L_asr + λ·L_turn`,**λ = 0.1**。

**推理时的一个重要设计**:自回归解码**只由 ASR 头驱动**,
状态头的预测**不反馈进解码循环**。
所以状态预测出错不会污染后续 ASR —— 这比 SoulX 的设计更稳健
(SoulX 的状态 token 会写进序列影响后续)。

**ASR-anchored 监督**:词级标注投影到帧级。
词边界 token `[W]` 锚在**词的起始**(而非 Voxtral 原版的词尾),
位置 `p_i = round(s_i/Δ) + n_τ`,Δ=80ms。
这样 ASR 和状态预测能更早可用。

**训练**:两阶段。Stage1 用 26k 小时中英 ASR 数据适配延迟流协议;
Stage2 加状态头(**用 ASR 头的拷贝初始化**),
在 EasyTurn 中文 126h + Fisher 英文 249h 上联合微调。
τ 每 batch 在 1~30 帧(80~2400ms)随机采样,所以单模型覆盖多种延迟配置。

**标注工具**:Qwen3-ForceAligner 做词级时间戳,Qwen3.5-Plus 做词级语义状态标注。

#### 延迟指标的定义(重要)

X2-Turn 的 latency 定义(原文 Eq.5):对跨越 `[s_i, e_i]` 的词,
状态在 `s_i + τ` 可得,故相对词尾的延迟为

```
L_i = τ - (e_i - s_i)
```

**可以为负** —— 若词长超过 τ,状态在词说完前就已可得。
报告的是全测试集平均。

对级联基线,报的是「推理时间 + 前端 VAD 延迟」,
原文明确说这**与 L_i 不严格等价**。
所以表 1 里流式与非流式的延迟数字**不能直接横向比较** ——
这是我读表时最该注意的一点。

#### τ 的消融(原文 Table 2)

| τ (ms) | ZH ACC_comp | ZH ACC_incomp | ZH 延迟 | EN ACC_comp | EN ACC_incomp | EN 延迟 |
|---|---|---|---|---|---|---|
| 480 | 91.00 | 93.00 | 288 | 92.10 | 84.60 | 225 |
| 400 | 88.70 | 94.30 | 208 | 85.20 | 85.30 | 145 |
| 320 | 87.33 | 94.00 | 120 | 82.70 | 87.60 | 65 |

有意思的是:**τ 降低时 `ACC_incomp` 反而略升**(ZH 93.00 → 94.00)。
只有 `ACC_comp` 明显下降。合理解释是看得少更倾向判「没说完」。

#### 关于 Phoenix-VAD 的定位重叠

`[F14]` 比 SoulX 早 6 个月,自我定位几乎逐条对应:

| | Phoenix-VAD (2025-09) | SoulX (2026-03) |
|---|---|---|
| 自述 | "plug-and-play full-duplex prediction module for semantic endpoint detection" | "plug-and-play streaming state prediction module... semantic VAD" |
| 骨干 | LLM | LLM |
| 流式手段 | sliding window 训练策略 | 逐块 160ms + KV cache |
| 解耦主张 | 可独立于对话模型优化 | 即插即用,避免灾难性遗忘 |

⚠️ 但注意:**X2-Turn 的参考文献里没有 Phoenix-VAD**。
可能原因:未被同行注意到、或作者认为不相关。
Phoenix-VAD 的 arXiv 备注写着 "It requires internal PR approval",
v3 曾撤稿 v4 重新上线,**也未见开源仓库** —— 影响力可能有限。

#### SoulX 的独特性还剩什么

| 主张 | 是否仍独特 |
|---|---|
| plug-and-play | ✗ Phoenix-VAD 早 6 个月;FlexDuo `[F20]` 也是 "pluggable" |
| LLM-based 语义端点 | ✗ Phoenix-VAD 同 |
| 联合流式 ASR | △ X2-Turn 做得更彻底(共享表征) |
| 双语评测集 | △ EasyTurn 已有中英测试集 |
| **KV cache 回滚纠错** | **✓ 未在其他工作中见到** |

KV 回滚是它级联架构的必然产物 ——
X2-Turn 那种共享表征方案根本不会有 ASR 改口问题。

#### 未核验 / 局限

- **表 1 数值全部来自 X2-Turn 自评**,存在自评偏向可能。
  SoulX 的配置(ASR 后端、`max_wait_num` 等)论文未说明。
- `latency_vad` 的**具体数值论文未给**,所以级联方法的真实延迟无法确定。
- **表 2 中大量「未核实」是诚实标注** ——
  TEN Turn Detection、EasyTurn、FastTurn、JAL-Turn 的参数量和许可
  我未逐一抓取原始页面确认。
- Smart Turn v2 的许可证:HF 页面未显示,搜索结果提到 GitHub 是 BSD-2,
  **我未直接核验 LICENSE 文件**。
- Smart Turn **v3** 我未核验(表 1 用的是 v3,我核验的是 v2 —— 两者参数量可能不同)。
- F19 (JAL-Turn) / F20 (FlexDuo) 仅从 X2-Turn 引述得知,未读原文。
- Phoenix-VAD 的性能数值、模型规模、是否开源均**未公开或未核实**。

</details>

---

## Q5:有没有引入视觉模态的 Semantic VAD?

### 答

**核心结论:有「视觉 + 话轮预测」的工作,但没有「视觉 + 语义完整性判断」的工作。**

这两件事必须分开看,因为它们回答的问题不同:

| | 问的问题 | 靠什么 | 有视觉工作吗 |
|---|---|---|---|
| **话轮预测**(hold/shift) | 下一个谁说话? | 声学节奏 + 视觉前兆 | **✓ 有,且已验证有效** |
| **语义完整性**(complete/incomplete) | 这句话说完了没? | 语义内容 | **✗ 检索未发现** |

**SoulX 属于第二类。所以「视觉 + Semantic VAD」目前是个空位。**

---

#### 1. MM-VAP(2025-05,ACL 2025 Findings)—— 最直接的证据

Trinity College Dublin,**代码开源**(`github.com/russelsa/mm-vap`)。

**它证明了视觉确实有用:**

| 模型 | 平衡准确率 | F1(shift) |
|---|---|---|
| 纯音频 VAP | 79% | 0.70 |
| **MM-VAP(音频+视觉)** | **83%** | **0.74** |
| 纯视觉 | 68% | 0.47 |

**视觉带来 4 个点提升**,而且**在所有静音时长上都优于纯音频**
(长静音 ≥750ms 时:78% vs 75%)。

**但它做的是 hold/shift 预测,不是语义完整性判断。**

---

#### 2. MM-VAP 的消融实验:哪种视觉特征真有用

这是全文最有价值的部分。它把视觉特征逐类去掉看影响:

| 视觉特征 | 维度 | 平衡准确率 | 相比全特征 |
|---|---|---|---|
| 全部特征 | 60 | **83%** | — |
| 仅 FAU(面部动作单元) | 17 | 81% | ↓2% |
| 仅头部姿态 | 6 | 80% | ↓3% |
| 仅面部关键点 | 30 | 74% | ↓11% |
| **仅注视(gaze)** | 6 | **67%** | **↓20%** |

**最反直觉的一点:单独用注视是最差的,掉 20 个点。**

论文的解释:视频会议场景本身扭曲注视行为
(摄像头位置 ≠ 视线焦点),且 gaze 提取受光照影响大。

**它还发现了一个具体可用的信号** —— 转折点前 200ms 的 FAU 分析显示:

> 下一个要说话的人,在开口前会出现**嘴部、唇部、下颌、下巴的活动增强**,
> 部分类似说话时的表情。

这是「准备说话的前兆动作」,纯音频完全捕捉不到。

---

#### 3. MuVAP(2026-06,Interspeech 2026)—— 视觉的另一种用法

KTH,Skantze 组(VAP 原作者)。同样用人脸,但**视觉的角色完全不同**。

- **视觉不是语义线索,而是"分离锚点"** ——
  单声道音频 + 单摄像头,用人脸把声音归属到正确的人
- 27.7M 参数(ASD 20.7M + VAP 4.9M + 融合 2.1M)
- 人脸处理:InsightFace + RetinaFace 检测,SCRFD 阈值 0.62,
  ArcFace 做身份跟踪,缺帧用灰度 127 填充
- 自建 **AVCC** 数据集:31 小时未剪辑单摄像头多方对话

**它的局限性自陈,恰好指向这个方向的下一步:**

> "视觉骨干依赖标准 ASD 模块。该模块分离说话人轨迹,
> 但**错过细微面部表情**。升级到更精细的视觉编码器
> 可能改善早期话轮转换检测。"

---

#### 4. Kurata et al.(2023,Interspeech)—— 标题最接近,但不能流式

标题直接叫《Multimodal Turn-Taking Model Using Visual Cues
for **End-of-Utterance Prediction**》—— 看起来正是「视觉 + 端点检测」。

**但 MM-VAP 指出了它的致命缺陷:**

> "它对**位于话轮末尾的 5 秒语音段**做 hold/shift 分类。
> **由于模型运行前必须已知话轮结束**,因此它不是 PTTM
> (预测式话轮模型)。"

**换句话说:它需要先知道话轮在哪结束,才能判断。** 这在实时系统里用不了。

---

#### 5. Barkhuysen et al.(2008,JASA)—— 人类层面的证据

给人看问句片段,判断属于句中还是句尾:

> **音频+视频条件下的准确率显著高于纯音频和纯视频。**

这是**人类**确实会用视觉信息判断 end-of-utterance 的实验证据 ——
说明这条路在原理上是通的,只是还没人用在语义 VAD 上。

---

#### 6. 视觉相关工作全景

```
视觉 + 话轮预测(已有大量工作)
   |
   +-- Roddy et al. (2018, ICMI)      多尺度 RNN,多模态连续话轮预测
   +-- Barkhuysen et al. (2008, JASA) 人类实验:视听优于单模态
   +-- Kurata et al. (2023)           视觉+端点预测,但需预知话轮结束
   +-- Onishi et al. (2024, IEICE)    多模态 VAP
   +-- Saga & Pelachaud (2025)        VAP + 音频/人脸编码器,捕捉细微表情
   +-- MM-VAP (2025, ACL Findings)    ★ 视觉带来 4 点提升,消融揭示 FAU 最有用
   +-- MuVAP (2026, Interspeech)      ★ 人脸作分离锚点,单摄像头多方
   +-- Cano et al. (2026)             社交机器人多模态 VAP

视觉 + 语义完整性判断(complete/incomplete)
   |
   +-- (检索未发现任何工作)          ← SoulX 所在的这一类
```

---

#### 7. 结论:空位确实存在,但要说清边界

**可以确证的:**

1. **视觉对话轮预测有效** —— MM-VAP 给出 4 个点的量化提升(T1,ACL Findings)
2. **有用的是表情(FAU)和头部姿态,不是注视** —— 消融数据支持
3. **人类本身会用视觉判断 end-of-utterance** —— Barkhuysen 2008(T1,JASA)

**不能确证的:**

**没有工作把视觉接到「语义完整性判断」上。**
这是 *absence of evidence*,不等于 *evidence of absence* ——
我针对该交叉点的两次检索都被严重污染(返回 YOLO、空间音频等无关结果),
覆盖度不如其他检索向量。

**所以准确表述是:检索未发现,而非确认不存在。**

**一个关键的互补性观察:**

| | 有视觉 | 有语义完整性判断 |
|---|---|---|
| MM-VAP / MuVAP | ✓ | ✗(只做 hold/shift) |
| SoulX-Duplug / X2-Turn | ✗ | ✓ |

> **视觉那条线有视觉而无语义;semantic VAD 这条线有语义而无视觉。
> 两者是互补的空缺 —— 这就是空位所在。**

<details>
<summary>思考过程 & 参考资料</summary>

#### 检索方法

四个向量并行:
- **A** 视觉 + 话轮预测(VAP 系多模态扩展)
- **B** 视觉 + 语义完整性判断 ← **核心目标**
- **C** 视听主动说话人检测 / 唇动 VAD
- **D** omni 多模态 LLM 的全双工交互

**核验**:MM-VAP 与 MuVAP **抓取 arXiv HTML 全文**,
含消融表与局限性自陈,非摘要转述。

⚠️ **向量 B 两次检索均被严重污染** —— 返回 YOLO 目标检测、
空间音频、视频描述等无关结果。这弱支持「该交叉点稀疏」,
但**不足以断言不存在**。这是 Q5 结论中置信度最低的一环。

#### 新增文献(编号从 F23 续)

| ID | 文献 | 出处 | 级 | 核验 |
|---|---|---|---|---|
| **F23** | Russell & Harte *MM-VAP: Visual Cues Enhance Predictive Turn-Taking for Two-Party Conversations* | **ACL 2025 Findings, pp.209-221** | **T1** | **抓全文** |
| **F4** | Qi & Skantze *MuVAP* | **Interspeech 2026**;arXiv:2606.16731 | **T1** | **抓全文** |
| **F24** | Kurata, Saeki, Fujie, Matsuyama *Multimodal Turn-Taking Model Using Visual Cues for End-of-Utterance Prediction* | **Interspeech 2023, pp.2658-2662**;doi:10.21437/interspeech.2023-578 | **T1** | 仅摘要 + MM-VAP 引述 |
| **F25** | Barkhuysen, Krahmer, Swerts *The interplay between the auditory and visual modality for end-of-utterance detection* | **JASA 123(1):354-365, 2008** | **T1** | 仅摘要 |
| **F26** | Roddy, Skantze, Harte *Multimodal Continuous Turn-Taking Prediction Using Multiscale RNNs* | ICMI 2018 | T1 | 经 F23 引述 |
| **F27** | Onishi et al. *Multimodal Voice Activity Projection for Turn-Taking* | IEICE 2024 | T1 | 经引述 |
| **F5** | Saga & Pelachaud *VAP with Multimodal Encoders* | arXiv:2506.03980 + 开源 | T2 | 摘要(前轮已核验) |
| **F28** | Cano et al. *Multimodal VAP for Social Robots* | 2026 | T2 | 仅题录 |
| **F29** | *Turn-taking in human face-to-face interaction is multimodal* | Phil Trans R Soc B 378(1875):20210473 | T1 | **被 Cloudflare 拦截,未取正文** |

#### F23 (MM-VAP) 方法细节

**视觉特征提取**:OpenFace,**60 维结构化特征**:
- 17 维 FAU(面部动作单元强度)
- 6 维 gaze(注视方向)
- 6 维头部姿态
- 30 维面部关键点(PCA 降维后)
- 1 维置信度

**关键:它不喂原始人脸 crop 给模型**,而是先提结构化特征。

**时间对齐**:视频帧率上采样到 50Hz 与音频对齐(线性插值)。
论文承认这**引入轻微未来信息泄漏**。

**数据**:Candor 语料,**710 小时**视频会议对话。

**局限性自陈**:
1. 视频会议场景**限制肢体语言**,且已知会扭曲对话节奏
2. gaze 表现差**可能部分归因于此** —— 真人面对面场景下或许更有用
3. 仅两方对话

#### F4 (MuVAP) 人脸处理管线

```
RetinaFace 检测 (SCRFD 阈值 0.62)
   → ArcFace 身份嵌入做跨帧跟踪
   → 人脸 crop → ASD 模块 → 说话人嵌入
   → 与单声道音频融合 → Role-Relative Projection
```

缺帧处理:用**灰度 127 填充**(中性灰),避免全黑影响卷积。

参数量分解:ASD 20.7M + VAP 4.9M + 融合 2.1M = **27.7M**。

#### 为什么「不要直接喂人脸 crop」

两篇工作都**不直接用原始 crop**:
- F23:OpenFace → 60 维结构化特征
- F4:crop → ASD 模块 → 嵌入

推测原因(**我的分析,非文献结论**):
1. 原始 crop 维度高、噪声大(光照、姿态、遮挡)
2. 160ms 只有 4~5 帧(30fps),信息量有限
3. 结构化特征已经压缩掉了与任务无关的外观信息

#### 数据是最大障碍

| 工作 | 数据 | 规模 |
|---|---|---|
| F23 MM-VAP | Candor(视频会议) | 710 小时 |
| F4 MuVAP | **自建 AVCC** | 31 小时未剪辑 |

F4 自建数据集的理由:**现有视听数据集有剪辑跳切,破坏因果追踪**。

而 semantic VAD 这条线的评测集(**EasyTurn**)**没有视频**。
所以要做「视觉 + semantic VAD」,**数据得从零开始**。

#### 一个未被验证的假设(H1)

F23 发现的是**听者侧**信号(下一个说话者开口前的嘴部活动)。
但 SoulX 是**单路、只看用户**,所以对它更相关的是**说话者侧**信号:

> **H1**:用户停顿时,若嘴部仍微动/保持口型(准备继续)→ `incomplete`;
> 若嘴部松弛、回看摄像头 → `complete`。

**这个假设无任何工作验证过。** 它的价值在于:
正好能替代 Q3 里指出的 SoulX 缺陷 —— 那个固定 1.28 秒硬等待。

⚠️ H1 是**我的推测**,基于 F23 的 FAU 发现 + F25 的人类实验证据外推,
非文献结论。

#### 未核验 / 局限

- **F29 被 Cloudflare 拦截**,所以「注视回避与话轮完成的精确统计量」未取到。
- **F24 (Kurata) 未抓到全文**,其具体数值未知;
  对它的批评转引自 F23,未独立核实。
- **F25 (Barkhuysen) 仅读摘要**,未见效应量。
- **F28 (Cano) 仅有题录**,未核验内容。
- **中文场景的视听对话数据**未检索到。
- 「不要直接喂 crop」的推理是**我的分析**,两篇论文未明确讨论此选择的原因。

</details>

---

## 附录 A:文献总表

<details>
<summary>展开</summary>

**核验方式**:全部**直抓 arXiv 摘要页原文**核对编号/标题/作者,
不采用搜索引擎返回的二手摘要。无法核验的一律不写进来。

**分级**:`T1` 同行评审正式发表 / `T2` 预印本(或预印本+代码)/ `T3` 灰色文献

| ID | 文献 | 出处 | 级 |
|---|---|---|---|
| F1 | Skantze *Turn-taking in Conversational Systems and HRI: A Review* | Computer Speech & Language 67:101178 (2021) | T1 |
| F2 | Ekstedt & Skantze *Voice Activity Projection* | arXiv:2205.09812;Interspeech 2023 | T1 |
| F3 | *FastTurn: Unifying Acoustic and Streaming Semantic Cues* | arXiv:2604.01897 (v6) | T2 |
| F13 | *SoulX-Duplug*(本 repo) | arXiv:2603.14877(under review) | T2 |
| F14 | *Phoenix-VAD: Streaming Semantic Endpoint Detection* | arXiv:2509.20410 (v4) | T2 |
| F15 | Li et al. *Easy Turn* | arXiv:2509.23938;**ICASSP 2026 pp.16957-16961** | **T1** |
| F16 | Fu et al. *X2-Turn* | arXiv:2608.10878 (v2) | T2 |
| F17 | Smart Turn v2 (pipecat-ai) | HF + GitHub | T3 |
| F18 | Ekstedt & Skantze *TurnGPT* | arXiv:2010.10874;**EMNLP 2020 Findings** | **T1** |
| F19 | Yang et al. *JAL-Turn* | arXiv:2603.26515 | T2 |
| F20 | Liao et al. *FlexDuo* | arXiv:2502.13472 | T2 |
| F21 | Liu et al. *Voxtral Realtime* | arXiv:2602.11298 | T2 |
| F22 | Zeghidour et al. *Delayed Streams Modeling* | arXiv:2509.08753 | T2 |
| **F4** | Qi & Skantze *MuVAP* | arXiv:2606.16731;**Interspeech 2026** | **T1** |
| **F5** | Saga & Pelachaud *VAP with Multimodal Encoders* | arXiv:2506.03980 | T2 |
| **F23** | Russell & Harte *MM-VAP* | **ACL 2025 Findings pp.209-221** | **T1** |
| **F24** | Kurata et al. *Multimodal Turn-Taking for End-of-Utterance Prediction* | **Interspeech 2023 pp.2658-2662** | **T1** |
| **F25** | Barkhuysen et al. *Auditory and visual modality for end-of-utterance detection* | **JASA 123(1):354-365, 2008** | **T1** |
| **F26** | Roddy et al. *Multimodal Continuous Turn-Taking with Multiscale RNNs* | ICMI 2018 | T1 |
| **F27** | Onishi et al. *Multimodal VAP for Turn-Taking* | IEICE 2024 | T1 |
| **F28** | Cano et al. *Multimodal VAP for Social Robots* | 2026 | T2 |
| **F29** | *Turn-taking in human face-to-face interaction is multimodal* | Phil Trans R Soc B 378(1875) | T1 |

> 编号 F1/F2/F3/F13 保持稳定,**不重排** —— 后续新增文献从 F14 起继续追加,
> 以免历史讨论中的引用失效。

### 关于 F3 的一个重要发现

**`[F3]` FastTurn 与 `[F13]` SoulX 作者重叠(均含 Lei Xie)。**

`[F3]` 摘要批评现有方案的两个缺陷:
(a) 依赖 VAD 缺乏语义理解;
(b) **依赖 ASR 模块引入额外延迟**,且在重叠语音与噪声下退化。

**(b) 正好击中 SoulX 的架构选择** —— SoulX 在 `speak` 判定时同步跑一次全量整轮 ASR
(`service/model.py:342-344`)。FastTurn 的解法是流式 CTC 解码 + 声学融合,
支持「部分观测即早决策」。

> 同一痛点、同一目标,两种解法。这是读 SoulX 时最该并排看的一篇。

### 注意

⚠️ arXiv 摘要页**不显示作者机构**。网上关于这些作者所属机构的说法属外部推断,
本文档不当作事实。

⚠️ 本文档**刻意不含任何性能数值**(准确率、延迟毫秒数)。
这些数字在 arXiv 摘要页均未给出,二手来源不可信。要填必须抓 PDF 正文。

</details>

---

## 附录 B:代码锚点

<details>
<summary>展开(行号对应 main 分支,已逐一核验)</summary>

| 位置 | 内容 | 意义 |
|---|---|---|
| `config/config.py:29-33` | 5 个内部状态 token id | 模型内部预测空间 |
| `service/model.py:398-400` | `_asr()` → `_state_predict()` | text-guided 的实现 |
| `service/model.py:703-711` | 比较 logits 大小 | 判别式分类,非生成 |
| `service/model.py:243-255` | RMS 远场过滤 | 纯能量启发式,阈值 0.02 |
| `service/model.py:342-344` | 整轮全量 ASR | FastTurn 批评的延迟来源 |
| `service/model.py:461-464` | `codebook → audio_projector` | 音频进 LLM 的路径 |
| `model/model.py:17-35` | `EncoderProjector`(通用 MLP) | 模态投影器 |
| `service/model.py:336-352` | `<user_complete>` → `speak` | 主触发路径 |
| `service/model.py:266-281` | `max_wait_num` 兜底 | 固定时长赌,≈1.28s |

### 内部 token vs 对外 state

**这两层不一样,容易混:**

| 层 | 取值 |
|---|---|
| 模型内部预测(词表 token) | `<\|user_idle\|>` `<\|user_nonidle\|>` `<\|user_complete\|>` `<\|user_incomplete\|>` `<\|user_backchannel\|>` |
| 对外 API 返回的 `state` | `idle` / `nonidle` / `speak` / `blank` |

已核验:`service/model.py` 中所有 return 点只出现这 4 个对外值。
`incomplete` 与 `backchannel` 不暴露 —— 它们的作用是让模块**继续等**,对外表现为 `idle`。

### speak 的触发逻辑(常被误解)

`text` 字段全代码只有两处 return,且都写死 `"state": "speak"`,
所以 **speak ⟺ 有 text**。

两条路径:
1. 模型输出 `<|user_complete|>`(语义判定说完了)
2. 先 `<|user_incomplete|>`,之后连续 idle 累计到 `max_wait_num=8`(≈1.28s)强行接话

两条都要求 `speech_detected == True`。
→ 所以 `<|user_complete|>` **不一定**产出 `speak`:若整轮压根没检测到语音,只返回 `idle`。

**`speak` 之后必定 `reset()`**,整轮上下文全清。

</details>

---

## 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.8 | 2026-08-21 | Q3 同步代码现状:`complete_bias` 已实现并升为 A 类可调参数,补 struct 模式坑与 developer_mode 打印格式 |
| v1.7 | 2026-08-20 | 新增 Q5:视觉模态相关工作调研,论证「视觉+语义完整性判断」为空位;新增 F23-F29 |
| v1.6 | 2026-08-20 | 表 1 扩为全景表:纳入所有已知方法,加「来源」列区分评测集,无数据留空 |
| v1.5 | 2026-08-20 | Q4 两表加提出时间列 + 时间线;纠错 TEN(Qwen2.5-7B)与 Smart Turn v3(Whisper Tiny 8M) |
| v1.4 | 2026-08-20 | Q4 重写:各方法逐一简介 + 两张横向对比表(准确率/延迟、开源/体量);新增 F19-F22 |
| v1.2 | 2026-08-20 | 新增 Q3:可调参数清单,分安全/危险/硬编码三类,附 complete_bias 改法 |
| v1.1 | 2026-08-20 | Q2 第 2 节补上 torch 数据格式:逐阶段 shape/dtype 与源码行号对照;补训练/冻结范围 |
| v1.0 | 2026-08-20 | Q2 整段重写:拆成「模型本身」与「整个系统」两部分,13 小节顺序推进 |
| v0.9 | 2026-08-20 | 新增第 8 步:澄清模型内部 5 个 token 与对外 4 个 state 的映射关系 |
| v0.8 | 2026-08-20 | 去掉 mermaid 图,信息流全部改为文字箭头版;补调用层次 |
| v0.7 | 2026-08-20 | 新增第 7 步:完整性判决的受限比较机制,及「无阈值不可调」的设计后果 |
| v0.6 | 2026-08-20 | 修正第 6 步流程图:区分「预测什么」与「是否转折点」两层判断;补充疑似挂起路径 |
| v0.5 | 2026-08-20 | Q2 正文改写为「一段声音的旅程」,术语首次出现即解释;技术细节移入折叠区 |
| v0.4 | 2026-08-20 | 新增 Q2:模型设计、输入输出、内部信息流,含 5 张流程图 |
| v0.3 | 2026-08-20 | 清空 Q2/Q3 及其衍生内容,仅保留 Q1;附录收敛到 Q1 所需范围 |
| v0.2 | 2026-08-20 | 改为问答体(结论前置 + 折叠细节);Q1 重构为 VAD → VAP → SoulX 递进 |
| v0.1 | 2026-08-19 | 初始版本(学术报告体,已重写) |
