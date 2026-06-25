import json

cells = []

def md(id, source):
    cells.append({"cell_type": "markdown", "id": id, "metadata": {}, "source": source.split("\n")})

def code(id, source):
    cells.append({"cell_type": "code", "id": id, "metadata": {}, "outputs": [], "execution_count": None, "source": source.split("\n")})

# ===== Cover =====
md("md-0", """# 第8讲：MLLM输出质量评估——BLEU与术语准确率
## 用自动评估指标量化微调模型的术语解释质量

> **本讲定位**：评估方法论讲——上承第3讲QLoRA微调，用BLEU和术语准确率对微调后的LLaMA-3术语解释做量化评估，为后续迭代优化提供数据依据。""")

# ===== 本讲定位 =====
md("md-1", """# 本讲定位

第3讲我们通过QLoRA让LLaMA-3学会了数字人文专业术语的解释。问题来了：微调后的模型到底"学得怎么样"？

仅凭肉眼扫几行输出不够严谨——我们需要**可量化、可复现、可对比**的自动评估。

本讲引入两个核心指标：
- **BLEU**：衡量生成文本与参考答案的n-gram重叠度（机器翻译/文本生成领域最通用的指标之一）
- **术语准确率**：衡量数字人文关键术语是否被正确识别和解释——这是比BLEU更贴近本课程场景的"业务指标"

> 评估是微调的"眼睛"——没有评估，你就不知道改超参数是在优化还是在倒退。""")

# ===== 本讲结构 =====
md("md-2", """# 本讲结构

## 1  环境准备
## 2  评估指标原理（简化版）
## 3  加载L03微调适配器
## 4  检查点：确认适配器可推理
## 5  准备评估数据（载入真实语料 + 生成模型输出）
## 6  计算BLEU分数
## 7  计算术语准确率
## 8  结果展示与解读
## 8.5  保存评估结果（供第9讲使用）
## 快速回顾""")

# ===== 透明度说明 =====
md("md-3", """# 透明度说明

> **模型来源**：本讲复用第3讲的QLoRA适配器（`qlora_adapter_lecture3/`），基础模型为`NousResearch/Meta-Llama-3-8B-Instruct`。适配器不包含完整模型权重，仅含LoRA增量（约55 MB），需叠加到基础模型上加载。
>
> **评估数据**：包含两组——（1）第3讲的10条占位示例（刻本、金石学、校勘等），（2）`real_examples.json`中的10条真实案例（甲骨文、古籍修复、皮影戏、敦煌壁画、活字印刷、青铜铭文、昆曲、简牍、拓片、二十四节气），最多20条。
>
> **评估工具**：BLEU使用`sacrebleu`库（社区标准实现），术语准确率通过关键词匹配计算——不涉及训练，无随机性，结果完全可复现。
>
> **指标局限**：BLEU仅衡量表面文本相似度，无法评估解释的"学术正确性"；术语准确率依赖于术语清单的完整性。两项指标需结合人工审读使用，不可替代专家判断。""")

# ===== Ch1: 环境准备 =====
md("md-4", """# 第1章  环境准备

- 安装`sacrebleu`（BLEU计算的社区标准工具）
- 复用第3讲的核心依赖：`transformers`、`peft`、`bitsandbytes`、`torch`
- 运行GPU自检——评估推理在CPU上速度极慢

> **显存需求**：4位量化基础模型约5–6 GB，T4 (16 GB) 可正常运行。20条数据的批量推理约需2–5分钟（取决于GPU和网络）。
>
> **本地用户**：如果已在`mllm-dh` conda环境中运行，可注释掉`!pip install`行。""")

code("code-5", """# ===== 第1章：环境准备 =====
# Colab 用户：运行此单元格自动安装依赖。
# 本地 mllm-dh 环境用户：可注释掉 !pip 行。

!pip install -q sacrebleu

import torch
import json
import os

if not torch.cuda.is_available():
    print("未检测到 GPU。评估推理在 CPU 上速度极慢，建议连接 GPU。")
    print("Colab 用户请选择：运行时 -> 更改运行时类型 -> T4 GPU。")
else:
    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU 已就绪：{device_name}")
    print(f"显存总量：{total_mem:.1f} GB")

print(f"使用设备：{'cuda' if torch.cuda.is_available() else 'cpu'}")""")

# ===== Ch2: 评估指标原理 =====
md("md-6", """# 第2章  评估指标原理（简化版）

## BLEU（Bilingual Evaluation Understudy）

BLEU衡量**生成文本**与**参考答案**之间的n-gram重叠度：

1. **n-gram Precision**：统计生成文本中每个n-gram（1-gram到4-gram）在参考答案中出现的比例
2. **Brevity Penalty（长度惩罚）**：如果生成文本比参考答案短太多，扣分——防止模型偷懒输出短句刷分
3. **最终分数**：几何平均 × 长度惩罚，输出0–100（越高越好）

重要提示：机器翻译领域常见的"30+算可用、50+算优秀"这类经验阈值，前提是参考答案措辞高度一致（翻译任务天然如此）。本讲的开放式术语解释生成则不同：模型用自己的话解释术语，即使语义正确，措辞也与参考答案差异极大，中文BLEU常低于10。因此本讲采用更务实的分级标准（见第8章），BLEU应作为**相对比较**的参考（如对比微调前后的变化），而非绝对质量的判定。

直观理解：BLEU-1看词对没对上，BLEU-4看短语顺不顺。

## 术语准确率

更贴近本课程场景的指标：

```
术语准确率 = 生成文本中正确出现的术语数 / 参考答案中的术语总数
```

例如：参考答案含"刻本、避讳、浙本"三个术语，生成文本只正确出现了"刻本"和"浙本"，准确率 = 2/3 ≈ 66.7%。

> 局限性：术语清单从explanation中自动提取（见第7章），如果提取遗漏了某个术语，分母也会随之缩小，**系统性高估准确率**。收到Kevin的标准术语清单后替换可解决此问题。术语准确率同样适合做**相对比较**（微调前 vs 微调后），而非绝对质量判定。""")

# ===== Ch3: 加载适配器 =====
md("md-7", """# 第3章  加载L03微调适配器

## 加载流程

1. 以4位量化方式加载基础模型（LLaMA-3 8B），显存需求约5–6 GB
2. 从`qlora_adapter_lecture3/`目录加载LoRA增量权重（约55 MB）
3. 挂载后，模型对数字人文术语的解释能力应显著优于未微调版本

> **适配器来源**：第3讲Colab运行结束后下载的zip文件。将解压后的`qlora_adapter_lecture3/`文件夹上传至Colab当前目录即可。
>
> **如果你没有适配器**：可从课程仓库的`shared/weights/`目录下载，或重新运行第3讲生成。""")

code("code-8", """# ===== 第3章：加载基础模型与LoRA适配器 =====

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH = "./qlora_adapter_lecture3"

ADAPTER_LOADED = os.path.exists(ADAPTER_PATH)

if not ADAPTER_LOADED:
    print("[WARN] 未找到适配器目录。")
    print(f"  预期路径：{ADAPTER_PATH}")
    print("  请将第3讲下载的 zip 解压后上传至 Colab 当前目录。")
    print("  本讲将以演示模式继续运行（使用预生成的示例输出）。")
    print("  演示数据仅供流程验证，不代表真实模型表现。")
    model = None
    tokenizer = None
else:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    # 计算精度
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"计算精度：{compute_dtype}")

    # 4位量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # 加载分词器（从适配器目录加载，确保chat_template与训练时一致）
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("分词器加载完成")

    # 加载4位量化基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print("基础模型加载完成")

    # 加载LoRA适配器
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.config.use_cache = True
    print(f"适配器已加载：{ADAPTER_PATH}")
    print(f"总参数量：{sum(p.numel() for p in model.parameters()) / 1e9:.2f} B")""")

# ===== Ch4: 检查点 =====
md("md-9", """# 第4章  检查点：确认适配器可推理

在进行批量评估之前，先用一条样本验证适配器加载正确、模型可正常推理。

这是一个**有意设置的检查点**：如果适配器路径错误、权重损坏或与基础模型不兼容，这里会立即报错——避免批量推理跑了一半才发现问题。""")

code("code-10", """# ===== 检查点：确认适配器可推理 =====

SYSTEM_PROMPT = (
    "你是一位专精于数字人文领域的学者，"
    "能够准确解释古籍文献、文化遗产与档案实践中的专业术语。"
)

if not ADAPTER_LOADED:
    print("[SKIP] 演示模式，跳过适配器推理检查点。")
else:
    test_transcript = "视频中提到这部书是刻本，与手写本不同。"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": test_transcript},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("=== 检查点输出 ===")
    print(f"输入：{test_transcript}")
    print(f"输出：{response}")
    print("=" * 20)
    print("适配器加载正常，可以继续评估。")""")

# ===== Ch5: 准备评估数据 =====
md("md-11", """# 第5章  准备评估数据

## 评估数据构成

本讲使用两组数据，最多20条：

| 数据来源 | 数量 | 内容示例 |
|----------|------|----------|
| L03占位数据 | 10条 | 刻本、缮写、金石学、校勘… |
| real_examples.json | 10条 | 甲骨文、古籍修复、皮影戏、敦煌壁画… |

每条数据包含两个关键字段：
- `transcript`：输入（视频转写片段，含专业术语）
- `explanation`：参考答案（标准术语解释，作为评估基准）

> 如自行准备数据，确保JSON文件为列表格式，每个元素含上述两个字段，否则运行时会报`KeyError`。

以下代码加载两组数据，并对每条transcript用微调模型生成解释。""")

code("code-12", """# ===== 第5章：加载评估数据 + 生成模型输出 =====

# ---- 第1组：L03占位数据（复刻自第3讲）----
RAW_DATA = [
    {"transcript": "视频中提到这部书是刻本，与手写本不同。",
     "explanation": "刻本（kè běn）指雕版印刷的书籍：工匠将文字反刻于木版，涂墨后压印于纸张。宋元刻本是版本学研究的核心文物。"},
    {"transcript": "文献里反复出现缮写一词，请问这是什么意思？",
     "explanation": "缮写指专职抄手对文本进行誊清、校正的抄录工序，是印刷术普及前文本传播的主要手段，也是研究字形演变的重要证据。"},
    {"transcript": "研究者说他的研究方向是金石学。",
     "explanation": "金石学是中国传统学问中研究铜器铭文（金）与石刻碑文（石）的学科，是现代铭刻学和考古学的前身，为历史地理与文字演变研究提供第一手资料。"},
    {"transcript": "档案整理时需要遵循原始顺序原则。",
     "explanation": "原始顺序原则（Principle of Original Order）是档案学核心原则之一：档案应按照其创建机构或个人最初形成时的顺序保存，以保留内在的背景关系与证据价值。"},
    {"transcript": "数字化之后，团队对每张图片做了标注。",
     "explanation": "数字化标注（digital annotation）指在数字化图像或文本上附加结构化元数据（如人名、地名、朝代、主题词），以支持计算机检索与人文分析。"},
    {"transcript": "不同版本之间存在异文，需要做校勘。",
     "explanation": "校勘（jiào kān）是文献学的核心方法：通过比对同一文本的不同传本，发现并考证文字差异（异文），以期恢复最接近原始面貌的文字。"},
    {"transcript": "这是一部传世本，与出土文献有所不同。",
     "explanation": "传世本指通过历代抄写、刻印而连续流传至今的文本版本；出土文献则指通过考古发掘获得的实物文本（如简牍、帛书）。二者互校是文献学研究的重要方法。"},
    {"transcript": "博物馆里展出了多件拓片。",
     "explanation": "拓片（碑帖）是将宣纸覆于石刻或铜器表面，用墨捶拓而成的平面复制品，是传递金石文字最重要的媒介，也是数字化采集的优先对象。"},
    {"transcript": "他的博士论文研究的是目录学。",
     "explanation": "目录学（bibliography）研究文献的著录、分类与目录编制方法，在中国学术传统中有独立的学科地位，现代数字图书馆的元数据标准（如 MARC、Dublin Core）可视为其数字延伸。"},
    {"transcript": "视频介绍了一处文化遗产的数字孪生项目。",
     "explanation": "数字孪生（digital twin）在文化遗产语境下指通过三维扫描、摄影测量等技术建立遗址或文物的高精度数字模型，用于远程研究、虚拟展览与损毁预防。"},
]

# ---- 第2组：真实数据（从real_examples.json加载）----
REAL_DATA_PATH = "real_examples.json"
if os.path.exists(REAL_DATA_PATH):
    with open(REAL_DATA_PATH, "r", encoding="utf-8") as f:
        real_data = json.load(f)
    # 校验字段，避免推理中途崩溃
    valid = []
    for item in real_data:
        if "transcript" in item and "explanation" in item:
            valid.append(item)
        else:
            print(f"  跳过缺少字段的条目: {item}")
    real_data = valid
    print(f"已加载真实数据：{REAL_DATA_PATH}（{len(real_data)} 条）")
else:
    print(f"未找到 {REAL_DATA_PATH}，仅使用占位数据。")
    print("如需真实数据，请从课程仓库 shared/corpus/ 目录上传。")
    real_data = []

all_data = RAW_DATA + real_data
print(f"评估数据总量：{len(all_data)} 条")

# ---- 批量推理：生成模型输出 ----
# 以下两个列表对应课程中提到的概念：
#   outputs  → l3_term_outputs（模型生成的解释）
#   references → l3_term_references（参考答案/基准）
references = []   # l3_term_references（参考答案）
outputs = []      # l3_term_outputs（模型生成）

if ADAPTER_LOADED:
    print("\\n正在用微调模型生成解释...")
    for i, item in enumerate(all_data):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["transcript"]},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        references.append(item["explanation"])
        outputs.append(generated)
        print(f"  [{i+1}/{len(all_data)}] {item['transcript'][:40]}...")
else:
    print("\\n[DEMO] 使用预生成的示例输出（供流程验证）...")
    # 为占位数据生成示例输出，部分术语故意缺失以展示评估的区分力
    DEMO_OUTPUTS_RAW = [
        "刻本是指通过雕版印刷技术制作的书籍，与手写本相比可以批量复制。",
        "缮写是指人工抄写和校正文本的过程，在古代是文字传播的重要方式。",
        "金石学是研究古代青铜器和石刻碑文的一门传统学科。",
        "档案整理应当遵循一定的规则，保持文件之间的内在关联。",
        "对数字图像添加描述性标签以便于检索和管理。",
        "校勘是通过对比不同版本来发现和纠正文献中的文字错误。",
        "传世本是指通过历代流传保存至今的古代文献版本。",
        "拓片是从石刻或铜器上用纸墨复制文字图案的传统技艺。",
        "目录学是研究文献分类、编目和著录方法的学科。",
        "数字孪生技术用于创建物理对象的虚拟模型，可应用于文化遗产保护。",
    ]
    for item in RAW_DATA:
        references.append(item["explanation"])
    outputs = DEMO_OUTPUTS_RAW.copy()
    # 真实数据用占位文本，BLEU 趋近于 0（演示极端情况）
    for item in real_data:
        references.append(item["explanation"])
        outputs.append("[演示模式] 模型未加载，此为占位输出。")

print(f"\\n推理完成：{len(references)} 条 references，{len(outputs)} 条 outputs")""")

# ===== Ch6: BLEU =====
md("md-13", """# 第6章  计算BLEU分数

## sacrebleu使用说明

`sacrebleu`是机器翻译/文本生成领域的社区标准BLEU实现，优势：
- 内置分词器（中文使用`zh`分词）
- 自动处理多参考译文
- 输出与WMT评测一致的分数

以下代码计算两个维度的BLEU：
1. **逐条BLEU**：每条样本单独计算，观察个体差异
2. **语料级BLEU**：所有样本合并计算，反映整体质量""")

code("code-14", """# ===== 第6章：计算BLEU分数 =====
from sacrebleu.metrics import BLEU

bleu = BLEU(tokenize="zh", effective_order=True)

print("===== 逐条BLEU分数 =====")
individual_bleu = []
for i, (ref, out) in enumerate(zip(references, outputs)):
    score = bleu.sentence_score(out, [ref])
    individual_bleu.append(score.score)
    print(f"  [{i+1:2d}] BLEU={score.score:.1f}  transcript: {all_data[i]['transcript'][:40]}...")

print()
print("===== 语料级BLEU =====")
corpus_score = bleu.corpus_score(outputs, [[r] for r in references])
print(f"BLEU = {corpus_score.score:.1f}")
# 注：中文开放式生成中BLEU-3/BLEU-4常为0（措辞差异大），属正常现象
print(f"  BLEU-1 (1-gram): {corpus_score.precisions[0]:.1f}")
print(f"  BLEU-2 (2-gram): {corpus_score.precisions[1]:.1f}")
print(f"  BLEU-3 (3-gram): {corpus_score.precisions[2]:.1f}")
print(f"  BLEU-4 (4-gram): {corpus_score.precisions[3]:.1f}")
print(f"  Brevity Penalty: {corpus_score.bp:.3f}")

# 汇总统计
import numpy as np
scores = np.array(individual_bleu)
print()
print("===== 逐条BLEU统计 =====")
print(f"  均值：{scores.mean():.1f}")
print(f"  中位数：{np.median(scores):.1f}")
print(f"  最低：{scores.min():.1f}（样本 {np.argmin(scores)+1}）")
print(f"  最高：{scores.max():.1f}（样本 {np.argmax(scores)+1}）")""")

# ===== Ch7: 术语准确率 =====
md("md-15", """# 第7章  计算术语准确率

## 计算方法

从每条参考答案（explanation）中提取关键术语，检查生成文本中是否包含该术语：

```
术语准确率 = 正确出现的术语数 / 参考答案中的术语总数
```

术语提取策略：从每条评估数据的 explanation 字段中提取与其对应的核心术语。例如"刻本（kè běn）"提取"刻本"，"原始顺序原则（Principle of Original Order）"提取"原始顺序原则"。

> 以下`TERM_LIST`为Kevin提供的标准术语清单，共10个核心数字人文术语。""")

code("code-16", """# ===== 第7章：计算术语准确率 =====

# ==== 术语清单（Kevin提供） ====
TERM_LIST = [
    "刻本", "缮写", "金石学", "原始顺序原则", "校勘",
    "传世本", "拓片", "目录学", "数字孪生", "甲骨文",
]
term_set = set(TERM_LIST)
print(f"术语清单：{len(TERM_LIST)} 个")
print(f"术语列表：{sorted(term_set)}")

# ==== 计算准确率 ====
term_hits = []
term_misses = []

print()
print(f"{'#':<5} {'术语':<20} {'结果':<8}")
print("-" * 35)

for i, (ref, out) in enumerate(zip(references, outputs)):
    ref_terms = [t for t in term_set if t in ref]
    if not ref_terms:
        continue  # 参考答案中没有匹配到术语，跳过
    hit = sum(1 for t in ref_terms if t in out)
    term_hits.append(hit)
    term_misses.append(len(ref_terms) - hit)
    missed = [t for t in ref_terms if t not in out]
    primary = ref_terms[0]  # 主术语
    result = "HIT" if hit >= len(ref_terms) else ("MISS" if hit == 0 else f"PARTIAL ({hit}/{len(ref_terms)})")
    print(f"{i:<5} {primary:<20} {result:<8}")
    if missed and hit > 0:
        print(f"      missed: {missed}")

total_hits = sum(term_hits)
total_terms = sum(term_hits) + sum(term_misses)
accuracy = total_hits / total_terms * 100 if total_terms > 0 else 0

print()
print(f"===== 术语准确率 =====")
print(f"  命中术语数：{total_hits}")
print(f"  术语总数：{total_terms}")
print(f"  术语准确率：{accuracy:.1f}%")
print(f"  评估条目数：{len(term_hits) + len(term_misses)}（{len(references) - len(term_hits) - len(term_misses)} 条因无匹配术语被跳过）")
if total_terms == 0:
    print("  WARNING: 未提取到任何术语，无法计算准确率。请检查TERM_LIST或数据格式。")""")

# ===== Ch8: 结果展示 =====
md("md-17", """# 第8章  结果展示与解读

## 评估输出示例

以下随机展示5条样本的参考答案与模型输出对比，供人工审读。

结合BLEU和术语准确率两个维度，可以初步判断（以下BLEU指逐条均值，注意：此阈值针对开放式术语解释场景，远低于MT任务常见的30+，详见第2章说明）：
- **BLEU ≥ 10** 且术语准确率 ≥ 50%：模型已具备基本的术语解释能力
- **BLEU < 5** 或术语准确率 < 30%：建议检查适配器是否正确加载，或训练数据是否覆盖当前术语
- 最重要的用法是**微调前后的相对比较**，而非孤立看绝对分数""")

code("code-18", """# ===== 第8章：结果展示 =====
import random

print("===== 随机抽样对比（5条）=====")
random.seed(42)
sample_indices = random.sample(range(len(all_data)), min(5, len(all_data)))

for idx in sample_indices:
    item = all_data[idx]
    ref = references[idx]
    out = outputs[idx]
    print(f"\\n【样本 {idx+1}】")
    print(f"  输入：{item['transcript'][:80]}")
    print(f"  参考：{ref[:120]}...")
    print(f"  生成：{out[:120]}...")
    print(f"  BLEU：{individual_bleu[idx]:.1f}")

print()
print("===== 评估总结 =====")
print(f"  语料级BLEU：{corpus_score.score:.1f}")
print(f"  逐条BLEU均值：{np.mean(individual_bleu):.1f}")
print(f"  术语准确率：{accuracy:.1f}%")
print(f"  评估样本数：{len(all_data)}")
print()
print("TIP: 评估结果仅反映模型在特定数据集上的表现。")
print("如需全面评估，建议增加样本量并进行人工交叉验证。")""")

# ===== Ch8.5: 保存评估结果 =====
md("md-19", """# 第8.5章  保存评估结果（供第9讲使用）

在第9讲中，我们会将本讲的评估数据可视化——生成论文风格的BLEU对比图和术语准确率图表。

为避免重复加载模型和推理，以下代码将本讲的所有评估结果保存到 `l08_evaluation_results.json`，第9讲直接读取即可。

保存的内容包括：
- 每条样本的逐条BLEU分数
- 语料级BLEU及各阶精度
- 术语准确率及逐条命中情况
- 评估摘要统计（均值、中位数、最高/最低）""")

code("code-20", """# ===== 保存评估结果（供第9讲使用）=====
import json
from datetime import datetime

# 整理术语命中结果
# 注意：部分条目因无匹配术语被跳过，term_details 的 index 不连续
# L09 使用时按 index 字段关联 samples，不要假设连续
term_details = []
for i, (ref, out) in enumerate(zip(references, outputs)):
    ref_terms = [t for t in term_set if t in ref]
    if not ref_terms:
        continue
    hit = sum(1 for t in ref_terms if t in out)
    primary = ref_terms[0]
    term_details.append({
        "index": i,
        "term": primary,
        "hit": hit > 0,
        "all_terms": ref_terms,
        "transcript": all_data[i]["transcript"][:60],
    })

results = {
    "_metadata": {
        "lecture": "L08",
        "model": MODEL_ID,
        "adapter": ADAPTER_PATH,
        "num_samples": len(all_data),
        "saved_at": datetime.now().isoformat(),
    },
    "bleu": {
        "corpus_score": corpus_score.score,
        "bleu1": corpus_score.precisions[0],
        "bleu2": corpus_score.precisions[1],
        "bleu3": corpus_score.precisions[2],
        "bleu4": corpus_score.precisions[3],
        "brevity_penalty": corpus_score.bp,
        "individual_scores": individual_bleu,
        "mean": float(np.mean(individual_bleu)),
        "median": float(np.median(individual_bleu)),
        "min": float(np.min(individual_bleu)),
        "max": float(np.max(individual_bleu)),
    },
    "term_accuracy": {
        "accuracy_pct": accuracy,
        "total_hits": total_hits,
        "total_terms": total_terms,
        "details": term_details,
    },
    "samples": [
        {"index": i, "transcript": d["transcript"], "reference": d["explanation"][:150],
         "output": outputs[i][:150], "bleu": individual_bleu[i]}
        for i, d in enumerate(all_data)
    ],
}

OUTPUT_PATH = "./l08_evaluation_results.json"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"评估结果已保存到: {OUTPUT_PATH}")
print(f"文件大小: {len(json.dumps(results, ensure_ascii=False)) / 1024:.1f} KB")
print()
print("该文件将被第9讲直接读取，用于生成可视化图表。")
print()
print("WARNING: Colab 用户请注意：运行时重置后该文件会丢失。")
print("   建议立即下载：左侧文件浏览器 → 右键 l08_evaluation_results.json → 下载。")
print("   或者挂载 Google Drive，将 OUTPUT_PATH 改为 Drive 路径。")""")

# ===== 快速回顾 =====
md("md-21", """# 快速回顾（Quick Check）

**本讲完成了以下工作：**

- **适配器加载**：将第3讲的QLoRA微调权重叠加到LLaMA-3 8B基础模型上
- **批量推理**：对评估数据（10条占位 + 可选10条真实）生成术语解释，形成l3_term_outputs
- **BLEU评估**：使用sacrebleu计算逐条BLEU和语料级BLEU，包括1-4 gram精度和长度惩罚
- **术语准确率**：从参考答案中提取术语并检查生成文本的术语覆盖率
- **结果展示**：随机抽样对比参考答案与模型输出，输出评估总结
- **结果保存**：将BLEU分数、术语准确率及逐条详情写入 `l08_evaluation_results.json`，供第9讲直接读取

> **和第3讲的连贯性**：L03做微调 → L08做评估 → L09可视化（L04-L07为多模态章节，与此流水线并行）。L08保存的结果文件是第9讲的直接输入，形成"训练→评估→可视化"完整流水线。
>
> **第9讲预告**：读取本讲保存的 `l08_evaluation_results.json`，生成论文风格的BLEU对比图、术语准确率柱状图和评估摘要表。""")

# ===== Build notebook =====
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

import os
os.makedirs("D:/dh-llm-course/lecture_8", exist_ok=True)
with open("D:/dh-llm-course/lecture_8/L8_evaluation.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("L8 notebook written successfully.")

# Count cells
for i, cell in enumerate(cells):
    ct = cell["cell_type"]
    cid = cell["id"]
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    print(f"  Cell {i:2d} [{ct:8s}] {cid:10s}: {src[:60].strip()}...")
