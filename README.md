# LLM for Digital Humanities — 10-Lecture Course

A hands-on course on applying Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) to Digital Humanities (DH) research. All notebooks run on Google Colab with a free T4 GPU; no paid API keys are required.

The course is delivered in **Chinese** and follows a single unifying scenario: processing a short DH video (oracle bone script, ancient manuscript restoration, intangible cultural heritage, Dunhuang murals, etc.) from raw footage through transcription, summarisation, fine-tuning, multimodal analysis, evaluation, and finally a complete end-to-end research pipeline.

---

## Repository Layout

```
dh-llm-course/
├── lecture_1/          # L01 — LLM basics + LLaMA-3 summarisation
├── lecture_2/          # L02 — Multimodal LLMs + CLIP cross-modal matching
├── lecture_3/          # L03 — QLoRA fine-tuning LLaMA-3
├── lecture_4/          # L04 — Corpus processing + precision visualisation
├── lecture_5/          # L05 — BLIP-2 video frame captioning
├── lecture_6/          # L06 — Whisper audio transcription pipeline
├── lecture_7/          # L07 — LLaVA-Next visual analysis
├── lecture_8/          # L08 — BLEU + term-accuracy evaluation
├── lecture_9/          # L09 — Publication-ready chart generation
│   └── outputs/L09/    # Generated figures (PNG)
├── lecture_10/         # L10 — Capstone end-to-end pipeline
├── presentation-slides/# PPTX slide decks for each lecture
└── shared/             # Shared data, model weights, and reference docs
    ├── corpus/         # Training data + sample images
    └── weights/        # Saved QLoRA adapter from L03
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python basics | Variables, functions, loops, lists — see `lecture_1/L1_prereq_python_basics.ipynb` |
| Google Colab free account | T4 GPU (16 GB VRAM) is sufficient for every notebook |
| Hugging Face account | Not required — all model IDs used are publicly accessible without a token |

No local GPU is needed. All notebooks auto-detect whether they are running on Colab and install dependencies accordingly.

---

## Lecture-by-Lecture Guide

### Prerequisite — Python Basics

**File**: `lecture_1/L1_prereq_python_basics.ipynb`

A self-contained Python refresher covering variables, control flow, functions, and list comprehensions. Complete this before Lecture 1 if you are new to Python.

---

### Lecture 1 — LLM Basics and Text Summarisation

**File**: `lecture_1/L1_llama3_summary.ipynb`  
**Slides**: `presentation-slides/L1_llama3_summary_slides.pptx`

**What you learn**:
- How LLMs generate text at a conceptual level (next-token prediction, temperature, sampling)
- Loading a 4-bit quantised LLaMA-3 8B model with BitsAndBytes NF4 to fit within a free T4 GPU (~5–6 GB VRAM instead of ~32 GB)
- Writing a `summarize_transcript(text)` function that produces a 2–3 sentence Chinese summary of a DH video transcript

**Model**: `NousResearch/Meta-Llama-3-8B-Instruct`  
**Key concepts**: 4-bit quantisation, inference pipeline, prompt engineering

---

### Lecture 2 — Multimodal LLMs and Cross-Modal Representation

**File**: `lecture_2/L02.ipynb`  
**Slides**: `presentation-slides/L02_slides.pptx`

**What you learn**:
- What multimodal embeddings are and why they matter for DH (retrieval, annotation, alignment)
- Using CLIP to encode video frames and ancient-manuscript text snippets into a shared embedding space
- Computing a cosine-similarity matrix between frames and text passages to find the best-matching pair

**Model**: `openai/clip-vit-base-patch32`  
**Key concepts**: contrastive learning, cross-modal similarity, embedding space, cosine similarity

**Note**: This lecture continues with the same short DH video introduced in Lecture 1.

---

### Lecture 3 — QLoRA Fine-Tuning LLaMA-3 for Domain Adaptation

**File**: `lecture_3/L03.ipynb` (Chinese); `lecture_3/L03_en.ipynb` (English translation)  
**Slides**: `presentation-slides/L03_slides.pptx`

**What you learn**:
- Why general-purpose LLMs struggle with specialised DH terminology (e.g., 刻本, 金石学, 缮写)
- QLoRA (Quantised Low-Rank Adaptation): how LoRA inserts small trainable matrices into a frozen base model, and how NF4 quantisation keeps memory use low
- Training on 10 sample term→explanation pairs, watching the loss curve converge, then swapping in real corpus data
- Saving the trained LoRA adapter for reuse in Lectures 4 and 8

**Model**: `NousResearch/Meta-Llama-3-8B-Instruct` + PEFT LoRA  
**Key concepts**: parameter-efficient fine-tuning, LoRA rank/alpha, SFTTrainer, adapter serialisation  
**Output**: `shared/weights/qlora_adapter_lecture3_archive/`

---

### Lecture 4 — Corpus Processing and Precision Visualisation

**File**: `lecture_4/L04.ipynb`  
**Slides**: `presentation-slides/L04_slides.pptx`

**What you learn**:
- Why raw video transcripts are not training-ready (repeated segments, formatting noise, unlabelled terms)
- Building a deduplication pipeline and an automatic term-annotation pipeline
- Logging training metrics and plotting a precision chart to compare baseline vs. fine-tuned outputs
- Fixed hyperparameter strategy for reproducible experiments

**Depends on**: The LoRA adapter from Lecture 3 (`shared/weights/qlora_adapter_lecture3_archive/`)  
**Key concepts**: corpus cleaning, exact/near-duplicate removal, term tagging, experiment logging

---

### Lecture 5 — Video Frame Captioning with BLIP-2

**File**: `lecture_5/L5_blip2_video_captioning.ipynb`  
**Slides**: `presentation-slides/L5_blip2_video_captioning_slides.pptx`

**What you learn**:
- How BLIP-2 differs from CLIP: BLIP-2 generates natural-language descriptions from images rather than computing similarity scores
- Extracting frames from a DH video and passing each frame through BLIP-2 to produce a per-frame caption
- Assembling the per-frame captions into a structured archival record

**Model**: `Salesforce/blip2-opt-2.7b` (~5 GB, runs on free T4)  
**Key concepts**: image-to-text generation, Q-Former bridge architecture, archival captioning

---

### Lecture 6 — Audio Transcription and the Full Archival Pipeline

**File**: `lecture_6/L6_whisper_transcribe.ipynb`  
**Slides**: `presentation-slides/L6_whisper_transcribe_slides.pptx`

**What you learn**:
- Using `ffmpeg` to extract audio from a video file
- Loading Whisper Small (~500 MB) for offline, multilingual speech-to-text (Chinese word error rate ~9% on Small)
- Chaining Whisper output directly into the LLaMA-3 summarisation function from Lecture 1 to produce a full "audio → transcript → archival summary" pipeline

**Model**: `openai/whisper-small`  
**Key concepts**: automatic speech recognition (ASR), word error rate (WER), timestamp-aligned transcription, pipeline composition

---

### Lecture 7 — LLaVA-Next Generative Visual Analysis

**File**: `lecture_7/L07_llava_next_dh_analysis.ipynb`  
**Slides**: `presentation-slides/L07_llava_next_dh_analysis_slides.pptx`

**What you learn**:
- The difference between CLIP (discriminative — "how similar are these?") and LLaVA-Next (generative — "describe this image")
- Loading LLaVA-Next (LLaVA-1.6 / Mistral-7B backbone, 4-bit quantised) and running three DH analysis tasks in a single inference pass:
  1. Text recognition and transcription from low-quality historical scans
  2. Cultural element identification
  3. Standardised archival description generation
- Building a reusable `analyse_image(image_path, prompt)` helper function

**Model**: `llava-hf/llava-v1.6-mistral-7b-hf` (4-bit quantised, ~7 GB)  
**Key concepts**: visual instruction tuning, image-text interleaving, archival metadata generation

---

### Lecture 8 — Quantitative Evaluation: BLEU and Term Accuracy

**File**: `lecture_8/L8_evaluation.ipynb`  
**Slides**: `presentation-slides/L8_evaluation_slides.pptx`

**What you learn**:
- Why visual inspection of model output is insufficient for scientific claims
- BLEU score: n-gram overlap between model output and reference answers; implemented with `sacrebleu`
- Term accuracy: a domain-specific metric counting how many DH key terms are correctly explained — more task-relevant than BLEU for this course
- Running both metrics on the fine-tuned LLaMA-3 adapter from Lecture 3 and saving results to JSON for Lecture 9

**Depends on**: `shared/weights/qlora_adapter_lecture3_archive/`  
**Output**: `lecture_8/l08_evaluation_results.json`  
**Key concepts**: automatic evaluation, BLEU-1/2/4, precision/recall for terminology, evaluation data design

---

### Lecture 9 — Publication-Ready Chart Generation

**File**: `lecture_9/L9_visualization.ipynb`  
**Slides**: `presentation-slides/L9_visualization_slides.pptx`

**What you learn**:
- Translating evaluation numbers into two fundamental chart types:
  - **Line chart**: BLEU score variation across samples (trend and consistency)
  - **Bar chart**: term hit/miss breakdown (categorical comparison)
- Applying academic-paper formatting conventions (font size, DPI, axis labels, minimal decoration) to produce figures ready for direct insertion into a paper or thesis

**Depends on**: `lecture_9/l08_evaluation_results.json` (copy of Lecture 8 output)  
**Output figures**: `lecture_9/outputs/L09/` — `bleu_line_basic.png`, `bleu_line_paper.png`, `term_bar_basic.png`, `term_bar_paper.png`  
**Key concepts**: matplotlib styling, DPI/figure size for print, legend placement, academic figure conventions

---

### Lecture 10 — Capstone: End-to-End DH Research Pipeline

**File**: `lecture_10/L10_capstone_pipeline.ipynb`  
**Slides**: `presentation-slides/L10_capstone_pipeline_slides.pptx`

**What you learn**:
- How every component built across Lectures 1–9 fits together as a single callable pipeline
- Given one DH short video as input, the pipeline automatically produces:
  1. Audio extraction and Whisper transcription (L06)
  2. LLaMA-3 archival summary (L01)
  3. CLIP frame–text alignment score (L02)
  4. BLIP-2 frame captions (L05)
  5. LLaVA-Next cultural analysis (L07)
  6. QLoRA-fine-tuned terminology explanations (L03)
  7. BLEU + term-accuracy evaluation (L08)
  8. Exported figures and a structured JSON research report (L09)

This capstone notebook is the prototype for a complete Digital Humanities MLLM research workflow.

---

## Shared Resources

### `shared/corpus/`

| File | Description |
|---|---|
| `real_examples.json` | 10 human-verified Chinese DH term→explanation pairs covering oracle bone script (甲骨文), ancient manuscript restoration (古籍修复), intangible cultural heritage (非遗), Dunhuang murals (敦煌壁画), movable type printing (活字印刷), bronze inscriptions (青铜铭文), Kunqu opera (昆曲), bamboo slips (简牍), rubbings (拓片), and the 24 Solar Terms (二十四节气). Used as fine-tuning training data in L03 and evaluation references in L08. |
| `ancient_object.jpg` | Sample DH image used in L07 for visual analysis demonstrations. |

### `shared/weights/qlora_adapter_lecture3_archive/`

Pre-trained QLoRA adapter for `NousResearch/Meta-Llama-3-8B-Instruct`, fine-tuned on `real_examples.json`. Used in Lectures 4 and 8.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL  = 'NousResearch/Meta-Llama-3-8B-Instruct'
ADAPTER_PATH = '../shared/weights/qlora_adapter_lecture3_archive'

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype='bfloat16')

tokenizer  = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,
                 quantization_config=bnb_config, device_map='auto')
model      = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
```

> Large binary files — do not commit to git without Git LFS configured.

---

## Models at a Glance

| Model | Lecture(s) | Task | VRAM (4-bit) |
|---|---|---|---|
| `NousResearch/Meta-Llama-3-8B-Instruct` | L01, L03, L04, L06, L10 | Text summarisation, QLoRA fine-tuning | ~5–6 GB |
| `openai/clip-vit-base-patch32` | L02, L10 | Cross-modal frame–text matching | < 1 GB |
| `Salesforce/blip2-opt-2.7b` | L05, L10 | Image captioning | ~5 GB |
| `openai/whisper-small` | L06, L10 | Speech-to-text transcription | ~1 GB |
| `llava-hf/llava-v1.6-mistral-7b-hf` | L07, L10 | Generative visual analysis | ~7 GB |

All models are downloaded automatically from Hugging Face on first run. No access tokens are required.

---

## Data Ethics Note

When substituting the placeholder transcripts with real interview recordings, field-work audio, or archival footage, ensure that:

- Informed consent has been obtained from participants where applicable.
- Sensitive materials comply with the data policies of the source institution.
- Model outputs used in publications accurately cite the underlying data provenance.

In Digital Humanities research, respect for research subjects takes precedence over model accuracy.

---

## Quick Start

1. Open any lecture notebook in [Google Colab](https://colab.research.google.com/) using **File → Open notebook → GitHub** and paste this repository URL, or upload the `.ipynb` file directly.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run the cells in order. The first cell auto-detects Colab and installs all required packages.
4. For Lectures 4 and 8, upload the `shared/weights/qlora_adapter_lecture3_archive/` folder when prompted, or adjust `ADAPTER_PATH` to point to your local copy.

---

## Lecture Dependency Graph

```
L01 (LLaMA-3 summarise)
  └─► L03 (QLoRA fine-tune) ──► L04 (corpus + precision)
        └─► L08 (evaluation) ──► L09 (charts)
L02 (CLIP matching)
L05 (BLIP-2 captioning)
L06 (Whisper + L01) ──► L10 (capstone, integrates all)
L07 (LLaVA-Next analysis)
```

Lectures 1–7 can be studied independently. Lecture 8 requires the adapter from Lecture 3. Lecture 9 requires the evaluation JSON from Lecture 8. Lecture 10 integrates all previous components.
