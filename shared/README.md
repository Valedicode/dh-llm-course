# Shared Resources

Course-wide data, model weights, and reference links used across lectures.

## corpus/

Training and reference data files shared across notebooks.

| File | Description | Used in |
|------|-------------|---------|
| `real_examples.json` | 10 real Chinese DH transcript→explanation pairs (甲骨文, 古籍修复, 非遗, 敦煌壁画, 活字印刷, 青铜铭文, 昆曲, 简牍, 拓片, 二十四节气) | L03 fine-tuning (replace placeholder data) |

## weights/

Model checkpoints and adapters. Large binary files — do not commit to git without LFS.

| Folder | Description | Used in |
|--------|-------------|---------|
| `qlora_adapter_lecture3_archive/` | Trained QLoRA adapter for LLaMA-3 8B (NousResearch/Meta-Llama-3-8B-Instruct), fine-tuned on `real_examples.json` | L03 (saved output), L04 (reloaded for pipeline) |

### Loading the adapter in a notebook

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = 'NousResearch/Meta-Llama-3-8B-Instruct'
ADAPTER_PATH = '../shared/weights/qlora_adapter_lecture3_archive'

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map='auto')
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
```

## Model IDs referenced in notebooks

| Model | Use case | HuggingFace link |
|-------|----------|-----------------|
| `NousResearch/Meta-Llama-3-8B-Instruct` | L01 summarisation, L03 QLoRA fine-tuning | https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct |
| `openai/clip-vit-base-patch32` | L02 CLIP video-text matching | https://huggingface.co/openai/clip-vit-base-patch32 |
