# Qwen 微调与推理示例（Unsloth + Alpaca SFT）

本项目基于 `unsloth` 框架，演示如何对大语言模型（示例为 Qwen 系列）进行 LoRA 监督微调（SFT），并完成推理与多格式导出。脚本文件为 `Qwen3-4B_Alpaca-2.py`。

---

## 目录结构

- `Qwen3-4B_Alpaca-2.py`：主训练与推理脚本。
- `outputs/`：训练日志与输出（运行脚本后自动生成）。
- `lora_model/`：保存的 LoRA 适配器与分词器（运行保存步骤后生成）。

> 注：`outputs/` 与 `lora_model/` 的目录在初次运行后生成。

---

## 环境与硬件要求

- 操作系统：推荐 Linux/WSL 或云端（Colab、Linux 服务器）。
- Python：建议 `Python 3.9+`。
- GPU：NVIDIA、已正确安装 CUDA；显存建议 ≥ 16GB（Tesla T4 可运行，结合 4bit 量化）。

> Windows 原生对 `bitsandbytes` 支持有限。若需 4bit 量化，推荐在 WSL/Linux/Colab 环境运行；或将脚本中的 `load_in_4bit=False` 以禁用 4bit（显存占用会增加）。

---

## 依赖安装

在已配置 CUDA 的环境中执行：

```bash
pip install unsloth transformers trl datasets peft accelerate bitsandbytes
# 根据你的 CUDA 版本选择正确的 PyTorch 轮子（示例为 CUDA 12.1）
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> 若在 Windows 且 `bitsandbytes` 安装失败：
> - 使用 WSL/云端环境；或
> - 将脚本中的 `load_in_4bit=False` 暂时禁用 4bit 量化。

---

## 快速开始

1) 准备模型路径/名称

- 脚本默认使用本地路径：
  - `model_name = "/root/autodl-tmp/models/Qwen/Qwen3-4B"`
- 如果你使用在线模型（例如 `unsloth/Qwen3-4B` 或 `Qwen/Qwen3-4B`），请：
  - 将 `model_name` 设置为对应的 Hugging Face 名称；
  - 把 `local_files_only=True` 改为 `False` 以允许在线下载。
- 在 Windows 下，请将路径替换为你本机的实际路径（示例：`e:\models\Qwen\Qwen3-4B`）。

2) 准备数据集

- 代码默认本地加载：`/root/autodl-tmp/datasets/yahma/alpaca-cleaned`
- 若使用在线数据集：将 `load_dataset("yahma/alpaca-cleaned", split="train")` 并设置网络可用；或者改为你本地数据路径。

3) 运行脚本

```bash
python Qwen3-4B_Alpaca-2.py
```

4) 训练完成后，脚本会进行示例推理，并将 LoRA 适配器保存在 `lora_model/` 中。

---

## 脚本功能概览

- 加载与量化：`FastLanguageModel.from_pretrained(...)`，支持 `4bit` 量化与自动 `dtype`（T4 通常为 FP16，Ampere+ 可用 BF16）。
- 注入 LoRA：`FastLanguageModel.get_peft_model(...)`，仅微调 1–10% 参数，节省显存。
- 数据处理：使用 Alpaca 三段式模板，并在末尾添加 `EOS_TOKEN`（防止生成无限延续）。
- 训练器：`trl.SFTTrainer` 进行监督微调（SFT），可选 `packing=True` 以提升短样本训练速度。
- 监控与统计：打印 GPU 型号、峰值显存、训练用时等信息。
- 推理：启用 `FastLanguageModel.for_inference(model)`，使用 `generate` 输出文本；支持 `TextStreamer` 流式输出。
- 保存与加载：保存 LoRA 适配器到 `lora_model/` 并再次加载进行推理。
- 导出：支持合并为 `float16`/`int4`、导出为 GGUF（`llama.cpp` 兼容）、以及推送到 Hugging Face Hub。

---

## 关键配置说明

- 模型与量化
  - `model_name`：本地路径或在线模型名称。
  - `max_seq_length=2048`：序列长度，越长越耗显存。
  - `dtype=None`：自动选择（T4 FP16，Ampere+ BF16）。
  - `load_in_4bit=True`：4bit 量化以减少显存；Windows 原生可能不可用。
  - `local_files_only=True`：仅离线加载；若需在线下载，设为 `False`。

- LoRA 参数
  - `r=16`，`lora_alpha=16`，`lora_dropout=0`，`bias="none"`。
  - `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`。
  - `use_gradient_checkpointing="unsloth"`：显存优化（约 30%）。

- 训练参数（`transformers.TrainingArguments`）
  - `per_device_train_batch_size=2`，`gradient_accumulation_steps=4`。
  - `warmup_steps=5`，`max_steps=60`（示例短训练）。
  - `learning_rate=2e-4`，`optim="adamw_8bit"`，`weight_decay=0.01`。
  - 精度：`fp16 = not is_bfloat16_supported()`，`bf16 = is_bfloat16_supported()`。
  - `lr_scheduler_type="linear"`，`seed=3407`，`logging_steps=1`，`output_dir="outputs"`。

- 数据模板（Alpaca）
  - 模板字段：`instruction`、`input`、`output`。
  - 格式化函数会生成单字段 `text` 供 `SFTTrainer` 使用。
  - 末尾必须拼接 `EOS_TOKEN`。

---

## 训练与推理流程（脚本内置）

1. 加载模型与分词器，并启用 4bit 与自动精度（可选）。
2. 注入 LoRA 适配器，设置显存优化（梯度检查点）。
3. 加载与格式化 Alpaca 数据集为 `text` 字段。
4. 用 `SFTTrainer` 启动训练，输出日志与内存占用统计。
5. 启用推理加速，进行示例推理与流式输出。
6. 保存 LoRA 适配器到 `lora_model/` 并示例性重新加载推理。

---

## 导出与部署

- 合并权重（适配 VLLM 等）：
  - `model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")`（float16）。
  - `save_method="merged_4bit"`（int4）。
  - 仅导出 LoRA：`save_method="lora"`。
  - 推送到 Hub：`model.push_to_hub_merged("hf/model", tokenizer, save_method=..., token="...")`。

- GGUF（llama.cpp 兼容）：
  - 默认 `q8_0`：`model.save_pretrained_gguf("model", tokenizer)`。
  - 其他可选：`f16`、`q4_k_m`、`q5_k_m`。
  - 推送到 Hub：`model.push_to_hub_gguf("hf/model", tokenizer, quantization_method=..., token="...")`。
  - 支持一次性导出多种量化：`quantization_method=["q4_k_m","q8_0","q5_k_m"]`。

---

## Windows 本地运行提示

- 路径替换：将脚本中的 Linux 路径替换为 Windows 路径（如 `e:\\datasets\\alpaca-cleaned`、`e:\\models\\Qwen\\...`）。
- 离线/在线切换：若需在线加载模型或数据，设置 `local_files_only=False` 并保证网络可用。
- 4bit 兼容：Windows 原生 `bitsandbytes` 可能不可用；可在 WSL/云端运行，或将 `load_in_4bit=False`。

---

## 常见问题排查（FAQ）

- 无限生成或解码异常：确认模板末尾拼接了 `EOS_TOKEN`。
- 数据集加载失败：检查路径是否存在；若为离线加载且路径错误，请改正或使用在线加载。
- 显存不足（OOM）：开启 `4bit`、保持 `use_gradient_checkpointing`、降低 `max_seq_length`、增大 `gradient_accumulation_steps`。
- 训练过慢：短样本任务可将 `packing=True`（注意与你的数据字段设置一致）。
- Windows 量化错误：优先在 WSL/Colab/服务器运行；或禁用 4bit 并确保满足显存需求。

---

## 参考与致谢

- Unsloth 项目与文档：https://unsloth.ai / https://github.com/unslothai/unsloth
- Hugging Face Transformers/TRL/PEFT：https://huggingface.co/docs
- 数据集：`yahma/alpaca-cleaned`

如需进一步定制（更换模型、数据路径、训练超参或导出格式），可直接修改 `Qwen3-4B_Alpaca-2.py` 中的对应变量。如需要，我也可以为你的 Windows/WSL 环境自动调整脚本参数并验证可运行性。