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

### LoRA 参数含义与调优建议

- `r`（秩/低维）：决定适配器容量与显存/计算开销。越大拟合能力越强，但更耗显存与时间。常用范围 8–64；当前值 `16` 属于稳健中档。
- `target_modules`：在这些层插入 LoRA。`q_proj/k_proj/v_proj/o_proj` 属于注意力投影层，`gate_proj/up_proj/down_proj` 属于前馈（MLP）层。覆盖注意力+MLP 通常效果更好，但显存占用最高。
- `lora_alpha`（缩放系数）：有效缩放约为 `lora_alpha / r`，控制 LoRA 分支输出幅度。`alpha=16` 且 `r=16`，有效缩放≈1，较为均衡。
- `lora_dropout`：仅作用于 LoRA 分支的 dropout（训练时启用，推理关闭）。`0` 适合小数据或希望最大化拟合；若过拟合可设 `0.05–0.1`。
- `bias`：是否训练偏置。`"none"` 不更新偏置，省显存且更稳；`"all"` 训练全部偏置；`"lora_only"` 仅为 LoRA 分支加偏置。一般推荐 `"none"`。
- `use_gradient_checkpointing`：梯度检查点策略。`"unsloth"` 使用优化实现，可降低约 30% 显存占用，但训练耗时会略增。
- `random_state`（随机种子）：保证初始化与数据打乱的可复现性，便于对比实验与复现。
- `use_rslora`：是否启用 Rank‑Stabilized LoRA（RS‑LoRA）。用于稳定不同 `r` 配置下的训练与收敛。为 `False` 时使用常规 LoRA。
- `loftq_config`：LoFTQ（面向 LoRA 微调的量化）配置。与 `4bit` 量化（如 QLoRA）协同，以减轻量化带来的精度损失；`None` 表示不启用。

调参建议：
- 显存吃紧：仅在注意力层使用 LoRA（如 `["q_proj","v_proj","o_proj"]`），或将 `r` 降至 `8–12`。
- 训练不稳定或梯度爆炸：开启 `use_rslora=True`，并将 `lora_dropout` 提高到 `0.05` 左右。
- 欠拟合：提升 `r` 至 `32`，或将 `lora_alpha` 增至 `32`（保持 `alpha/r≈1` 或略大）。
- 过拟合：提高 `lora_dropout`、缩小 `target_modules` 覆盖范围，或降低 `alpha/r`。
- 长序列/大批次：保持 `use_gradient_checkpointing="unsloth"`，必要时适当减小 `r` 控制显存。
- 计划量化部署：结合 QLoRA 时提供合适的 `loftq_config`，以减少量化后精度回退。

快速判断与当前配置：
- 建议让有效缩放 `lora_alpha / r` 落在 `0.5–2` 区间，通常更稳。
- 当前配置（`r=16`、`alpha=16`、`dropout=0`、覆盖注意力+MLP）在中等规模数据与显存充足场景下较为平衡；若数据较小或易过拟合，建议将 `dropout` 先调至 `0.05` 做对比实验。

- 训练参数（`transformers.TrainingArguments`）
  - `per_device_train_batch_size=2`，`gradient_accumulation_steps=4`。
  - `warmup_steps=5`，`max_steps=60`（示例短训练）。
  - `learning_rate=2e-4`，`optim="adamw_8bit"`，`weight_decay=0.01`。
  - 精度：`fp16 = not is_bfloat16_supported()`，`bf16 = is_bfloat16_supported()`。
  - `lr_scheduler_type="linear"`，`seed=3407`，`logging_steps=1`，`output_dir="outputs"`。

### 训练参数含义与调优建议（TrainingArguments）

- `per_device_train_batch_size`：每张 GPU/每个进程的微批大小。与 `gradient_accumulation_steps` 和设备数共同决定有效批次大小。
- `gradient_accumulation_steps`：梯度累积步数。每累积 N 个微批执行一次优化更新，在显存有限时模拟更大的批次。
- `warmup_steps`：线性预热步数，前 N 步将学习率从 0 升至目标值，缓解初期不稳定。也可用 `warmup_ratio`（如 0.03–0.1）。
- `max_steps`：总优化步数；当 > 0 时会覆盖 `num_train_epochs`，到达该步数后停止训练。
- `num_train_epochs`：训练轮数；仅在 `max_steps <= 0` 时生效。二者不建议同时设为正值。
- `learning_rate`：学习率。LoRA/QLoRA 常用范围 `1e-4 ~ 3e-4`，较大数据/更长训练可适当降低；小数据/专业领域（如医疗）建议 `1e-4 ~ 2e-4` 更稳。
- `fp16/bf16`：混合精度。Ampere+ GPU 倾向使用 BF16（稳定性更好）；否则使用 FP16。示例写法保证两者只会启用其一。
- `logging_steps`：日志打印间隔（以优化步为单位）。数值越小日志越密集、开销略增；可设为 10–50 以平衡噪声与性能。
- `optim`：优化器。`adamw_8bit` 使用 bitsandbytes 的 8-bit 优化器，显存友好；在不支持环境（如部分 Windows）可改为 `adamw_torch`。
- `weight_decay`：权重衰减。LoRA 仅训练少量参数，`0.0 ~ 0.01` 常见；过大可能抑制适配器学习。
- `lr_scheduler_type`：学习率调度器。`linear` 预热后线性下降；`cosine`/`cosine_with_restarts` 更平滑，后期泛化常更好。
- `seed`：随机种子，保证初始化与数据打乱的可复现。
- `output_dir`：训练输出目录。
- `report_to`：训练日志上报目标。`"none"` 不上报；可设为 `"wandb"`、`"tensorboard"` 等。

关键关系：
- 有效批次大小（单机）≈ `per_device_train_batch_size × gradient_accumulation_steps`；多卡需再乘以设备数（world size）。
- 每个 epoch 的优化步数 ≈ `ceil(样本数 / 有效批次大小)`。
- 当 `max_steps > 0` 时优先生效，`num_train_epochs` 被忽略；若希望按轮数训练，请将 `max_steps` 设为 `-1`。

调优建议：
- 显存与吞吐：显存吃紧时先降 `per_device_train_batch_size` 并提 `gradient_accumulation_steps` 保持有效批次；必要时降低序列长度与 LoRA 覆盖范围（见上文 LoRA 小节）。
- 学习率与调度：小数据/易过拟合任务（如医学指令集）建议 `learning_rate=1e-4 ~ 2e-4`、增大 `warmup_steps` 或 `warmup_ratio`（3%–10%）；大数据/长训练倾向 `cosine` 或 `cosine_with_restarts`。
- 训练步数 vs 轮数：若需精确控制预算，用 `max_steps`（如 240/1000/5000）；若需按轮数训练，设 `max_steps=-1` 并用 `num_train_epochs`。
- 精度与优化器：支持 BF16 的 GPU 建议 `bf16=True, fp16=False`；Windows/bitsandbytes 不可用时，将 `optim` 改为 `adamw_torch`，并参考下文 4bit 兼容提示。
- 正则化与稳定性：LoRA 通常不需要较大 `weight_decay`；初期震荡或损失不降时，增大 `warmup_steps`、降低 `learning_rate`，或在 LoRA 配置中适度提高 `lora_dropout`。
- 监控与可视化：将 `report_to` 设为 `"wandb"` 或 `"tensorboard"` 开启可视化；将 `logging_steps` 设为 10–50 以减少噪声并提升吞吐。

示例提示：
- 若脚本中同时设置了 `max_steps` 与 `num_train_epochs`，训练会在达到 `max_steps` 后提前停止；想跑满若干 epoch，请将 `max_steps=-1`。

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