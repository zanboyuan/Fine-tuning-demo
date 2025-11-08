#!/usr/bin/env python
# coding: utf-8
"""
脚本说明（中文）：
- 使用 Unsloth 框架对 Qwen3-4B 进行监督微调（SFT），包含模型加载、LoRA 适配器配置、
  Alpaca 格式数据集准备、训练、推理、LoRA 保存与重新加载、合并保存（16bit/4bit）、
  以及 GGUF 导出示例。
- 注意：本脚本为演示用途，路径如模型与数据集的本地位置需要按你的环境实际修改；
  若未启用 CUDA 或显存不足，请调整 `max_seq_length`、batch 大小、是否 4bit 等参数。
"""

# ### 使用 Unsloth 框架对 Qwen3-4B 模型进行微调的示例代码
# ### 本代码可以在免费的 Tesla T4 Google Colab 实例上运行 https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-Alpaca.ipynb

# In[1]:


# 导入必要的库
from unsloth import FastLanguageModel
import torch

# 设置模型参数（关键：根据你的 GPU 能力选择合适的配置）
max_seq_length = 2048  # 设置最大序列长度，支持 RoPE 缩放
dtype = None  # 数据类型，None 表示自动检测。Tesla T4 使用 Float16，Ampere+ 使用 Bfloat16
load_in_4bit = True  # 使用 4bit 量化来减少内存使用

# 加载预训练模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name 可使用 Hugging Face 名称（例如 "unsloth/Qwen3-4B"）或本地路径。
    # 出于演示，这里使用本地路径；请按需修改为你的环境路径或在线仓库名。
    # model_name = "unsloth/Qwen3-4B",  # 示例：在线仓库名
    model_name = "/root/autodl-tmp/models/Qwen/Qwen3-4B",  # 示例：本地模型路径
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    local_files_only=True,  # 仅从本地文件加载模型，避免在受限环境下触发下载
)


# In[2]:


# 添加 LoRA 适配器（关键：仅更新少量参数，显著降低微调成本）
model = FastLanguageModel.get_peft_model(
    model, # Qwen3-4B
    r = 16,  # LoRA秩，建议使用8、16、32、64、128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],  # 需要应用LoRA的模块
    lora_alpha = 16,  # LoRA缩放因子
    lora_dropout = 0,  # LoRA dropout率，0为优化设置
    bias = "none",    # 偏置项设置，none为优化设置
    use_gradient_checkpointing = "unsloth",  # 使用梯度检查点，可减少约30%显存使用
    random_state = 3407,  # 随机种子
    use_rslora = False,  # 是否使用rank stabilized LoRA
    loftq_config = None,  # LoftQ配置
)


# ### 数据准备
# 定义 Alpaca 格式的提示模板（注意：这是训练/推理文本的结构，不建议改动关键词）
alpaca_prompt = """Below is an instruction that describes a task, 
paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 获取结束标记（关键：生成任务需要 EOS，避免无穷生成）
EOS_TOKEN = tokenizer.eos_token

# 定义数据格式化函数：将 instruction/input/output 拼接为可训练文本，并追加 EOS
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # 必须添加EOS_TOKEN，否则生成会无限继续
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# 加载 Alpaca 数据集（可替换为 Hugging Face 仓库名或本地路径）
# 例如：`yahma/alpaca-cleaned`（在线）或本地目录路径。
from datasets import load_dataset
dataset = load_dataset("/root/autodl-tmp/datasets/yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)


# ### 模型训练

# 设置训练参数和训练器（关键：根据显存/速度需求调整 batch、accumulation、max_steps 等）
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# 定义训练参数
training_args = TrainingArguments(
        per_device_train_batch_size = 2,  # 每个设备的训练批次大小
        gradient_accumulation_steps = 4,  # 梯度累积步数
        warmup_steps = 5,  # 预热步数
        max_steps = 60,  # 最大训练步数
        learning_rate = 2e-4,  # 学习率
        fp16 = not is_bfloat16_supported(),  # 是否使用 FP16（若不支持 BF16）
        bf16 = is_bfloat16_supported(),  # 是否使用 BF16（若硬件支持更推荐）
        logging_steps = 1,  # 日志记录步数
        optim = "adamw_8bit",  # 优化器
        weight_decay = 0.01,  # 权重衰减
        lr_scheduler_type = "linear",  # 学习率调度器类型
        seed = 3407,  # 随机种子
        output_dir = "outputs",  # 输出目录
        report_to = "none",  # 报告方式
    )

# 创建SFTTrainer实例 => SFT 监督微调的训练器
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,  # 短序列场景可设为 True，提高吞吐；长序列建议保持 False
    args = training_args,
)


# In[5]:


# 显示当前 GPU 内存状态（便于评估显存占用与是否需要调小参数）
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# In[6]:


# 开始训练（返回训练统计信息，含运行时长等）
trainer_stats = trainer.train()


# In[7]:


# 显示训练后的显存与时间统计（关键：观察 LoRA 训练额外显存占用比例）
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# ### 模型推理

# In[8]:


# 模型推理（一次性生成）：
# 先启用 Unsloth 原生 2x 推理加速，再按模板构造输入并生成输出。
FastLanguageModel.for_inference(model)  # 启用原生 2x 推理
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.",  # 指令
        "1, 1, 2, 3, 5, 8",  # 输入
        "",  # 输出留空用于生成
    )
], return_tensors = "pt").to("cuda")

# 生成输出
outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)


# In[9]:


# 使用 TextStreamer 进行连续推理（流式打印到终端，适合长文本）
FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.",  # 指令
        "1, 1, 2, 3, 5, 8",  # 输入
        "",  # 输出留空用于生成
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)


# ### 微调模型保存
# **[注意]** 这里只是LoRA参数，不是完整模型。

# In[10]:


# 保存 LoRA 适配器与分词器（注意：仅保存 LoRA 权重，不是完整基础模型）
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")


# In[11]:


# 加载保存的 LoRA 模型进行推理（演示重新加载后的使用方式）
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model",  # 训练时使用的模型
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # 启用原生2倍速推理

# 使用重新加载的模型进行推理示例（流式输出）
inputs = tokenizer(
[
    alpaca_prompt.format(
        "What is a famous tall tower in Paris?",  # 指令
        "",  # 输入
        "",  # 输出留空用于生成
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)


# You can also use Hugging Face's `AutoModelForPeftCausalLM`. Only use this if you do not have `unsloth` installed. It can be hopelessly slow, since `4bit` model downloading is not supported, and Unsloth's **inference is 2x faster**.

# In[15]:


# 使用 Hugging Face 的 AutoPeftModelForCausalLM 加载模型（不推荐：下载与推理速度较慢）
if False:
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        "lora_model",  # 训练时使用的模型
        load_in_4bit = load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("lora_model")


# ### 保存为 float16（用于 VLLM）
# 
# 支持直接保存为 `float16`（选择 `merged_16bit`）或 `int4`（选择 `merged_4bit`）。
# 也可仅保存 `lora` 适配器作为回退方案。若需推送到 Hugging Face，
# 可使用 `push_to_hub_merged` 并在 https://huggingface.co/settings/tokens 获取个人 token。

# In[16]:


# 保存为float16格式（用于VLLM）
# 支持merged_16bit（float16）、merged_4bit（int4）和lora适配器
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# 保存为4bit格式
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# 仅保存LoRA适配器
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")


# ### GGUF / llama.cpp 转换
# 原生支持保存为 GGUF / llama.cpp 使用的格式，默认量化为 `q8_0`，也支持 `q4_k_m` 等多种方法。
# 本地保存使用 `save_pretrained_gguf`，推送到 Hugging Face 使用 `push_to_hub_gguf`。
# 
# 常用量化方法（完整列表见 Wiki：https://github.com/unslothai/unsloth/wiki#gguf-quantization-options）：
# * `q8_0`：转换速度快，资源占用较高，但通常可接受。
# * `q4_k_m`：推荐，对 attention.wv 与 feed_forward.w2 的一半使用 Q6_K，其余使用 Q4_K。
# * `q5_k_m`：推荐，对 attention.wv 与 feed_forward.w2 的一半使用 Q6_K，其余使用 Q5_K。
# 
# 【新】若希望微调并自动导出到 Ollama，可参考官方 Ollama Notebook 示例。

# In[17]:


# GGUF/llama.cpp转换
# 支持多种量化方法：
# q8_0：快速转换，资源使用较高但一般可接受
# q4_k_m：推荐使用，对attention.wv和feed_forward.w2使用Q6_K，其他使用Q4_K
# q5_k_m：推荐使用，对attention.wv和feed_forward.w2使用Q6_K，其他使用Q5_K

# 保存为8bit Q8_0格式
if False: model.save_pretrained_gguf("model", tokenizer,)
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# 保存为16bit GGUF格式
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# 保存为q4_k_m GGUF格式
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# 同时保存多种GGUF格式
if False:
    model.push_to_hub_gguf(
        "hf/model",  # 更改为你的用户名
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "",  # 从https://huggingface.co/settings/tokens获取token
    )


# 现在可在 llama.cpp 或 UI 系统（如 Jan / Open WebUI）中使用生成的 GGUF 文件。
# Jan 安装：https://github.com/janhq/jan ，Open WebUI：https://github.com/open-webui/open-webui
# 
# 到此结束。如需帮助或关注最新动态，可加入 Unsloth 的 Discord 频道。
# 更多示例：
# 1. 推理微调 - Llama GRPO（免费 Colab 示例）
# 2. 微调后导出至 Ollama（免费 Notebook 示例）
# 3. 视觉微调 - Llama 3.2 Radiography 用例（免费 Colab 示例）
# 以及 DPO/ORPO/继续预训练/对话式微调等，详见官方文档。
