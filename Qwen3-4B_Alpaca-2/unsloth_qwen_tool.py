"""
Unsloth Qwen 微调工具模块

此模块将使用 Unsloth 对 Qwen 系列进行微调的常见流程抽象为可复用的工具类，
涵盖：模型加载、LoRA 适配器配置、使用 Alpaca 风格模板的数据集准备、监督微调训练、
推理以及多种保存/导出方式（普通 LoRA、合并为 16bit/4bit、GGUF）。

使用示例：

from unsloth_qwen_tool import UnslothQwenFineTuner

tool = UnslothQwenFineTuner(
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
tool.load_model(model_name_or_path="/path/to/Qwen3-4B", local_files_only=True)
tool.add_lora_adapter()
ds = tool.prepare_alpaca_dataset(dataset_path_or_name="yahma/alpaca-cleaned", split="train")
trainer = tool.build_trainer(
    train_dataset=ds,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",
)
stats = tool.train(trainer)
tool.enable_inference()
text = tool.generate(
    instruction="Continue the fibonnaci sequence.",
    inp="1, 1, 2, 3, 5, 8",
    max_new_tokens=64,
)
print(text)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported


class UnslothQwenFineTuner:
    """Unsloth + Qwen 微调与推理的可复用工具类。

    主要职责：
    - 加载基础模型与分词器，并设置最大长度、数据类型、4bit 量化等。
    - 为注意力与前馈等模块注入 LoRA 适配器，支持梯度检查点以节省显存。
    - 基于 Alpaca 模板准备数据集，将 `instruction/input/output` 拼接为训练文本。
    - 构建并运行 SFTTrainer（监督微调），通过 `TrainingArguments` 控制训练参数。
    - 启用 Unsloth 的原生 2x 推理加速，进行一次性或流式生成。
    - 保存/加载 LoRA 参数；合并保存为 16bit/4bit；导出或推送 GGUF。
    """

    def __init__(
        self,
        max_seq_length: int = 2048,
        dtype: Optional[str] = None,
        load_in_4bit: bool = True,
        prompt_template: str | None = None,
    ) -> None:
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self._prompt_template = (
            prompt_template
            or (
                "Below is an instruction that describes a task,\n"
                "paired with an input that provides further context.\n"
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{}\n\n"
                "### Input:\n{}\n\n"
                "### Response:\n{}"
            )
        )
        self._eos_token = None

    # --------------------- Core: Model & LoRA ---------------------
    def load_model(self, model_name_or_path: str, local_files_only: bool = True) -> None:
        """加载基础模型与分词器。

        参数：
        - model_name_or_path：模型名称或本地路径。
        - local_files_only：仅从本地加载（避免在线下载）。
        """
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            local_files_only=local_files_only,
        )
        self.model = model
        self.tokenizer = tokenizer
        self._eos_token = getattr(tokenizer, "eos_token", None)

    def add_lora_adapter(
        self,
        r: int = 16,
        target_modules: Optional[Iterable[str]] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        use_gradient_checkpointing: str | bool = "unsloth",
        random_state: int = 3407,
        use_rslora: bool = False,
        loftq_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """为当前已加载的模型附加 LoRA 适配器。

        关键参数：
        - r：LoRA 秩（通常取值 8/16/32/64/128）。
        - target_modules：要注入 LoRA 的模块列表。
        - use_gradient_checkpointing：启用梯度检查点以降低显存占用。
        - use_rslora / loftq_config：可选的稳定/量化配置。
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before adding LoRA adapters.")
        target_modules = target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=list(target_modules),
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=loftq_config,
        )

    # --------------------- Core: Prompting & Dataset ---------------------
    def build_prompt(self, instruction: str, inp: str = "", response: str = "") -> str:
        """构造 Alpaca 风格的提示字符串，并在可能时追加 EOS 标记。"""
        text = self._prompt_template.format(instruction, inp, response)
        if self._eos_token:
            return text + self._eos_token
        return text

    def _formatting_prompts_map(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = [self.build_prompt(i, x, o) for i, x, o in zip(instructions, inputs, outputs)]
        return {"text": texts}

    def prepare_alpaca_dataset(
        self,
        dataset_path_or_name: str,
        split: str = "train",
        num_proc: int = 2,
    ) -> Dataset:
        """加载并格式化 Alpaca 风格数据集（支持本地路径或 Hugging Face 仓库名）。"""
        ds = load_dataset(dataset_path_or_name, split=split)
        ds = ds.map(self._formatting_prompts_map, batched=True, num_proc=num_proc)
        return ds

    # --------------------- Core: Training ---------------------
    def build_trainer(
        self,
        train_dataset: Dataset,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 5,
        max_steps: int = 60,
        learning_rate: float = 2e-4,
        logging_steps: int = 1,
        optim: str = "adamw_8bit",
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "linear",
        seed: int = 3407,
        output_dir: str = "outputs",
        report_to: str = "none",
        packing: bool = False,
        dataset_text_field: str = "text",
        dataset_num_proc: int = 2,
    ) -> SFTTrainer:
        """创建 SFTTrainer 用于监督微调（SFT）。

        说明：自动根据设备是否支持 bfloat16 来设置 `fp16/bf16`；
        可通过 `packing=True` 在短序列场景提升训练吞吐。"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before building trainer.")

        training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=logging_steps,
            optim=optim,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            seed=seed,
            output_dir=output_dir,
            report_to=report_to,
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            dataset_text_field=dataset_text_field,
            max_seq_length=self.max_seq_length,
            dataset_num_proc=dataset_num_proc,
            packing=packing,
            args=training_args,
        )
        return trainer

    def train(self, trainer: SFTTrainer) -> Any:
        """执行训练并返回训练统计信息对象。"""
        stats = trainer.train()
        return stats

    # --------------------- Core: GPU Stats ---------------------
    def gpu_memory_stats(self) -> Dict[str, Any]:
        """返回当前 GPU 显存使用统计（如果 CUDA 可用）。"""
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        return {
            "cuda_available": True,
            "gpu_name": gpu_stats.name,
            "max_memory_gb": max_memory,
            "reserved_memory_gb": start_gpu_memory,
        }

    def print_gpu_memory_stats(self) -> None:
        """打印当前 GPU 显存使用统计信息。"""
        stats = self.gpu_memory_stats()
        if not stats.get("cuda_available"):
            print("CUDA not available.")
            return
        print(f"GPU = {stats['gpu_name']}. Max memory = {stats['max_memory_gb']} GB.")
        print(f"{stats['reserved_memory_gb']} GB of memory reserved.")

    # --------------------- Core: Inference ---------------------
    def enable_inference(self) -> None:
        """为当前模型启用 Unsloth 原生 2x 推理加速模式。"""
        if self.model is None:
            raise RuntimeError("Model must be loaded before enabling inference.")
        FastLanguageModel.for_inference(self.model)

    def _extract_response_text(self, decoded: str) -> str:
        """从解码后的文本中提取 `### Response:` 之后的内容。"""
        marker = "### Response:"
        idx = decoded.find(marker)
        if idx == -1:
            return decoded
        return decoded[idx + len(marker):].strip()

    def generate(
        self,
        instruction: str,
        inp: str = "",
        max_new_tokens: int = 128,
        use_cache: bool = True,
        stream: bool = False,
    ) -> str:
        """针对单条指令/输入对进行生成。

        说明：
        - `stream=True` 时使用 `TextStreamer` 进行连续输出（直接打印到终端），函数返回空字符串。
        - `stream=False` 时返回完整的生成文本（已尝试抽取响应正文）。
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generation.")
        if stream:
            FastLanguageModel.for_inference(self.model)
        prompt = self.build_prompt(instruction, inp, "")
        inputs = self.tokenizer([prompt], return_tensors="pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if stream:
            text_streamer = TextStreamer(self.tokenizer)
            _ = self.model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens)
            return ""
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=use_cache)
        decoded = self.tokenizer.batch_decode(outputs)[0]
        return self._extract_response_text(decoded)

    # --------------------- Core: Saving & Reloading ---------------------
    def save_lora(self, output_dir: str) -> None:
        """将 LoRA 适配器与分词器一并保存到指定目录。"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before saving.")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_lora(self, lora_dir: str) -> None:
        """从目录中加载已保存的 LoRA 模型与分词器。"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_dir,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        self.model = model
        self.tokenizer = tokenizer
        self._eos_token = getattr(tokenizer, "eos_token", None)

    # --------------------- Optional: Merged Saves ---------------------
    def save_merged(self, save_dir: str, save_method: str = "merged_16bit") -> None:
        """保存合并后的模型（支持 16bit、4bit 或仅 lora）。"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before saving.")
        self.model.save_pretrained_merged(save_dir, self.tokenizer, save_method=save_method)

    def push_merged_to_hub(
        self,
        repo_id: str,
        token: str,
        save_method: str = "merged_16bit",
    ) -> None:
        """将合并后的模型推送到 Hugging Face Hub。"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before pushing.")
        self.model.push_to_hub_merged(repo_id, self.tokenizer, save_method=save_method, token=token)

    # --------------------- Optional: GGUF Saves ---------------------
    def save_gguf(self, save_dir: str, quantization_method: str | None = None) -> None:
        """保存为 GGUF / llama.cpp 可用的格式，量化方式可选。"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before saving.")
        if quantization_method:
            self.model.save_pretrained_gguf(save_dir, self.tokenizer, quantization_method=quantization_method)
        else:
            self.model.save_pretrained_gguf(save_dir, self.tokenizer)

    def push_gguf_to_hub(
        self,
        repo_id: str,
        token: str,
        quantization_method: str | None = None,
    ) -> None:
        """将 GGUF 格式模型推送到 Hugging Face Hub。"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before pushing.")
        if quantization_method:
            self.model.push_to_hub_gguf(repo_id, self.tokenizer, quantization_method=quantization_method, token=token)
        else:
            self.model.push_to_hub_gguf(repo_id, self.tokenizer, token=token)