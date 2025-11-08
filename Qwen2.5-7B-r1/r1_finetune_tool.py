"""
R1FineTuneTool

封装 Qwen2.5-7B-R1 的 GRPO 微调、推理与权重保存流程。

特性：
- 构建/量化加载模型（Unsloth FastLanguageModel）
- 配置并注入 LoRA（PEFT）
- 加载 GSM8K 数据并构造 r1 风格 XML 格式提示
- 复用/封装奖励函数（正确性、整数、格式匹配、XML计数）
- 训练（TRL.GRPOTrainer）与日志配置
- 推理（fast_generate + vLLM SamplingParams）
- 保存 LoRA、合并导出 merged_16bit/merged_4bit

使用示例：

from Qwen2.5-7B-r1.r1_finetune_tool import R1FineTuneTool, R1FineTuneConfig

cfg = R1FineTuneConfig(
    model_name="/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct",
    dataset_repo_or_path="openai/gsm8k",
    dataset_subset="main",
    local_files_only=True,
)

tool = R1FineTuneTool(cfg)
model, tokenizer = tool.prepare_model()
tool.prepare_lora(model)
dataset = tool.get_gsm8k_questions(split="train")
trainer = tool.build_trainer(model, tokenizer, dataset)
trainer.train()
tool.save_lora(model, "grpo_saved_lora")

# 推理
text = tool.build_chat_text(tokenizer, user_text="Calculate pi.")
output_text = tool.fast_generate(model, text)
print(output_text)

# 合并导出（用于 vLLM 推理）
tool.save_merged(model, tokenizer, out_dir="model", save_method="merged_16bit")

"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
import re

import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams


DEFAULT_SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


@dataclass
class R1FineTuneConfig:
    # 模型加载相关
    model_name: str
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    fast_inference: bool = True
    max_lora_rank: int = 32
    gpu_memory_utilization: float = 0.6
    local_files_only: bool = False

    # LoRA 相关
    lora_rank: int = 32
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407

    # 数据集相关
    dataset_repo_or_path: str = "openai/gsm8k"
    dataset_subset: str = "main"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # 训练相关
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 6
    max_steps: int = 250
    save_steps: int = 250
    max_grad_norm: float = 0.1
    report_to: str = "none"
    output_dir: str = "outputs"
    max_prompt_length: int = 256

    # 推理相关
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 1024


class R1FineTuneTool:
    def __init__(self, cfg: R1FineTuneConfig):
        self.cfg = cfg

    # ============ 文本/格式处理 ============
    @staticmethod
    def xml_cot_format(reasoning: str, answer: str) -> str:
        return f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer}\n</answer>\n"

    @staticmethod
    def extract_xml_answer(text: str) -> str:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

    @staticmethod
    def extract_hash_answer(text: str) -> Optional[str]:
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    # ============ 数据集 ============
    def get_gsm8k_questions(self, split: str = "train") -> Dataset:
        data = load_dataset(self.cfg.dataset_repo_or_path, self.cfg.dataset_subset)[split]
        sys_prompt = self.cfg.system_prompt
        data = data.map(lambda x: {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": self.extract_hash_answer(x["answer"]),
        })
        return data

    # ============ 奖励函数 ============
    def correctness_reward_func(self, prompts, completions, answer, **kwargs) -> List[float]:
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [self.extract_xml_answer(r) for r in responses]
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    def int_reward_func(self, completions, **kwargs) -> List[float]:
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [self.extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

    def strict_format_reward_func(self, completions, **kwargs) -> List[float]:
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def soft_format_reward_func(self, completions, **kwargs) -> List[float]:
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    @staticmethod
    def _count_xml(text: str) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
        return count

    def xmlcount_reward_func(self, completions, **kwargs) -> List[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [self._count_xml(c) for c in contents]

    def build_reward_funcs(self):
        return [
            self.xmlcount_reward_func,
            self.soft_format_reward_func,
            self.strict_format_reward_func,
            self.int_reward_func,
            self.correctness_reward_func,
        ]

    # ============ 模型与 LoRA ============
    def prepare_model(self) -> Tuple[Any, Any]:
        cfg = self.cfg
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            fast_inference=cfg.fast_inference,
            max_lora_rank=cfg.max_lora_rank,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            local_files_only=cfg.local_files_only,
        )
        return model, tokenizer

    def prepare_lora(self, model: Any) -> Any:
        cfg = self.cfg
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_rank,
            target_modules=cfg.target_modules,
            lora_alpha=cfg.lora_rank,
            use_gradient_checkpointing=cfg.use_gradient_checkpointing,
            random_state=cfg.random_state,
        )
        return model

    # ============ 训练器构建 ============
    def build_training_args(self) -> GRPOConfig:
        cfg = self.cfg
        training_args = GRPOConfig(
            learning_rate=cfg.learning_rate,
            adam_beta1=cfg.adam_beta1,
            adam_beta2=cfg.adam_beta2,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler_type,
            optim=cfg.optim,
            logging_steps=cfg.logging_steps,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            num_generations=cfg.num_generations,
            max_prompt_length=cfg.max_prompt_length,
            max_completion_length=cfg.max_seq_length - cfg.max_prompt_length,
            max_steps=cfg.max_steps,
            save_steps=cfg.save_steps,
            max_grad_norm=cfg.max_grad_norm,
            report_to=cfg.report_to,
            output_dir=cfg.output_dir,
        )
        return training_args

    def build_trainer(self, model: Any, tokenizer: Any, train_dataset: Dataset) -> GRPOTrainer:
        training_args = self.build_training_args()
        reward_funcs = self.build_reward_funcs()
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
        )
        return trainer

    # ============ 推理 ============
    def build_chat_text(self, tokenizer: Any, user_text: str, system_prompt: Optional[str] = None) -> str:
        sys_prompt = system_prompt or self.cfg.system_prompt
        text = tokenizer.apply_chat_template([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_text},
        ], tokenize=False, add_generation_prompt=True)
        return text

    def fast_generate(self, model: Any, text: str, temperature: Optional[float] = None,
                       top_p: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        cfg = self.cfg
        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else cfg.temperature,
            top_p=top_p if top_p is not None else cfg.top_p,
            max_tokens=max_tokens if max_tokens is not None else cfg.max_tokens,
        )
        output = model.fast_generate(
            text,
            sampling_params=sampling_params,
        )[0].outputs[0].text
        return output

    # ============ 权重保存 ============
    @staticmethod
    def save_lora(model: Any, out_dir: str) -> None:
        model.save_lora(out_dir)

    @staticmethod
    def save_merged(model: Any, tokenizer: Any, out_dir: str, save_method: str = "merged_16bit") -> None:
        # 支持 merged_16bit（float16）、merged_4bit（int4）和 lora 适配器
        model.save_pretrained_merged(out_dir, tokenizer, save_method=save_method)