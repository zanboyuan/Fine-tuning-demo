"""封装医疗问诊微调流程的工具类。

该模块不修改原始脚本，提供可复用的类以便：
- 加载并清洗医疗问答数据
- 初始化模型与LoRA适配器
- 构建并运行SFT训练
- 保存LoRA与合并后的权重

快速上手（一键式）：

from medical_finetune_tool import MedicalFineTuneConfig, MedicalFineTuner

config = MedicalFineTuneConfig(
    model_name="/root/autodl-tmp/models/Qwen/Qwen3-4B",
    data_dir="【数据集】中文医疗数据",
)

tuner = MedicalFineTuner(config)
tuner.run(
    save_lora_dir="lora_model_medical",
    save_merged_dir="model_medical",  # 可选
    merged_method="merged_16bit",     # 可选：merged_16bit / merged_4bit
)

进阶示例（手动流程）：

config = MedicalFineTuneConfig(
    model_name="/root/autodl-tmp/models/Qwen/Qwen3-4B",
    data_dir="【数据集】中文医疗数据",
    max_steps=20,
    num_train_epochs=1,
)
tuner = MedicalFineTuner(config)
tuner.init_model()             # 加载模型与LoRA
tuner.load_medical_data()      # 读取并清洗数据
tuner.prepare_dataset()        # 应用提示模板，构造text字段
tuner.create_trainer()         # 创建 SFTTrainer
train_stats = tuner.train()    # 运行训练
tuner.save_lora("lora_model_medical_demo")  # 保存LoRA
# 可选：保存合并权重
# tuner.save_merged("model_medical_demo", save_method="merged_16bit")

"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
import torch


@dataclass
class MedicalFineTuneConfig:
    """可配置项，默认值与现有脚本保持一致。

    Attributes:
        model_name: 基础模型目录（本地路径或HF模型名）。
        max_seq_length: 最大序列长度（支持RoPE缩放）。
        dtype: 精度类型；None表示自动（常见为float16/bfloat16）。
        load_in_4bit: 是否使用4bit量化加载以节省显存。
        local_files_only: 仅从本地缓存/目录加载模型，不联网下载。

        lora_r: LoRA秩（推荐：8/16/32/64/128）。
        lora_target_modules: 应用LoRA的模块列表（Q/K/V/O及MLP）。
        lora_alpha: LoRA缩放因子。
        lora_dropout: LoRA dropout；0通常更快。
        lora_bias: LoRA对偏置的处理方式：none/all等。
        use_gradient_checkpointing: 梯度检查点策略（unsloth可显著降显存）。
        random_state: 随机种子，保证可重复性。
        use_rslora: 是否启用Rank-Stabilized LoRA。
        loftq_config: LoftQ配置（量化+LoRA联合），一般为None。

        data_dir: 数据集根目录（包含各科室子目录）。
        departments: 科室目录映射：子目录名 -> 中文名称（用于遍历与日志）。
        max_qa_len: 问/答最大长度（字符数）；超出将被过滤。

        per_device_train_batch_size: 每设备训练批次大小。
        gradient_accumulation_steps: 梯度累积步数（等效增大总batch）。
        warmup_steps: 学习率预热步数。
        max_steps: 最大训练步数（与num_train_epochs配合使用）。
        num_train_epochs: 训练轮数（若设置max_steps，则优先按步数停止）。
        learning_rate: 学习率。
        logging_steps: 日志打印步数间隔。
        optim: 优化器类型（8bit优化器可降显存）。
        weight_decay: 权重衰减。
        lr_scheduler_type: 学习率调度策略。
        seed: 全局随机种子。
        output_dir: 训练输出目录。
        report_to: 训练日志上报后端（none禁用）。
        packing: 是否对短样本进行序列打包以提速。
        dataset_num_proc: 数据集map并行进程数。

        medical_prompt: 提示模板；使用 format(input, output) 生成训练文本。
    """
    # 模型与加载参数
    model_name: str = "/root/autodl-tmp/models/Qwen/Qwen3-4B"  # 基础模型目录（本地路径或HF模型名）
    max_seq_length: int = 2048  # 最大序列长度（支持RoPE缩放）
    dtype: Optional[str] = None  # 精度类型；None表示自动（常见为float16/bfloat16）
    load_in_4bit: bool = True  # 是否使用4bit量化加载以节省显存
    local_files_only: bool = True  # 仅从本地缓存/目录加载模型，不联网下载

    # LoRA参数
    lora_r: int = 16  # LoRA秩（推荐：8/16/32/64/128）
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )  # 需要应用LoRA的模块列表
    lora_alpha: int = 16  # LoRA缩放因子
    lora_dropout: float = 0.0  # LoRA dropout；0通常更快
    lora_bias: str = "none"  # LoRA对偏置的处理方式：none/all等
    use_gradient_checkpointing: str = "unsloth"  # 梯度检查点策略（unsloth可显著降显存）
    random_state: int = 3407  # 随机种子，保证可重复性
    use_rslora: bool = False  # 是否启用Rank-Stabilized LoRA
    loftq_config: Optional[Dict[str, Any]] = None  # LoftQ配置（量化+LoRA联合），一般为None

    # 数据参数
    data_dir: str = "【数据集】中文医疗数据"  # 数据集根目录（包含各科室子目录）
    departments: Dict[str, str] = field(
        default_factory=lambda: {
            "IM_内科": "内科",
            "Surgical_外科": "外科",
            "Pediatric_儿科": "儿科",
            "Oncology_肿瘤科": "肿瘤科",
            "OAGD_妇产科": "妇产科",
            "Andriatria_男科": "男科",
        }
    )  # 科室目录映射：子目录名 -> 中文名称（用于遍历与日志）
    max_qa_len: int = 200  # 问/答最大长度（字符数）；超出将被过滤

    # 训练参数
    per_device_train_batch_size: int = 2  # 每设备训练批次大小
    gradient_accumulation_steps: int = 4  # 梯度累积步数（等效增大总batch）
    warmup_steps: int = 5  # 学习率预热步数
    max_steps: int = 240  # 最大训练步数（与num_train_epochs配合使用）
    num_train_epochs: int = 3  # 训练轮数（若设置max_steps，则优先按步数停止）
    learning_rate: float = 2e-4  # 学习率
    logging_steps: int = 1  # 日志打印步数间隔
    optim: str = "adamw_8bit"  # 优化器类型（8bit优化器可降显存）
    weight_decay: float = 0.01  # 权重衰减
    lr_scheduler_type: str = "linear"  # 学习率调度策略
    seed: int = 3407  # 全局随机种子
    output_dir: str = "outputs"  # 训练输出目录
    report_to: str = "none"  # 训练日志上报后端（none禁用）
    packing: bool = False  # 是否对短样本进行序列打包以提速
    dataset_num_proc: int = 2  # 数据集map并行进程数

    # 提示模板
    medical_prompt: str = (
        "你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。\n\n"
        "### 问题：\n{}\n\n### 回答：\n{}"
    )  # 提示模板；使用 format(input, output) 生成训练文本


class MedicalFineTuner:
    """提供加载数据、初始化模型、训练和保存的封装。"""
    def __init__(self, config: MedicalFineTuneConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        self._eos_token = None

    @staticmethod
    def _read_csv_with_encoding(file_path: str) -> pd.DataFrame:
        """尝试使用多种编码读取CSV，提升兼容性。

        Args:
            file_path: CSV文件的绝对或相对路径。

        Returns:
            读取后的 `pandas.DataFrame`。
        """
        encodings = ["gbk", "gb2312", "gb18030", "utf-8"]
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法使用任何编码读取文件: {file_path}")

    def load_medical_data(self) -> Dataset:
        """遍历科室目录，读取CSV，清洗并构建Dataset。"""
        data: List[Dict[str, str]] = []
        for dept_dir, dept_name in self.config.departments.items():
            dept_path = os.path.join(self.config.data_dir, dept_dir)
            if not os.path.exists(dept_path):
                print(f"目录不存在: {dept_path}")
                continue

            print(f"\n处理{dept_name}数据...")
            csv_files = [f for f in os.listdir(dept_path) if f.endswith(".csv")]

            for csv_file in csv_files:
                file_path = os.path.join(dept_path, csv_file)
                print(f"正在处理文件: {csv_file}")

                try:
                    df = self._read_csv_with_encoding(file_path)
                    print(f"文件 {csv_file} 的列名: {df.columns.tolist()}")

                    for _, row in df.iterrows():
                        try:
                            question = None
                            answer = None

                            if "question" in row:
                                question = str(row["question"]).strip()
                            elif "问题" in row:
                                question = str(row["问题"]).strip()
                            elif "ask" in row:
                                question = str(row["ask"]).strip()

                            if "answer" in row:
                                answer = str(row["answer"]).strip()
                            elif "回答" in row:
                                answer = str(row["回答"]).strip()
                            elif "response" in row:
                                answer = str(row["response"]).strip()

                            if not question or not answer:
                                continue

                            if (
                                len(question) > self.config.max_qa_len
                                or len(answer) > self.config.max_qa_len
                            ):
                                continue

                            data.append(
                                {
                                    "instruction": "请回答以下医疗相关问题",
                                    "input": question,
                                    "output": answer,
                                }
                            )
                        except Exception as e:
                            print(f"处理数据行时出错: {e}")
                            continue
                except Exception as e:
                    print(f"处理文件 {csv_file} 时出错: {e}")
                    continue

        if not data:
            raise ValueError("没有成功处理任何数据！")

        print(f"\n成功处理 {len(data)} 条数据")
        self.dataset = Dataset.from_list(data)
        return self.dataset

    def init_model(self) -> None:
        """加载基础模型与分词器，并应用LoRA配置。

        Raises:
            RuntimeError: 当模型或分词器加载失败时抛出。
        """
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,
            local_files_only=self.config.local_files_only,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=self.config.lora_target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            random_state=self.config.random_state,
            use_rslora=self.config.use_rslora,
            loftq_config=self.config.loftq_config,
        )

        self._eos_token = self.tokenizer.eos_token

    def prepare_dataset(self) -> Dataset:
        """根据提示模板将样本格式化为text字段，供SFT训练。

        Raises:
            RuntimeError: 当 tokenizer 未初始化时抛出（需先调用 init_model）。
        """
        if self.dataset is None:
            self.load_medical_data()

        eos_token = self._eos_token or (self.tokenizer.eos_token if self.tokenizer else None)
        if eos_token is None:
            raise RuntimeError("tokenizer 尚未初始化，请先调用 init_model()。")

        medical_prompt = self.config.medical_prompt

        def formatting_prompts_func(examples):
            instructions = examples["instruction"]
            inputs = examples["input"]
            outputs = examples["output"]
            texts = []
            for instruction, input_text, output_text in zip(instructions, inputs, outputs):
                text = medical_prompt.format(input_text, output_text) + eos_token
                texts.append(text)
            return {"text": texts}

        self.dataset = self.dataset.map(formatting_prompts_func, batched=True)
        return self.dataset

    def create_training_arguments(self) -> TrainingArguments:
        """创建TrainingArguments，默认设置与原脚本一致。

        Returns:
            transformers.TrainingArguments: 由配置构建的训练参数对象（自动切换bf16/fp16）。
        """
        return TrainingArguments(
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=self.config.logging_steps,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            seed=self.config.seed,
            output_dir=self.config.output_dir,
            report_to=self.config.report_to,
        )

    def create_trainer(self) -> SFTTrainer:
        """用模型、tokenizer和数据集构建SFTTrainer。"""
        if self.model is None or self.tokenizer is None:
            self.init_model()

        if self.dataset is None or (
            self.dataset is not None and "text" not in self.dataset.column_names
        ):
            self.prepare_dataset()

        training_args = self.create_training_arguments()

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=self.config.dataset_num_proc,
            packing=self.config.packing,
            args=training_args,
        )
        return self.trainer

    def train(self):
        """运行训练流程并返回统计信息。

        Returns:
            Any: 训练统计信息，具体结构由 `trl.SFTTrainer.train` 决定。
        """
        if self.trainer is None:
            self.create_trainer()
        return self.trainer.train()

    def save_lora(self, output_dir: str = "lora_model_medical") -> None:
        """保存LoRA适配器权重和tokenizer。

        Args:
            output_dir: 输出目录名称或路径，用于保存LoRA权重与tokenizer。
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型尚未初始化，请先调用 init_model() 或 train()。")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def save_merged(self, merged_dir: str = "model_medical", save_method: str = "merged_16bit") -> None:
        """保存合并后的全量权重，支持16bit或4bit。

        Args:
            merged_dir: 合并后模型输出目录。
            save_method: 合并方式，可选 `"merged_16bit"` 或 `"merged_4bit"`。
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型尚未初始化，请先调用 init_model() 或 train()。")
        self.model.save_pretrained_merged(merged_dir, self.tokenizer, save_method=save_method)

    def run(
        self,
        save_lora_dir: Optional[str] = "lora_model_medical",
        save_merged_dir: Optional[str] = None,
        merged_method: str = "merged_16bit",
    ):
        """一键式执行：初始化、数据准备、训练与保存。

        Args:
            save_lora_dir: LoRA权重保存目录；为None则不保存LoRA。
            save_merged_dir: 合并后模型保存目录；为None则不保存合并权重。
            merged_method: 合并权重方式，`merged_16bit` 或 `merged_4bit`。

        Returns:
            Any: 训练统计信息。
        """
        self.init_model()
        self.load_medical_data()
        self.prepare_dataset()
        self.create_trainer()
        stats = self.train()

        if save_lora_dir:
            self.save_lora(save_lora_dir)
        if save_merged_dir:
            self.save_merged(save_merged_dir, merged_method)

        return stats

def demo_basic_usage() -> None:
    """更完整的调用示例：手动控制各步骤与保存。

    使用较小训练步数与epoch方便快速演示；根据需要调参。
    """
    config = MedicalFineTuneConfig(
        model_name=os.environ.get("QWEN_MODEL_PATH", "/root/autodl-tmp/models/Qwen/Qwen3-4B"),
        data_dir=os.environ.get("MEDICAL_DATA_DIR", "【数据集】中文医疗数据"),
        max_steps=20,
        num_train_epochs=1,
        logging_steps=1,
    )
    tuner = MedicalFineTuner(config)
    tuner.init_model()
    tuner.load_medical_data()
    tuner.prepare_dataset()
    tuner.create_trainer()
    stats = tuner.train()
    print("手动流程训练统计：", stats)
    tuner.save_lora("lora_model_medical_demo")
    save_merged_dir = os.environ.get("SAVE_MERGED_DIR")
    if save_merged_dir:
        tuner.save_merged(save_merged_dir, os.environ.get("MERGED_METHOD", "merged_16bit"))
        print(f"合并权重已保存到 `{save_merged_dir}`。")

if __name__ == "__main__":
    # Demo 入口：通过环境变量选择演示流程
    # QWEN_MODEL_PATH: 基础模型目录；MEDICAL_DATA_DIR: 数据集根目录
    # SAVE_MERGED_DIR: 可选，设置后会保存合并权重；MERGED_METHOD: merged_16bit/merged_4bit
    # MEDICAL_DEMO_FLOW: 选择 `basic`（手动流程）或 `oneclick`（一键式），默认 oneclick
    demo_flow = os.environ.get("MEDICAL_DEMO_FLOW", "oneclick").lower()

    if demo_flow == "basic":
        print("运行手动流程 Demo（步数降低，仅供演示）...")
        try:
            demo_basic_usage()
        except Exception as e:
            print("手动流程 Demo 运行失败：", e)
            print("请检查模型路径与数据目录设置，并确认依赖已安装。")
    else:
        print("开始一键式 Demo（步数降低，仅供演示）...")
        config = MedicalFineTuneConfig(
            model_name=os.environ.get("QWEN_MODEL_PATH", "/root/autodl-tmp/models/Qwen/Qwen3-4B"),
            data_dir=os.environ.get("MEDICAL_DATA_DIR", "【数据集】中文医疗数据"),
            max_steps=20,           # 为演示降低训练步数
            num_train_epochs=1,     # 为演示降低epoch数量
            logging_steps=1,
        )
        tuner = MedicalFineTuner(config)
        try:
            stats = tuner.run(
                save_lora_dir="lora_model_medical_demo",
                save_merged_dir=os.environ.get("SAVE_MERGED_DIR") or None,
                merged_method=os.environ.get("MERGED_METHOD", "merged_16bit"),
            )
            print("训练统计：", stats)
            print("LoRA模型已保存到 `lora_model_medical_demo`。")
            if os.environ.get("SAVE_MERGED_DIR"):
                print(f"合并权重已保存到 `{os.environ.get('SAVE_MERGED_DIR')}`。")
        except Exception as e:
            print("一键式 Demo 运行失败：", e)
            print("请检查模型路径与数据目录设置，并确认依赖已安装。")