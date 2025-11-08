"""
VisionInstrumentFineTuneTool

封装 Qwen2.5-VL 视觉模型的仪表盘（里程表等）识别任务微调与推理流程。

特性：
- 模型加载（Unsloth FastVisionModel）与 LoRA 适配器配置
- 从 Excel 加载数据并转换为视觉对话样本（文本 + 图片）
- 构建 SFT 训练器（TRL.SFTTrainer + UnslothVisionDataCollator）
- 推理接口：给定图片与指令生成文本结果
- 权重保存：LoRA 适配器保存，支持可选合并权重保存

快速上手（一键式微调）：

from Qwen2.5-vl-7B-finetuning.vision_instrument_finetune_tool import (
    VisionFineTuneConfig, VisionInstrumentTuner,
)

cfg = VisionFineTuneConfig(
    model_name="/root/autodl-tmp/models/Qwen/Qwen2.5-VL-3B-Instruct",
    excel_path="qwen-vl-train.xlsx",
)
tuner = VisionInstrumentTuner(cfg)
stats = tuner.run(
    save_lora_dir="instrument_lora_model",
    save_merged_dir=None,           # 可选：设置目录以保存合并权重
    merged_method="merged_16bit",  # 可选：merged_16bit / merged_4bit
)

# 推理示例
text = tuner.generate_from_image(
    image_path="images/1-vehicle-odometer-reading.jpg",
    instruction="你是一名汽车保险承保专家。请从图片中提取里程表读数等关键信息。",
)
print(text)

进阶示例（手动流程）：

cfg = VisionFineTuneConfig(
    model_name="/root/autodl-tmp/models/Qwen/Qwen2.5-VL-3B-Instruct",
    excel_path="qwen-vl-train.xlsx",
    max_steps=20,
)
tuner = VisionInstrumentTuner(cfg)
tuner.init_model()              # 加载模型与分词器
tuner.prepare_lora()            # 配置 LoRA 适配器
df = tuner.load_excel_dataset(cfg.excel_path)
dataset = tuner.convert_excel_to_training_format(df)
tuner.create_trainer(dataset)   # 构建训练器
train_stats = tuner.train()     # 训练
tuner.save_lora("instrument_lora_model_demo")
# 可选：tuner.save_merged("instrument_merged_demo", save_method="merged_16bit")

"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import os
import pandas as pd
from PIL import Image
import torch

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer


@dataclass
class VisionFineTuneConfig:
    """视觉仪表盘识别微调配置。

    Attributes:
        model_name: 基础视觉模型路径或HF名称，例如 Qwen2.5-VL-3B-Instruct。
        load_in_4bit: 是否以4bit量化加载，降低显存占用。
        use_gradient_checkpointing: 梯度检查点策略，"unsloth" 可显著节省显存。

        finetune_vision_layers: 是否微调视觉层。
        finetune_language_layers: 是否微调语言层。
        finetune_attention_modules: 是否微调注意力模块。
        finetune_mlp_modules: 是否微调 MLP 模块。

        r: LoRA 秩（数值越大表达能力越强，但可能过拟合）。
        lora_alpha: LoRA alpha 参数，通常与 r 相同。
        lora_dropout: LoRA dropout，0 通常更快。
        bias: LoRA 偏置设置，常用 "none"。
        random_state: 随机种子，保证可重复性。
        use_rslora: 是否使用 Rank-Stabilized LoRA。
        loftq_config: LoftQ 配置（量化 + LoRA 联合），通常为 None。

        excel_path: 训练数据的 Excel 文件路径。
        image_column: Excel 中图片路径列名。
        prompt_column: Excel 中用户指令列名。
        response_column: Excel 中助手回复列名。

        per_device_train_batch_size: 每设备训练批次大小。
        gradient_accumulation_steps: 梯度累积步数（等效增大总 batch）。
        warmup_steps: 预热步数。
        max_steps: 最大训练步数（与 num_train_epochs 二选一）。
        num_train_epochs: 训练轮数（设定后可替代 max_steps）。
        learning_rate: 学习率。
        logging_steps: 日志打印步数间隔。
        optim: 优化器类型，"adamw_8bit" 可显著降低显存。
        weight_decay: 权重衰减。
        lr_scheduler_type: 学习率调度策略。
        seed: 随机种子。
        output_dir: 训练输出目录。
        report_to: 训练日志上报后端（none 禁用）。
        max_seq_length: 最大序列长度（用于 SFTTrainer）。
        remove_unused_columns: 是否移除无用列（视觉任务需 False）。
        dataset_text_field: 数据集文本字段（视觉任务通常为空字符串）。
        dataset_kwargs: 数据集设置，视觉任务需 {"skip_prepare_dataset": True}。

        inference_temperature: 推理采样温度。
        inference_min_p: 推理最小概率阈值（nucleus-like）。
        inference_max_new_tokens: 推理最大生成 token 数。
    """

    # 模型与加载
    model_name: str = "/root/autodl-tmp/models/Qwen/Qwen2.5-VL-3B-Instruct"
    load_in_4bit: bool = False
    use_gradient_checkpointing: str = "unsloth"

    # LoRA 与微调开关
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[Dict[str, Any]] = None

    # 数据集（Excel）
    excel_path: str = "qwen-vl-train.xlsx"
    image_column: str = "image"
    prompt_column: str = "prompt"
    response_column: str = "response"

    # 训练参数
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 30
    num_train_epochs: Optional[int] = None
    learning_rate: float = 2e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs"
    report_to: str = "none"
    max_seq_length: int = 2048
    remove_unused_columns: bool = False
    dataset_text_field: str = ""
    dataset_kwargs: Dict[str, Any] = field(default_factory=lambda: {"skip_prepare_dataset": True})

    # 推理参数
    inference_temperature: float = 1.5
    inference_min_p: float = 0.1
    inference_max_new_tokens: int = 128


class VisionInstrumentTuner:
    """仪表盘识别任务的视觉模型微调与推理封装。"""

    def __init__(self, config: VisionFineTuneConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset: Optional[List[Dict[str, Any]]] = None

    # ============ 模型与 LoRA ============
    def init_model(self) -> None:
        """加载基础视觉模型与分词器。"""
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.config.model_name,
            load_in_4bit=self.config.load_in_4bit,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
        )

    def prepare_lora(self) -> None:
        """配置并注入 LoRA 适配器。"""
        if self.model is None:
            raise RuntimeError("模型未初始化，请先调用 init_model()。")
        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers=self.config.finetune_vision_layers,
            finetune_language_layers=self.config.finetune_language_layers,
            finetune_attention_modules=self.config.finetune_attention_modules,
            finetune_mlp_modules=self.config.finetune_mlp_modules,
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            random_state=self.config.random_state,
            use_rslora=self.config.use_rslora,
            loftq_config=self.config.loftq_config,
        )

    # ============ 数据集处理 ============
    def load_excel_dataset(self, file_path: str) -> pd.DataFrame:
        """加载 Excel 数据集。

        Args:
            file_path: Excel 文件路径。

        Returns:
            加载的 `pandas.DataFrame`。
        """
        try:
            df = pd.read_excel(file_path)
            print(f"Excel文件列名: {list(df.columns)}")
            print(f"数据集形状: {df.shape}")
            return df
        except Exception as e:
            raise RuntimeError(f"读取Excel文件失败: {e}")

    def convert_excel_to_training_format(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """将 Excel 数据转换为训练所需的视觉对话样本。

        每条样本包含一条用户消息（图片 + 文本）和一条助手回复（文本）。

        Args:
            df: 由 `load_excel_dataset` 加载的 DataFrame。

        Returns:
            视觉对话样本列表，供 `SFTTrainer` 使用。
        """
        cfg = self.config
        converted_data: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            image_path = row.get(cfg.image_column, None)
            prompt = row.get(cfg.prompt_column, None)
            response = row.get(cfg.response_column, None)

            if pd.isna(image_path) or not image_path:
                print(f"警告：图片路径为空（样本 {idx}）")
                continue
            if not os.path.exists(str(image_path)):
                print(f"警告：图片文件不存在 {image_path}")
                continue
            if pd.isna(prompt) or pd.isna(response):
                print(f"警告：prompt/response 为空（样本 {idx}）")
                continue

            try:
                image = Image.open(str(image_path)).convert("RGB")
            except Exception as e:
                print(f"处理图片失败 {image_path}: {e}")
                continue

            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": str(prompt)},
                            {"type": "image", "image": image},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": str(response)},
                        ],
                    },
                ]
            }
            converted_data.append(conversation)
            if (idx + 1) % 50 == 0:
                print(f"已转换 {idx + 1} 条样本...")

        print(f"成功转换 {len(converted_data)} 条训练样本")
        self.dataset = converted_data
        return converted_data

    # ============ 训练器与训练 ============
    def create_trainer(self, train_dataset: Optional[List[Dict[str, Any]]] = None) -> SFTTrainer:
        """构建视觉 SFT 训练器。

        Args:
            train_dataset: 视觉对话样本列表；若为 None 将使用 `self.dataset`。

        Returns:
            已构建好的 `trl.SFTTrainer`。
        """
        if self.model is None or self.tokenizer is None:
            self.init_model()
        if train_dataset is None:
            if self.dataset is None:
                raise RuntimeError("数据集为空，请先转换 Excel 数据为训练格式。")
            train_dataset = self.dataset

        FastVisionModel.for_training(self.model)
        args = SFTConfig(
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            seed=self.config.seed,
            output_dir=self.config.output_dir,
            report_to=self.config.report_to,
            # 视觉微调必需配置
            remove_unused_columns=self.config.remove_unused_columns,
            dataset_text_field=self.config.dataset_text_field,
            dataset_kwargs=self.config.dataset_kwargs,
        )
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=train_dataset,
            args=args,
            max_seq_length=self.config.max_seq_length,
        )
        return self.trainer

    def train(self):
        """运行训练并返回统计信息。"""
        if self.trainer is None:
            raise RuntimeError("训练器为空，请先调用 create_trainer()。")
        stats = self.trainer.train()
        return stats

    # ============ 推理 ============
    def generate_from_image(self, image_path: str, instruction: str, stream: bool = False) -> str:
        """对给定图片与指令进行推理生成文本。

        Args:
            image_path: 图片绝对或相对路径。
            instruction: 文本指令，例如“请识别里程表读数”。
            stream: 是否使用 `TextStreamer` 流式打印生成文本。

        Returns:
            生成的文本内容；当 `stream=True` 时返回空字符串（结果已打印）。
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未初始化，请先调用 init_model() 或完成训练。")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"加载图片失败 {image_path}: {e}")

        FastVisionModel.for_inference(self.model)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ]}
        ]
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = self.tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)

        if stream:
            text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            _ = self.model.generate(
                **inputs,
                streamer=text_streamer,
                max_new_tokens=self.config.inference_max_new_tokens,
                use_cache=True,
                temperature=self.config.inference_temperature,
                min_p=self.config.inference_min_p,
            )
            return ""
        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.inference_max_new_tokens,
                use_cache=True,
                temperature=self.config.inference_temperature,
                min_p=self.config.inference_min_p,
            )
            text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return text

    # ============ 权重保存 ============
    def save_lora(self, out_dir: str = "instrument_lora_model") -> None:
        """保存 LoRA 适配器权重与 tokenizer。"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未初始化，请先调用 init_model() 或训练完成后再保存。")
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)

    def save_merged(self, out_dir: str = "instrument_merged_model", save_method: str = "merged_16bit") -> None:
        """保存合并后的全量权重，支持 16bit/4bit（如模型支持）。"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未初始化，请先调用 init_model() 或训练完成后再保存。")
        # 某些视觉模型可能不支持合并导出；若报错，请使用 save_lora。
        try:
            self.model.save_pretrained_merged(out_dir, self.tokenizer, save_method=save_method)
        except Exception as e:
            raise RuntimeError(f"合并权重保存失败：{e}")

    # ============ 一键式流程 ============
    def run(
        self,
        save_lora_dir: Optional[str] = "instrument_lora_model",
        save_merged_dir: Optional[str] = None,
        merged_method: str = "merged_16bit",
    ) -> Any:
        """一键执行：模型初始化、LoRA配置、数据处理、训练与保存。"""
        self.init_model()
        self.prepare_lora()
        df = self.load_excel_dataset(self.config.excel_path)
        dataset = self.convert_excel_to_training_format(df)
        self.create_trainer(dataset)
        stats = self.train()

        if save_lora_dir:
            self.save_lora(save_lora_dir)
        if save_merged_dir:
            self.save_merged(save_merged_dir, merged_method)
        return stats


def demo_manual() -> None:
    """手动流程 Demo：便于逐步调试每个环节。"""
    cfg = VisionFineTuneConfig(
        model_name=os.environ.get("QWEN_VL_MODEL_PATH", "/root/autodl-tmp/models/Qwen/Qwen2.5-VL-3B-Instruct"),
        excel_path=os.environ.get("VL_EXCEL_PATH", "qwen-vl-train.xlsx"),
        max_steps=20,
        logging_steps=1,
    )
    tuner = VisionInstrumentTuner(cfg)
    tuner.init_model()
    tuner.prepare_lora()
    df = tuner.load_excel_dataset(cfg.excel_path)
    dataset = tuner.convert_excel_to_training_format(df)
    tuner.create_trainer(dataset)
    stats = tuner.train()
    print("手动流程训练统计：", stats)
    tuner.save_lora("instrument_lora_model_demo")

    test_image = os.environ.get("VL_TEST_IMAGE_PATH", "images/1-vehicle-odometer-reading.jpg")
    test_instruction = os.environ.get("VL_TEST_INSTRUCTION", "你是一名汽车保险承保专家。请从图片中提取里程表读数等关键信息。")
    print("推理结果（手动流程）：")
    print(tuner.generate_from_image(test_image, test_instruction)[:1000])


def demo_oneclick() -> None:
    """一键式 Demo：简化调用与环境变量控制。"""
    cfg = VisionFineTuneConfig(
        model_name=os.environ.get("QWEN_VL_MODEL_PATH", "/root/autodl-tmp/models/Qwen/Qwen2.5-VL-3B-Instruct"),
        excel_path=os.environ.get("VL_EXCEL_PATH", "qwen-vl-train.xlsx"),
        max_steps=20,
        logging_steps=1,
    )
    tuner = VisionInstrumentTuner(cfg)
    print("开始一键式 Demo（步数降低，仅供演示）...")
    stats = tuner.run(
        save_lora_dir=os.environ.get("VL_SAVE_LORA_DIR", "instrument_lora_model_demo"),
        save_merged_dir=os.environ.get("VL_SAVE_MERGED_DIR") or None,
        merged_method=os.environ.get("VL_MERGED_METHOD", "merged_16bit"),
    )
    print("训练统计：", stats)

    test_image = os.environ.get("VL_TEST_IMAGE_PATH", "images/1-vehicle-odometer-reading.jpg")
    test_instruction = os.environ.get("VL_TEST_INSTRUCTION", "你是一名汽车保险承保专家。请从图片中提取里程表读数等关键信息。")
    print("推理结果（一键式）：")
    print(tuner.generate_from_image(test_image, test_instruction)[:1000])


if __name__ == "__main__":
    # 通过环境变量选择 Demo 流程：VL_DEMO_FLOW=manual 或 oneclick（默认 oneclick）
    demo_flow = os.environ.get("VL_DEMO_FLOW", "oneclick").lower()
    try:
        if demo_flow == "manual":
            demo_manual()
        else:
            demo_oneclick()
    except Exception as e:
        print("Demo 运行失败：", e)
        print("请检查模型路径、Excel 数据与图片路径设置，并确认依赖已安装。")