#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Qwen2-VL 3B 视觉模型微调脚本 - 汽车保险承保专家
基于Unsloth框架进行高效微调
"""

import json
import os
from PIL import Image
from unsloth import FastVisionModel  
import torch

# 支持的4bit预量化模型列表
# fourbit_models = [
#     "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
#     "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit", 
#     "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",
# ]

print("正在加载Qwen2.5-VL-3B模型...")
# 加载预训练模型和分词器
model, tokenizer = FastVisionModel.from_pretrained(
    "/root/autodl-tmp/models/Qwen/Qwen2.5-VL-3B-Instruct",  # 使用Qwen2.5-VL-3B模型
    #load_in_4bit = True,  # 使用4bit量化减少显存使用
    use_gradient_checkpointing = "unsloth",  # 使用梯度检查点节省显存
)

print("配置LoRA参数...")
# 配置LoRA适配器进行参数高效微调
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers = True,  # 微调视觉层
    finetune_language_layers = True,  # 微调语言层  
    finetune_attention_modules = True,  # 微调注意力模块
    finetune_mlp_modules = True,  # 微调MLP模块
    
    r = 16,  # LoRA秩，越大精度越高但可能过拟合
    lora_alpha = 16,  # LoRA alpha参数，建议等于r
    lora_dropout = 0,  # LoRA dropout
    bias = "none",  # 偏置设置
    random_state = 3407,  # 随机种子
    use_rslora = False,  # 是否使用rank stabilized LoRA
    loftq_config = None,  # LoftQ配置
)


# In[2]:


import pandas as pd
import os

print("加载训练数据...")
# 加载Excel格式的训练数据
def load_excel_dataset(file_path):
    """加载Excel格式的数据集"""
    try:
        df = pd.read_excel(file_path)
        print(f"Excel文件列名: {list(df.columns)}")
        print(f"数据集形状: {df.shape}")
        return df
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return None

# 数据转换函数 - 处理Excel格式数据
def convert_excel_to_training_format(df):
    """将Excel格式转换为训练格式，加载本地图片"""
    converted_data = []
    
    for idx, row in df.iterrows():
        # Excel格式: id, prompt, image, response
        image_path = row["image"]
        prompt = row["prompt"]
        response = row["response"]
        
        if pd.notna(image_path) and os.path.exists(image_path):
            try:
                # 加载图片
                image = Image.open(image_path).convert('RGB')
                
                # 创建训练样本
                conversation = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "image": image}
                            ]
                        },
                        {
                            "role": "assistant", 
                            "content": [
                                {"type": "text", "text": response}
                            ]
                        }
                    ]
                }
                converted_data.append(conversation)
                print(f"成功处理样本 {idx + 1}: {image_path}")
            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {e}")
        else:
            print(f"警告：图片文件不存在或路径为空 {image_path}")
    
    return converted_data

# 加载并转换数据集
train_df = load_excel_dataset("qwen-vl-train.xlsx")
if train_df is not None:
    converted_dataset = convert_excel_to_training_format(train_df)
else:
    print("无法加载数据集，程序退出")
    exit()

print(f"成功加载 {len(converted_dataset)} 个训练样本")


# In[3]:


converted_dataset


# In[4]:


# 查看第一个样本的结构
print("第一个训练样本结构：")
print(f"用户消息文本: {converted_dataset[0]['messages'][0]['content'][0]['text']}")
print(f"助手回复: {converted_dataset[0]['messages'][1]['content'][0]['text']}")

# 训练前推理测试
print("\n训练前模型推理测试...")
FastVisionModel.for_inference(model)  # 切换到推理模式

# 加载测试图片
test_image = Image.open("images/1-vehicle-odometer-reading.jpg").convert('RGB')
test_instruction = "你是一名汽车保险承保专家。这里有一张车辆里程表的图片。请从中提取关键信息。"

# 构建推理输入
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": test_instruction}
    ]}
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    test_image,
    input_text, 
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

# 生成回复
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
print("训练前模型输出:")
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                   use_cache=True, temperature=1.5, min_p=0.1)


# In[5]:


# 开始训练
print("\n开始训练模型...")
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# 切换到训练模式
FastVisionModel.for_training(model)

# 配置训练器
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),  # 必须使用视觉数据整理器
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,  # 每设备批次大小
        gradient_accumulation_steps=4,  # 梯度累积步数
        warmup_steps=5,  # 预热步数
        max_steps=30,  # 训练步数，可改为 num_train_epochs=1 进行完整训练
        learning_rate=2e-4,  # 学习率
        logging_steps=1,  # 日志记录步数
        optim="adamw_8bit",  # 使用8bit AdamW优化器
        weight_decay=0.01,  # 权重衰减
        lr_scheduler_type="linear",  # 线性学习率调度
        seed=3407,  # 随机种子
        output_dir="outputs",  # 输出目录
        report_to="none",  # 不使用wandb等记录工具
        
        # 视觉微调必需配置
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    ),
    max_seq_length=2048,  # 序列最大长度参数移到这里
)

# 显示显存使用情况
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. 最大显存 = {max_memory} GB.")
print(f"{start_gpu_memory} GB 显存已使用.")

# 执行训练
trainer_stats = trainer.train()

# 显示训练完成后的显存和时间统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"训练用时: {trainer_stats.metrics['train_runtime']} 秒")
print(f"训练用时: {round(trainer_stats.metrics['train_runtime']/60, 2)} 分钟")
print(f"峰值显存使用: {used_memory} GB")
print(f"LoRA训练显存使用: {used_memory_for_lora} GB")
print(f"显存使用率: {used_percentage}%")
print(f"LoRA显存使用率: {lora_percentage}%")

# 训练后推理测试
print("\n训练后模型推理测试...")
FastVisionModel.for_inference(model)  # 切换到推理模式


# In[6]:


# 使用相同的测试样本
inputs = tokenizer(
    test_image,
    input_text,
    add_special_tokens=False, 
    return_tensors="pt",
).to("cuda")

print("训练后模型输出:")
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                   use_cache=True, temperature=1.5, min_p=0.1)

# 保存模型
print("\n保存LoRA适配器...")
model.save_pretrained("car_insurance_lora_model")  # 本地保存
tokenizer.save_pretrained("car_insurance_lora_model")

print("训练完成！模型已保存到 car_insurance_lora_model 目录")


# In[ ]:


# from unsloth import FastVisionModel
# model, tokenizer = FastVisionModel.from_pretrained(
#     model_name="car_insurance_lora_model",
#     load_in_4bit=True,
# )
# FastVisionModel.for_inference(model)

