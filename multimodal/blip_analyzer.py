"""
BLIP 多模态图像分析模块

使用 Salesforce BLIP 模型对症状相关图片生成文字描述，
描述结果将与用户文本症状融合，用于增强 RAG 检索查询。

懒加载设计：模型在首次调用 analyze_image() 时才加载，
避免应用启动时的性能损耗（模型约 900MB）。
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

_processor = None
_model = None


def _get_model():
    """懒加载单例：首次调用时加载模型，后续复用。"""
    global _processor, _model
    if _model is None:
        print("正在加载 BLIP 图像理解模型（首次加载约需 30s）...")
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _model.eval()
    return _processor, _model


def analyze_image(image_path: str) -> str:
    """
    对给定图片生成自然语言描述。

    Args:
        image_path: 图片文件路径（支持 PNG / JPG / JPEG）

    Returns:
        英文图像描述字符串，例如 "a close-up of red rash on skin"
    """
    processor, model = _get_model()
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption
