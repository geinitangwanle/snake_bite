#!/usr/bin/env python3
"""
Snake Image Classification Prediction Script
Author: PyTorch Expert Assistant
Description: Predict snake species from individual images using the trained model
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import argparse
from snake_config import ClassConfig

class_config = ClassConfig()


class SnakePredictor:
    """简单的蛇类分类预测器。"""
    
    def __init__(self, model_path, class_names=class_config.get_classes("english"), device=None):
        """
        初始化蛇类预测器。
        
        参数：
            model_path (str): 训练好的模型文件路径 (.pth)
            class_names (list): 类别名称列表（可选）
            device (str): 使用的设备（'cuda', 'mps', 'cpu'）
        """
        self.model_path = model_path
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
        self.class_names = class_names or self._get_class_names_from_dataset()
        self.model = self._load_model()
        self.transform = self._get_transform()
        print(f"蛇类预测器已在 {self.device} 上初始化，类别：{self.class_names}")
    
    def _get_class_names_from_dataset(self):
        """从数据集目录获取类别名称或使用默认值。"""
        dataset_dirs = ["dataset/train", "dataset/test", "dataset/validation"]
        for dataset_dir in dataset_dirs:
            if os.path.exists(dataset_dir):
                class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')]
                if class_dirs:
                    return sorted(class_dirs)
        return ['Northern Watersnake', 'Common Garter snake', 'DeKays Brown snake', 'Black Rat snake', 'Western Diamondback rattlesnake']
    
    def _load_model(self):
        """加载训练好的模型。"""
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        model.classifier = nn.Linear(model.classifier[1].in_features, len(self.class_names))
        model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        return model.to(self.device).eval()
    
    def _get_transform(self):
        """获取图像预处理变换。"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_image(self, image_path, top_k=3):
        """
        从图像中预测蛇类种类。
        
        参数：
            image_path (str): 图像文件路径
            top_k (int): 返回的前k个预测结果
            
        返回：
            dict: 包含概率的预测结果
        """
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, top_k)
                
                #确保张量转换为Python数值
                top_probs = top_probs.cpu().squeeze()
                top_indices = top_indices.cpu().squeeze()
                
            return {
                'image_path': image_path,
                'top_prediction': {
                    'class': self.class_names[int(top_indices[0])], 
                    'confidence': f"{float(top_probs[0])*100:.2f}%"
                },
                'all_predictions': [
                    {
                        'class': self.class_names[int(idx)], 
                        'confidence': f"{float(prob)*100:.2f}%"
                    } for idx, prob in zip(top_indices, top_probs)
                ]
            }
        except Exception as e:
            return {'image_path': image_path, 'error': str(e)}

    def predict_batch(self, image_paths, top_k=3):
        """
        为多张图像预测蛇类种类。
        
        参数：
            image_paths (list): 图像文件路径列表
            top_k (int): 返回的前k个预测结果
            
        返回：
            list: 预测结果列表
        """
        return [self.predict_image(image_path, top_k) for image_path in image_paths]


def main():
    """命令行使用的主函数。"""
    parser = argparse.ArgumentParser(description='从图像中预测蛇类种类')
    parser.add_argument('images', nargs='+', help='图像文件路径')
    parser.add_argument('--model', default='snake_classifier_mobilenetv2.pth', help='训练好的模型文件路径')
    parser.add_argument('--top-k', type=int, default=3, help='显示的前k个预测结果')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'], default=None, help='用于预测的设备')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"错误：模型文件 '{args.model}' 未找到！")
        return

    valid_images = [img for img in args.images if os.path.exists(img)]
    if not valid_images:
        print("错误：未找到有效的图像文件！")
        return

    predictor = SnakePredictor(model_path=args.model, device=args.device)
    results = predictor.predict_batch(valid_images, args.top_k)

    for result in results:
        if 'error' in result:
            print(f"处理 {result['image_path']} 时出错：{result['error']}")
        else:
            print(f"图像：{result['image_path']}")
            print(f"最佳预测：{result['top_prediction']['class']} ({result['top_prediction']['confidence']})")
            if args.top_k > 1:
                print("最佳预测：")
                for pred in result['all_predictions']:
                    print(f"  {pred['class']} - {pred['confidence']}")

if __name__ == "__main__":
    main() 
