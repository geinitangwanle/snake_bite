# class_config.py
"""
蛇类分类器的类别配置管理
"""

import json
import os

class ClassConfig:
    """类别配置管理器"""
    
    def __init__(self):
        self.config = {
            "snake_classes": {
                "chinese": [
                    "北方水蛇",
                    "普通袜带蛇", 
                    "德凯棕蛇",
                    "黑鼠蛇",
                    "西部菱斑响尾蛇"
                ],
                "english": [
                    "Northern Watersnake",
                    "Common Garter snake",
                    "DeKays Brown snake", 
                    "Black Rat snake",
                    "Western Diamondback rattlesnake"
                ],
                "scientific": [
                    "Nerodia sipedon",
                    "Thamnophis sirtalis",
                    "Storeria dekayi",
                    "Pantherophis obsoletus", 
                    "Crotalus atrox"
                ]
            },
            "default_language": "chinese"
        }
    
    def get_classes(self, language="chinese"):
        """获取指定语言的类别名称"""
        return self.config["snake_classes"].get(language, self.config["snake_classes"]["chinese"])
    
    def save_config(self, filepath="class_config.json"):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
    
    def load_config(self, filepath="class_config.json"):
        """从文件加载配置"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

# 使用示例
if __name__ == "__main__":
    config = ClassConfig()
    
    # 获取中文类别名称
    chinese_classes = config.get_classes("chinese")
    print("中文类别：", chinese_classes)
    
    # 获取英文类别名称
    english_classes = config.get_classes("english")
    print("英文类别：", english_classes)
    
    # 获取学名
    scientific_classes = config.get_classes("scientific")
    print("学名：", scientific_classes)