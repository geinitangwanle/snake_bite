#!/usr/bin/env python3
"""
AI深度分析模块
集成DeepSeek API，提供基于用户位置和蛇类特征的深度分析
"""

import requests
import json
import os
import re
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class AIAnalyzer:
    """AI深度分析器，使用DeepSeek API进行分析"""
    
    def __init__(self, api_key: str = None):
        """
        初始化AI分析器
        
        参数:
            api_key (str): DeepSeek API密钥
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY', 'sk-53fa7d59f32246f4ab6acc78bd66127e')
        self.base_url = "https://api.deepseek.com/chat/completions"
        print(f"🔑 AI分析器已初始化，API密钥: {self.api_key[:10]}...")
    
    def clean_markdown_output(self, text: str) -> str:
        """
        清理输出文本中的Markdown格式符号
        
        参数:
            text (str): 原始文本
        
        返回:
            str: 清理后的文本
        """
        if not text:
            return text
        
        # 移除井号标题符号 (# ## ### 等)
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        
        # 移除粗体标记 (**)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        
        # 移除斜体标记 (*)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        
        # 移除代码块标记 (```)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # 移除行内代码标记 (`)
        text = re.sub(r'`(.*?)`', r'\1', text)
        
        # 移除链接格式 [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # 移除多余的空行，保持最多两行连续空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 去除行首行尾的空白字符
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        # 移除开头和结尾的空行
        text = text.strip()
        
        return text
    
    def get_user_location(self, location_text: str = None) -> Dict[str, str]:
        """
        获取用户地理位置信息
        
        参数:
            location_text (str): 用户输入的位置信息
        
        返回:
            dict: 包含位置信息的字典
        """
        if location_text:
            # 用户手动输入位置
            return {
                'location': location_text.strip(),
                'source': 'user_input',
                'details': f"用户指定位置: {location_text}"
            }
        else:
            # 默认提示用户输入位置
            return {
                'location': '未指定',
                'source': 'unknown',
                'details': '请在分析文本中提供您的具体位置信息，以获得更准确的建议'
            }
    
    def deepseek_chat(self, message: str) -> str:
        """
        调用DeepSeek API进行对话
        
        参数:
            message (str): 要发送的消息
        
        返回:
            str: API返回的响应消息
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ],
            "stream": False
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            
            if response.status_code == 200:
                json_response = response.json()
                if "choices" in json_response and len(json_response["choices"]) > 0:
                    raw_content = json_response["choices"][0]["message"]["content"]
                    # 清理Markdown格式
                    cleaned_content = self.clean_markdown_output(raw_content)
                    return cleaned_content
                else:
                    return "❌ API响应中没有找到内容"
            else:
                return f"❌ API调用错误: {response.status_code} - {response.text}"
        
        except requests.exceptions.RequestException as e:
            return f"❌ 网络请求错误: {e}"
        except json.JSONDecodeError as e:
            return f"❌ JSON解析错误: {e}"
        except Exception as e:
            return f"❌ 未知错误: {e}"
    
    def analyze_snake_encounter(self, 
                              snake_species: str,
                              confidence: str,
                              user_description: str,
                              location_info: Dict[str, str],
                              snake_knowledge: str) -> str:
        """
        分析蛇类遭遇情况并提供深度建议
        
        参数:
            snake_species (str): 识别的蛇类种类
            confidence (str): 识别置信度
            user_description (str): 用户描述
            location_info (dict): 位置信息
            snake_knowledge (str): 蛇类知识信息
        
        返回:
            str: AI深度分析结果
        """
        
        # 构建分析提示词
        analysis_prompt = f"""
你是一位专业的爬虫学专家和野生动物安全顾问。请基于以下信息提供详细的专业分析和建议：

**AI识别结果：**
- 蛇类种类：{snake_species}
- 识别置信度：{confidence}

**用户位置信息：**
- 位置：{location_info.get('location', '未指定')}
- 详细信息：{location_info.get('details', '无')}

**用户描述和问题：**
{user_description}

**蛇类基础知识：**
{snake_knowledge}

请从以下几个方面提供专业分析：

1. **识别结果评估**：基于用户描述验证AI识别的准确性，指出可能的误判风险

2. **地域相关性分析**：结合用户位置分析该蛇类在当地的分布情况和常见程度

3. **即时安全建议**：针对当前情况的紧急处理方案

4. **行为预测**：基于环境和季节预测蛇的可能行为

5. **预防措施**：针对该地区的长期预防建议

6. **医疗建议**：如果是有毒蛇类，提供具体的医疗处理建议

7. **生态价值**：说明该蛇类的生态作用和保护意义

请用中文回答，语言专业但易懂，重点突出安全性建议。如果信息不足，请明确指出需要补充哪些信息以提供更准确的建议。请不要使用Markdown格式，直接用纯文本回答。
"""
        
        print(f"🤖 正在调用DeepSeek API进行深度分析...")
        return self.deepseek_chat(analysis_prompt)
    
    def quick_safety_check(self, snake_species: str, user_location: str) -> str:
        """
        快速安全检查
        
        参数:
            snake_species (str): 蛇类种类
            user_location (str): 用户位置
        
        返回:
            str: 快速安全建议
        """
        safety_prompt = f"""
用户在{user_location}遇到了{snake_species}，请立即提供简短的安全建议（100字以内）：
1. 是否有毒？
2. 立即应该做什么？
3. 不应该做什么？

请用中文回答，不要使用任何Markdown格式符号。
"""
        return self.deepseek_chat(safety_prompt)

# 创建全局AI分析器实例
ai_analyzer = AIAnalyzer()

# 测试函数
def test_analyzer():
    """测试AI分析器功能"""
    print("🧪 测试AI分析器...")
    
    # 测试基础对话
    response = ai_analyzer.deepseek_chat("你好，请简单介绍一下自己。")
    print(f"基础对话测试: {response[:100]}...")
    
    # 测试位置获取
    location = ai_analyzer.get_user_location("北京市朝阳区")
    print(f"位置信息: {location}")
    
    # 测试Markdown清理功能
    test_text = "## 这是标题\n\n**这是粗体**内容\n\n*这是斜体*\n\n`代码`内容\n\n```\n代码块\n```\n\n[链接](http://example.com)"
    cleaned = ai_analyzer.clean_markdown_output(test_text)
    print(f"Markdown清理测试:")
    print(f"原文: {test_text}")
    print(f"清理后: {cleaned}")

if __name__ == "__main__":
    test_analyzer() 