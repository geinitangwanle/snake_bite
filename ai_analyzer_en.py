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
                'details': f"User specified location: {location_text}"
            }
        else:
            # 默认提示用户输入位置
            return {
                'location': 'Not specified',
                'source': 'unknown',
                'details': 'Please provide your specific location information in the analysis text for more accurate advice'
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
                    return "❌ No content found in API response"
            else:
                return f"❌ API call error: {response.status_code} - {response.text}"
        
        except requests.exceptions.RequestException as e:
            return f"❌ Network request error: {e}"
        except json.JSONDecodeError as e:
            return f"❌ JSON parsing error: {e}"
        except Exception as e:
            return f"❌ Unknown error: {e}"
    
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
You are a professional herpetologist and wildlife safety consultant. Please provide detailed professional analysis and recommendations based on the following information:

**AI Identification Results:**
- Snake species: {snake_species}
- Identification confidence: {confidence}

**User Location Information:**
- Location: {location_info.get('location', 'Not specified')}
- Details: {location_info.get('details', 'None')}

**User Description and Questions:**
{user_description}

**Snake Basic Knowledge:**
{snake_knowledge}

Please provide professional analysis from the following aspects:

1. **Identification Result Assessment**: Verify the accuracy of AI identification based on user description, point out possible misidentification risks

2. **Regional Relevance Analysis**: Analyze the distribution and commonness of this snake species in the local area based on user location

3. **Immediate Safety Advice**: Emergency response plan for the current situation

4. **Behavior Prediction**: Predict possible snake behavior based on environment and season

5. **Prevention Measures**: Long-term prevention recommendations for this region

6. **Medical Advice**: If it's a venomous snake species, provide specific medical treatment recommendations

7. **Ecological Value**: Explain the ecological role and conservation significance of this snake species

Please respond in English with professional but understandable language, emphasizing safety recommendations. If information is insufficient, please clearly indicate what additional information is needed to provide more accurate advice. Please do not use Markdown formatting, respond with plain text only.
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
A user encountered {snake_species} in {user_location}. Please provide immediate safety advice (within 100 words):
1. Is it venomous?
2. What should they do immediately?
3. What should they NOT do?

Please respond in English and do not use any Markdown formatting symbols.
"""
        return self.deepseek_chat(safety_prompt)

# 创建全局AI分析器实例
ai_analyzer = AIAnalyzer()

# 测试函数
def test_analyzer():
    """测试AI分析器功能"""
    print("🧪 测试AI分析器...")
    
    # 测试基础对话
    response = ai_analyzer.deepseek_chat("Hello, please introduce yourself briefly.")
    print(f"基础对话测试: {response[:100]}...")
    
    # 测试位置获取
    location = ai_analyzer.get_user_location("Beijing, Chaoyang District")
    print(f"位置信息: {location}")
    
    # 测试Markdown清理功能
    test_text = "## This is a title\n\n**This is bold** content\n\n*This is italic*\n\n`code` content\n\n```\ncode block\n```\n\n[link](http://example.com)"
    cleaned = ai_analyzer.clean_markdown_output(test_text)
    print(f"Markdown清理测试:")
    print(f"原文: {test_text}")
    print(f"清理后: {cleaned}")

if __name__ == "__main__":
    test_analyzer() 