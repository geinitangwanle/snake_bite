import requests

def deepseek_chat(api_key, message):
    """
    调用DeepSeek API进行对话。

    参数：
        api_key (str): DeepSeek API密钥
        message (str): 要发送的消息

    返回：
        str: API返回的响应消息
    """
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
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
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            json_response = response.json()
            # 根据DeepSeek API格式提取响应
            if "choices" in json_response and len(json_response["choices"]) > 0:
                return json_response["choices"][0]["message"]["content"]
            else:
                return "No response content found in API response."
        else:
            return f"Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.RequestException as e:
        return f"Request Error: {e}"
    except requests.exceptions.JSONDecodeError as e:
        return f"JSON Decode Error: {e}"

# 示例用法
api_key = "sk-53fa7d59f32246f4ab6acc78bd66127e"
message = "Hello, how are you?"
response = deepseek_chat(api_key, message)
print("DeepSeek Response:", response)
