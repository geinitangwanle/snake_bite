#!/usr/bin/env python3
"""
Snake Image Classification GUI Application with Fixed Map Display
Author: PyTorch Expert Assistant  
Description: Gradio-based web interface for snake species prediction with working interactive map
"""
# 美化了界面
import os
import gradio as gr
from predict import SnakePredictor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from snake_knowledge import get_snake_knowledge, snake_kb
from ai_analyzer import ai_analyzer
import plotly.graph_objects as go
import requests
import json

# 全局变量存储预测器和当前识别结果
predictor = None
current_prediction = None
current_knowledge = None
current_location = {"lat": 39.9042, "lon": 116.4074, "address": "北京市"}

def create_interactive_map(lat=39.9042, lon=116.4074, zoom=10):
    """创建交互式地图 - 完全修复版本"""
    try:
        print(f"🗺️ 创建地图: 纬度={lat}, 经度={lon}, 缩放={zoom}")
        
        # 使用 Scattermapbox (更稳定的版本)
        fig = go.Figure()
        
        # 添加标记点
        fig.add_trace(go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode='markers',
            marker=dict(
                size=14,
                color='red',
                symbol='circle'
            ),
            text=[f"📍 当前位置<br>纬度: {lat:.4f}<br>经度: {lon:.4f}"],
            hoverinfo="text",
            hovertemplate="<b>%{text}</b><extra></extra>",
            name="您的位置"
        ))
        
        # 更新布局
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                bearing=0,
                center=dict(lat=lat, lon=lon),
                pitch=0,
                zoom=zoom
            ),
            hovermode='closest',
            margin=dict(r=0, t=0, l=0, b=0),
            width=500,
            height=400,
            showlegend=False
        )
        
        print(f"✅ 地图创建成功")
        return fig
        
    except Exception as e:
        print(f"❌ 创建地图失败: {e}")
        
        # 创建一个替代的简单图表
        try:
            fig = go.Figure()
            
            # 添加一个简单的散点图作为地图的替代
            fig.add_trace(go.Scatter(
                x=[lon],
                y=[lat],
                mode='markers',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='circle'
                ),
                text=[f"位置: {lat:.4f}, {lon:.4f}"],
                hoverinfo="text",
                name="您的位置"
            ))
            
            fig.update_layout(
                title="位置坐标图 (地图功能暂不可用)",
                xaxis_title="经度",
                yaxis_title="纬度",
                width=500,
                height=400,
                showlegend=False
            )
            
            print(f"✅ 使用替代坐标图")
            return fig
            
        except Exception as e2:
            print(f"❌ 创建替代图表也失败: {e2}")
            
            # 最后的备选方案：空图表
            fig = go.Figure()
            fig.add_annotation(
                text=f"地图加载失败<br>位置: {lat:.4f}, {lon:.4f}<br>错误: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                width=500,
                height=400,
                title="地图显示错误"
            )
            return fig

def get_location_from_ip():
    """尝试通过IP获取用户大概位置"""
    try:
        print("🌍 正在获取IP位置...")
        response = requests.get('http://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            lat = data.get('latitude', 39.9042)
            lon = data.get('longitude', 116.4074)
            city = data.get('city', '')
            country = data.get('country_name', '')
            
            address = f"{city}, {country}" if city and country else "位置检测成功"
            print(f"✅ 检测到用户位置: {address} ({lat}, {lon})")
            return lat, lon, address
        else:
            print("⚠️ 无法获取IP位置，使用默认位置(北京)")
            return 39.9042, 116.4074, "北京市"
            
    except Exception as e:
        print(f"❌ IP位置检测失败: {e}, 使用默认位置")
        return 39.9042, 116.4074, "北京市"

def update_location_manually(lat_input, lon_input):
    """手动更新位置"""
    global current_location
    
    try:
        lat = float(lat_input) if lat_input else current_location["lat"]
        lon = float(lon_input) if lon_input else current_location["lon"]
        
        # 验证坐标范围
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return (
                create_interactive_map(current_location["lat"], current_location["lon"]), 
                "❌ 坐标超出有效范围！纬度: -90到90, 经度: -180到180"
            )
        
        # 更新当前位置
        current_location["lat"] = lat
        current_location["lon"] = lon
        current_location["address"] = f"经度: {lon:.4f}, 纬度: {lat:.4f}"
        
        # 创建新地图
        new_map = create_interactive_map(lat, lon, zoom=12)
        location_text = f"✅ 位置已更新: 纬度 {lat:.4f}, 经度 {lon:.4f}"
        
        print(f"📍 位置手动更新: {location_text}")
        return new_map, location_text
        
    except ValueError:
        return (
            create_interactive_map(current_location["lat"], current_location["lon"]),
            "❌ 请输入有效的数字坐标"
        )
    except Exception as e:
        print(f"❌ 位置更新错误: {e}")
        return (
            create_interactive_map(current_location["lat"], current_location["lon"]),
            f"❌ 位置更新失败: {str(e)}"
        )

def initialize_model():
    """初始化模型"""
    global predictor
    model_path = "snake_classifier_mobilenetv2.pth"
    
    chinese_classes = ['北方水蛇', '普通袜带蛇', '德凯棕蛇', '黑鼠蛇', '西部菱斑响尾蛇']
    
    try:
        if os.path.exists(model_path):
            predictor = SnakePredictor(
                model_path=model_path,
                class_names=chinese_classes
            )
            print(f"✅ 模型加载成功：{model_path}")
            print(f"📋 类别：{chinese_classes}")
            return True
        else:
            print(f"⚠️ 警告：模型文件 '{model_path}' 未找到！")
            return False
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return False

def create_pie_chart(predictions):
    """创建扇形图"""
    try:
        labels = []
        sizes = []
        
        for pred in predictions:
            labels.append(pred['class'])
            confidence = float(pred['confidence'].replace('%', ''))
            sizes.append(confidence)
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        wedges, texts, = ax.pie(
            sizes, 
            labels=None,
            colors=colors[:len(labels)],
            autopct=None,
            startangle=90,
        )
        
        ax.axis('equal')
        ax.legend(wedges, labels, title="蛇类种类", loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=12, title_fontsize=14)
        
        plt.tight_layout(pad=2.0)
        fig.patch.set_facecolor('white')
        
        print(f"🔧 扇形图创建成功")
        return fig
        
    except Exception as e:
        print(f"❌ 创建扇形图失败：{e}")
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        ax.text(0.5, 0.5, f'图表生成失败：{str(e)}', ha='center', va='center', transform=ax.transAxes)
        fig.patch.set_facecolor('white')
        return fig

def ai_deep_analysis(user_description):
    """AI深度分析功能"""
    global current_location
    
    print(f"🧠 开始AI深度分析...")
    
    if current_prediction is None:
        return """
❌ 无法进行深度分析

请先上传蛇类图片进行识别，然后再使用AI深度分析功能。

使用步骤：
1. 上传蛇类图片
2. 等待AI识别完成
3. 确认或调整您的位置坐标
4. 详细描述您的情况
5. 点击"开始AI深度分析"按钮
        """.strip()
    
    if not user_description or not user_description.strip():
        return """
❌ 描述信息不足

请在文本框中详细描述您的情况，例如：
- 遇到蛇的具体环境（花园、森林、水边等）
- 蛇的行为表现
- 您的担心和疑问
- 需要哪方面的建议

这些信息将帮助AI提供更准确的分析建议。
        """.strip()
    
    try:
        location_info = ai_analyzer.get_user_location(current_location["address"])
        
        snake_species = current_prediction['class']
        confidence = current_prediction['confidence']
        
        snake_knowledge = current_knowledge or "暂无详细知识信息"
        
        print(f"🔍 分析参数 - 种类：{snake_species}, 置信度：{confidence}, 位置：{location_info['location']}")
        
        analysis_result = ai_analyzer.analyze_snake_encounter(
            snake_species=snake_species,
            confidence=confidence,
            user_description=user_description,
            location_info=location_info,
            snake_knowledge=snake_knowledge
        )
        
        formatted_result = f"""
🤖 AI深度分析报告

分析对象：{snake_species} (置信度: {confidence})
用户位置：{location_info['location']}
坐标：纬度 {current_location['lat']:.4f}, 经度 {current_location['lon']:.4f}
分析时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

{analysis_result}

---

⚠️ 重要提醒：
本分析基于AI模型，仅供参考。遇到蛇类时请优先考虑安全，必要时请咨询当地专业人士或相关部门。
        """.strip()
        
        print(f"✅ AI深度分析完成")
        return formatted_result
        
    except Exception as e:
        print(f"❌ AI深度分析错误：{str(e)}")
        import traceback
        traceback.print_exc()
        
        error_message = f"""
❌ 分析过程中出现错误

错误信息：{str(e)}

可能的解决方案：
1. 检查网络连接是否正常
2. 确认API服务是否可用
3. 稍后重试分析功能

如果问题持续存在，请联系技术支持。
        """.strip()
        
        return error_message

def predict_snake(image):
    """预测蛇类种类"""
    global current_prediction, current_knowledge
    
    print(f"🔍 开始预测，收到图像：{type(image)}")
    
    if predictor is None:
        current_prediction = None
        current_knowledge = None
        return (
            "❌ 模型未加载！请检查模型文件是否存在。",
            None,
            "模型未加载",
            "模型未加载，无法获取蛇类知识信息",
            "请先上传图片进行识别后再使用AI分析功能"
        )
    
    if image is None:
        current_prediction = None
        current_knowledge = None
        return (
            "请上传一张蛇类图片",
            None,
            "等待上传图片...",
            "等待上传图片以获取蛇类知识信息...",
            "请先上传图片进行识别后再使用AI分析功能"
        )
    
    try:
        print(f"📐 图像尺寸：{image.size}")
        
        temp_path = "temp_prediction.jpg"
        image.save(temp_path)
        print(f"💾 图像已保存：{temp_path}")
        
        result = predictor.predict_image(temp_path, top_k=5)
        print(f"🤖 预测完成")
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if 'error' in result:
            error_msg = f"预测失败：{result['error']}"
            current_prediction = None
            current_knowledge = None
            return error_msg, None, error_msg, error_msg, "请先成功识别图片后再使用AI分析功能"
        
        best_pred = result['top_prediction']
        best_result = f"🏆 最佳预测：{best_pred['class']} ({best_pred['confidence']})"
        
        current_prediction = best_pred
        
        detailed_text = "🐍 详细预测结果：\n\n"
        for i, pred in enumerate(result['all_predictions'], 1):
            detailed_text += f"{i}. {pred['class']}: {pred['confidence']}\n"
        
        top_snake_name = best_pred['class']
        snake_knowledge_info = get_snake_knowledge(top_snake_name)
        
        current_knowledge = snake_knowledge_info
        
        if not snake_knowledge_info or "暂无" in snake_knowledge_info:
            snake_knowledge_info = f"""
🐍 {top_snake_name}

📊 AI预测置信度：{best_pred['confidence']}

ℹ️ 基本信息
AI识别出这是 {top_snake_name}，置信度为 {best_pred['confidence']}。

⚠️ 安全提醒
无论遇到任何蛇类，都应该：
- 保持安全距离，不要试图接近或触摸
- 慢慢后退，避免突然动作
- 如被咬伤，立即就医
- 拍照记录（安全距离）有助于医生判断

💡 建议
如需更多关于此蛇类的信息，建议咨询当地爬虫学专家或查阅专业资料。
            """.strip()
        
        pie_chart = create_pie_chart(result['all_predictions'])
        
        plt.close('all')
        
        ai_analysis_prompt = f"""
✅ 图片识别完成！

识别结果：{best_pred['class']} (置信度: {best_pred['confidence']})

现在您可以：
1. 在地图上确认或调整您的位置坐标
2. 在"详细描述"框中描述遇到蛇的具体情况和您的疑问
3. 点击"🧠 开始AI深度分析"按钮获得专业建议

AI将基于识别结果、您的位置和描述提供针对性的安全建议和专业分析。
        """.strip()
        
        print(f"✅ 预测成功完成")
        return best_result, pie_chart, detailed_text, snake_knowledge_info, ai_analysis_prompt
        
    except Exception as e:
        print(f"❌ 预测错误：{str(e)}")
        import traceback
        traceback.print_exc()
        error_msg = f"处理错误：{str(e)}"
        current_prediction = None
        current_knowledge = None
        return error_msg, None, error_msg, error_msg, "请先成功识别图片后再使用AI分析功能"

# 初始化模型
print("🔄 正在初始化模型...")
model_loaded = initialize_model()

# 创建界面
def create_interface():
    global current_location
    
    # 获取用户当前位置作为地图初始位置
    default_lat, default_lon, default_address = get_location_from_ip()
    current_location = {"lat": default_lat, "lon": default_lon, "address": default_address}
    
    print(f"🗺️ 准备创建初始地图: {default_lat}, {default_lon}")
    
    # 创建初始地图 - 延迟创建以避免初始化问题
    try:
        initial_map = create_interactive_map(default_lat, default_lon)
        print("✅ 初始地图创建成功")
    except Exception as e:
        print(f"❌ 初始地图创建失败: {e}")
        initial_map = None
    
    # 自定义CSS样式
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    
    .header-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .status-success {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .status-error {
        background: linear-gradient(135deg, #f44336, #da190b);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
    }
    
    .section-header {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 6px 20px rgba(17, 153, 142, 0.3);
    }
    
    .info-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #333;
        font-size: 0.9rem;
    }
    """
    
    # 标题HTML
    title_html = """
    <div class="header-section">
        <h1 style="margin: 0; font-size: 2.5rem; text-align: center;">
            🐍 智能蛇类识别系统
        </h1>
        <h2 style="margin: 0.5rem 0 0 0; font-size: 1.2rem; text-align: center; opacity: 0.9;">
            🧠 AI深度分析 + 🗺️ 地理定位
        </h2>
        <p style="margin: 1rem 0 0 0; text-align: center; font-size: 1rem; opacity: 0.8;">
            上传蛇类图片，AI帮您识别种类并提供专业的深度分析建议
        </p>
    </div>
    """
    
    # 模型状态HTML
    if model_loaded:
        status_html = '<div class="status-success">✅ 模型已成功加载，AI分析功能已就绪</div>'
    else:
        status_html = '<div class="status-error">❌ 模型未加载，请检查模型文件</div>'
    
    # 使用说明
    usage_info = """
    ### 🚀 完整使用指南
    
    #### 📷 第一步：基础识别功能
    1. 上传图片：点击或拖拽蛇类图片到上传区域
    2. 自动识别：图片上传后会自动开始AI识别
    3. 查看结果：在右侧查看识别结果、概率分布和详细知识信息
    
    #### 🗺️ 第二步：精确定位
    1. 自动定位：系统会自动检测您的大概位置
    2. 手动调整：输入精确的纬度和经度坐标
    3. 位置确认：在交互式地图上确认您的位置
    
    #### 🧠 第三步：AI深度分析 ⭐ 
    1. 详细描述：描述遇到蛇的环境、行为和您的疑问
    2. 获取分析：点击分析按钮，获得基于DeepSeek AI的专业建议
    3. 安全指导：获得针对性的安全建议和应急处理方案
    
    ### 🐍 支持识别的蛇类
    - 🔵 北方水蛇 - 无毒，水生环境
    - 🟢 普通袜带蛇 - 无毒，性格温和
    - 🟤 德凯棕蛇 - 无毒，体型较小
    - ⚫ 黑鼠蛇 - 无毒，体型较大
    - 🔴 西部菱斑响尾蛇 - ⚠️ 剧毒！极度危险
    
    ### 🎯 AI分析特色功能
    - 🌍 地域性建议：基于地理位置的当地相关建议
    - 🔮 行为预测：分析蛇类在特定环境下的可能行为
    - ⚡ 安全评估：专业的风险评估和应急处理建议
    - 📚 生态教育：了解蛇类的生态价值和保护意义
    
    ### ⚠️ 重要安全提醒
    - 🚨 识别结果仅供参考，不能替代专业判断
    - 🛡️ 遇到任何蛇类都应保持安全距离
    - 🏥 如被蛇咬伤，立即拨打急救电话并就医
    - 👨‍⚕️ AI分析基于通用知识，具体情况请咨询当地专家
    """
    
    # 创建界面
    with gr.Blocks(title="🐍 蛇类识别器 + AI分析", css=custom_css, theme=gr.themes.Soft()) as demo:
        
        # 标题和状态
        gr.HTML(title_html)
        gr.HTML(status_html)
        
        # 第一部分：图片识别区域
        gr.HTML('<div class="section-header"><h2>📷 第一步：上传图片进行AI识别</h2></div>')
        
        with gr.Row(equal_height=True):
            # 左侧：图像上传
            with gr.Column(scale=1):
                with gr.Group():
                    image_input = gr.Image(
                        label="🖼️ 上传蛇类图片",
                        type="pil"
                    )
                    
                    gr.HTML("""
                    <div class="info-card">
                        📝 <strong>支持格式：</strong>JPG, PNG, JPEG<br>
                        🔄 <strong>自动处理：</strong>图片会在上传后自动识别<br>
                        ⚡ <strong>处理速度：</strong>通常在几秒内完成识别
                    </div>
                    """)
            
            # 右侧：识别结果
            with gr.Column(scale=1):
                with gr.Group():
                    best_output = gr.Textbox(
                        label="🏆 最佳预测结果",
                        lines=3,
                        value="🔄 等待上传图片...",
                        interactive=False
                    )
                    
                    pie_chart_output = gr.Plot(
                        label="📊 置信度分布图"
                    )
                    
                    detail_output = gr.Textbox(
                        label="📋 详细预测结果",
                        lines=6,
                        value="🔄 等待上传图片...",
                        interactive=False
                    )
        
        # 第二部分：知识库信息
        gr.HTML('<div class="section-header"><h2>📚 蛇类知识档案</h2></div>')
        
        with gr.Row():
            with gr.Column():
                knowledge_output = gr.Textbox(
                    label="📖 详细知识信息",
                    lines=12,
                    value="📖 等待上传图片以获取蛇类知识信息...",
                    interactive=False,
                    max_lines=15
                )
        
        # 第三部分：AI深度分析区域
        gr.HTML('<div class="section-header"><h2>🧠 第二步：AI深度分析功能</h2></div>')
        
        with gr.Row(equal_height=True):
            # 左侧：位置和描述输入
            with gr.Column(scale=1):
                # 地图部分
                with gr.Group():
                    gr.HTML("<h3 style='margin: 0 0 1rem 0; color: #11998e;'>🗺️ 您的位置</h3>")
                    
                    # 地图组件 - 使用条件显示
                    if initial_map is not None:
                        map_plot = gr.Plot(
                            value=initial_map,
                            label="📍 交互式地图"
                        )
                    else:
                        map_plot = gr.Plot(
                            label="📍 交互式地图 (加载中...)"
                        )
                    
                    gr.HTML("""
                    <div class="info-card">
                        🎯 <strong>位置作用：</strong>AI会基于您的位置提供当地相关的蛇类分布信息和安全建议<br>
                        🗺️ <strong>地图说明：</strong>红色标记显示您的当前位置，可通过下方坐标输入精确调整
                    </div>
                    """)
                
                # 坐标输入部分
                with gr.Group():
                    gr.HTML("<h4 style='margin: 1rem 0 0.5rem 0; color: #667eea;'>📐 精确坐标输入</h4>")
                    
                    with gr.Row():
                        lat_input = gr.Number(
                            value=default_lat,
                            label="🌐 纬度 (Latitude)",
                            precision=4,
                            info="范围: -90 到 90"
                        )
                        
                        lon_input = gr.Number(
                            value=default_lon, 
                            label="🌐 经度 (Longitude)",
                            precision=4,
                            info="范围: -180 到 180"
                        )
                    
                    update_location_btn = gr.Button(
                        "📍 更新地图位置",
                        variant="secondary",
                        size="sm"
                    )
                    
                    location_status = gr.Textbox(
                        label="📍 当前位置状态",
                        value=f"📍 当前位置: {default_address}",
                        interactive=False,
                        lines=2
                    )
                
                # 情况描述部分
                with gr.Group():
                    gr.HTML("<h4 style='margin: 1rem 0 0.5rem 0; color: #764ba2;'>📝 详细情况描述</h4>")
                    
                    description_input = gr.Textbox(
                        label="🔍 遇到蛇的详细情况",
                        placeholder="""请详细描述您遇到蛇的情况，例如：

🏞️ 环境描述：
• 在什么地方遇到的？（花园、森林、水边、室内等）
• 周围环境如何？（湿润、干燥、有植被等）

🐍 蛇的行为：
• 蛇当时在做什么？（爬行、静止、攻击姿态等）
• 蛇的大小和颜色特征？

❓ 您的疑问：
• 最担心什么问题？
• 需要哪方面的专业建议？
• 是否有其他特殊情况？

💡 描述越详细，AI分析越准确！""",
                        lines=10,
                        info="详细的描述有助于AI进行准确的风险评估和专业建议"
                    )
                    
                    analyze_button = gr.Button(
                        "🧠 开始AI深度分析",
                        variant="primary",
                        size="lg"
                    )
            
            # 右侧：AI分析结果
            with gr.Column(scale=1):
                with gr.Group():
                    ai_analysis_output = gr.Textbox(
                        label="🤖 AI深度分析报告",
                        lines=35,
                        value="🔄 请先上传图片进行识别，然后描述具体情况以获取AI深度分析...",
                        interactive=False,
                        max_lines=40
                    )
        
        # 事件绑定
        image_input.change(
            fn=predict_snake,
            inputs=image_input,
            outputs=[best_output, pie_chart_output, detail_output, knowledge_output, ai_analysis_output]
        )
        
        update_location_btn.click(
            fn=update_location_manually,
            inputs=[lat_input, lon_input],
            outputs=[map_plot, location_status]
        )
        
        analyze_button.click(
            fn=ai_deep_analysis,
            inputs=[description_input],
            outputs=ai_analysis_output
        )
        
        # 使用说明
        with gr.Accordion("📖 详细使用说明", open=False):
            gr.Markdown(usage_info)
    
    return demo

# 创建并启动界面
demo = create_interface()

if __name__ == "__main__":
    print("🚀 启动修复版蛇类识别系统...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True
    )