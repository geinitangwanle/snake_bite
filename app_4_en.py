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

from snake_knowledge_en import get_snake_knowledge, snake_kb
from ai_analyzer_en import ai_analyzer
import plotly.graph_objects as go
import requests
import json

# 全局变量存储预测器和当前识别结果
predictor = None
current_prediction = None
current_knowledge = None
current_location = {"lat": 39.9042, "lon": 116.4074, "address": "Beijing, China"}

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
            text=[f"📍 Current Location<br>Latitude: {lat:.4f}<br>Longitude: {lon:.4f}"],
            hoverinfo="text",
            hovertemplate="<b>%{text}</b><extra></extra>",
            name="Your Location"
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
                text=[f"Location: {lat:.4f}, {lon:.4f}"],
                hoverinfo="text",
                name="Your Location"
            ))
            
            fig.update_layout(
                title="Coordinate Chart (Map unavailable)",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
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
                text=f"Map loading failed<br>Location: {lat:.4f}, {lon:.4f}<br>Error: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                width=500,
                height=400,
                title="Map Display Error"
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
            
            address = f"{city}, {country}" if city and country else "Location detected successfully"
            print(f"✅ 检测到用户位置: {address} ({lat}, {lon})")
            return lat, lon, address
        else:
            print("⚠️ 无法获取IP位置，使用默认位置(北京)")
            return 39.9042, 116.4074, "Beijing, China"
            
    except Exception as e:
        print(f"❌ IP位置检测失败: {e}, 使用默认位置")
        return 39.9042, 116.4074, "Beijing, China"

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
                "❌ Coordinates out of valid range! Latitude: -90 to 90, Longitude: -180 to 180"
            )
        
        # 更新当前位置
        current_location["lat"] = lat
        current_location["lon"] = lon
        current_location["address"] = f"Longitude: {lon:.4f}, Latitude: {lat:.4f}"
        
        # 创建新地图
        new_map = create_interactive_map(lat, lon, zoom=12)
        location_text = f"✅ Location updated: Latitude {lat:.4f}, Longitude {lon:.4f}"
        
        print(f"📍 位置手动更新: {location_text}")
        return new_map, location_text
        
    except ValueError:
        return (
            create_interactive_map(current_location["lat"], current_location["lon"]),
            "❌ Please enter valid numeric coordinates"
        )
    except Exception as e:
        print(f"❌ 位置更新错误: {e}")
        return (
            create_interactive_map(current_location["lat"], current_location["lon"]),
            f"❌ Location update failed: {str(e)}"
        )

def initialize_model():
    """初始化模型"""
    global predictor
    model_path = "snake_classifier_mobilenetv2.pth"
    
    chinese_classes = ['Northern Water Snake', 'Common Garter Snake', 'DeKay\'s Brownsnake', 'Black Rat Snake', 'Western Diamondback Rattlesnake']
    
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
        ax.legend(wedges, labels, title="Snake Species", loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=12, title_fontsize=14)
        
        plt.tight_layout(pad=2.0)
        fig.patch.set_facecolor('white')
        
        print(f"🔧 扇形图创建成功")
        return fig
        
    except Exception as e:
        print(f"❌ 创建扇形图失败：{e}")
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        ax.text(0.5, 0.5, f'Chart generation failed: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        fig.patch.set_facecolor('white')
        return fig

def ai_deep_analysis(user_description):
    """AI深度分析功能"""
    global current_location
    
    print(f"🧠 开始AI深度分析...")
    
    if current_prediction is None:
        return """
❌ Unable to perform deep analysis

Please upload a snake image for identification first, then use the AI deep analysis feature.

Usage Steps:
1. Upload snake image
2. Wait for AI identification to complete
3. Confirm or adjust your location coordinates
4. Describe your situation in detail
5. Click the "Start AI Deep Analysis" button
        """.strip()
    
    if not user_description or not user_description.strip():
        return """
❌ Insufficient description

Please describe your situation in detail in the text box, for example:
- Specific environment where you encountered the snake (garden, forest, waterside, etc.)
- Snake's behavior
- Your concerns and questions
- What type of advice you need

This information will help AI provide more accurate analysis and advice.
        """.strip()
    
    try:
        location_info = ai_analyzer.get_user_location(current_location["address"])
        
        snake_species = current_prediction['class']
        confidence = current_prediction['confidence']
        
        snake_knowledge = current_knowledge or "No detailed knowledge information available"
        
        print(f"🔍 分析参数 - 种类：{snake_species}, 置信度：{confidence}, 位置：{location_info['location']}")
        
        analysis_result = ai_analyzer.analyze_snake_encounter(
            snake_species=snake_species,
            confidence=confidence,
            user_description=user_description,
            location_info=location_info,
            snake_knowledge=snake_knowledge
        )
        
        formatted_result = f"""
🤖 AI Deep Analysis Report

Analysis Subject: {snake_species} (Confidence: {confidence})
User Location: {location_info['location']}
Coordinates: Latitude {current_location['lat']:.4f}, Longitude {current_location['lon']:.4f}
Analysis Time: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

{analysis_result}

---

⚠️ Important Reminder:
This analysis is based on AI models and is for reference only. When encountering snakes, prioritize safety and consult local professionals or relevant authorities when necessary.
        """.strip()
        
        print(f"✅ AI深度分析完成")
        return formatted_result
        
    except Exception as e:
        print(f"❌ AI深度分析错误：{str(e)}")
        import traceback
        traceback.print_exc()
        
        error_message = f"""
❌ Error occurred during analysis

Error information: {str(e)}

Possible solutions:
1. Check if network connection is normal
2. Confirm if API service is available
3. Try the analysis feature again later

If the problem persists, please contact technical support.
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
            "❌ Model not loaded! Please check if model file exists.",
            None,
            "Model not loaded",
            "Model not loaded, unable to get snake knowledge information",
            "Please upload an image for identification first before using AI analysis"
        )
    
    if image is None:
        current_prediction = None
        current_knowledge = None
        return (
            "Please upload a snake image",
            None,
            "Waiting for image upload...",
            "Waiting for image upload to get snake knowledge information...",
            "Please upload an image for identification first before using AI analysis"
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
            error_msg = f"Prediction failed: {result['error']}"
            current_prediction = None
            current_knowledge = None
            return error_msg, None, error_msg, error_msg, "Please successfully identify an image first before using AI analysis"
        
        best_pred = result['top_prediction']
        best_result = f"🏆 Best Prediction: {best_pred['class']} ({best_pred['confidence']})"
        
        current_prediction = best_pred
        
        detailed_text = "🐍 Detailed Prediction Results:\n\n"
        for i, pred in enumerate(result['all_predictions'], 1):
            detailed_text += f"{i}. {pred['class']}: {pred['confidence']}\n"
        
        top_snake_name = best_pred['class']
        snake_knowledge_info = get_snake_knowledge(top_snake_name)
        
        current_knowledge = snake_knowledge_info
        
        if not snake_knowledge_info or "暂无" in snake_knowledge_info:
            snake_knowledge_info = f"""
🐍 {top_snake_name}

📊 AI Prediction Confidence: {best_pred['confidence']}

ℹ️ Basic Information
AI identified this as {top_snake_name} with confidence of {best_pred['confidence']}.

⚠️ Safety Reminder
When encountering any snake species, you should:
- Maintain a safe distance, do not attempt to approach or touch
- Slowly back away, avoid sudden movements
- If bitten, seek medical attention immediately
- Taking photos (from safe distance) helps doctors make judgments

💡 Suggestion
For more information about this snake species, consult local herpetology experts or professional references.
            """.strip()
        
        pie_chart = create_pie_chart(result['all_predictions'])
        
        plt.close('all')
        
        ai_analysis_prompt = f"""
✅ Image identification completed!

Identification result: {best_pred['class']} (Confidence: {best_pred['confidence']})

Now you can:
1. Confirm or adjust your location coordinates on the map
2. Describe the specific situation of encountering the snake and your questions in the "Detailed Description" box
3. Click the "🧠 Start AI Deep Analysis" button to get professional advice

AI will provide targeted safety advice and professional analysis based on the identification results, your location, and description.
        """.strip()
        
        print(f"✅ 预测成功完成")
        return best_result, pie_chart, detailed_text, snake_knowledge_info, ai_analysis_prompt
        
    except Exception as e:
        print(f"❌ 预测错误：{str(e)}")
        import traceback
        traceback.print_exc()
        error_msg = f"Processing error: {str(e)}"
        current_prediction = None
        current_knowledge = None
        return error_msg, None, error_msg, error_msg, "Please successfully identify an image first before using AI analysis"

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
            🐍 Intelligent Snake Identification System
        </h1>
        <h2 style="margin: 0.5rem 0 0 0; font-size: 1.2rem; text-align: center; opacity: 0.9;">
            🧠 AI Deep Analysis + 🗺️ Geolocation
        </h2>
        <p style="margin: 1rem 0 0 0; text-align: center; font-size: 1rem; opacity: 0.8;">
            Upload snake images, AI helps you identify species and provides professional deep analysis advice
        </p>
    </div>
    """
    
    # 模型状态HTML
    if model_loaded:
        status_html = '<div class="status-success">✅ Model loaded successfully, AI analysis function is ready</div>'
    else:
        status_html = '<div class="status-error">❌ Model not loaded, please check model file</div>'
    
    # 使用说明
    usage_info = """
    ### 🚀 Complete Usage Guide
    
    #### 📷 Step 1: Basic Identification Function
    1. Upload Image: Click or drag snake images to the upload area
    2. Auto Identification: AI identification will start automatically after image upload
    3. View Results: Check identification results, probability distribution, and detailed knowledge information on the right
    
    #### 🗺️ Step 2: Precise Location
    1. Auto Location: System will automatically detect your approximate location
    2. Manual Adjustment: Enter precise latitude and longitude coordinates
    3. Location Confirmation: Confirm your location on the interactive map
    
    #### 🧠 Step 3: AI Deep Analysis ⭐ 
    1. Detailed Description: Describe the environment, behavior of the snake you encountered and your questions
    2. Get Analysis: Click the analysis button to get professional advice based on DeepSeek AI
    3. Safety Guidance: Get targeted safety advice and emergency response plans
    
    ### 🐍 Supported Snake Species
    - 🔵 Northern Water Snake - Non-venomous, aquatic environment
    - 🟢 Common Garter Snake - Non-venomous, gentle nature
    - 🟤 DeKay's Brownsnake - Non-venomous, smaller size
    - ⚫ Black Rat Snake - Non-venomous, larger size
    - 🔴 Western Diamondback Rattlesnake - ⚠️ Highly venomous! Extremely dangerous
    
    ### 🎯 AI Analysis Special Features
    - 🌍 Regional Advice: Local relevant advice based on geographic location
    - 🔮 Behavior Prediction: Analysis of possible snake behavior in specific environments
    - ⚡ Safety Assessment: Professional risk assessment and emergency response advice
    - 📚 Ecological Education: Understanding ecological value and conservation significance of snakes
    
    ### ⚠️ Important Safety Reminders
    - 🚨 Identification results are for reference only and cannot replace professional judgment
    - 🛡️ Maintain safe distance when encountering any snake species
    - 🏥 If bitten by a snake, immediately call emergency services and seek medical attention
    - 👨‍⚕️ AI analysis is based on general knowledge, consult local experts for specific situations
    """
    
    # 创建界面
    with gr.Blocks(title="🐍 Snake Identifier + AI Analysis", css=custom_css, theme=gr.themes.Soft()) as demo:
        
        # 标题和状态
        gr.HTML(title_html)
        gr.HTML(status_html)
        
        # 第一部分：图片识别区域
        gr.HTML('<div class="section-header"><h2>📷 Step 1: Upload Image for AI Identification</h2></div>')
        
        with gr.Row(equal_height=True):
            # 左侧：图像上传
            with gr.Column(scale=1):
                with gr.Group():
                    image_input = gr.Image(
                        label="🖼️ Upload Snake Image",
                        type="pil"
                    )
                    
                    gr.HTML("""
                    <div class="info-card">
                        📝 <strong>Supported Formats:</strong> JPG, PNG, JPEG<br>
                        🔄 <strong>Auto Processing:</strong> Images will be automatically identified after upload<br>
                        ⚡ <strong>Processing Speed:</strong> Usually completes identification within seconds
                    </div>
                    """)
            
            # 右侧：识别结果
            with gr.Column(scale=1):
                with gr.Group():
                    best_output = gr.Textbox(
                        label="🏆 Best Prediction Result",
                        lines=3,
                        value="🔄 Waiting for image upload...",
                        interactive=False
                    )
                    
                    pie_chart_output = gr.Plot(
                        label="📊 Confidence Distribution Chart"
                    )
                    
                    detail_output = gr.Textbox(
                        label="📋 Detailed Prediction Results",
                        lines=6,
                        value="🔄 Waiting for image upload...",
                        interactive=False
                    )
        
        # 第二部分：知识库信息
        gr.HTML('<div class="section-header"><h2>📚 Snake Knowledge Archive</h2></div>')
        
        with gr.Row():
            with gr.Column():
                knowledge_output = gr.Textbox(
                    label="📖 Detailed Knowledge Information",
                    lines=12,
                    value="📖 Waiting for image upload to get snake knowledge information...",
                    interactive=False,
                    max_lines=15
                )
        
        # 第三部分：AI深度分析区域
        gr.HTML('<div class="section-header"><h2>🧠 Step 2: AI Deep Analysis Function</h2></div>')
        
        with gr.Row(equal_height=True):
            # 左侧：位置和描述输入
            with gr.Column(scale=1):
                # 地图部分
                with gr.Group():
                    gr.HTML("<h3 style='margin: 0 0 1rem 0; color: #11998e;'>🗺️ Your Location</h3>")
                    
                    # 地图组件 - 使用条件显示
                    if initial_map is not None:
                        map_plot = gr.Plot(
                            value=initial_map,
                            label="📍 Interactive Map"
                        )
                    else:
                        map_plot = gr.Plot(
                            label="📍 Interactive Map (Loading...)"
                        )
                    
                    gr.HTML("""
                    <div class="info-card">
                        🎯 <strong>Location Purpose:</strong> AI will provide local relevant snake distribution information and safety advice based on your location<br>
                        🗺️ <strong>Map Description:</strong> Red marker shows your current location, can be adjusted precisely through coordinate input below
                    </div>
                    """)
                
                # 坐标输入部分
                with gr.Group():
                    gr.HTML("<h4 style='margin: 1rem 0 0.5rem 0; color: #667eea;'>📐 Precise Coordinate Input</h4>")
                    
                    with gr.Row():
                        lat_input = gr.Number(
                            value=default_lat,
                            label="🌐 Latitude",
                            precision=4,
                            info="Range: -90 to 90"
                        )
                        
                        lon_input = gr.Number(
                            value=default_lon, 
                            label="🌐 Longitude",
                            precision=4,
                            info="Range: -180 to 180"
                        )
                    
                    update_location_btn = gr.Button(
                        "📍 Update Map Location",
                        variant="secondary",
                        size="sm"
                    )
                    
                    location_status = gr.Textbox(
                        label="📍 Current Location Status",
                        value=f"📍 Current Location: {default_address}",
                        interactive=False,
                        lines=2
                    )
                
                # 情况描述部分
                with gr.Group():
                    gr.HTML("<h4 style='margin: 1rem 0 0.5rem 0; color: #764ba2;'>📝 Detailed Situation Description</h4>")
                    
                    description_input = gr.Textbox(
                        label="🔍 Detailed Snake Encounter Situation",
                        placeholder="""Please describe in detail the situation when you encountered the snake, for example:

🏞️ Environment Description:
• Where did you encounter it? (garden, forest, waterside, indoors, etc.)
• What was the surrounding environment like? (humid, dry, vegetated, etc.)

🐍 Snake Behavior:
• What was the snake doing at the time? (crawling, stationary, aggressive posture, etc.)
• Size and color characteristics of the snake?

❓ Your Questions:
• What are you most concerned about?
• What type of professional advice do you need?
• Are there any other special circumstances?

💡 The more detailed the description, the more accurate the AI analysis!""",
                        lines=10,
                        info="Detailed descriptions help AI conduct accurate risk assessment and professional advice"
                    )
                    
                    analyze_button = gr.Button(
                        "🧠 Start AI Deep Analysis",
                        variant="primary",
                        size="lg"
                    )
            
            # 右侧：AI分析结果
            with gr.Column(scale=1):
                with gr.Group():
                    ai_analysis_output = gr.Textbox(
                        label="🤖 AI Deep Analysis Report",
                        lines=35,
                        value="🔄 Please upload an image for identification first, then describe the specific situation to get AI deep analysis...",
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
        with gr.Accordion("📖 Detailed Usage Instructions", open=False):
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