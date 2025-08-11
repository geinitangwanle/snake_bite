#!/usr/bin/env python3
"""
Snake Image Classification GUI Application
Author: PyTorch Expert Assistant
Description: Gradio-based web interface for snake species prediction
"""
# 添加了AI深度分析功能
import os
import gradio as gr
from predict import SnakePredictor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from snake_knowledge import get_snake_knowledge, snake_kb
from ai_analyzer_en import ai_analyzer

# 全局变量存储预测器和当前识别结果
predictor = None
current_prediction = None
current_knowledge = None

def initialize_model():
    """初始化模型"""
    global predictor
    model_path = "snake_classifier_mobilenetv2.pth"
    
    # 定义中文类别名称
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
        # 提取类别名称和概率
        labels = []
        sizes = []
        
        for pred in predictions:
            labels.append(pred['class'])
            # 去除百分号并转换为浮点数
            confidence = float(pred['confidence'].replace('%', ''))
            sizes.append(confidence)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        
        # 设置颜色
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # 创建扇形图
        wedges, texts, = ax.pie(
            sizes, 
            labels=None,
            colors=colors[:len(labels)],
            autopct=None,
            startangle=90,
            #explode=[0.05] * len(labels),
        )
        
        
        # 确保圆形
        ax.axis('equal')
        
        # 添加图例
        # 调整图例位置，让它更美观
        ax.legend(wedges, labels, title="蛇类种类", loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=12, title_fontsize=14)
        
        
        plt.tight_layout(pad=2.0)  # 增加内边距
        
        # 设置背景色
        fig.patch.set_facecolor('white')
        
        print(f"🔧 扇形图创建成功，返回类型：{type(fig)}")
        return fig
        
    except Exception as e:
        print(f"❌ 创建扇形图失败：{e}")
        # 返回空图
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        ax.text(0.5, 0.5, f'图表生成失败：{str(e)}', ha='center', va='center', transform=ax.transAxes)
        fig.patch.set_facecolor('white')
        return fig

def ai_deep_analysis(user_description, location_input):
    """
    AI深度分析功能
    
    参数:
        user_description (str): 用户输入的描述和问题
        location_input (str): 用户输入的地理位置
    
    返回:
        str: AI深度分析结果
    """
    print(f"🧠 开始AI深度分析...")
    
    # 检查是否有当前的预测结果
    if current_prediction is None:
        return """
❌ **无法进行深度分析**

请先上传蛇类图片进行识别，然后再使用AI深度分析功能。

**使用步骤：**
1. 上传蛇类图片
2. 等待AI识别完成
3. 在下方输入您的位置和问题描述
4. 点击"开始AI深度分析"按钮
        """.strip()
    
    # 检查用户输入
    if not user_description or not user_description.strip():
        return """
❌ **描述信息不足**

请在文本框中详细描述您的情况，例如：
- 遇到蛇的具体环境（花园、森林、水边等）
- 蛇的行为表现
- 您的担心和疑问
- 需要哪方面的建议

这些信息将帮助AI提供更准确的分析建议。
        """.strip()
    
    try:
        # 获取位置信息
        location_info = ai_analyzer.get_user_location(location_input)
        
        # 获取当前识别结果
        snake_species = current_prediction['class']
        confidence = current_prediction['confidence']
        
        # 获取蛇类知识信息
        snake_knowledge = current_knowledge or "暂无详细知识信息"
        
        print(f"🔍 分析参数 - 种类：{snake_species}, 置信度：{confidence}, 位置：{location_info['location']}")
        
        # 调用AI分析器进行深度分析
        analysis_result = ai_analyzer.analyze_snake_encounter(
            snake_species=snake_species,
            confidence=confidence,
            user_description=user_description,
            location_info=location_info,
            snake_knowledge=snake_knowledge
        )
        
        # 格式化返回结果
        formatted_result = f"""
🤖 **AI深度分析报告**

**分析对象：** {snake_species} (置信度: {confidence})
**用户位置：** {location_info['location']}
**分析时间：** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

{analysis_result}

---

⚠️ **重要提醒：** 
本分析基于AI模型，仅供参考。遇到蛇类时请优先考虑安全，必要时请咨询当地专业人士或相关部门。
        """.strip()
        
        print(f"✅ AI深度分析完成")
        return formatted_result
        
    except Exception as e:
        print(f"❌ AI深度分析错误：{str(e)}")
        import traceback
        traceback.print_exc()
        
        error_message = f"""
❌ **分析过程中出现错误**

错误信息：{str(e)}

**可能的解决方案：**
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
    
    # 检查模型
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
    
    # 检查图像
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
        
        # 保存临时图像
        temp_path = "temp_prediction.jpg"
        image.save(temp_path)
        print(f"💾 图像已保存：{temp_path}")
        
        # 进行预测
        result = predictor.predict_image(temp_path, top_k=5)
        print(f"🤖 预测完成")
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if 'error' in result:
            error_msg = f"预测失败：{result['error']}"
            current_prediction = None
            current_knowledge = None
            return error_msg, None, error_msg, error_msg, "请先成功识别图片后再使用AI分析功能"
        
        # 格式化最佳预测结果
        best_pred = result['top_prediction']
        best_result = f"🏆 最佳预测：{best_pred['class']} ({best_pred['confidence']})"
        
        # 保存当前预测结果供AI分析使用
        current_prediction = best_pred
        
        # 详细结果
        detailed_text = "🐍 详细预测结果：\n\n"
        for i, pred in enumerate(result['all_predictions'], 1):
            detailed_text += f"{i}. {pred['class']}: {pred['confidence']}\n"
        
        # 获取最高预测结果的知识库信息
        top_snake_name = best_pred['class']
        snake_knowledge_info = get_snake_knowledge(top_snake_name)
        
        # 保存当前知识信息供AI分析使用
        current_knowledge = snake_knowledge_info
        
        # 如果没有找到知识库信息，提供基本信息
        if not snake_knowledge_info or "暂无" in snake_knowledge_info:
            snake_knowledge_info = f"""
🐍 **{top_snake_name}** 

📊 **AI预测置信度：{best_pred['confidence']}**

ℹ️ **基本信息**
AI识别出这是 {top_snake_name}，置信度为 {best_pred['confidence']}。

⚠️ **安全提醒**
无论遇到任何蛇类，都应该：
- 保持安全距离，不要试图接近或触摸
- 慢慢后退，避免突然动作
- 如被咬伤，立即就医
- 拍照记录（安全距离）有助于医生判断

💡 **建议**
如需更多关于此蛇类的信息，建议咨询当地爬虫学专家或查阅专业资料。
            """.strip()
        
        # 创建扇形图
        pie_chart = create_pie_chart(result['all_predictions'])
        
        # 关闭之前的图形，避免内存问题
        plt.close('all')
        
        # AI分析提示信息
        ai_analysis_prompt = f"""
✅ **图片识别完成！**

识别结果：{best_pred['class']} (置信度: {best_pred['confidence']})

现在您可以：
1. 在下方"位置信息"框中输入您的地理位置
2. 在"详细描述"框中描述遇到蛇的具体情况和您的疑问
3. 点击"🧠 开始AI深度分析"按钮获得专业建议

AI将基于识别结果、您的位置和描述提供针对性的安全建议和专业分析。
        """.strip()
        
        # 添加调试信息
        print(f"🔧 pie_chart 类型：{type(pie_chart)}")
        print(f"🔧 best_result 类型：{type(best_result)}")
        print(f"🔧 detailed_text 类型：{type(detailed_text)}")
        print(f"🔧 knowledge_info 类型：{type(snake_knowledge_info)}")

        print(f"✅ 预测成功完成")
        # 返回顺序：best_output, pie_chart_output, detail_output, knowledge_output, ai_analysis_output
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
    # 标题HTML
    title_html = """
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>🐍 智能蛇类识别系统 + AI深度分析</h1>
        <p>上传蛇类图片，AI帮您识别种类并提供专业的深度分析建议</p>
    </div>
    """
    
    # 模型状态HTML
    if model_loaded:
        status_html = "<div style='color: green; text-align: center;'>✅ 模型已成功加载，AI分析功能已就绪</div>"
    else:
        status_html = "<div style='color: red; text-align: center;'>❌ 模型未加载，请检查模型文件</div>"
    
    # 使用说明
    usage_info = """
    ### 🚀 如何使用
    
    #### 📷 **基础识别功能**
    1. **上传图片**：点击或拖拽图片到上传区域
    2. **自动识别**：图片上传后会自动开始识别
    3. **查看结果**：在右侧查看识别结果、概率分布和详细知识信息
    
    #### 🧠 **AI深度分析功能** ⭐ 新功能
    1. **完成图片识别**：先上传图片并获得识别结果
    2. **输入位置信息**：在"地理位置"框中输入您的具体位置
    3. **详细描述情况**：描述遇到蛇的环境、行为和您的疑问
    4. **获取专业分析**：点击分析按钮，获得基于DeepSeek AI的专业建议
    
    ### 🐍 支持识别的蛇类
    - **北方水蛇** (Northern Watersnake) - 无毒，水生
    - **普通袜带蛇** (Common Garter Snake) - 无毒，温和
    - **德凯棕蛇** (DeKay's Brown Snake) - 无毒，小型
    - **黑鼠蛇** (Black Rat Snake) - 无毒，大型
    - **西部菱斑响尾蛇** (Western Diamondback Rattlesnake) - ⚠️ 剧毒！
    
    ### 🧠 AI深度分析特色
    - **地域性建议**：基于您的地理位置提供当地相关建议
    - **行为预测**：分析蛇类在特定环境下的可能行为
    - **安全评估**：专业的风险评估和应急处理建议
    - **生态教育**：了解蛇类的生态价值和保护意义
    
    ### ⚠️ 安全注意事项
    - 识别结果仅供参考，不能替代专业判断
    - 遇到任何蛇类都应保持安全距离
    - 如被蛇咬伤，立即就医
    - AI分析基于通用知识，具体情况请咨询当地专家
    """
    
    # 图片上传提示
    upload_info = """
    <div style="margin-top: 10px; color: #666; font-size: 12px;">
        支持格式：JPG, PNG, JPEG<br>
        图片会在上传后自动识别，然后可进行AI深度分析
    </div>
    """
    
    # 创建界面
    with gr.Blocks(title="🐍 蛇类识别器 + AI分析") as demo:
        
        # 标题
        gr.HTML(title_html)
        
        # 显示模型状态
        gr.HTML(status_html)
        
        with gr.Row():
            # 左侧：图像上传
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📷 上传蛇类图片",
                    type="pil"
                )
                
                gr.HTML(upload_info)
            
            # 右侧：结果显示
            with gr.Column(scale=1):
                best_output = gr.Textbox(
                    label="🏆 最佳预测结果",
                    lines=2,
                    value="等待上传图片...",
                    interactive=False
                )
                
                # 使用Plot组件显示扇形图
                pie_chart_output = gr.Plot(
                    label="📊 概率分布扇形图"
                )
                
                detail_output = gr.Textbox(
                    label="📋 详细预测结果",
                    lines=6,
                    value="等待上传图片...",
                    interactive=False
                )
        
        # 蛇类知识信息显示区域
        with gr.Row():
            with gr.Column():
                knowledge_output = gr.Textbox(
                    label="📚 蛇类知识档案",
                    lines=15,
                    value="等待上传图片以获取蛇类知识信息...",
                    interactive=False,
                    max_lines=20
                )
        
        # 新增：AI深度分析区域
        with gr.Row():
            gr.HTML("<h3>🧠 AI深度分析功能</h3>")
        
        with gr.Row():
            with gr.Column(scale=1):
                location_input = gr.Textbox(
                    label="📍 地理位置",
                    placeholder="请输入您的具体位置，例如：北京市朝阳区、上海市浦东新区、广东省深圳市等",
                    lines=2,
                    info="详细的位置信息有助于AI提供更准确的地域性建议"
                )
                
                description_input = gr.Textbox(
                    label="📝 详细描述您的情况",
                    placeholder="""请详细描述您遇到蛇的情况，例如：
• 在什么环境下遇到的？（花园、森林、水边、室内等）
• 蛇当时的行为表现如何？
• 您最担心什么问题？
• 需要哪方面的专业建议？
• 是否有其他特殊情况？

描述越详细，AI分析越准确！""",
                    lines=8,
                    info="详细的描述有助于AI进行准确的风险评估和建议"
                )
                
                analyze_button = gr.Button(
                    "🧠 开始AI深度分析",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                ai_analysis_output = gr.Textbox(
                    label="🤖 AI深度分析结果",
                    lines=20,
                    value="请先上传图片进行识别后再使用AI分析功能",
                    interactive=False,
                    max_lines=25
                )
        
        # 图片上传时自动预测
        image_input.change(
            fn=predict_snake,
            inputs=image_input,
            outputs=[best_output, pie_chart_output, detail_output, knowledge_output, ai_analysis_output]
        )
        
        # AI深度分析按钮点击事件
        analyze_button.click(
            fn=ai_deep_analysis,
            inputs=[description_input, location_input],
            outputs=ai_analysis_output
        )
        
        # 使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown(usage_info)
    
    return demo

# 创建并启动界面
demo = create_interface()

if __name__ == "__main__":
    print("🚀 启动蛇类识别系统...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True
    )