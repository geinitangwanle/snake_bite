### SnakeBite — Snake Image Classification & Safety Assistant (中英双语)

A Gradio-based web app and toolkit for snake species identification using a MobileNetV2 classifier, enriched with a bilingual knowledge base and optional DeepSeek-powered safety analysis.

基于 Gradio 的蛇类识别与安全助手，使用 MobileNetV2 迁移学习模型进行分类，并提供中英文知识库与可选的 DeepSeek AI 深度分析。

---

### Features 功能
- **Web UI**: Gradio 界面，交互式地图（Plotly/OpenStreetMap）。
- **Image Classification**: MobileNetV2 迁移学习模型，支持 CPU/GPU/MPS。
- **Bilingual Knowledge Base**: 中英文蛇类资料与安全提示。
- **AI Deep Analysis (optional)**: 集成 DeepSeek API，基于位置与描述给出深度建议。
- **CLI + Scripts**: 训练、预测与评估脚本一应俱全。

---

### Project Structure 目录结构
- `app_4.py` / `app_4_en.py`: 中文/英文版 Gradio 界面
- `predict.py`: 单图/批量推理脚本
- `train.py`: 训练脚本（MobileNetV2 迁移学习）
- `evaluate.py`: 评估脚本（报告、混淆矩阵、按类指标图表）
- `snake_config.py`: 类别配置（中/英/学名）
- `snake_knowledge.py` / `snake_knowledge_en.py`: 知识库
- `ai_analyzer.py` / `ai_analyzer_en.py`: DeepSeek 分析模块
- `requirements.txt`: 依赖
- `dataset/`: 数据集目录（train/validation/test）
- `snake_classifier_mobilenetv2.pth`: 预训练权重（如已提供）

---

### Requirements 环境要求
- Python 3.10+
- macOS/Linux/Windows（可选 GPU 或 Apple Silicon MPS 加速）

Install dependencies 安装依赖：
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

---

### Environment Variables 环境变量
Optional DeepSeek API for AI analysis 可选的 DeepSeek API：
- Create `.env` and set 设置 `.env`：
```bash
DEEPSEEK_API_KEY=your_api_key_here
```
Note: Do not commit real keys. 请勿提交真实密钥。

---

### Run the App 运行应用
- Chinese 中文版：
```bash
python app_4.py
```
- English 英文版：
```bash
python app_4_en.py
```
Default address 默认访问地址：`http://127.0.0.1:7860`

Change port 修改端口：编辑 `app_4*.py` 末尾的 `demo.launch(server_port=...)`。

Internet is required for map tiles and optional IP geolocation. 地图与 IP 定位需要网络。

---

### Quick Predict (CLI) 命令行快速推理
Use the provided model or your trained one 使用提供或自训练的模型：
```bash
python predict.py /path/to/img1.jpg /path/to/img2.jpg \
  --model snake_classifier_mobilenetv2.pth \
  --top-k 3 \
  --device cuda|mps|cpu
```
Output includes top prediction and confidences. 输出包含前 k 预测与置信度。

---

### Dataset Layout 数据集结构
Place images in the following folders 按如下目录放置图像：
```
dataset/
  train/
    ClassA/
    ClassB/
    ...
  validation/
    ClassA/
    ClassB/
    ...
  test/
    ClassA/
    ClassB/
    ...
```
Classes are detected from `dataset/train/*`. 类别将从 `dataset/train/*` 自动检测。

Built-in classes 示例内置类别（中/英/学名见 `snake_config.py`）：
- 北方水蛇 / Northern Watersnake
- 普通袜带蛇 / Common Garter snake
- 德凯棕蛇 / DeKays Brown snake
- 黑鼠蛇 / Black Rat snake
- 西部菱斑响尾蛇 / Western Diamondback rattlesnake

---

### Train 训练
```bash
python train.py
```
- Uses MobileNetV2 with frozen backbone and a new classifier.
- Saves model to `snake_classifier_mobilenetv2.pth`.
- Requires `dataset/train`, `dataset/validation`, `dataset/test` to exist.

要点：
- 使用 MobileNetV2 迁移学习（冻结主干，仅训练分类头）。
- 训练结束保存至 `snake_classifier_mobilenetv2.pth`。
- 需要确保三套分割数据集目录存在。

---

### Evaluate 评估
```bash
python evaluate.py
```
Generates 输出：
- `test_confusion_matrix.png`
- `test_per_class_metrics.png`
- `test_classification_report.txt`
- CSV/JSON summaries 指标汇总（CSV/JSON）

---

### Technical Notes 技术说明
- Devices 设备：自动选择 `cuda` > `mps` > `cpu`（可在 CLI 指定）。
- Preprocessing 预处理：ImageNet 均值方差、224×224 中心裁剪/随机裁剪。
- Map 地图：Plotly Scattermapbox + OpenStreetMap 瓦片。
- DeepSeek：需要有效 `DEEPSEEK_API_KEY`，用于 AI 深度分析（可选）。

---

### Safety Disclaimer 安全声明
- The model may be wrong; do not rely on it for medical decisions.
- For venomous species, keep distance and seek professional help.

- 模型可能出错；不要将其用于医疗决策。
- 若怀疑是毒蛇，请保持距离并寻求专业帮助。

---

