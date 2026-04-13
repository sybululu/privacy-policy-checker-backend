# 隐私政策合规审查系统

基于BERT的隐私政策违规风险识别系统，支持A/B版本切换。

## 版本说明

### A版本（当前默认）
- 模型：CAPP-130数据实践分类（11类）
- 功能：11类数据实践识别 + 规则映射 → 12类违规风险
- 用途：快速交付，基于浙大CAPP-130模型

### B版本（预留）
- 模型：自定义12类违规识别
- 功能：直接输出12类违规分类
- 用途：之后替换为自行训练的模型

## 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# A版本运行
python app.py

# B版本运行
APP_VERSION=B python app.py
```

## HF Space 部署

1. 上传 `app.py`, `config.py`, `mapper.py`, `requirements.txt`
2. 设置环境变量（可选）：`APP_VERSION=A` 或 `APP_VERSION=B`
3. Space 自动启动

## 文件说明

```
├── app.py           # 主程序
├── config.py        # 版本配置
├── mapper.py        # 规则映射（A版）
└── requirements.txt # 依赖
```

## API 接口

### 分析接口

**POST** `/api/analyze`

Request:
```json
{
  "text": "隐私政策文本..."
}
```

Response:
```json
{
  "result": "违规风险分析结果...",
  "version": "A"
}
```

## 技术栈

- PyTorch 2.0+
- Transformers 4.30+
- Gradio 4.0+
- HuggingFace Hub

## 参考

- CAPP-130数据集：https://github.com/EnlightenedAI/CAPP-130
- 论文：CAPP-130: A Corpus of Chinese Application Privacy Policy Summarization and Interpretation (NeurIPS 2023)
