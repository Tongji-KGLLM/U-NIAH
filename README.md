# U-NIAH

U-NIAH (Unified Needle In A Haystack) 是一个用于测试大语言模型长文本理解能力的框架。它通过在长文本中插入特定的"针"（关键信息），然后测试模型是否能够准确检索和理解这些信息。

## 主要功能

1. **LLM长文本理解能力测试**：测试大语言模型在不同上下文长度和不同深度位置检索信息的能力
2. **RAG检索增强生成测试**：评估基于RAG的系统在长文本中检索和生成答案的性能

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/U-NIAH.git
cd U-NIAH
```

### 2. 创建虚拟环境

```bash
# 创建conda虚拟环境
conda create -n uniah python=3.12

# 激活虚拟环境
conda activate uniah
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境

#### 配置model_config.yaml

`config/model_config.yaml`文件已包含常见模型的配置。如需添加新模型，请按照以下格式：

```yaml
models:
  your-model-name:
    provider: OpenAI  # 或 Anthropic
    api_key: YOUR_API_KEY_ENV_VAR  # 对应.env中的环境变量名
    base_url: YOUR_API_BASE_ENV_VAR  # 对应.env中的环境变量名
    max_context: 32000  # 模型的最大上下文窗口大小
```
目前SDK Provider 只支持OpenAI或Anthropic，绝大部分供应商，例如DeepSeek、SiliconFlow、阿里Qwen、智谱、MiniMax等，都可以通过OpenAI的SDK来调用。
在实验过程中，会自动根据设定模型最大上下文长度来进行约束。例如实际模型最大长度16K，则会自动过滤掉超出最大长度的Context实验设定

#### 创建.env文件
根据配置模型，创建`.env`文件，添加API密钥和API_BASE(Base_url)，例如：
```
# OpenAI API
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1

# Anthropic API
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_API_BASE=https://api.anthropic.com

# DeepSeek API
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1

```

### 测试案例说明
目前所有测试用例都在`config/needle_cases.yaml`中定义。
具体，每个测试用例包含以下组件，用户看也可以自定义案例：

```yaml
case_name:  # 测试用例名称
  needles:  # 要插入到文本中的"针"（关键信息）
    - "针1的内容"
    - "针2的内容"
    # 可以添加更多针
  question: "要问模型的问题"  # 用于测试模型是否能检索到针中的信息
  true_answer: "正确答案"  # 用于评估模型回答的标准
```
例如，U-NIAH兼容的NIAH的原始示例：

```yaml
pizza_ingredients:
  needles:
    - " Figs are one of the secret ingredients needed to build the perfect pizza. "
    - " Prosciutto is one of the secret ingredients needed to build the perfect pizza. "
    - " Goat cheese is one of the secret ingredients needed to build the perfect pizza. "
  question: "What are the secret ingredients needed to build the perfect pizza?"
  true_answer: "Figs, Prosciutto, and Goat cheese are the secret ingredients needed to build the perfect pizza."
```

## 使用方法


### 运行LLM多针测试

运行LLM多针测试可以评估大语言模型在不同上下文长度和深度位置下检索信息的能力。以下是一些典型的使用场景：

#### 基本使用

```bash
# 最简单的使用方式 - 测试单个模型和单个案例
python run_llm_multi_needle_test.py --model-names gpt-3.5-turbo --case-names pizza_ingredients

# 测试多个模型
python run_llm_multi_needle_test.py --model-names gpt-3.5-turbo deepseek-chat --case-names pizza_ingredients

# 测试多个案例
python run_llm_multi_needle_test.py --model-names gpt-3.5-turbo --case-names pizza_ingredients rainbow_potion
```

#### 自定义测试参数

```bash
# 自定义上下文长度和深度百分比
python run_llm_multi_needle_test.py \
    --model-names gpt-3.5-turbo \
    --case-names pizza_ingredients \
    --context-lengths 3000 8000 16000 \
    --document-depth-percents 20 50 80

# 调整并发请求数和休眠时间
python run_llm_multi_needle_test.py \
    --model-names gpt-3.5-turbo \
    --case-names pizza_ingredients \
    --num-concurrent-requests 3 \
    --base-sleep-time 1.0

# 仅生成上下文而不执行测试
python run_llm_multi_needle_test.py \
    --model-names gpt-3.5-turbo \
    --case-names pizza_ingredients \
    --only-context
```

#### 命令行参数说明

必需参数：
- `--model-names`：要测试的模型名称列表，例如：gpt-3.5-turbo
- `--case-names`：要测试的案例名称列表，例如：pizza_ingredients rainbow_potion

可选参数：
- `--eval-model`：评估模型名称，默认为"gpt-4o",评估模型也需要在model_config.yaml中配置
- `--context-lengths`：上下文长度列表，默认包含从1K,3K,8K,12K,16K,24K,30K,48K...127K...999K的多个长度
- `--document-depth-percents`：文档深度百分比列表，默认为[10,20,...,100]
- `--num-concurrent-requests`：并发请求数，默认为1
- `--final-context-length-buffer`：最终上下文长度缓冲区，默认为300
- `--base-sleep-time`：基础睡眠时间（秒），默认为0.5，避免超过模型的请求限制
- `--haystack-dir`：haystack目录，默认为"PaulGrahamEssays"
- `--depth-interval-type`：深度区间类型，默认为"linear"
- `--no-save-results`：不保存测试结果
- `--no-save-contexts`：不保存上下文文件
- `--no-print-status`：不打印进行状态
- `--only-context`：仅生成上下文文件
- `--no-dynamic-sleep`：禁用动态休眠




### 运行RAG多针测试

运行RAG多针测试可以评估基于检索增强生成（RAG）的系统在不同上下文长度和深度位置下检索信息的能力。以下是一些典型的使用场景：

#### 基本使用

```bash
# 最简单的使用方式 - 测试单个模型和单个案例
python run_rag_multi_needle_test.py --model-names gpt-3.5-turbo --case-names pizza_ingredients

# 测试多个模型
python run_rag_multi_needle_test.py --model-names  gpt-3.5-turbo  claude-3 --case-names pizza_ingredients

# 测试多个案例
python run_rag_multi_needle_test.py --model-names  gpt-3.5-turbo --case-names pizza_ingredients needle_in_needle
```

#### 不同上下文模式测试

RAG测试支持三种上下文模式（context_mode）：topk、half和full，分别代表不同的检索策略：

```bash
# topk模式 - 仅使用相似度最高的前k个文档块作为上下文（默认模式）
python run_rag_multi_needle_test.py \
    --model-names gpt-3.5-turbo  \
    --case-names pizza_ingredients \
    --context-mode topk \
    --top-k-context 5

# half模式 - 检索文档填充一半的给定上下文长度
python run_rag_multi_needle_test.py \
    --model-names gpt-3.5-turbo  \
    --case-names pizza_ingredients\
    --context-mode half

# full模式 -  检索文档填充全部的给定上下文长度
python run_rag_multi_needle_test.py \
    --model-names gpt-3.5-turbo  \
    --case-names pizza_ingredients\
    --context-mode full
```

#### 其他自定义测试参数

```bash
# 自定义分块大小和重叠大小
python run_rag_multi_needle_test.py \
    --model-names gpt-3.5-turbo \
    --case-names pizza_ingredients \
    --chunk-size 800 \
    --chunk-overlap 150

# 反转上下文顺序（将相关性较低的文档块放在前面）
python run_rag_multi_needle_test.py \
    --model-names gpt-3.5-turbo \
    --case-names pizza_ingredients \
    --reverse

```


#### 命令行参数说明

必需参数：
- `--model-names`：要测试的模型名称列表，例如：gpt-3.5-turbo,deepseek-chat
- `--case-names`：要测试的案例名称列表，例如：needle_in_needle pizza_ingredients

可选参数：
- `--eval-model`：评估模型名称，默认为"gpt-4o"
- `--context-lengths`：上下文长度列表，默认包含从1K到999K的多个长度
- `--document-depth-percents`：文档深度百分比列表，默认为[10,20,...,100]
- `--context-mode`：上下文模式，可选值为"topk"、"half"、"full"，默认为"topk"
- `--top-k-context`：当context_mode为topk时使用的top-k值，默认为5
- `--reverse`：是否反转上下文顺序，默认为False
- `--chunk-size`：分块大小，默认为600
- `--chunk-overlap`：分块重叠大小，默认为100
- `--embedding-model`：嵌入模型名称，默认为"text-embedding-3-small"
- `--top-k-retrieval`：检索时使用的top-k值，默认为None
- `--num-concurrent-requests`：并发请求数，默认为1
- `--final-context-length-buffer`：最终上下文长度缓冲区，默认为300
- `--base-sleep-time`：基础睡眠时间（秒），默认为0.6
- `--haystack-dir`：数据目录，默认为"PaulGrahamEssays"
- `--only-build-rag-context`：仅构建RAG上下文，默认为False
- `--enable-dynamic-sleep`：启用动态休眠，默认为True





## 结果分析

测试结果将保存在以下目录：

- RAG测试结果：`rag_multi_needle/results/`
- RAG上下文：`rag_multi_needle/contexts/`
- LLM测试结果：`llm_multi_needle/results/`

单个测试案例的分析可以使用CreateVizFromLLMTesting.ipynb

### 使用可视化脚本分析结果

项目提供了两个批量可视化处理脚本，：

```bash
# 处理LLM测试结果
python Needle_vis_llm.py

# 处理RAG测试结果
python Needle_vis_rag.py
```

#### Needle_vis_llm.py 使用说明

此脚本用于处理LLM多针测试的结果，生成热力图并汇总数据。

**输入**：
- 测试结果目录（result）
- 结果文件格式：JSON/JSONL文件，包含depth_percent、context_length和score字段

**输出**：
- 热力图：显示不同上下文长度和文档深度下的得分情况
- 汇总数据：`visualizations/llm_all_experiments_data.csv`

**使用方法**：

1. 基本使用：
```python
# 在脚本末尾设置base_dir为结果目录
base_dir = "llm_multi_needle/results"
visualizer = NeedleVisualizer(base_dir)
all_experiments_data = visualizer.process_all_experiments()
```

1. 高级选项：
```python
# 排除特定文件夹
exclude_dirs = ["visualizations", "other_folder"]

# 强制重新处理所有实验
visualizer = NeedleVisualizer(base_dir, exclude_dirs, force_reprocess=True)
all_experiments_data = visualizer.process_all_experiments()
```
 Needle_vis_rag.py 使用方法一致

两个脚本都会在结果目录和`visualizations`子目录中生成热力图，并将所有实验数据汇总到CSV文件中，方便进一步分析。

## 其他
本项目是基于[NIAH框架](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)的改进版本，感谢原作者的贡献。
在兼容性上，本框架对原始NIAH做了一些额外的改进，因此和原始版本相比，可能存在一些差异。
具体包括：
1. 在Context构建过程中，Needle只能插入在完整的句子（例如".", "?",".....。" 等）后面。因为在测试的时候发现，在一些特殊场景下，插入Neddle前半部分和被截断的语句组成了新的具有特定含义的词组，导致模型理解出现偏差。
2. 在拼接Haystack时，强制按照首字母的顺序拼接txt文件。目的是为了在后期拓展补充语料（例如1M+Token）时，保证和之前的测试的Context是保持一致。

