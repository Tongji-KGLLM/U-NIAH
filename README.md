# U-NIAH

[中文版本 / Chinese Version](README_ZH.md)

U-NIAH (Unified Needle In A Haystack) is a framework for testing the long-text understanding capabilities of large language models. It tests whether models can accurately retrieve and understand specific information by inserting "needles" (key information) into long texts.

For detailed information, please refer to the paper: [U-NIAH: Unified RAG and LLM Evaluation for Long Context Needle-In-A-Haystack](https://arxiv.org/abs/2503.00353)

## Key Features

1. **LLM Long-text Understanding Testing**: Tests the ability of large language models to retrieve information at different context lengths and depth positions
2. **RAG Retrieval-Augmented Generation Testing**: Evaluates the performance of RAG-based systems in retrieving and generating answers from long texts

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/U-NIAH.git
cd U-NIAH
```

### 2. Create Virtual Environment

```bash
# Create conda virtual environment
conda create -n uniah python=3.12

# Activate virtual environment
conda activate uniah
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration

#### Configure model_config.yaml

The `config/model_config.yaml` file already contains configurations for common models. To add new models, follow this format:

```yaml
models:
  your-model-name:
    provider: OpenAI  # or Anthropic
    api_key: YOUR_API_KEY_ENV_VAR  # Environment variable name in .env
    base_url: YOUR_API_BASE_ENV_VAR  # Environment variable name in .env
    max_context: 32000  # Maximum context window size of the model
```

Currently, SDK Provider only supports OpenAI or Anthropic. Most providers such as DeepSeek, SiliconFlow, Alibaba Qwen, Zhipu, MiniMax, etc., can be called through OpenAI's SDK.

During experiments, the system will automatically constrain based on the model's maximum context length setting. For example, if the actual model's maximum length is 16K, context experiment settings exceeding the maximum length will be automatically filtered out.

#### Create .env File
Based on your configured models, create a `.env` file and add API keys and API_BASE (Base_url), for example:
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

### Test Case Description
Currently, all test cases are defined in `config/needle_cases.yaml`.
Specifically, each test case contains the following components, and users can also customize cases:

```yaml
case_name:  # Test case name
  needles:  # "Needles" (key information) to insert into text
    - "Content of needle 1"
    - "Content of needle 2"
    # More needles can be added
  question: "Question to ask the model"  # Used to test if the model can retrieve information from needles
  true_answer: "Correct answer"  # Standard for evaluating the model's response
```

For example, the original NIAH example compatible with U-NIAH:

```yaml
pizza_ingredients:
  needles:
    - " Figs are one of the secret ingredients needed to build the perfect pizza. "
    - " Prosciutto is one of the secret ingredients needed to build the perfect pizza. "
    - " Goat cheese is one of the secret ingredients needed to build the perfect pizza. "
  question: "What are the secret ingredients needed to build the perfect pizza?"
  true_answer: "Figs, Prosciutto, and Goat cheese are the secret ingredients needed to build the perfect pizza."
```

## Usage

### Running LLM Multi-Needle Testing

Running LLM multi-needle testing evaluates the ability of large language models to retrieve information at different context lengths and depth positions. Here are some typical usage scenarios:

#### Basic Usage

```bash
# Simplest usage - test a single model and single case
python run_llm_multi_needle_test.py --model-names gpt-3.5-turbo --case-names pizza_ingredients

# Test multiple models
python run_llm_multi_needle_test.py --model-names gpt-3.5-turbo deepseek-chat --case-names pizza_ingredients

# Test multiple cases
python run_llm_multi_needle_test.py --model-names gpt-3.5-turbo --case-names pizza_ingredients rainbow_potion
```

#### Custom Test Parameters

```bash
# Custom context lengths and depth percentages
python run_llm_multi_needle_test.py \
    --model-names gpt-3.5-turbo \
    --case-names pizza_ingredients \
    --context-lengths 3000 8000 16000 \
    --document-depth-percents 20 50 80

# Adjust concurrent requests and sleep time
python run_llm_multi_needle_test.py \
    --model-names gpt-3.5-turbo \
    --case-names pizza_ingredients \
    --num-concurrent-requests 3 \
    --base-sleep-time 1.0

# Generate context only without executing tests
python run_llm_multi_needle_test.py \
    --model-names gpt-3.5-turbo \
    --case-names pizza_ingredients \
    --only-context
```

#### Command Line Arguments

Required arguments:
- `--model-names`: List of model names to test, e.g., gpt-3.5-turbo
- `--case-names`: List of case names to test, e.g., pizza_ingredients rainbow_potion

Optional arguments:
- `--eval-model`: Evaluation model name, defaults to "gpt-4o". The evaluation model also needs to be configured in model_config.yaml
- `--context-lengths`: List of context lengths, defaults to multiple lengths from 1K, 3K, 8K, 12K, 16K, 24K, 30K, 48K...127K...999K
- `--document-depth-percents`: List of document depth percentages, defaults to [10,20,...,100]
- `--num-concurrent-requests`: Number of concurrent requests, defaults to 1
- `--final-context-length-buffer`: Final context length buffer, defaults to 300
- `--base-sleep-time`: Base sleep time (seconds), defaults to 0.5, to avoid exceeding model request limits
- `--haystack-dir`: Haystack directory, defaults to "PaulGrahamEssays"
- `--depth-interval-type`: Depth interval type, defaults to "linear"
- `--no-save-results`: Don't save test results
- `--no-save-contexts`: Don't save context files
- `--no-print-status`: Don't print progress status
- `--only-context`: Only generate context files
- `--no-dynamic-sleep`: Disable dynamic sleep

### Running RAG Multi-Needle Testing

Running RAG multi-needle testing evaluates the ability of Retrieval-Augmented Generation (RAG) systems to retrieve information at different context lengths and depth positions. Here are some typical usage scenarios:

#### Basic Usage

```bash
# Simplest usage - test a single model and single case
python run_rag_multi_needle_test.py --model-names gpt-3.5-turbo --case-names pizza_ingredients

# Test multiple models
python run_rag_multi_needle_test.py --model-names gpt-3.5-turbo claude-3 --case-names pizza_ingredients

# Test multiple cases
python run_rag_multi_needle_test.py --model-names gpt-3.5-turbo --case-names pizza_ingredients needle_in_needle
```

#### Different Context Mode Testing

RAG testing supports three context modes (context_mode): topk, half, and full, representing different retrieval strategies:

```bash
# topk mode - only use top k document chunks with highest similarity scores as context (default mode)
python run_rag_multi_needle_test.py \
    --model-names gpt-3.5-turbo \
    --case-names pizza_ingredients \
    --context-mode topk \
    --top-k-context 5

# half mode - retrieve documents to fill half of the given context length
python run_rag_multi_needle_test.py \
    --model-names gpt-3.5-turbo \
    --case-names pizza_ingredients \
    --context-mode half

# full mode - retrieve documents to fill the entire given context length
python run_rag_multi_needle_test.py \
    --model-names gpt-3.5-turbo \
    --case-names pizza_ingredients \
    --context-mode full
```

#### Other Custom Test Parameters

```bash
# Custom chunk size and overlap size
python run_rag_multi_needle_test.py \
    --model-names gpt-3.5-turbo \
    --case-names pizza_ingredients \
    --chunk-size 800 \
    --chunk-overlap 150

# Reverse context order (put less relevant document chunks first)
python run_rag_multi_needle_test.py \
    --model-names gpt-3.5-turbo \
    --case-names pizza_ingredients \
    --reverse
```

#### Command Line Arguments

Required arguments:
- `--model-names`: List of model names to test, e.g., gpt-3.5-turbo,deepseek-chat
- `--case-names`: List of case names to test, e.g., needle_in_needle pizza_ingredients

Optional arguments:
- `--eval-model`: Evaluation model name, defaults to "gpt-4o"
- `--context-lengths`: List of context lengths, defaults to multiple lengths from 1K to 999K
- `--document-depth-percents`: List of document depth percentages, defaults to [10,20,...,100]
- `--context-mode`: Context mode, options are "topk", "half", "full", defaults to "topk"
- `--top-k-context`: Top-k value when context_mode is topk, defaults to 5
- `--reverse`: Whether to reverse context order, defaults to False
- `--chunk-size`: Chunk size, defaults to 600
- `--chunk-overlap`: Chunk overlap size, defaults to 100
- `--embedding-model`: Embedding model name, defaults to "text-embedding-3-small"
- `--top-k-retrieval`: Top-k value for retrieval, defaults to None
- `--num-concurrent-requests`: Number of concurrent requests, defaults to 1
- `--final-context-length-buffer`: Final context length buffer, defaults to 300
- `--base-sleep-time`: Base sleep time (seconds), defaults to 0.6
- `--haystack-dir`: Data directory, defaults to "PaulGrahamEssays"
- `--only-build-rag-context`: Only build RAG context, defaults to False
- `--enable-dynamic-sleep`: Enable dynamic sleep, defaults to True

## Result Analysis

Test results will be saved in the following directories:

- RAG test results: `rag_multi_needle/results/`
- RAG contexts: `rag_multi_needle/contexts/`
- LLM test results: `llm_multi_needle/results/`

Individual test case analysis can be done using CreateVizFromLLMTesting.ipynb

### Using Visualization Scripts for Result Analysis

The project provides two batch visualization processing scripts:

```bash
# Process LLM test results
python Needle_vis_llm.py

# Process RAG test results
python Needle_vis_rag.py
```

#### Needle_vis_llm.py Usage

This script processes LLM multi-needle test results, generates heat maps, and summarizes data.

**Input**:
- Test result directory (result)
- Result file format: JSON/JSONL files containing depth_percent, context_length, and score fields

**Output**:
- Heat maps: Show scoring patterns at different context lengths and document depths
- Summary data: `visualizations/llm_all_experiments_data.csv`

**Usage**:

1. Basic usage:
```python
# Set base_dir to results directory at the end of the script
base_dir = "llm_multi_needle/results"
visualizer = NeedleVisualizer(base_dir)
all_experiments_data = visualizer.process_all_experiments()
```

2. Advanced options:
```python
# Exclude specific folders
exclude_dirs = ["visualizations", "other_folder"]

# Force reprocess all experiments
visualizer = NeedleVisualizer(base_dir, exclude_dirs, force_reprocess=True)
all_experiments_data = visualizer.process_all_experiments()
```

Needle_vis_rag.py usage is identical.

Both scripts generate heat maps in the results directory and `visualizations` subdirectory, and consolidate all experiment data into CSV files for further analysis.

## Other Information

This project is an improved version based on the [NIAH framework](https://github.com/gkamradt/LLMTest_NeedleInAHaystack). We thank the original authors for their contributions.

In terms of compatibility, this framework has made some additional improvements over the original NIAH, so there may be some differences compared to the original version.

Specifically including:
1. During context construction, needles can only be inserted after complete sentences (e.g., ".", "?", ".....。", etc.). This is because during testing, we found that in some special scenarios, the first half of the inserted needle and truncated statements formed new word groups with specific meanings, causing model understanding deviations.
2. When concatenating haystacks, txt files are forcibly concatenated in alphabetical order. The purpose is to ensure consistency with previous test contexts when expanding and supplementing corpora (e.g., 1M+ tokens) in later stages.

## Citation
```  
@article{gao2025u,
  title={U-NIAH: Unified RAG and LLM Evaluation for Long Context Needle-In-A-Haystack},
  author={Gao, Yunfan and Xiong, Yun and Wu, Wenlong and Huang, Zijing and Li, Bohan and Wang, Haofen},
  journal={arXiv preprint arXiv:2503.00353},
  year={2025}
}
``` 