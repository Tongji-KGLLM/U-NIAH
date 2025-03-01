import sys
import os
import asyncio
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import threading
from tqdm import tqdm
import argparse

# 设置运行目录为当前文件所在位置
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

import yaml
from needlehaystack.providers import OpenAI, Anthropic
from needlehaystack.evaluators import OpenAIEvaluator
from needlehaystack.rag_multi_needle_haystack_tester import RAGMultiNeedleHaystackTester
from dotenv import load_dotenv

def load_test_cases(file_path):
    """加载测试案例配置文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_model_config(file_path: str) -> Dict[str, Any]:
    """加载模型配置文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_model_provider(model_name: str, model_config: Dict[str, Any]):
    """根据模型名称获取对应的Provider实例"""
    if model_name not in model_config['models']:
        raise ValueError(f"未找到模型 {model_name} 的配置")
    
    config = model_config['models'][model_name]
    provider_name = config['provider']
    api_key = os.getenv(config['api_key'])
    base_url = os.getenv(config['base_url'])
    
    if provider_name == 'OpenAI':
        return OpenAI(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )
    elif provider_name == 'Anthropic':
        return Anthropic(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )
    else:
        raise ValueError(f"不支持的提供商: {provider_name}")

def run_single_test(
    model_name: str, 
    case_name: str, 
    model_config: Dict, 
    test_cases: Dict,
    context_lengths: List[int] = None,
    document_depth_percents: List[int] = None,
    num_concurrent_requests: int = 2,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    context_mode: str = "half",
    reverse: bool = True,
    top_k_retrieval: int = None,
    top_k_context: int = None,
    embedding_model: str = "text-embedding-3-small",
    save_results: bool = True,
    save_contexts: bool = True,
    final_context_length_buffer: int = 300,
    print_ongoing_status: bool = True,
    only_build_rag_context: bool = False,
    enable_dynamic_sleep: bool = True,
    base_sleep_time: float = 0.3,
    haystack_dir: str = "PaulGrahamEssays",
    eval_model: str = "gpt-4o"
):
    """运行单个模型和案例的测试"""
    # 创建新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # 获取模型最大上下文长度
    max_context = model_config['models'][model_name]['max_context']
    
    # 如果没有指定context_lengths，使用默认值
    if context_lengths is None:
        context_lengths = [1000, 2000, 3000, 5000, 8000, 12000, 16000, 24000, 30000, 48000, 
                         64000, 96000, 127000, 156000, 192000, 256000, 386000, 512000, 786000, 999000]

    # 过滤掉超过模型最大上下文长度的值
    context_lengths = [l for l in context_lengths if l <= max_context]

    # 如果没有指定document_depth_percents，使用默认值
    if document_depth_percents is None:
        document_depth_percents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # 获取模型提供者
    model_to_test = get_model_provider(model_name, model_config)
    
    # 根据配置文件确定LLM提供商
    provider_name = model_config['models'][model_name]['provider']
    llm_provider = provider_name.lower()  # 转换为小写以匹配期望的格式

    # 获取评估模型的API密钥和基础URL，如果不存在则抛出错误
    eval_api_key = os.getenv(model_config['models'][eval_model]['api_key'])
    if eval_api_key is None:
        raise ValueError(f"未找到评估模型 {eval_model} 的API密钥")
    
    eval_base_url = os.getenv(model_config['models'][eval_model]['base_url'])
    if eval_base_url is None:
        raise ValueError(f"未找到评估模型 {eval_model} 的基础URL")

    # 获取评估器
    evaluator = OpenAIEvaluator(
        model_name=eval_model,
        question_asked=test_cases[case_name]["question"],
        true_answer=test_cases[case_name]["true_answer"],
        api_key=eval_api_key,
        base_url=eval_base_url,
    )


    # 创建RAG多needle测试器，使用传入的参数
    tester = RAGMultiNeedleHaystackTester(
        model_to_test=model_to_test,
        evaluator=evaluator,
        needles=test_cases[case_name]["needles"],
        retrieval_question=test_cases[case_name]["question"],
        haystack_dir=haystack_dir,
        context_lengths=context_lengths,
        document_depth_percents=document_depth_percents,
        num_concurrent_requests=num_concurrent_requests,
        save_results=save_results,
        save_contexts=save_contexts,
        final_context_length_buffer=final_context_length_buffer,
        print_ongoing_status=print_ongoing_status,
        api_key=os.getenv(model_config['models'][model_name]['api_key']),
        base_url=os.getenv(model_config['models'][model_name]['base_url']),
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k_retrieval=top_k_retrieval,
        context_mode=context_mode,
        reverse=reverse,
        top_k_context=top_k_context,
        llm_provider=llm_provider,
        case_name=case_name,
        only_build_rag_context=only_build_rag_context,
        enable_dynamic_sleep=enable_dynamic_sleep,
        base_sleep_time=base_sleep_time,
    )
    
    # 开始测试
    tester.start_test()
    
    # 关闭事件循环
    loop.close()
    
    return True, f"成功完成 {model_name} - {case_name} 的测试"

# 添加新的检查函数
def check_and_run_missing_experiments(model_name: str, case_name: str, model_config: dict, test_cases: dict, test_params: dict):
    """
    检查指定模型和测试用例的实验结果是否完整，并运行缺失的实验
    
    Args:
        model_name: 模型名称
        case_name: 测试用例名称
        model_config: 模型配置
        test_cases: 测试用例配置
        test_params: 测试参数
    """
    # 获取模型最大上下文长度
    max_context = model_config['models'][model_name]['max_context']
    
    # 获取所有可能的context_lengths（根据最大上下文过滤）
    all_context_lengths = [l for l in test_params["context_lengths"] if l <= max_context]
    
    # 构建结果文件夹路径
    params_str = f"{case_name}"
    params_str += f"_{test_params['context_mode']}"
    params_str += f"{test_params['top_k_context'] if test_params['top_k_context'] else ''}"
    params_str += f"_{'rev' if test_params['reverse'] else 'norm'}"
    
    safe_model_name = model_name.replace(".", "_").replace("/", "_")
    results_base_path = f'./rag_multi_needle/results/{case_name}/{params_str}/{safe_model_name}'
    
    # 检查每个组合是否都有结果文件
    missing_experiments = []
    for context_length in all_context_lengths:
        for depth_percent in test_params["document_depth_percents"]:
            result_file = f'{safe_model_name}_len_{context_length}_depth_{int(depth_percent)}_results.json'
            result_path = os.path.join(results_base_path, result_file)
            
            if not os.path.exists(result_path):
                missing_experiments.append((context_length, depth_percent))
    
    # 如果有缺失的实验，运行它们
    if missing_experiments:
        print(f"\n发现 {model_name} 在 {case_name} 用例中有 {len(missing_experiments)} 个缺失的实验点")
        print("正在补充运行缺失的实验...")
        
        # 更新测试参数，只运行缺失的实验点
        missing_test_params = test_params.copy()
        missing_test_params["context_lengths"] = list(set(exp[0] for exp in missing_experiments))
        missing_test_params["document_depth_percents"] = list(set(exp[1] for exp in missing_experiments))
        
        # 运行缺失的实验
        success, message = run_single_test(
            model_name,
            case_name,
            model_config,
            test_cases,
            **missing_test_params
        )
        
        if success:
            print(f"成功补充完成缺失的实验")
        else:
            print(f"补充实验过程中出现错误: {message}")
    else:
        print(f"\n{model_name} 在 {case_name} 用例中的所有实验点都已完成")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RAG多针测试脚本')
    
    # 模型和案例参数
    parser.add_argument('--model-names', nargs='+', default=["gpt-3.5-turbo"],
                        help='要测试的模型名称列表，例如：gpt-4o claude-3-5-sonnet-20241022')
    parser.add_argument('--case-names', nargs='+', default=["needle_in_needle"],
                        help='要测试的案例名称列表，例如：pizza_ingredients rainbow_potion')
    parser.add_argument('--eval-model', type=str, default="gpt-4o",
                        help='评估模型名称，默认为gpt-4o')
    
    # 上下文长度和文档深度参数
    parser.add_argument('--context-lengths', nargs='+', type=int,
                        default=[1000,3000, 5000, 8000, 12000, 16000, 24000, 30000, 48000, 64000,
                                96000, 127000, 156000, 192000, 256000, 386000, 512000, 786000, 999000],
                        help='上下文长度列表，默认为完整区间')
    parser.add_argument('--document-depth-percents', nargs='+', type=int,
                        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help='文档深度百分比列表，默认为完整区间')
    
    # RAG特定参数
    parser.add_argument('--context-mode', type=str, default="topk",
                        choices=["half", "topk", "full"],
                        help='上下文模式，可选：half, topk, full，默认为topk')
    parser.add_argument('--top-k-context', type=int, default=5,
                        help='当context_mode为topk时，使用的top-k值，默认为5')
    parser.add_argument('--reverse', action='store_true', default=False,
                        help='是否反转上下文顺序，默认为False')
    parser.add_argument('--chunk-size', type=int, default=600,
                        help='分块大小，默认为600')
    parser.add_argument('--chunk-overlap', type=int, default=100,
                        help='分块重叠大小，默认为100')
    parser.add_argument('--embedding-model', type=str, default="text-embedding-3-small",
                        help='嵌入模型名称，默认为text-embedding-3-small')
    parser.add_argument('--top-k-retrieval', type=int, default=None,
                        help='检索时使用的top-k值，默认为None')
    parser.add_argument('--haystack-dir', type=str, default="PaulGrahamEssays",
                        help='文档目录，默认为PaulGrahamEssays')
    
    # 其他通用参数
    parser.add_argument('--num-concurrent-requests', type=int, default=1,
                        help='并发请求数，默认为1')
    parser.add_argument('--final-context-length-buffer', type=int, default=300,
                        help='最终上下文长度缓冲区，默认为300')
    parser.add_argument('--base-sleep-time', type=float, default=0.6,
                        help='基础睡眠时间，默认为0.6秒')
    parser.add_argument('--enable-dynamic-sleep', action='store_true', default=True,
                        help='是否启用动态睡眠，默认为True')
    parser.add_argument('--only-build-rag-context', action='store_true', default=False,
                        help='是否仅构建RAG上下文，默认为False')

    
    return parser.parse_args()

def main():
    # 加载环境变量
    load_dotenv()
    
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    model_config = load_model_config('config/model_config.yaml')
    test_cases = load_test_cases('config/needle_cases.yaml')
    
    # 从命令行参数获取模型和案例列表
    model_names = args.model_names
    case_names = args.case_names
    
    # 创建所有测试组合
    test_combinations = list(product(model_names, case_names))
    total_tests = len(test_combinations)
    
    # 从命令行参数构建测试参数
    test_params = {
        "context_lengths": args.context_lengths,
        "document_depth_percents": args.document_depth_percents,    
        "context_mode": args.context_mode,
        "top_k_context": args.top_k_context,
        "reverse": args.reverse,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "num_concurrent_requests": args.num_concurrent_requests,
        "enable_dynamic_sleep": args.enable_dynamic_sleep,
        "base_sleep_time": args.base_sleep_time,
        "only_build_rag_context": args.only_build_rag_context,
        "haystack_dir": args.haystack_dir,
        "eval_model": args.eval_model,
    }

    # 创建进度条
    progress_bar = tqdm(total=total_tests, desc="测试进度")
    progress_lock = threading.Lock()
    
    def update_progress(*args):
        with progress_lock:
            progress_bar.update(1)
    
    # 使用线程池执行测试
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        results = []  # 保留results列表用于最终摘要
        for model_name, case_name in test_combinations:
            # 运行主要测试
            future = executor.submit(
                run_single_test,
                model_name,
                case_name,
                model_config,
                test_cases,
                **test_params
            )
            future.add_done_callback(update_progress)
            futures.append(future)
            
            # 等待当前测试完成
            success, message = future.result()
            results.append(message)
            

    
    progress_bar.close()
    
    # 打印测试结果摘要
    print("\n测试结果摘要:")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()