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
from needlehaystack.providers import OpenAI,Anthropic
from needlehaystack.evaluators import OpenAIEvaluator
from needlehaystack.llm_multi_needle_haystack_tester import LLMMultiNeedleHaystackTester
from dotenv import load_dotenv
load_dotenv(".env")
# 加载环境变量

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
    num_concurrent_requests: int = 5,
    save_results: bool = True,
    save_contexts: bool = True,
    final_context_length_buffer: int = 300,
    print_ongoing_status: bool = True,
    only_context: bool = False,
    enable_dynamic_sleep: bool = True,
    base_sleep_time: float = 0.3,
    document_depth_percent_interval_type: str = 'linear',
    haystack_dir: str = "StarlightHarryPorter",
    eval_model: str = "gpt-4o"
):
    """运行单个模型和案例的测试"""
    try:
        # 获取模型最大上下文长度
        max_context = model_config['models'][model_name]['max_context']
        
        # 如果没有指定context_lengths，使用默认值
        if context_lengths is None:
            context_lengths = [3000, 5000, 8000, 12000, 16000, 24000, 30000,48000,64000, 
                             96000, 127000, 156000, 192000,256000,386000,512000,786000,999000] 

        context_lengths = [l for l in context_lengths if l <= max_context]
        
        # 如果没有指定document_depth_percents，使用默认值
        if document_depth_percents is None:
            document_depth_percents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        # 获取模型提供者
        model_to_test = get_model_provider(model_name, model_config)
        
        eval_model = "gpt-4o"

        # 获取评估模型的API密钥和基础URL，如果不存在则抛出错误
        eval_api_key = os.getenv(model_config['models'][eval_model]['api_key'])
        if eval_api_key is None:
            raise ValueError(f"未找到评估模型 {eval_model} 的API密钥")
        
        eval_base_url = os.getenv(model_config['models'][eval_model]['base_url'])
        if eval_base_url is None:
            raise ValueError(f"未找到评估模型 {eval_model} 的基础URL")

        # 获取评估器
        evaluator = None if only_context else OpenAIEvaluator(
            model_name=eval_model,
            question_asked=test_cases[case_name]["question"],
            true_answer=test_cases[case_name]["true_answer"],
            api_key=eval_api_key ,
            base_url=eval_base_url
        )
        
        # 创建多needle测试器
        tester = LLMMultiNeedleHaystackTester(
            model_to_test=model_to_test,
            evaluator=evaluator,
            needles=test_cases[case_name]["needles"],
            retrieval_question=test_cases[case_name]["question"],
            haystack_dir=haystack_dir,
            context_lengths=context_lengths,
            document_depth_percents=document_depth_percents,
            document_depth_percent_interval_type=document_depth_percent_interval_type,
            num_concurrent_requests=num_concurrent_requests,
            save_results=save_results,
            save_contexts=save_contexts,
            final_context_length_buffer=final_context_length_buffer,
            print_ongoing_status=print_ongoing_status,
            case_name=case_name,
            only_context=only_context,
            enable_dynamic_sleep=enable_dynamic_sleep,
            base_sleep_time=base_sleep_time
        )
        
        # 开始测试
        tester.start_test()
        return True, f"成功完成 {model_name} - {case_name} 的测试"
    except Exception as e:
        return False, f"测试 {model_name} - {case_name} 失败: {str(e)}"

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LLM多针测试脚本')
    
    # 必需参数
    parser.add_argument('--model-names', nargs='+', required=True,
                        default="gpt-4o-mini",
                        help='要测试的模型名称列表，例如：gpt-4 claude-3')
    parser.add_argument('--case-names', nargs='+', required=True,
                        default="pizza_ingredients",
                        help='要测试的案例名称列表，例如：pizza_ingredients rainbow_potion')
    parser.add_argument('--eval-model', type=str, default="gpt-4o",
                        help='评估模型名称，默认为gpt-4o')
    
    # 可选参数 - 上下文长度和文档深度
    parser.add_argument('--context-lengths', nargs='+', type=int,
                        default=[1000, 2000, 3000, 5000, 8000, 12000, 16000, 24000, 30000, 48000, 64000,
                                96000, 127000, 156000, 192000, 256000, 386000, 512000, 786000, 999000],
                        help='上下文长度列表，默认为完整区间')
    parser.add_argument('--document-depth-percents', nargs='+', type=int,
                        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help='文档深度百分比列表，默认为完整区间')
    
    # 其他可选参数
    parser.add_argument('--num-concurrent-requests', type=int, default=1,
                        help='并发请求数，默认为1')
    parser.add_argument('--final-context-length-buffer', type=int, default=300,
                        help='最终上下文长度缓冲区，默认为300')
    parser.add_argument('--base-sleep-time', type=float, default=0.5,
                        help='基础睡眠时间，默认为0.5秒')
    parser.add_argument('--haystack-dir', type=str, default='PaulGrahamEssays',
                        help='数据目录，默认为PaulGrahamEssays')
    parser.add_argument('--depth-interval-type', type=str, default='linear',
                        choices=['linear'],
                        help='文档深度区间类型，默认为linear')
    
    # 布尔标志
    parser.add_argument('--no-save-results', action='store_false', dest='save_results',
                        help='不保存结果')
    parser.add_argument('--no-save-contexts', action='store_false', dest='save_contexts',
                        help='不保存上下文')
    parser.add_argument('--no-print-status', action='store_false', dest='print_ongoing_status',
                        help='不打印进行状态')
    parser.add_argument('--only-context', action='store_true',
                        help='仅处理上下文')
    parser.add_argument('--no-dynamic-sleep', action='store_false', dest='enable_dynamic_sleep',
                        help='禁用动态睡眠')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    load_dotenv()
    
    # 加载配置文件
    model_config = load_model_config('config/model_config.yaml')
    test_cases = load_test_cases('config/needle_cases.yaml')

    # 定义测试参数
    test_params = {
        "context_lengths": args.context_lengths,
        "document_depth_percents": args.document_depth_percents,
        "document_depth_percent_interval_type": args.depth_interval_type,
        "num_concurrent_requests": args.num_concurrent_requests,
        "save_results": args.save_results,
        "save_contexts": args.save_contexts,
        "final_context_length_buffer": args.final_context_length_buffer,
        "print_ongoing_status": args.print_ongoing_status,
        "only_context": args.only_context,
        "enable_dynamic_sleep": args.enable_dynamic_sleep,
        "base_sleep_time": args.base_sleep_time,
        "haystack_dir": args.haystack_dir,
        "eval_model": args.eval_model
    }
    
    # 创建所有测试组合
    test_combinations = list(product(args.model_names, args.case_names))
    total_tests = len(test_combinations)
    
    # 创建进度条
    progress_bar = tqdm(total=total_tests, desc="测试进度")
    progress_lock = threading.Lock()
    
    def update_progress(*args):
        with progress_lock:
            progress_bar.update(1)
    
    # 使用线程池执行测试
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for model_name, case_name in test_combinations:
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
        
        # 收集结果
        results = []
        for future in futures:
            success, message = future.result()
            results.append(message)
    
    progress_bar.close()
    
    # 打印测试结果摘要
    print("\n测试结果摘要:")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()