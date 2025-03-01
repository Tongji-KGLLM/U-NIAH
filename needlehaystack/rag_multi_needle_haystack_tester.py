import asyncio
import glob
import json
import os
import time
from asyncio import Semaphore
from datetime import datetime, timezone

import numpy as np

from .evaluators import Evaluator
from .llm_needle_haystack_tester import LLMNeedleHaystackTester
from .providers import ModelProvider
from .RAG_support.NaiveRAG import query_rag_async, setup_rag_pipeline,RAGQueryError


class RAGMultiNeedleHaystackTester(LLMNeedleHaystackTester):
    """
    Extends LLMNeedleHaystackTester to support testing with multiple needles in the haystack.
    
    Attributes:
        needles (list): A list of needles (facts) to insert into the haystack (context).
        model_to_test (ModelProvider): The model being tested.
        evaluator (Evaluator): The evaluator used to assess the model's performance.
        print_ongoing_status (bool): Flag to print ongoing status messages.
        eval_set (str): The evaluation set identifier.
    """
    def __init__(self, *args, 
                 needles=[], 
                 model_to_test: ModelProvider = None,
                 evaluator: Evaluator = None, 
                 print_ongoing_status = True,
                 eval_set = "multi-needle-eval-sf",
                 api_key = None,
                 base_url = None,
                 embedding_model = "text-embedding-3-small",
                 chunk_size = 600,
                 chunk_overlap = 100,
                 top_k_retrieval = 50,
                 context_mode = "full",
                 reverse = True,
                 top_k_context = None,
                 case_name = None,
                 haystack_dir = None,
                 llm_provider = "openai",
                 only_build_rag_context = False,
                 enable_dynamic_sleep=False,  # 是否启用动态休眠
                 base_sleep_time=2,          # 基础休眠时间（秒）
                 **kwargs):

        # 添加验证逻辑
        if context_mode != "topk" and top_k_context is not None:
            raise ValueError("top_k_context 只能在 context_mode='topk' 时设置，其他模式必须为 None")

        super().__init__(*args, model_to_test=model_to_test, **kwargs)
        self.needles = needles
        self.evaluator = evaluator
        self.evaluation_model = evaluator  
        self.model_to_test = model_to_test
        self.eval_set = eval_set
        self.model_name = self.model_to_test.model_name
        self.print_ongoing_status = print_ongoing_status
        self.insertion_percentages = []
        self.api_key = api_key
        self.base_url = base_url
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_retrieval = top_k_retrieval
        self.context_mode = context_mode
        self.reverse = reverse
        self.top_k_context = top_k_context
        self.case_name = case_name
        self.haystack_dir = haystack_dir
        self.llm_provider = llm_provider
        self.only_build_rag_context = only_build_rag_context
        self.enable_dynamic_sleep = enable_dynamic_sleep
        self.base_sleep_time = base_sleep_time

    async def insert_needles(self, context, depth_percent, context_length):
        """
        Inserts multiple needles (specific facts or pieces of information) into the original context string at 
        designated depth percentages, effectively distributing these needles throughout the context. This method 
        is designed to test a model's ability to retrieve specific information (needles) from a larger body of text 
        (haystack) based on the placement depth of these needles.

        The method first encodes the context and each needle into tokens to calculate their lengths in tokens. 
        It then adjusts the context length to accommodate the final buffer length. This is crucial for ensuring 
        that the total token count (context plus needles) does not exceed the maximum allowable context length, 
        which might otherwise lead to information being truncated.

        This approach calculates the initial insertion point for the first needle as before but then calculates even 
        spacing for the remaining needles based on the remaining context length. It ensures that needles are 
        distributed as evenly as possible throughout the context after the first insertion. 
        
        Args:
            context (str): The original context string.
            depth_percent (float): The depth percent at which to insert the needles.
            context_length (int): The total length of the context in tokens, adjusted for final buffer.
        
        Returns:
            str: The new context with needles inserted.
        """
        tokens_context = self.model_to_test.encode_text_to_tokens(context)
        tokens_context = self.ensure_complete_sentence(tokens_context)
        context_length -= self.final_context_length_buffer

        # Calculate the total length of all needles in tokens
        total_needles_length = sum(len(self.model_to_test.encode_text_to_tokens(needle)) for needle in self.needles)

        # Ensure context length accounts for needles
        if len(tokens_context) + total_needles_length > context_length:
            tokens_context = tokens_context[:context_length - total_needles_length]
            tokens_context = self.ensure_complete_sentence(tokens_context)

        
        # To evenly distribute the needles, we calculate the intervals they need to be inserted.
        depth_percent_interval = (100 - depth_percent) / len(self.needles)
        
        # Reset the insertion percentages list for the current context
        self.insertion_percentages = []

        # Insert needles at calculated points
        for needle in self.needles:

            tokens_needle = self.model_to_test.encode_text_to_tokens(needle)

            if depth_percent == 100:
                # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
                tokens_context = tokens_context + tokens_needle
            else:
                # Go get the position (in terms of tokens) to insert your needle
                insertion_point = int(len(tokens_context) * (depth_percent / 100))

                # tokens_new_context represents the tokens before the needle
                tokens_new_context = tokens_context[:insertion_point]

                # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
                period_tokens = self.model_to_test.encode_text_to_tokens('.')
                
                # Then we iteration backwards until we find the first period
                while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                    insertion_point -= 1
                    tokens_new_context = tokens_context[:insertion_point]
                    
                # Insert the needle into the context at the found position
                tokens_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]

                # Log 
                insertion_percentage = (insertion_point / len(tokens_context)) * 100
                self.insertion_percentages.append(insertion_percentage)
                print(f"Inserted '{needle}' at {insertion_percentage:.2f}% of the context, total length now: {len(tokens_context)} tokens")
                
                # Adjust depth for next needle
                depth_percent += depth_percent_interval  

        new_context = self.model_to_test.decode_tokens(tokens_context)
        return new_context

    def encode_and_trim(self, context, context_length):
        """
        Encodes the context to tokens and trims it to the specified length.
        
        Args:
            context (str): The context to encode and trim.
            context_length (int): The desired length of the context in tokens.
        
        Returns:
            str: The encoded and trimmed context.
        """
        tokens = self.model_to_test.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.model_to_test.decode_tokens(tokens, context_length)
        return context

    async def generate_context(self, context_length, depth_percent):
        """
        Generates a context of a specified length and inserts needles at given depth percentages.
        
        Args:
            context_length (int): The total length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.
        
        Returns:
            str: The context with needles inserted.
        """
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context = await self.insert_needles(context, depth_percent, context_length)
        return context
    
    def calculate_dynamic_sleep_time(self, context_length):
        """
        根据上下文长度计算动态休眠时间
        Args:
            context_length (int): 上下文长度（token数）
        Returns:
            float: 需要等待的秒数
        """
        base_sleep = self.base_sleep_time
        
        # 根据上下文长度分级设置等待时间倍数
        if context_length <= 4000:
            multiplier = 1
        elif context_length <= 8000:
            multiplier = 3
        elif context_length <= 16000:
            multiplier = 5
        elif context_length <= 32000:
            multiplier = 8
        elif context_length <= 64000:
            multiplier = 12
        elif context_length <= 128000:
            multiplier = 20
        else:
            multiplier = 25
        
        sleep_time = base_sleep * multiplier
        # 设置最大休眠时间为60秒
        return min(sleep_time, 60)
    


    async def evaluate_and_log(self, context_length, depth_percent):
        """
        Evaluates the model's performance with the generated context and logs the results.
        
        Args:
            context_length (int): The length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.
        """
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                return

        # Go generate the required length context and place your needle statement in
        context = await self.generate_context(context_length, depth_percent)

        # 构建包含更多参数信息的文件夹名
        params_str = f"{self.case_name}"
        params_str += f"_{self.context_mode}"
        params_str += f"{self.top_k_context if self.top_k_context else ''}"
        params_str += f"_{"rev" if self.reverse else "norm"}"
        params_str += f"_needles_{len(self.needles)}"
   
   
        # context_file_location = f'{self.model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'
        
        model_name = self.model_name.replace(".", "_")
        model_name = model_name.replace("/", "_")
        print("model_name", model_name)

        # 确保 context_file_location 也安全处理模型名称
        safe_model_name = self.model_name.replace(".", "_").replace("/", "_")
        context_file_location = f'{safe_model_name}_len_{context_length}_depth_{int(depth_percent*100)}'
        collection_name = f'{self.haystack_dir[0:5]}_{self.case_name[0:5]}_len_{context_length}_depth_{int(depth_percent*100)}_RAG'


        test_start_time = time.time()

        # LangSmith
        ## TODO: Support for other evaluators 
        if self.evaluator.__class__.__name__ == "LangSmithEvaluator":  
            print("EVALUATOR: LANGSMITH")
            chain = self.model_to_test.get_langchain_runnable(context)
            self.evaluator.evaluate_chain(chain, context_length, depth_percent, self.model_to_test.model_name, self.eval_set, len(self.needles), self.needles, self.insertion_percentages)
            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time

        else:
            print("EVALUATOR: OpenAI Model")


            
            # # Prepare your message to send to the model you're going to evaluate
            # prompt = self.model_to_test.generate_prompt(context, self.retrieval_question)
            # # Go see if the model can answer the question to pull out your random fact
            # response = await self.model_to_test.evaluate_model(prompt)


            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_name = f"log_{context_file_location}_reverse_{self.reverse}_ragfull_{self.context_mode}_topk_{self.top_k_context}_{self.case_name}_{timestamp}.jsonl"

    
            rag_pipeline = setup_rag_pipeline(
                file_path=context,
                collection_name=collection_name,
                persist_directory=f"./rag_multi_needle/chroma_db/{self.case_name}",
                context_mode=self.context_mode,
                max_length=context_length,
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name,
                embedding_model=self.embedding_model,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                top_k_retrieval=self.top_k_retrieval,
                reverse=self.reverse,
                top_k_context=self.top_k_context,
                llm_provider=self.llm_provider,
                log_file=f"./rag_multi_needle/logs/{self.case_name}/{params_str}/{safe_model_name}/{log_file_name}",
                only_build_rag_context=self.only_build_rag_context
            )
            
            # 在开始时就创建所需的文件夹路径
            base_path = './rag_multi_needle/contexts'
            model_folder = f'{self.haystack_dir}_{self.case_name}/{params_str}/{self.model_name.replace(".", "_")}'
            context_folder = os.path.join(base_path, model_folder)
            
            # 确保文件夹存在
            if not os.path.exists(context_folder):
                os.makedirs(context_folder, exist_ok=True)

            if not self.only_build_rag_context:
                # 使用新的动态休眠逻辑
                if self.enable_dynamic_sleep:
                    sleep_time = self.calculate_dynamic_sleep_time(context_length)
                    print(f"Dynamic sleeping for {sleep_time:.2f} seconds based on context length {context_length}")
                    await asyncio.sleep(sleep_time)
                    
                # 添加重试逻辑
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        response = await query_rag_async(
                            question=self.retrieval_question,
                            rag_pipeline=rag_pipeline,
                            detail_result_file=f"./rag_multi_needle/detail_results/{self.haystack_dir}_{self.case_name}/{params_str}/{safe_model_name}/{log_file_name}"
                        )
                        break  # 如果成功,跳出循环
                        
                    except RAGQueryError as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            # 最后一次重试时,尝试重建向量存储
                            try:
                                # 删除整个collection目录
                                collection_dir = e.persist_directory
                                if os.path.exists(collection_dir):
                                    print(f"删除向量存储目录: {collection_dir}")
                                    import shutil
                                    shutil.rmtree(collection_dir)
                                
                                # 重新构建RAG pipeline
                                rag_pipeline = setup_rag_pipeline(
                                    file_path=context,
                                    collection_name=e.collection_name,
                                    persist_directory=os.path.dirname(e.persist_directory),
                                    context_mode=self.context_mode,
                                    max_length=context_length,
                                    api_key=self.api_key,
                                    base_url=self.base_url,
                                    model_name=self.model_name,
                                    embedding_model=self.embedding_model,
                                    chunk_size=self.chunk_size,
                                    chunk_overlap=self.chunk_overlap,
                                    top_k_retrieval=self.top_k_retrieval,
                                    reverse=self.reverse,
                                    top_k_context=self.top_k_context,
                                    llm_provider=self.llm_provider,
                                    detail_result_file=f"./rag_multi_needle/detail_results/{self.haystack_dir}_{self.case_name}/{params_str}/{safe_model_name}/{log_file_name}"
                                )
                                
                                # 使用新的pipeline重试
                                response = await query_rag_async(
                                    question=self.retrieval_question,
                                    rag_pipeline=rag_pipeline,
                                    detail_result_file=f"./rag_multi_needle/detail_results/{self.haystack_dir}_{self.case_name}/{params_str}/{safe_model_name}/{log_file_name}"
                                )
                                break
                                
                            except Exception as rebuild_error:
                                print(f"重建向量存储失败: {str(rebuild_error)}")
                                raise  # 重建失败时直接抛出异常
                        
                        print(f"查询失败,第 {retry_count} 次重试: {str(e)}")
                        if self.enable_dynamic_sleep:
                            sleep_time = self.calculate_dynamic_sleep_time(context_length)
                            print(f"等待 {sleep_time:.2f} 秒后重试...")
                            await asyncio.sleep(sleep_time)
                        else:
                            await asyncio.sleep(5)
                
                response_content = response["answer"]
                score = self.evaluation_model.evaluate_response(response_content)

                test_end_time = time.time()
                test_elapsed_time = test_end_time - test_start_time

                # 保存结果
                results = {
                    'model': self.model_to_test.model_name,
                    'context_length': int(context_length),
                    'depth_percent': float(depth_percent),
                    'version': self.results_version,
                    'needles': self.needles,
                    'model_response': response_content,
                    'score': score,
                    'test_duration_seconds': test_elapsed_time,
                    'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
                }
                self.testing_results.append(results)

                if self.print_ongoing_status:
                    print(f"-- Test Summary -- ")
                    print(f"Duration: {test_elapsed_time:.1f} seconds")
                    print(f"Context: {context_length} tokens")
                    print(f"Depth: {depth_percent}%")
                    print(f"Score: {score}")
                    print(f"Response: {response_content}\n")

            else:
                # 对 only_build_rag_context 模式也添加重试逻辑
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        response = await query_rag_async(
                            question=self.retrieval_question,
                            rag_pipeline=rag_pipeline,
                            detail_result_file=f"./rag_multi_needle/detail_results/{self.haystack_dir}_{self.case_name}/{params_str}/{self.model_name.replace('.', '_')}/{log_file_name}"
                        )
                        break  # 如果成功，跳出循环
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            print(f"最终失败，已重试 {max_retries} 次: {str(e)}")
                            raise  # 重试次数用完后，抛出异常
                        print(f"查询失败，正在进行第 {retry_count} 次重试: {str(e)}")
                        await asyncio.sleep(5)  # 等待5秒后重试

                prepared_context = response.get("prepared_context")
                prompt_messages = response.get("prompt_messages")
                
                test_end_time = time.time()
                test_elapsed_time = test_end_time - test_start_time
                
                # 保存构建的上下文和prompt
                if self.save_contexts:
                    # 保存prepared context
                    context_file_path = os.path.join(
                        context_folder, 
                        f'{context_file_location}_prepared_context.txt'
                    )
                    try:
                        with open(context_file_path, 'w', encoding='utf-8') as f:
                            f.write(prepared_context)
                        print(f"Successfully wrote prepared context to: {context_file_path}")
                    except Exception as e:
                        print(f"Error writing prepared context file: {e}")

                    # 保存prompt messages
                    prompt_file_path = os.path.join(
                        context_folder, 
                        f'{context_file_location}_prompt.txt'
                    )
                    try:
                        with open(prompt_file_path, 'w', encoding='utf-8') as f:
                            f.write(prompt_messages)
                        print(f"Successfully wrote prompt to: {prompt_file_path}")
                    except Exception as e:
                        print(f"Error writing prompt file: {e}")

                if self.print_ongoing_status:
                    print(f"-- Context Build Summary -- ")
                    print(f"Duration: {test_elapsed_time:.1f} seconds")
                    print(f"Context: {context_length} tokens")
                    print(f"Depth: {depth_percent}%")
                    print(f"Context saved to: {context_file_path}")
                    print(f"Prompt saved to: {prompt_file_path}\n")

            # 保存原始上下文
            if self.save_contexts:
                context_file_path = os.path.join(
                    context_folder, 
                    f'{context_file_location}_context.txt'
                )
                try:
                    print(f"Attempting to write to: {context_file_path}")
                    with open(context_file_path, 'w', encoding='utf-8') as f:
                        f.write(context)
                    print(f"Successfully wrote context to: {context_file_path}")
                except Exception as e:
                    print(f"Error writing context file: {e}")
                    print(f"Current working directory: {os.getcwd()}")

            # 保存结果（只在非only_build_rag_context模式下）
            if self.save_results and not self.only_build_rag_context:
                results['file_name'] = context_file_location
                base_path = './rag_multi_needle/results'
                model_folder = f'{self.haystack_dir}_{self.case_name}/{params_str}/{safe_model_name}'
                results_folder = os.path.join(base_path, model_folder)
                
                if not os.path.exists(results_folder):
                    os.makedirs(results_folder, exist_ok=True)
                
                results_file_path = os.path.join(
                    results_folder, 
                    f'{context_file_location}_results.json'
                )
                
                try:
                    print(f"Attempting to write results to: {results_file_path}")
                    with open(results_file_path, 'w') as f:
                        json.dump(results, f)
                    print(f"Successfully wrote results to: {results_file_path}")
                except Exception as e:
                    print(f"Error writing results file: {e}")
                    print(f"Current working directory: {os.getcwd()}")

            if self.seconds_to_sleep_between_completions:
                await asyncio.sleep(self.seconds_to_sleep_between_completions)

    async def bound_evaluate_and_log(self, sem, *args):
            async with sem:
                await self.evaluate_and_log(*args)

    async def run_test(self):
        sem = Semaphore(self.num_concurrent_requests)

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(sem, context_length, depth_percent)
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needles: {self.needles}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())

    def ensure_complete_sentence(self, tokens_context):
        """确保上下文在完整句子处截断"""
        # 将tokens转换回文本
        context = self.model_to_test.decode_tokens(tokens_context, preserve_sentence=True)
        
        # 查找最后一个完整句子的结束位置
        sentence_endings = ['. ', '! ', '? ']  # 定义句子结束标记
        last_end = -1
        for ending in sentence_endings:
            pos = context.rfind(ending)
            last_end = max(last_end, pos)
        
        if last_end == -1:
            return tokens_context  # 如果找不到句子结束标记，返回原始tokens
        
        # 在最后一个完整句子处截断文本
        truncated_context = context[:last_end + 2]  # +2 是为了包含句号和空格
        
        # 将截断后的文本转换回tokens
        return self.model_to_test.encode_text_to_tokens(truncated_context)
