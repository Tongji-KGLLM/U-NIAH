from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List, Dict, Optional
from langgraph.graph import START, StateGraph
from langchain_anthropic import ChatAnthropic
import os
import logging
import datetime
import json
import asyncio
from tiktoken import encoding_for_model
import tiktoken
import aiofiles
from asyncio import Lock

def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    计算文本的token数量
    """
    encoding = encoding_for_model(model_name)
    return len(encoding.encode(text))

def prepare_context(retrieved_contents: List[Dict], 
                   template: str, 
                   question: str,
                   max_length: int = 8000,
                   context_mode: str = "full",  # 可选值: "full", "half", "topk"
                   token_count_model: str = "gpt-4",
                   reverse: bool = True,
                   topk_context: Optional[int] = None) -> str:
    """
    准备上下文内容
    Args:
        retrieved_contents: 检索到的文档列表
        template: 提示模板
        question: 用户问题
        max_length: 最大token长度
        context_mode: 上下文处理模式 ("full": 完整长度, "half": 一半长度, "topk": 指定数量)
        token_count_model: 计算token的模型名称
        reverse: 是否倒序处理文档（True时检索分数最高的文档会在最后）
        topk_context: 在topk模式下使用的文档数量
    """
    # 根据context_mode确定处理方式
    if context_mode == "topk":
        if topk_context is None:
            raise ValueError("topk模式下，topk_context不能为None")
        docs_to_process = retrieved_contents[:topk_context]
        if reverse:
            return "\n\n".join(doc["content"] for doc in reversed(docs_to_process))
        return "\n\n".join(doc["content"] for doc in docs_to_process)
    
    # 计算模板和问题的token数
    template_tokens = count_tokens(template.format(context="", question=question), token_count_model)
    available_tokens = max_length - template_tokens
    
    # 如果是half模式，将可用token减半
    if context_mode == "half":
        available_tokens = available_tokens // 2
    
    # 获取所有文档内容
    contents = [doc["content"] for doc in retrieved_contents]
    combined_text = ""
    current_tokens = 0
    
    # 根据reverse参数决定处理顺序
    if reverse:
        # reverse=True时，从后往前处理，让高分文档在最后
        for content in contents:
            content_tokens = count_tokens(content, token_count_model)
            if current_tokens + content_tokens <= available_tokens:
                combined_text = content + "\n\n" + combined_text
                current_tokens += content_tokens
            else:
                # 计算剩余可用token数
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 0:
                    encoding = encoding_for_model(token_count_model)
                    tokens = encoding.encode(content)
                    truncated_tokens = tokens[:remaining_tokens]
                    truncated_text = encoding.decode(truncated_tokens)
                    combined_text = truncated_text + "\n\n" + combined_text
                break
    else:
        # reverse=False时，从前往后处理，让高分文档在最前
        for content in contents:
            content_tokens = count_tokens(content, token_count_model)
            if current_tokens + content_tokens <= available_tokens:
                combined_text = combined_text + content + "\n\n"
                current_tokens += content_tokens
            else:
                # 计算剩余可用token数
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 0:
                    encoding = encoding_for_model(token_count_model)
                    tokens = encoding.encode(content)
                    truncated_tokens = tokens[:remaining_tokens]
                    truncated_text = encoding.decode(truncated_tokens)
                    combined_text = combined_text + truncated_text + "\n\n"
                break
            
    return combined_text.strip()

def setup_rag_pipeline(
    file_path: str,
    collection_name: str = None,
    persist_directory: str = None,
    context_mode: str = "full",  
    max_length: int = 8000,
    api_key: str = None,
    base_url: str = None,
    model_name: str = "gpt-4o-mini",
    llm_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    top_k_retrieval: int = None,
    reverse: bool = True,
    top_k_context: Optional[int] = None,
    log_file: str = None,
    detail_result_file: str = None,
    only_build_rag_context: bool = False,
) -> Dict:
    """
    设置RAG管道并处理问题
    Args:
        file_path: 文档路径
        collection_name: 向量存储集合名称
        persist_directory: 向量存储持久化目录
        context_mode: 上下文处理模式 ("full": 完整长度, "half": 一半长度, "topk": 指定数量)
        max_length: 最大token长度
        api_key: OpenAI API密钥
        base_url: API基础URL
        model_name: 模型名称
        embedding_model: 嵌入模型名称
        chunk_size: 文档分块大小
        chunk_overlap: 文档分块重叠大小
        top_k_retrieval: 检索的文档数量
        reverse: 是否将高分文档放在最后
        topk_context: 限制使用的文档数量
    """
    # 设置日志文件路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_file is None:
        log_dir = f"./rag_needle/logs_{context_mode}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"rag_log_{timestamp}.jsonl")
    else:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # 清除现有的处理器，避免重复
    logger.handlers.clear()
    logger.setLevel(logging.INFO)   
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    logger.info(f"日志文件路径: {os.path.abspath(log_file)}")
    
    # 如果未指定collection_name，则使用文件名
    if collection_name is None:
        collection_name = os.path.basename(file_path)  # 提取文件名，包括扩展名
        collection_name = os.path.splitext(collection_name)[0]  # 移除扩展名
        collection_name += "_rag"
    print(f"当前collection_name: {collection_name}")
    
    # 记录初始配置
    config = {
        "file_path": file_path[0:100],
        "model_name": model_name,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "persist_directory": persist_directory,
        "collection_name": collection_name,
        "top_k": top_k_retrieval,
        "log_file": log_file,
        "context_mode": context_mode,
        "max_length": max_length
    }
    logger.info("\n" + "="*50)
    logger.info("初始化RAG Pipeline")
    logger.info("-"*50)
    logger.info(f"配置信息:\n{json.dumps(config, indent=2, ensure_ascii=False)}")
    logger.info("="*50 + "\n")
    # print(f"\n当前配置:\n{json.dumps(config, indent=2, ensure_ascii=False)}\n")
    
    # 初始化模型
    logger.info("正在初始化LLM和Embedding模型...")
    
    # 根据provider选择不同的LLM实现
    if llm_provider.lower() == "openai":
        llm = ChatOpenAI(
            model=model_name, 
            base_url=base_url, 
            api_key=api_key,
            temperature=0
        )
    elif llm_provider.lower() == "anthropic":
        if not base_url.endswith("/anthropic"):
            base_url = os.getenv("ANTHROPIC_API_BASE")
        llm = ChatAnthropic(
            model=model_name, 
            base_url=base_url, 
            api_key=api_key,
            temperature=0
        )
    else:
        raise ValueError(f"不支持的LLM provider: {llm_provider}。目前只支持 'openai' 和 'anthropic'")

    # 记录使用的LLM provider
    config["llm_provider"] = llm_provider
    logger.info(f"使用的LLM provider: {llm_provider}")

    embed_model = OpenAIEmbeddings(
        model=embedding_model, 
        api_key=os.getenv("OPENAI_API_KEY"), 
        base_url=os.getenv("OPENAI_API_BASE"),
    )

    # 加载和分割文档 - 移到向量存储处理之前
    logger.info(f"正在加载文档")
    if file_path.endswith('.txt'):
        loader = TextLoader(file_path)
        docs = loader.load()
        doc = Document(page_content=docs[0].page_content, metadata={"source": file_path})
    else:
        doc = Document(page_content=file_path)

    # 文档分割
    logger.info("正在进行文档分割...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents([doc])
    logger.info(f"文档已分割成 {len(all_splits)} 个子文档")

    # 设置top_k_retrieval
    if top_k_retrieval is None:
        top_k_retrieval = len(all_splits)
    logger.info(f"上下文检索top_k_retrieval: {top_k_retrieval}")

    # 设置向量存储路径
    if persist_directory is None:
        persist_directory = "./chroma_rag_db/"+collection_name
    else:
        persist_directory = persist_directory+"/"+collection_name

    # 检查并清理现有的向量存储
    if os.path.exists(persist_directory):
        logger.info(f"发现已存在的向量存储目录: {persist_directory}")
        try:
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embed_model,
                persist_directory=persist_directory,
            )
            logger.info("成功加载已存在的向量存储")
        except Exception as e:
            logger.error(f"加载现有向量存储失败: {str(e)}")
            logger.info("删除并重建向量存储...")
            import shutil
            shutil.rmtree(persist_directory)
            # 创建新的向量存储
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embed_model,
                persist_directory=persist_directory,
            )
            # 添加文档到向量存储
            document_ids = vector_store.add_documents(documents=all_splits)
            logger.info(f"已添加 {len(document_ids)} 个文档到新创建的向量存储")
    else:
        logger.info("创建新的向量存储...")
        # 创建新的向量存储
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embed_model,
            persist_directory=persist_directory,
        )
        # 添加文档到向量存储
        document_ids = vector_store.add_documents(documents=all_splits)
        logger.info(f"已添加 {len(document_ids)} 个文档到向量存储")

    
    # 去掉IDK的设置
    # If you don't know the answer, just say that you don't know, don't try to make up an answer. 

    # 设置RAG提示模板
    template = """Use the following pieces of context to answer the question at the end.
    Keep the answer as concise as possible.
    Don't give information outside the document or repeat your findings
    Context:
    {context}

    Question: 
    {question}

    Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    if detail_result_file is None:
        # 如果没有指定详细结果文件路径，使用默认路径
        if context_mode == "full":
            detail_dir = "./rag_needle/detail_results_full"
        else:
            detail_dir = "./rag_needle/detail_results"
        os.makedirs(detail_dir, exist_ok=True)
        detail_result_file = os.path.join(detail_dir, f"rag_detail_{timestamp}.jsonl")
    else:
        # 如果指定了详细结果文件路径，确保目录存在
        os.makedirs(os.path.dirname(detail_result_file), exist_ok=True)
   

    # 定义检索和生成函数
    def retrieve(state: Dict):
        logger.info("\n" + "-"*50)
        logger.info("开始文档检索")
        logger.info("-"*50)
        retrieved_docs = vector_store.similarity_search(state["question"], k=top_k_retrieval)
        retrieved_contents = [{"content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs]
        
        # 准备上下文
        docs_content = prepare_context(
            retrieved_contents=retrieved_contents,
            template=template,
            question=state["question"],
            max_length=max_length,
            context_mode=context_mode,  # 使用新的context_mode参数
            token_count_model="gpt-3.5-turbo",
            reverse=reverse,
            topk_context=top_k_context
        )
        
        return {
            "context": retrieved_docs, 
            "retrieved_contents": retrieved_contents,
            "question": state["question"],
            "prepared_context": docs_content
        }

    def generate(state: Dict):
        if only_build_rag_context:
            # 构建prompt但不执行模型调用
            messages = custom_rag_prompt.invoke({
                "question": state["question"],
                "context": state["prepared_context"]
            })
            return {
                "answer": None,
                "retrieved_contents": state["retrieved_contents"],
                "question": state["question"],
                "prepared_context": state["prepared_context"],
                "prompt_messages": messages.to_string()  # 添加prompt信息
            }
            
        logger.info("\n" + "-"*50)
        logger.info("开始生成回答")
        logger.info("-"*50)
        
        # 使用retrieve阶段已经准备好的上下文
        messages = custom_rag_prompt.invoke({
            "question": state["question"],
            "context": state["prepared_context"]
        })
        response = llm.invoke(messages)
        logger.info(f"生成的回答:\n{response.content}")
        logger.info("-"*50 + "\n")
        return {
            "answer": response.content,
            "retrieved_contents": state["retrieved_contents"],
            "question": state["question"],
            "prompt_messages": messages.to_string()  # 在返回结果中也包含prompt
        }

    # 构建图
    graph_builder = StateGraph(Dict).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

  
    
    return {
        "graph": graph,
        "vector_store": vector_store,
        "logger": logger,
        "config": config
    }

def query_rag(
    question: str,
    rag_pipeline: Dict,
    detail_result_file: str
) -> Dict:
    """
    使用RAG管道处理问题
    """
    logger = rag_pipeline["logger"]
    graph = rag_pipeline["graph"]
    config = rag_pipeline["config"]
    
    # 记录查询信息
    query_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "config": config
    }
    logger.info("\n" + "="*50)
    logger.info("新的查询请求")
    logger.info("-"*50)
    logger.info(f"查询信息:\n{json.dumps(query_info, indent=2, ensure_ascii=False)}")
    logger.info("="*50 + "\n")
    
    # 执行查询
    response = graph.invoke({"question": question})
    
    # 记录完整的查询结果
    query_result = {
        **query_info,
        "answer": response["answer"],
        "retrieved_documents": response["retrieved_contents"],
        "prompt_messages": response.get("prompt_messages", "")  # 添加prompt信息
    }
    logger.info("\n" + "="*50)
    logger.info("查询完成")
    logger.info("-"*50)
    logger.info(f"查询结果:\n{json.dumps(query_result, indent=2, ensure_ascii=False)}")
    logger.info("="*50 + "\n")
    
    # 在记录完整查询结果后，添加保存到文件的逻辑
    os.makedirs(os.path.dirname(detail_result_file), exist_ok=True)
    with open(detail_result_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(query_result, ensure_ascii=False) + '\n')
    return response

async def query_rag_async(
    question: str,
    rag_pipeline: Dict,
    detail_result_file: str = None
) -> Dict:
    """
    异步使用RAG管道处理问题
    """
    logger = rag_pipeline["logger"]
    graph = rag_pipeline["graph"]
    config = rag_pipeline["config"]
    vector_store = rag_pipeline["vector_store"]
    
    # 创建一个新的锁，绑定到当前事件循环
    file_write_lock = Lock()
    
    # 记录查询信息
    query_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "config": config
    }
    logger.info("\n" + "="*50)
    logger.info("新的异步查询请求")
    logger.info("-"*50)
    logger.info(f"查询信息:\n{json.dumps(query_info, indent=2, ensure_ascii=False)}")
    logger.info("="*50 + "\n")
    
    try:
        # 执行异步查询
        response = await graph.ainvoke({"question": question})
        
        # 记录完整的查询结果
        query_result = {
            **query_info,
            "answer": response["answer"],
            "retrieved_documents": response["retrieved_contents"],
            "prompt_messages": response.get("prompt_messages", "")
        }
        logger.info("\n" + "="*50)
        logger.info("异步查询完成")
        logger.info("-"*50)
        logger.info("="*50 + "\n")
        
        # 使用异步锁和异步文件写入（如果提供了文件路径）
        if detail_result_file:
            os.makedirs(os.path.dirname(detail_result_file), exist_ok=True)
            async with file_write_lock:
                async with aiofiles.open(detail_result_file, 'a', encoding='utf-8') as f:
                    await f.write(json.dumps(query_result, ensure_ascii=False) + '\n')
        
        return response
        
    except Exception as e:
        logger.error(f"RAG查询失败: {str(e)}")
        # 向上层抛出特定的异常,包含重要信息
        raise RAGQueryError(
            error=str(e),
            persist_directory=config["persist_directory"],
            collection_name=config["collection_name"],
            config=config
        ) from e

# 添加自定义异常类
class RAGQueryError(Exception):
    """RAG查询失败时的自定义异常"""
    def __init__(self, error: str, persist_directory: str, collection_name: str, config: dict):
        self.error = error
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.config = config
        super().__init__(f"RAG查询失败: {error}")

