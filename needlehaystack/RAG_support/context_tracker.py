from typing import List, Dict, Optional, Tuple
from.NaiveRAG import count_tokens, encoding_for_model

default_template = """Use the following pieces of context to answer the question at the end.
    Keep the answer as concise as possible.
    Don't give information outside the document or repeat your findings
    Context:
    {context}

    Question: 
    {question}

    Answer:"""


def track_used_context(retrieved_contents: List[Dict],
                      template: str = default_template,
                      question: str=None,
                      max_length: int = 8000,
                      context_mode: str = "full",
                      token_count_model: str = "gpt-4",
                      reverse: bool = True,
                      topk_context: Optional[int] = None) -> List[Dict]:
    """
    跟踪并返回在不同设定下最终被使用的上下文内容
    Args:
        retrieved_contents: 检索到的文档列表
        template: 提示模板
        question: 用户问题
        max_length: 最大token长度
        context_mode: 上下文处理模式 ("full": 完整长度, "half": 一半长度, "topk": 指定数量)
        token_count_model: 计算token的模型名称
        reverse: 是否倒序处理文档（True时检索分数最高的文档会在最后）
        topk_context: 在topk模式下使用的文档数量
    Returns:
        List[Dict]: 包含所有被使用的文档内容的列表，每个文档包含原始内容和是否被截断的信息
    """
    used_contents = []

    # 处理topk模式
    if context_mode == "topk":
        if topk_context is None:
            raise ValueError("topk模式下，topk_context不能为None")
        docs_to_process = retrieved_contents[:topk_context]
        if reverse:
            docs_to_process = list(reversed(docs_to_process))
        
        for doc in docs_to_process:
            used_contents.append({
                "content": doc["content"],
                "truncated": False,
                "original_score": doc.get("score", None),
                "original_index": retrieved_contents.index(doc)
            })
        return used_contents

    # 计算模板和问题的token数
    template_tokens = count_tokens(template.format(context="", question=question), token_count_model)
    available_tokens = max_length - template_tokens

    # 如果是half模式，将可用token减半
    if context_mode == "half":
        available_tokens = available_tokens // 2

    # 获取所有文档内容
    contents = [doc["content"] for doc in retrieved_contents]
    current_tokens = 0

    # 根据reverse参数决定处理顺序
    if reverse:
        # reverse=True时，从后往前处理
        for i, content in enumerate(contents):
            content_tokens = count_tokens(content, token_count_model)
            if current_tokens + content_tokens <= available_tokens:
                used_contents.insert(0, {
                    "content": content,
                    "truncated": False,
                    "original_score": retrieved_contents[i].get("score", None),
                    "original_index": i
                })
                current_tokens += content_tokens
            else:
                # 处理需要截断的文档
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 0:
                    encoding = encoding_for_model(token_count_model)
                    tokens = encoding.encode(content)
                    truncated_tokens = tokens[:remaining_tokens]
                    truncated_text = encoding.decode(truncated_tokens)
                    used_contents.insert(0, {
                        "content": truncated_text,
                        "truncated": True,
                        "original_content": content,
                        "original_score": retrieved_contents[i].get("score", None),
                        "original_index": i
                    })
                break
    else:
        # reverse=False时，从前往后处理
        for i, content in enumerate(contents):
            content_tokens = count_tokens(content, token_count_model)
            if current_tokens + content_tokens <= available_tokens:
                used_contents.append({
                    "content": content,
                    "truncated": False,
                    "original_score": retrieved_contents[i].get("score", None),
                    "original_index": i
                })
                current_tokens += content_tokens
            else:
                # 处理需要截断的文档
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 0:
                    encoding = encoding_for_model(token_count_model)
                    tokens = encoding.encode(content)
                    truncated_tokens = tokens[:remaining_tokens]
                    truncated_text = encoding.decode(truncated_tokens)
                    used_contents.append({
                        "content": truncated_text,
                        "truncated": True,
                        "original_content": content,
                        "original_score": retrieved_contents[i].get("score", None),
                        "original_index": i
                    })
                break

    return used_contents