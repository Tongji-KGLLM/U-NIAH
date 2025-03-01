import os
import pkg_resources
import tiktoken

from operator import itemgetter
from typing import Optional

from anthropic import AsyncAnthropic
from anthropic import Anthropic as AnthropicModel
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate

from .model import ModelProvider

class Anthropic(ModelProvider):
    DEFAULT_MODEL_KWARGS: dict = dict(
        max_tokens=400,
        temperature=0
    )

    def __init__(self,
                 model_name: str = "claude-3-5-haiku",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 api_key: str = None,
                 base_url: str = None
                 ):
        """
        :param model_name: The name of the model. Default is 'claude'.
        :param model_kwargs: Model configuration. Default is {max_tokens_to_sample: 300, temperature: 0}
        """

        if "claude" not in model_name:
            raise ValueError("If the model provider is 'anthropic', the model name must include 'claude'. See https://docs.anthropic.com/claude/reference/selecting-a-model for more details on Anthropic models")
        

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.api_key = api_key
        self.base_url = base_url
        self.model = AsyncAnthropic(api_key=self.api_key,base_url=self.base_url)
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

        resource_path = pkg_resources.resource_filename('needlehaystack', 'providers/Anthropic_prompt.txt')

        # Generate the prompt structure for the Anthropic model
        # Replace the following file with the appropriate prompt structure
        with open(resource_path, 'r') as file:
            self.prompt_structure = file.read()

    async def evaluate_model(self, prompt: str) -> str:
        # 将字符串 prompt 转换为 Anthropic 期望的消息格式
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = await self.model.messages.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.model_kwargs.get('max_tokens', 400),
            temperature=self.model_kwargs.get('temperature', 0)
        )
        return response.content[0].text

    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        return self.prompt_structure.format(
            retrieval_question=retrieval_question,
            context=context)
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
    
    # def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
    #     # Assuming you have a different decoder for Anthropic
    #     return self.tokenizer.decode(tokens[:context_length])
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None, preserve_sentence: bool = False) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode.
                If not provided, decodes all tokens.
            preserve_sentence (bool, optional): If True, ensures the decoded text ends with a complete sentence
                within the context length. Defaults to False.

        Returns:
            str: The decoded text string.
        """
        if context_length is None:
            return self.tokenizer.decode(tokens)
        
        truncated_text = self.tokenizer.decode(tokens[:context_length])
        
        if not preserve_sentence:
            return truncated_text.rstrip()
        
        # 查找最后一个完整句子的结束位置
        sentence_endings = ['. ', '! ', '? ', '。', '！', '？']
        last_end = -1
        
        for ending in sentence_endings:
            pos = truncated_text.rfind(ending)
            last_end = max(last_end, pos)
        
        # 如果找到了句子结束符，在那里截断
        if last_end != -1:
            return truncated_text[:last_end + 1].rstrip()
        
        # 如果没找到句子结束符，返回原始截断的文本
        return truncated_text.rstrip()
    
    def get_langchain_runnable(self, context: str) -> str:
        """
        Creates a LangChain runnable that constructs a prompt based on a given context and a question, 
        queries the Anthropic model, and returns the model's response. This method leverages the LangChain 
        library to build a sequence of operations: extracting input variables, generating a prompt, 
        querying the model, and processing the response.

        Args:
            context (str): The context or background information relevant to the user's question. 
            This context is provided to the model to aid in generating relevant and accurate responses.

        Returns:
            str: A LangChain runnable object that can be executed to obtain the model's response to a 
            dynamically provided question. The runnable encapsulates the entire process from prompt 
            generation to response retrieval.

        Example:
            To use the runnable:
                - Define the context and question.
                - Execute the runnable with these parameters to get the model's response.
        """

        template = """Human: You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
        <document_content>
        {context} 
        </document_content>
        Here is the user question:
        <question>
         {question}
        </question>
        Don't give information outside the document or repeat your findings.
        Assistant: Here is the most relevant information in the documents:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        # Create a LangChain runnable
        model = ChatAnthropic(temperature=0, model=self.model_name,base_url=self.base_url,api_key=self.api_key,**self.model_kwargs)
        chain = ( {"context": lambda x: context,
                  "question": itemgetter("question")} 
                | prompt 
                | model 
                )
        return chain