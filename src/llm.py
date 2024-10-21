from openai import OpenAI
import re
import queue
import os
import json
import time

class Qwen_API:
    def __init__(self, api_key = None, base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        api_key = api_key if api_key else os.getenv("DASHSCOPE_API_KEY")
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url,
        )
       
    def infer(self, user_input, user_messages, chat_mode):
        # prompt 
        if len(user_messages) == 1:
            if chat_mode == "单轮对话 (一次性回答问题)":
                user_messages[0]['content'] = '你负责为一个语音聊天系统生成对话文本输出，使用长度接近的短句，确保语气情感丰富、友好，并且响应迅速以保持用户的参与感。请你以“好的”、“没问题”、“明白了”等短句作为回复的开头。'
            else:
                with open('src/prompt.txt', 'r') as f:
                    user_messages[0]['content'] = f.read()
        user_messages.append({'role': 'user', 'content': user_input})
        print(user_messages)

        completion = self.client.chat.completions.create(
            model="qwen-turbo",
            messages=user_messages
        )
        print(completion)
        chat_response = completion.choices[0].message.content
        user_messages.append({'role': 'assistant', 'content': chat_response})

        if len(user_messages) > 10:
            user_messages.pop(0)
  
        print(f'[Qwen API] {chat_response}')
        return chat_response, user_messages


    def infer_stream(self, user_input, user_messages, llm_queue, chunk_size, chat_mode):
        print(f"[LLM] User input: {user_input}")
        time_cost = []
        start_time = time.time()
        # prompt 
        if len(user_messages) == 1:
            if chat_mode == "单轮对话 (一次性回答问题)":
                user_messages[0]['content'] = '你负责为一个语音聊天系统生成对话文本输出，使用短句，确保语气情感丰富、友好，并且响应迅速以保持用户的参与感。请你以“好的”、“没问题”、“明白了”、“当然可以”等短句作为回复的开头。'
            else:
                with open('src/prompt.txt', 'r') as f:
                    user_messages[0]['content'] = f.read()
        print(f"[LLM] user_messages: {user_messages}")
        user_messages.append({'role': 'user', 'content': user_input})
        completion = self.client.chat.completions.create(
            model="qwen-turbo",
            messages=user_messages,
            stream=True
        )
        
        chat_response = ""
        buffer = ""
        sentence_buffer = ""
        sentence_split_pattern = re.compile(r'(?<=[,;.!?，；：。:！？》、”])')
        fp_flag = True
        print("[LLM] Start LLM streaming...")
        for chunk in completion:
            chat_response_chunk = chunk.choices[0].delta.content
            chat_response += chat_response_chunk
            buffer += chat_response_chunk

            sentences = sentence_split_pattern.split(buffer)
            
            if not sentences:
                continue
            
            for i in range(len(sentences) - 1):
                sentence = sentences[i].strip()
                sentence_buffer += sentence

                if fp_flag or len(sentence_buffer) >= chunk_size:
                    llm_queue.put(sentence_buffer)
                    time_cost.append(round(time.time()-start_time, 2))
                    start_time = time.time()
                    print(f"[LLM] Put into queue: {sentence_buffer}")
                    sentence_buffer = ""
                    fp_flag = False
            
            buffer = sentences[-1].strip()

        sentence_buffer += buffer
        if sentence_buffer:
            llm_queue.put(sentence_buffer)
            print(f"[LLM] Put into queue: {sentence_buffer}")

        llm_queue.put(None)
        
        user_messages.append({'role': 'assistant', 'content': chat_response})
        if len(user_messages) > 10:
            user_messages.pop(0)
        
        print(f"[LLM] Response: {chat_response}\n")
        
        return chat_response, user_messages, time_cost

    # def infer_stream_split(self, user_input, user_messages, llm_queue):
    #     print(f"[LLM] User input: {user_input}")
    #     print(user_messages)
    #     user_messages.append({'role': 'user', 'content': user_input})
    #     completion = self.client.chat.completions.create(
    #         model="qwen-turbo",
    #         messages=user_messages,
    #         stream=True
    #     )
        
    #     chat_response = ""
    #     buffer = ""
    #     sentence_split_pattern = re.compile(r'(?<=[,;.!?，；：。:！？、》”])')
    #     print("[LLM] Start LLM streaming...")
    #     for chunk in completion:
    #         chat_response_chunk = chunk.choices[0].delta.content
    #         chat_response += chat_response_chunk
    #         buffer += chat_response_chunk

    #         sentences = sentence_split_pattern.split(buffer)
            
    #         if not sentences:
    #             continue
            
    #         for i in range(len(sentences) - 1):
    #             sentence = sentences[i].strip()
    #             llm_queue.put(sentence)
    #             print(f"[LLM] Put into queue: {sentence}")
            
    #         buffer = sentences[-1].strip()

    #     if buffer:
    #         llm_queue.put(sentence_buffer)
    #         print(f"[LLM] Put into queue: {sentence_buffer}")

    #     llm_queue.put(None)
        
    #     user_messages.append({'role': 'assistant', 'content': chat_response})
    #     if len(user_messages) > 10:
    #         user_messages.pop(0)
        
    #     print(f"[LLM] Response: {chat_response}\n")
        
    #     return chat_response, user_messages

if __name__ == "__main__":
    qwen = Qwen_API()
    user_input = "你好，讲一个故事"
    user_messages = [{'role': 'system', 'content': '你是一个聊天机器人，请你尽可能简短地回复用户的问题，使用短句，仅使用简体中文和英文。'}]
    # chat_response, _ = qwen.infer(user_input, user_messages)
    llm_queue = queue.Queue()
    qwen.infer_stream_split(user_input, user_messages, llm_queue)
    # qwen.infer(user_input, user_messages)
    while llm_queue.qsize() > 0:
        print(llm_queue.get())