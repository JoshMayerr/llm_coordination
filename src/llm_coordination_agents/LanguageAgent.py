# Author: Saaket Agashe
# Date: 2023-10-12
# License: MIT

from llm_coordination_agents.LanguageEngine import LLMEngineOpenAI, LLMEngineAzureOpenAI, LLMEnginevLLM

class LLMAgent:
    def __init__(self, engine_params=None, system_prompt=None, engine=None):
        if engine is None:
            if engine_params is not None:
                engine_type = engine_params.get('engine_type')
                if engine_type == 'openai':
                    self.engine = LLMEngineOpenAI(**engine_params)
                elif engine_type == 'azure':
                    self.engine = LLMEngineAzureOpenAI(**engine_params)
                elif engine_type == "vllm":
                    self.engine = LLMEnginevLLM(**engine_params)
                else:
                    raise ValueError("engine_type must be either 'openai' or 'azure'")
            else:
                raise ValueError("engine_params must be provided")
        else:
            self.engine = engine

        self.messages = []  # Empty messages

        if system_prompt:
            self.add_system_prompt(system_prompt)
        else:
            self.add_system_prompt("You are a helpful assistant.")
    
    def reset(self, keep_sys_prompt=False):
        if keep_sys_prompt:
            self.messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
        else:
            self.messages = []    
            
    def add_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        if len(self.messages) > 0:
            self.messages[0] = {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
        else:
            self.messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
    
    def remove_message_at(self, index):
        '''Remove a message at a given index'''
        if index < len(self.messages):
            self.messages.pop(index)

    def add_message(self, text_content, role=None):
        '''Add a new message to the list of messages'''
        # For API-style inference from OpenAI and AzureOpenAI 

        # infer role from previous message
        if self.messages[-1]["role"] == "system":
            role = "user"
        elif self.messages[-1]["role"] == "user":
            role = "assistant"
        elif self.messages[-1]["role"] == "assistant":
            role = "user"

        message = {"role": role, "content": [{"type": "text", "text": text_content}]}

        self.messages.append(message)
    
    def get_response(self, user_message=None, messages=None, temperature=0., max_new_tokens=None, **kwargs):
        '''Generate the next response based on previous messages'''
        if messages is None:
            messages = self.messages
        if user_message:
            messages.append({"role": "user", "content": [{"type": "text", "text": user_message}]})
            
        return self.engine.generate(messages, temperature=temperature, max_new_tokens=max_new_tokens, **kwargs)
