import os
import backoff
from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
    AzureOpenAI,
    OpenAI
)

class LLMEngine:
    pass

class LLMEngineOpenAI(LLMEngine):
    def __init__(self, api_key=None, model=None, rate_limit=-1, **kwargs):
        assert model is not None, "model must be provided"
        self.model = model

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENAI_API_KEY")
        
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit

        self.llm_client = OpenAI(api_key=self.api_key)

    @backoff.on_exception(backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60)
    def generate(self, messages, temperature=0., max_new_tokens=None, **kwargs):
        '''Generate the next message based on previous messages'''
        return self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            **kwargs,
        ).choices[0].message.content



class LLMEngineAzureOpenAI(LLMEngine):
    def __init__(self, api_key=None, azure_endpoint=None, model=None, api_version=None, rate_limit=-1, **kwargs):
        assert model is not None, "model must be provided"
        self.model = model

        assert api_version is not None, "api_version must be provided"
        self.api_version = api_version

        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("An API Key needs to be provided in either the api_key parameter or as an environment variable named AZURE_OPENAI_API_KEY")
        
        self.api_key = api_key

        azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_API_BASE")
        if azure_endpoint is None:
            raise ValueError("An Azure API endpoint needs to be provided in either the azure_endpoint parameter or as an environment variable named AZURE_OPENAI_API_BASE")
        
        self.azure_endpoint = azure_endpoint
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit

        self.llm_client = AzureOpenAI(azure_endpoint=self.azure_endpoint, api_key=self.api_key, api_version=self.api_version)

    # @backoff.on_exception(backoff.expo, (APIConnectionError, APIError, RateLimitError), max_tries=10)
    def generate(self, messages, temperature=0., max_new_tokens=None, **kwargs):
        '''Generate the next message based on previous messages'''
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            **kwargs,
        )
        return completion.choices[0].message.content
    

class LLMEnginevLLM(LLMEngine):
    def __init__(self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs):
        assert model is not None, "model must be provided"
        self.model = model
        self.api_key = api_key

        self.base_url = base_url or os.getenv("vLLM_ENDPOINT_URL")
        if self.base_url is None:
            raise ValueError("An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named vLLM_ENDPOINT_URL")

        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit

        self.llm_client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    # @backoff.on_exception(backoff.expo, (APIConnectionError, APIError, RateLimitError), max_tries=10)
    # TODO: Default params chosen for the Qwen model
    def generate(self, messages, temperature=0., top_p=0.8, repetition_penalty=1.05, max_new_tokens=512, **kwargs):
        '''Generate the next message based on previous messages'''
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature,
            top_p=top_p,
            extra_body={"repetition_penalty": repetition_penalty},
        )
        return completion.choices[0].message.content