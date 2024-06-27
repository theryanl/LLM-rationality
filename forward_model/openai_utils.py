"""
python -m forward_model.openai_utils
"""
from typing import List, Literal, Optional
from dataclasses import dataclass

import openai
import os
import time
import json
from rich import print

from filelock import FileLock

MODEL_CONFIGS = {   
    "gpt-3.5-turbo-16k": {
        "deployment_name": "gpt-3.5-turbo-1106",
        "prompt_cost_per_token": 0.001 / 1000,
        "response_cost_per_token": 0.002 / 1000,
    },
    "gpt-3.5-turbo-instruct": {
        "deployment_name": "gpt-3.5-turbo-instruct",
        "prompt_cost_per_token": 0.0015 / 1000,
        "response_cost_per_token": 0.002 / 1000,
    },
    "gpt-4": {
        "deployment_name": "gpt-4",
        "prompt_cost_per_token": 0.03 / 1000,
        "response_cost_per_token": 0.06 / 1000,
    },
    "gpt-4-0125-preview": {
        "deployment_name": "gpt-4-0125-preview",
        "prompt_cost_per_token": 0.01 / 1000,
        "response_cost_per_token": 0.03 / 1000,
    },
    "gpt-4-1106-preview": {
        "deployment_name": "gpt-4-1106-preview",
        "prompt_cost_per_token": 0.01 / 1000,
        "response_cost_per_token": 0.03 / 1000,
    },
    
}


engine_env_mappings = {
    "gpt-35-turbo-0301": {
        "OPENAI_API_KEY": "OPENAI_API_KEY",
        "OPENAI_ORG_ID": "OPENAI_ORG_ID",
        "api": "openai"
    },
    "gpt-35-turbo-16k": {
        "OPENAI_API_KEY": "OPENAI_API_KEY",
        "OPENAI_ORG_ID": "OPENAI_ORG_ID",
        "api": "openai"
    },
    "gpt-4": {
        "OPENAI_API_KEY": "OPENAI_API_KEY_4",
        "OPENAI_ORG_ID": "OPENAI_ORG_ID_4",
        "api": "openai"
    },
    "gpt-4-1106-preview": {
        "OPENAI_API_KEY": "OPENAI_API_KEY_4",
        "OPENAI_ORG_ID": "OPENAI_ORG_ID_4",
        "api": "openai"
    },
    "gpt-4-0125-preview": {
        "OPENAI_API_KEY": "OPENAI_API_KEY_4",
        "OPENAI_ORG_ID": "OPENAI_ORG_ID_4",
        "api": "openai"
    }

}


def get_credentials(engine, azure=None):
    if azure:
        return {
            "api_args": {
                "api_key": os.environ.get(engine_env_mappings[engine]["OPENAI_API_KEY"]),
                "api_base": os.environ.get(engine_env_mappings[engine]["OPENAI_ORG_ID"]),
                "api_type": 'azure',
                "api_version": '2023-05-15',
                "engine": engine,
            },
            "mode": 'Chat'
        }
    else:
        return {
            "api_args": {
                "api_key": os.environ.get(engine_env_mappings[engine]["OPENAI_API_KEY"]),
                "engine": engine,
            },
            "mode": 'Chat'
        }
        

@dataclass
class OpenAI:
    model: Literal["gpt-3.5-turbo-16k", "gpt-3.5-turbo", "gpt-4", "gpt-4-0125-preview", "gpt-4-1106-preview"] = "gpt-4-0125-preview"

    temperature: float = 0.7
    
    max_tokens: int = 1000

    system_prompt: Optional[str] = None

    max_retries = 1

    log_file_path = "openai_usage.jsonl"

    def complete(self, conversation: List[str]) -> str:
        model = self.model
        print(f"model: {model}")
        config = MODEL_CONFIGS[model]
        deployment_name = config["deployment_name"]

        args_dict = get_credentials(model, azure=True)["api_args"]
        args_dict.update(
            {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        )
        print(f"args_dict: {args_dict}")
        retry_count = 0

        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        for i, prompt in enumerate(conversation):
            messages.append({"role": ("user" if i % 2 == 0 else "assistant"), "content": prompt})
            
        while True:
            try:
                response = openai.ChatCompletion.create(
                    messages=messages,
                    **args_dict
                )
                break
            except Exception as error:
                if "Please retry after" in str(error):
                    timeout = int(str(error).split("Please retry after ")[1].split(" second")[0]) + 2
                    print(f"Wait {timeout}s before OpenAI API retry ({error})")
                    time.sleep(timeout)
                elif retry_count < self.max_retries:
                    print(f"OpenAI API retry for {retry_count} times ({error})")
                    time.sleep(2)
                    retry_count += 1
                else:
                    print(f"OpenAI API failed for {retry_count} times ({error})")
                    return None

        self.log_usage(config, response.usage)

        generation = response.choices[0].message.content
        return generation

    def log_usage(self, config, usage):
        usage_log = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
        usage_log["prompt_cost"] = config["prompt_cost_per_token"] * usage.prompt_tokens
        usage_log["completion_cost"] = config["response_cost_per_token"] * usage.completion_tokens
        usage_log["cost"] = usage_log["prompt_cost"] + usage_log["completion_cost"]
        usage_log["model"] = config["deployment_name"]
        usage_log["user"] = os.getlogin()

        with FileLock(self.log_file_path + ".lock"):
            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(usage_log) + "\n")