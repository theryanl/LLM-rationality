from typing import List, Literal, Optional
from dataclasses import dataclass

import openai
import os
import time
import json
from rich import print
import anthropic


@dataclass
class Claude3:
    model: Literal["claude-3-opus-20240229"] = "claude-3-opus-20240229"

    temperature: float = 0.0
    
    max_tokens: int = 700

    system_prompt: Optional[str] = None
    
    def complete(self, conversation: List[str]) -> str:
        client = anthropic.Anthropic(
            api_key="", # redacted for anonymity purposes
            )
        messages = []
        for i, prompt in enumerate(conversation):
            messages.append({"role": ("user" if i % 2 == 0 else "assistant"), "content": [{"type": "text", "text":prompt}]})
            
        while True:
            try:
                response = client.messages.create(
                    model = self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                print(f"response: {response}")
                break 
            except Exception as error:
                print(f"Error: {error}")
                time.sleep(55)
        return response.content[0].text