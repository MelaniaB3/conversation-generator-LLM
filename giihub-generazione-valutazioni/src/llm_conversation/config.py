"""
Derived from: https://github.com/famiu/llm_conversation (original repository)
Modified by: Melania Balestri, 2025  adaptations for APP/IE matrix configuration and hosted usage
License: GNU AGPL v3.0 (see LICENSE in project root)
"""

from typing import List, Optional
from pydantic import BaseModel


class AgentConfig(BaseModel):
    name: str
    model: str
    temperature: float
    ctx_size: int
    system_prompt: str


class Settings(BaseModel):
    initial_message: Optional[str] = None


class Config(BaseModel):
    agents: List[AgentConfig]
    settings: Settings


def load_config(file_path: str) -> Config:
    """Load configuration from a JSON file."""
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Config(**data)
