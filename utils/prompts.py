"""Prompt-building utilities for Qwen2-VL."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

DEFAULT_CLASSIFICATION_PROMPT = (
    "You are given two images. Decide whether the statement is true.\n"
    "Statement: {statement}\nAnswer with either True or False."
)


def format_nlvr2_prompt(statement: str, template: str = DEFAULT_CLASSIFICATION_PROMPT) -> str:
    """Return the default NLVR2 classification prompt."""

    return template.format(statement=statement.strip())


def build_user_content(images: List[Any], text: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for image in images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": text})
    return content


def build_conversation(
    images: List[Any],
    user_text: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": build_user_content(images, user_text)})
    return messages


def attach_assistant_response(messages: List[Dict[str, Any]], response: str) -> List[Dict[str, Any]]:
    """Return a new conversation list with the assistant response appended."""

    extended = list(messages)
    extended.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    return extended
