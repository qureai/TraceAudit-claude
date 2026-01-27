import os
import json
import time
import logging
from typing import Optional

import dotenv
from portkey_ai import Portkey

logger = logging.getLogger(__name__)


def create_portkey_client(portkey_api_key: str = None, provider_name: str = None) -> Portkey:
    """Create a Portkey client with the given API key and virtual key."""
    dotenv.load_dotenv()

    api_key = portkey_api_key or os.getenv("PORTKEY_API_KEY")
    virtual_key = provider_name or os.getenv("PROVIDER_NAME")

    if not api_key:
        raise ValueError("PORTKEY_API_KEY not found in environment")
    if not virtual_key:
        raise ValueError("PROVIDER_NAME not found in environment")

    return Portkey(api_key=api_key, virtual_key=virtual_key)


def get_response(
    portkey_client: Portkey,
    system_prompt: str,
    user_prompt: str,
    model: str,
    model_params: dict = None,
    max_retries: int = 3
) -> str:
    """Get a response from the LLM using Portkey with retry logic."""
    model_params = model_params or {}
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(max_retries):
        try:
            completion = portkey_client.chat.completions.create(
                model=model, messages=messages, **model_params
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s


def get_json_response(
    portkey_client: Portkey,
    system_prompt: str,
    user_prompt: str,
    model: str,
    model_params: dict = None,
    max_retries: int = 3
) -> Optional[dict]:
    """
    Get a JSON response from the LLM and parse it.

    Returns parsed JSON dict or None if parsing fails.
    """
    response = get_response(
        portkey_client=portkey_client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        model_params=model_params,
        max_retries=max_retries
    )

    if not response:
        return None

    # Try to extract JSON from response
    try:
        # First try direct parsing
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in response (might be wrapped in markdown)
    try:
        # Look for JSON block
        if '```json' in response:
            json_start = response.find('```json') + 7
            json_end = response.find('```', json_start)
            if json_end > json_start:
                return json.loads(response[json_start:json_end].strip())

        # Look for { } pattern
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass

    logger.warning(f"Failed to parse JSON from LLM response: {response[:100]}...")
    return None


# Singleton client for reuse
_portkey_client: Optional[Portkey] = None


def get_portkey_client() -> Portkey:
    """Get or create a singleton Portkey client."""
    global _portkey_client
    if _portkey_client is None:
        _portkey_client = create_portkey_client()
    return _portkey_client
