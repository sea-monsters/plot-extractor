"""LLM-enhanced policy router for ambiguous chart cases.

Triggers only when the rule-based guesser is uncertain (low confidence gap).
Supports Anthropic and OpenAI vision APIs via environment variables.

Environment variables:
    ANTHROPIC_API_KEY  -- use Claude vision models
    OPENAI_API_KEY     -- use GPT-4o / GPT-4-turbo vision
    LLM_BASE_URL       -- custom OpenAI-compatible endpoint
    LLM_API_KEY        -- key for custom endpoint
    LLM_MODEL          -- model name override (default varies by provider)
"""
import base64
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from plot_extractor.core.chart_type_guesser import ImageFeatures
from plot_extractor.core.policy_router import ExtractionPolicy
from plot_extractor.core.adaptive_strategy import compute_adaptive_policy


CHART_TYPES = [
    "simple_linear", "log_y", "log_x", "loglog", "scatter",
    "multi_series", "dense", "dual_y", "inverted_y", "no_grid",
]

# Confidence-gap threshold: if top1 - top2 < this, invoke LLM
LLM_TRIGGER_GAP = 0.15


def _encode_image(image_path: Path) -> str:
    """Base64-encode an image for vision API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_to_data_url(image_path: Path) -> str:
    """Convert image to a data URL for API submission."""
    ext = image_path.suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    mime = f"image/{ext}"
    b64 = _encode_image(image_path)
    return f"data:{mime};base64,{b64}"


def _encode_image_array(image_array: np.ndarray) -> str:
    """Base64-encode a numpy image array for vision API."""
    import cv2
    _, buf = cv2.imencode(".png", image_array)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _image_array_to_data_url(image_array: np.ndarray) -> str:
    """Convert numpy image array to a data URL for API submission."""
    b64 = _encode_image_array(image_array)
    return f"data:image/png;base64,{b64}"


def _build_vision_prompt() -> str:
    return (
        "You are a chart analysis assistant. Look at the chart image and "
        "classify it precisely. Respond ONLY with a JSON object in this format:\n"
        "{\n"
        '  "chart_type": "one of: simple_linear, log_y, log_x, loglog, scatter, '
        'multi_series, dense, dual_y, inverted_y, no_grid",\n'
        '  "confidence": 0.0 to 1.0,\n'
        '  "noise_level": "clean" | "moderate" | "heavy",\n'
        '  "rotation_degrees": number or 0,\n'
        '  "special_notes": "brief notes, e.g. grid lines, multiple colors, thick lines"\n'
        "}\n"
        "Be concise. chart_type must be exactly one of the allowed values."
    )


def _parse_llm_json(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response (handles markdown fences)."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening fence
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _llm_probabilities(parsed: Optional[Dict], rule_probs: Dict[str, float]) -> Dict[str, float]:
    """Convert LLM response into a probability distribution over chart types."""
    probs = {t: 0.0 for t in CHART_TYPES}
    if parsed is None:
        return probs

    llm_type = parsed.get("chart_type", "").strip().lower()
    llm_conf = float(parsed.get("confidence", 0.7))
    llm_conf = max(0.0, min(1.0, llm_conf))

    if llm_type in probs:
        probs[llm_type] = llm_conf
        # Distribute remaining mass uniformly among other types
        remaining = 1.0 - llm_conf
        others = [t for t in CHART_TYPES if t != llm_type]
        for t in others:
            probs[t] = remaining / max(len(others), 1)
    else:
        # Unknown type: fallback to rule-based
        return probs

    return probs


def _blend_probs(rule_probs: Dict[str, float], llm_probs: Dict[str, float], alpha: float = 0.6) -> Dict[str, float]:
    """Blend rule-based and LLM probabilities. alpha weights LLM."""
    blended = {}
    for t in CHART_TYPES:
        blended[t] = (1.0 - alpha) * rule_probs.get(t, 0.0) + alpha * llm_probs.get(t, 0.0)
    # Renormalize
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}
    return blended


def _detect_provider() -> Tuple[str, str, str]:
    """Detect LLM provider from environment and return (provider, base_url, model).

    When running inside Claude Code, ANTHROPIC_API_KEY is already set
    (Claude Code needs it to talk to the API).  The plugin re-uses the
    same credentials automatically — no extra configuration required.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        # Claude Code runtime: reuse the same API key.
        # Default to Haiku for speed/cost; override with LLM_MODEL env var if needed.
        model = os.environ.get("LLM_MODEL", "claude-haiku-4-5")
        return "anthropic", "https://api.anthropic.com/v1/messages", model
    if os.environ.get("OPENAI_API_KEY"):
        model = os.environ.get("LLM_MODEL", "gpt-5.4-nano")
        return "openai", "https://api.openai.com/v1/chat/completions", model
    if os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_API_KEY"):
        model = os.environ.get("LLM_MODEL", "default")
        return "custom", os.environ["LLM_BASE_URL"], model
    return "", "", ""


def _call_anthropic(image_data_url: str, api_key: str, model: str) -> Optional[Dict]:
    """Call Anthropic Messages API with vision."""
    if not HAS_REQUESTS:
        return None
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data_url.split(",")[1]}},
                    {"type": "text", "text": _build_vision_prompt()},
                ],
            }
        ],
    }
    try:
        resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]
        return _parse_llm_json(text)
    except Exception:
        return None


def _call_openai(image_data_url: str, api_key: str, model: str) -> Optional[Dict]:
    """Call OpenAI Chat Completions API with vision."""
    if not HAS_REQUESTS:
        return None
    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_vision_prompt()},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
    }
    try:
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return _parse_llm_json(text)
    except Exception:
        return None


def _call_custom(image_data_url: str, base_url: str, api_key: str, model: str) -> Optional[Dict]:
    """Call custom OpenAI-compatible endpoint with vision."""
    if not HAS_REQUESTS:
        return None
    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_vision_prompt()},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
    }
    try:
        resp = requests.post(base_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return _parse_llm_json(text)
    except Exception:
        return None


def llm_available() -> bool:
    """Check whether any LLM API credentials are configured."""
    provider, _, _ = _detect_provider()
    return provider != "" and HAS_REQUESTS


def should_use_llm(type_probs: Dict[str, float]) -> bool:
    """Determine whether rule-based guesser is uncertain enough to warrant LLM."""
    sorted_probs = sorted(type_probs.values(), reverse=True)
    if len(sorted_probs) < 2:
        return False
    gap = sorted_probs[0] - sorted_probs[1]
    return gap < LLM_TRIGGER_GAP


def _build_axis_prompt(axis_direction: str, n_ticks: int) -> str:
    return (
        f"You are reading a chart {axis_direction}-axis. "
        f"The image shows {n_ticks} tick marks along this axis. "
        "Read the numeric label at each tick mark (top-to-bottom for y-axis, "
        "left-to-right for x-axis). If a tick has no visible label, use null. "
        'Respond ONLY with JSON: {"axis_type": "linear" | "log", '
        '"tick_values": [v1, v2, ... or null], "min": number, "max": number}'
    )


def _parse_axis_response(parsed: Optional[Dict]) -> Optional[Dict]:
    if not parsed or "tick_values" not in parsed:
        return None
    return parsed


def llm_read_axis_labels(
    image_data_url: str,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    axis_direction: str,
    n_ticks: int,
) -> Optional[Dict]:
    """Ask LLM to read axis tick labels from a cropped axis region."""
    if not llm_available():
        return None

    prompt = _build_axis_prompt(axis_direction, n_ticks)
    if provider == "anthropic":
        parsed = _call_anthropic_for_axis(image_data_url, api_key, model, prompt)
    elif provider == "openai":
        parsed = _call_openai_for_axis(image_data_url, api_key, model, prompt)
    elif provider == "custom":
        parsed = _call_custom_for_axis(image_data_url, base_url, api_key, model, prompt)
    else:
        parsed = None
    return _parse_axis_response(parsed)


def _call_anthropic_for_axis(image_data_url: str, api_key: str, model: str, prompt: str) -> Optional[Dict]:
    if not HAS_REQUESTS:
        return None
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data_url.split(",")[1]}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    try:
        resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]
        return _parse_llm_json(text)
    except Exception:
        return None


def _call_openai_for_axis(image_data_url: str, api_key: str, model: str, prompt: str) -> Optional[Dict]:
    if not HAS_REQUESTS:
        return None
    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
    }
    try:
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return _parse_llm_json(text)
    except Exception:
        return None


def _call_custom_for_axis(image_data_url: str, base_url: str, api_key: str, model: str, prompt: str) -> Optional[Dict]:
    if not HAS_REQUESTS:
        return None
    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
    }
    try:
        resp = requests.post(base_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return _parse_llm_json(text)
    except Exception:
        return None


def compute_llm_enhanced_policy(
    image_path: Path,
    features: ImageFeatures,
    type_probs: Dict[str, float],
    use_llm: bool = False,
) -> Tuple[ExtractionPolicy, Dict[str, float]]:
    """Build policy with optional LLM enhancement.

    Returns (policy, final_type_probs).
    """
    if not use_llm or not llm_available() or not should_use_llm(type_probs):
        policy = compute_adaptive_policy(features, type_probs)
        return policy, type_probs

    provider, base_url, model = _detect_provider()
    image_data_url = _image_to_data_url(image_path)

    if provider == "anthropic":
        parsed = _call_anthropic(image_data_url, os.environ["ANTHROPIC_API_KEY"], model)
    elif provider == "openai":
        parsed = _call_openai(image_data_url, os.environ["OPENAI_API_KEY"], model)
    elif provider == "custom":
        parsed = _call_custom(image_data_url, base_url, os.environ["LLM_API_KEY"], model)
    else:
        parsed = None

    llm_probs = _llm_probabilities(parsed, type_probs)
    if any(v > 0 for v in llm_probs.values()):
        blended = _blend_probs(type_probs, llm_probs, alpha=0.6)
        policy = compute_adaptive_policy(features, blended)
        return policy, blended

    policy = compute_adaptive_policy(features, type_probs)
    return policy, type_probs
