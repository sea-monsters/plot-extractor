"""JSON schemas for plot extraction output.

Defines structured output format for MCP responses and validation.
"""
from typing import Dict, Any


# Schema definition for extraction result
EXTRACTION_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "success": {
            "type": "boolean",
            "description": "Whether extraction succeeded",
        },
        "error": {
            "type": ["string", "null"],
            "description": "Error message if failed",
        },
        "data": {
            "type": ["object", "null"],
            "description": "Extracted series data",
            "properties": {
                "series_name": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "y": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                    },
                },
            },
        },
        "axes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["x", "y"]},
                    "side": {
                        "type": "string",
                        "enum": ["bottom", "top", "left", "right"],
                    },
                    "type": {"type": "string", "enum": ["linear", "log"]},
                    "inverted": {"type": "boolean"},
                    "value_range": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
            },
        },
        "ssim": {
            "type": ["number", "null"],
            "description": "SSIM score of rebuilt plot vs original",
        },
        "panel_count": {
            "type": "integer",
            "description": "Number of panels detected",
        },
        "diagnostics": {
            "type": "object",
            "description": "Internal extraction diagnostics",
        },
    },
    "required": ["success", "error", "data"],
}


def validate_extraction_result(result: Dict[str, Any]) -> bool:
    """
    Validate extraction result against schema.

    Args:
        result: Extraction result dict

    Returns:
        True if valid, False otherwise
    """
    # Basic validation: check required fields
    required_fields = ["success", "error", "data"]
    for field in required_fields:
        if field not in result:
            return False

    # Type checks
    if not isinstance(result["success"], bool):
        return False

    if result["success"] and result["data"] is None:
        return False

    return True
