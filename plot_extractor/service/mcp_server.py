"""MCP server for plot extraction service.

Exposes plot extraction as an MCP tool that can be called by Claude Code
or other MCP clients.
"""
from pathlib import Path
from typing import Dict, Any, List
from plot_extractor.main import extract_from_image


def mcp_extract_plot(
    image_path: str,
    use_ocr: bool = False,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """
    MCP tool: Extract data from a plot image.

    Args:
        image_path: Path to PNG/JPG chart image
        use_ocr: Enable OCR tick label reading
        use_llm: Enable LLM vision enhancement

    Returns:
        Structured extraction result with data, axes, confidence, diagnostics
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return {
            "success": False,
            "error": f"Image not found: {image_path}",
            "data": None,
        }

    result = extract_from_image(
        image_path,
        use_ocr=use_ocr,
        use_llm=use_llm,
    )

    if not result:
        return {
            "success": False,
            "error": "Extraction failed",
            "data": None,
        }

    return {
        "success": True,
        "error": None,
        "data": result["data"],
        "axes": _format_axes(result["calibrated_axes"]),
        "ssim": result["ssim_crop"],
        "panel_count": result["panel_count"],
        "diagnostics": result["diagnostics"],
    }


def _format_axes(calibrated_axes) -> List[Dict[str, Any]]:
    """Format calibrated axes for JSON output."""
    axes = []
    for ca in calibrated_axes:
        axes.append({
            "direction": ca.axis.direction,
            "side": ca.axis.side,
            "type": ca.axis_type,
            "inverted": bool(ca.inverted),
            "value_range": [
                float(min(v for _, v in ca.tick_map)),
                float(max(v for _, v in ca.tick_map)),
            ] if ca.tick_map else None,
        })
    return axes
