from __future__ import annotations

import ast
import importlib.util
import re

if importlib.util.find_spec("nbformat") is not None:
    import nbformat  # type: ignore[import]


def remove_embedded_media(content: str, passes: int = 3) -> str:
    """
    Strip out embedded image, video, and audio base64 data URIs from a notebook JSON string.

    This runs multiple regex passes to clean up both inline `"image/png"`/`"image/jpeg"`
    blocks and `"data:*;base64,..."` URIs.

    Args:
        content: The JSON content as a string (e.g. the text of an .ipynb file).
        passes: Number of times to re-apply each regex removal (default: 3).

    Returns:
        The cleaned content string with all matched media data replaced by empty strings.
    """
    # remove explicit image/png or image/jpeg blocks
    if any(kw in content for kw in ("image/png", "image/jpeg")):
        pattern = r'"(?:image\/png|image\/jpeg)":\s*"([^"]*)"'
        for _ in range(passes):
            content = re.sub(pattern, '""', content, re.DOTALL)

    # remove generic base64 data URIs for video, image, audio
    if any(
        kw in content
        for kw in (
            "data:video/mp4;base64,",
            "data:image/jpeg;base64,",
            "data:image/png;base64,",
            "data:audio/x-wav;base64,",
        )
    ):
        pattern = r'"data:(?:video/mp4|image/jpeg|image/png|audio/x-wav);base64,([^"]+)"'
        for _ in range(passes):
            content = re.sub(pattern, '""', content, re.DOTALL)

    return content


def parse_notebook_blob(raw: str) -> nbformat.NotebookNode:
    """
    Parse raw notebook text into a NotebookNode.
    - If `raw` looks like a Python-repr'd string, un-repr it first.
    - Then call nbformat.reads() to get a NotebookNode.
    """
    # 1) Strip any outer Python quotes, if present
    s = raw.strip()
    if s.startswith(("'", '"')) and s.endswith(("'", '"')):
        try:
            raw = ast.literal_eval(s)
        except Exception:
            # not a literal repr after all, leave it alone
            pass

    # 2) Parse the JSON into a NotebookNode
    try:
        nb = nbformat.reads(raw, as_version=nbformat.NO_CONVERT)
    except nbformat.NotJSONError as e:
        # If it really isn’t JSON, re-raise or handle as you see fit
        raise ValueError(f"Failed to parse notebook JSON: {e}") from e
    return nb
