import base64
from typing import Optional


def encode_image(image_path: str) -> Optional[str]:
    """Return the contents of ``image_path`` as a base64 string.

    Parameters
    ----------
    image_path: str
        Path to the image file to encode.

    Returns
    -------
    str or None
        The base64-encoded contents of the file, or ``None`` if reading fails.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception:
        return None
