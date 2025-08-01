import re
import unicodedata
from typing import Optional


def normalize_text_aggressive(text: str) -> str:
    """Aggressively normalize text for maximum matching flexibility."""
    if not text:
        return ""

    # First, handle encoding artifacts BEFORE Unicode normalization
    # This is crucial because unicodedata.normalize might change the artifacts
    encoding_fixes = {
        '‚Äô': "'",  # Main apostrophe artifact
        '‚Äôs': "'s", '‚Äôt': "'t", '‚Äôm': "'m", '‚Äôre': "'re",
        '‚Äôve': "'ve", '‚Äôll': "'ll", '‚Äôd': "'d",
        '‚Ä¶': ' ', '‚Äù': '"', '‚Äú': '"', '‚Äî': '-', '‚Ä¢': "'",
        '‚Äò': "'", '‚Äúa': '"a',  # Common patterns
    }

    for old, new in encoding_fixes.items():
        text = text.replace(old, new)

    # Unicode normalization after fixing encoding artifacts
    text = unicodedata.normalize('NFKD', text)

    # Standard Unicode replacements
    unicode_fixes = {
        # Unicode quotes and apostrophes
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-', '\u2026': '...',
        # Common variations
        ''': "'", ''': "'", '"': '"', '"': '"',
        # Remove zero-width characters
        '\u200b': '', '\u200c': '', '\u200d': '', '\ufeff': '',
    }

    for old, new in unicode_fixes.items():
        text = text.replace(old, new)

    # Clean up any remaining apostrophe patterns (with length check for safety)
    if len(text) < 100000:  # Only apply regex patterns to reasonable-sized text
        try:
            text = re.sub(r"[''‚Äô]+s\b", "'s", text)  # Possessives
            text = re.sub(r"[''‚Äô]+t\b", "'t", text)  # Contractions
            text = re.sub(r"[''‚Äô]+m\b", "'m", text)
            text = re.sub(r"[''‚Äô]+re\b", "'re", text)
            text = re.sub(r"[''‚Äô]+ve\b", "'ve", text)
            text = re.sub(r"[''‚Äô]+ll\b", "'ll", text)
            text = re.sub(r"[''‚Äô]+d\b", "'d", text)
        except Exception:
            # If regex fails, just continue without this cleanup
            pass

    # Normalize whitespace - convert all whitespace to single spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def letters_only(text: str) -> str:
    """Keep only lowercase letters a-z, remove everything else."""
    if not text:
        return ""
    # Convert to lowercase and keep only a-z letters
    return re.sub(r'[^a-z]', '', text.lower())


def robust_find_improved(text: str, excerpt: str) -> Optional[str]:
    """Fast and robust text matching using letters-only approach with fallbacks."""
    if not excerpt.strip():
        return None

    # Primary strategy: letters-only matching (super fast and robust)
    text_letters = letters_only(text)
    excerpt_letters = letters_only(excerpt)

    if excerpt_letters and excerpt_letters in text_letters:
        return excerpt  # Return original excerpt if letters-only match found

    # Fallback 1: First 20 characters of letters-only (for partial matches at start)
    if len(excerpt_letters) >= 20:
        excerpt_first_20 = excerpt_letters[:20]
        if excerpt_first_20 in text_letters:
            return excerpt

    # Fallback 2: Last 20 characters of letters-only (for partial matches at end)
    if len(excerpt_letters) >= 20:
        excerpt_last_20 = excerpt_letters[-20:]
        if excerpt_last_20 in text_letters:
            return excerpt

    # Fallback 3: First 10 + last 10 characters (for middle truncation issues)
    if len(excerpt_letters) >= 20:
        excerpt_first_10 = excerpt_letters[:10]
        excerpt_last_10 = excerpt_letters[-10:]
        if excerpt_first_10 in text_letters and excerpt_last_10 in text_letters:
            return excerpt

    # Fallback 4: For shorter excerpts, try partial matching
    if 10 <= len(excerpt_letters) < 20:
        excerpt_first_half = excerpt_letters[:len(excerpt_letters)//2]
        excerpt_second_half = excerpt_letters[len(excerpt_letters)//2:]
        if len(excerpt_first_half) >= 5 and len(excerpt_second_half) >= 5:
            if excerpt_first_half in text_letters and excerpt_second_half in text_letters:
                return excerpt

    # Fallback 5: Very short excerpts (< 10 letters): try normalized approach
    if len(excerpt_letters) < 10:
        text_norm = normalize_text_aggressive(text).lower()
        excerpt_norm = normalize_text_aggressive(excerpt).lower()

        if excerpt_norm in text_norm:
            return excerpt

    return None


def strict_find(text: str, excerpt: str) -> bool:
    """Strict matching for failure analysis - only direct and normalized matching, no fallbacks."""
    if not excerpt.strip():
        return False

    # Strategy 1: Direct case-insensitive match
    text_lower = text.lower()
    excerpt_lower = excerpt.lower().strip()
    if excerpt_lower in text_lower:
        return True

    # Strategy 2: Normalized matching (handles encoding issues)
    text_norm = normalize_text_aggressive(text).lower()
    excerpt_norm = normalize_text_aggressive(excerpt).lower()
    if excerpt_norm in text_norm:
        return True

    # Strategy 3: Letters-only matching (most basic level)
    text_letters = letters_only(text)
    excerpt_letters = letters_only(excerpt)
    if excerpt_letters and excerpt_letters in text_letters:
        return True

    return False

