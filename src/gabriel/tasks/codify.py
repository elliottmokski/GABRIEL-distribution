import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm.auto import tqdm

from ..utils import (
    Teleprompter,
    get_all_responses,
    normalize_text_aggressive,
    letters_only,
    robust_find_improved,
    strict_find,
)


class Codify:
    """Pipeline for coding passages of text according to specified categories."""

    def __init__(self, teleprompter: Optional[Teleprompter] = None) -> None:
        self.teleprompter = teleprompter or Teleprompter()
        self.hit_rate_stats = {}  # Track hit rates across all texts

    @staticmethod
    def view(
        df: pd.DataFrame,
        text_column: str,
        categories: Optional[Union[List[str], str]] = None,
    ):
        """Convenience wrapper around :func:`view_coded_passages`.

        This helper makes it easy to visualise coding results produced by
        :class:`Codify`. It simply forwards the provided DataFrame to the
        passage viewer utility.
        """
        from ..utils import view_coded_passages

        return view_coded_passages(df, text_column, categories)

    def parse_json(self, response_text: Any) -> Optional[dict]:
        """Parse JSON response, handling various input types."""
        # If it's already a dict, return it
        if isinstance(response_text, dict):
            return response_text
        
        # If it's not a string, we can't parse it
        if not isinstance(response_text, str):
            return None
        
        # If it's an empty string, return empty dict
        if not response_text.strip():
            return {}
            
        try:
            # First attempt: parse as JSON directly
            parsed = json.loads(response_text)
            
            # If it's a dict, we're done
            if isinstance(parsed, dict):
                return parsed
            
            # If it's a list and has elements, try to parse the first element as JSON
            if isinstance(parsed, list) and parsed:
                first_element = parsed[0]
                if isinstance(first_element, str):
                    return json.loads(first_element)
                elif isinstance(first_element, dict):
                    return first_element
                    
            return None
            
        except json.JSONDecodeError:
            return None

    def chunk_by_words(self, text: str, max_words: int) -> List[str]:
        """Split text into chunks by word count."""
        words = text.split()
        if len(words) <= max_words:
            return [text]
        return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

    def _extract_key_words(self, text: str, n: int = 5) -> tuple:
        """Extract first n and last n words from text."""
        words = text.split()
        if len(words) <= n * 2:
            return ' '.join(words), ''
        return ' '.join(words[:n]), ' '.join(words[-n:])


    def find_snippet_in_text(self, text: str, beginning_excerpt: str, ending_excerpt: str) -> Optional[str]:
        """Fast snippet finding that returns actual text from the original document."""
        if not beginning_excerpt:
            return None
        
        # Handle short excerpts (no ending)
        if not ending_excerpt:
            match = robust_find_improved(text, beginning_excerpt)
            if match:
                # Find the actual position in the original text
                start_pos, end_pos, match_type = self._find_actual_position_with_type(text, beginning_excerpt)
                if start_pos is not None:
                    # Always ensure word boundaries and add some context
                    start_pos = self._find_word_start(text, start_pos)
                    end_pos = self._find_word_end(text, end_pos)
                    
                    # If using fallback matching, expand to include more context
                    if match_type in ['first_20', 'last_20', 'first_half', 'second_half']:
                        start_pos, end_pos = self._expand_fallback_match(text, start_pos, end_pos, beginning_excerpt)
                    else:
                        # Add minimal context for exact matches
                        words_after = self._get_n_words_after(text, end_pos, 5)
                        end_pos = min(len(text), end_pos + len(words_after))
                    
                    return text[start_pos:end_pos].strip()
            return None
        
        # Handle longer snippets with both beginning and ending
        begin_match = robust_find_improved(text, beginning_excerpt)
        end_match = robust_find_improved(text, ending_excerpt)
        
        if not begin_match and not end_match:
            return None
        elif begin_match and not end_match:
            # Beginning found but ending not found - include beginning + 20 words after
            begin_start, begin_end, _ = self._find_actual_position_with_type(text, beginning_excerpt)
            if begin_start is not None:
                # Find word boundary at beginning end
                word_end = self._find_word_end(text, begin_end)
                words_after = self._get_n_words_after(text, word_end, 20)
                
                # Calculate end position for the 20 words after
                after_end = min(len(text), word_end + len(words_after))
                result = text[begin_start:after_end].strip()
                return result if result else None
            return None
        elif not begin_match and end_match:
            # Ending found but beginning not found - include 20 words before + ending
            end_start, end_end, _ = self._find_actual_position_with_type(text, ending_excerpt)
            if end_start is not None:
                # Find word boundary at ending start
                word_start = self._find_word_start(text, end_start)
                words_before = self._get_n_words_before(text, word_start, 20)
                
                # Calculate start position for the 20 words before
                before_start = max(0, word_start - len(words_before))
                result = text[before_start:end_end].strip()
                return result if result else None
            return None
        else:
            # Both beginning and ending found - extract the actual snippet from original text
            begin_start, begin_end, begin_type = self._find_actual_position_with_type(text, beginning_excerpt)
            end_start, end_end, end_type = self._find_actual_position_with_type(text, ending_excerpt)
            
            if begin_start is not None and end_start is not None:
                # Expand fallback matches
                if begin_type in ['first_20', 'last_20', 'first_half', 'second_half']:
                    begin_start, begin_end = self._expand_fallback_match(text, begin_start, begin_end, beginning_excerpt)
                if end_type in ['first_20', 'last_20', 'first_half', 'second_half']:
                    end_start, end_end = self._expand_fallback_match(text, end_start, end_end, ending_excerpt)
                
                # Make sure ending comes after beginning
                if end_start >= begin_start:
                    return text[begin_start:end_end].strip()
                else:
                    # Ending comes before beginning, just return beginning snippet
                    return self.find_snippet_in_text(text, beginning_excerpt, "")
            elif begin_start is not None:
                # Only beginning found
                return self.find_snippet_in_text(text, beginning_excerpt, "")
            
            return None

    def _find_actual_position(self, text: str, excerpt: str, _recursion_depth: int = 0) -> tuple:
        """Find the actual character positions of an excerpt in the original text."""
        result = self._find_actual_position_with_type(text, excerpt, _recursion_depth)
        return result[0], result[1]  # Return just position, not match type

    def _find_actual_position_with_type(self, text: str, excerpt: str, _recursion_depth: int = 0) -> tuple:
        """Find the actual character positions and match type using the SAME permissive strategies as robust_find_improved."""
        if not excerpt.strip():
            return None, None, None
        
        # Prevent infinite recursion
        if _recursion_depth > 1:
            return None, None, None
        
        # Strategy 1: Try direct matching first (fastest)
        text_lower = text.lower()
        excerpt_lower = excerpt.lower().strip()
        idx = text_lower.find(excerpt_lower)
        if idx != -1:
            return idx, idx + len(excerpt_lower), 'exact'
        
        # Strategy 2: Try with our aggressive normalization
        text_norm = normalize_text_aggressive(text)
        excerpt_norm = normalize_text_aggressive(excerpt)
        
        idx = text_norm.lower().find(excerpt_norm.lower())
        if idx != -1:
            # Map back to original text position approximately
            start_pos, end_pos = self._map_normalized_to_original(text, text_norm, idx, len(excerpt_norm))
            return start_pos, end_pos, 'normalized'
        
        # Strategy 3: Letters-only matching (same as robust_find_improved)
        text_letters = letters_only(text)
        excerpt_letters = letters_only(excerpt)
        
        if excerpt_letters and excerpt_letters in text_letters:
            letters_idx = text_letters.find(excerpt_letters)
            ratio = letters_idx / len(text_letters) if text_letters else 0
            approx_start = int(ratio * len(text))
            return approx_start, approx_start + len(excerpt), 'letters_only'
        
        # Strategy 4: First 20 characters fallback (same as robust_find_improved)
        if len(excerpt_letters) >= 20:
            excerpt_first_20 = excerpt_letters[:20]
            if excerpt_first_20 in text_letters:
                letters_idx = text_letters.find(excerpt_first_20)
                ratio = letters_idx / len(text_letters) if text_letters else 0
                approx_start = int(ratio * len(text))
                return approx_start, approx_start + len(excerpt), 'first_20'
        
        # Strategy 5: Last 20 characters fallback (same as robust_find_improved)
        if len(excerpt_letters) >= 20:
            excerpt_last_20 = excerpt_letters[-20:]
            if excerpt_last_20 in text_letters:
                letters_idx = text_letters.find(excerpt_last_20)
                ratio = letters_idx / len(text_letters) if text_letters else 0
                approx_start = int(ratio * len(text))
                return approx_start, approx_start + len(excerpt), 'last_20'
        
        # Strategy 6: First + Last 10 fallback (same as robust_find_improved)
        if len(excerpt_letters) >= 20:
            excerpt_first_10 = excerpt_letters[:10]
            excerpt_last_10 = excerpt_letters[-10:]
            if excerpt_first_10 in text_letters and excerpt_last_10 in text_letters:
                letters_idx = text_letters.find(excerpt_first_10)
                ratio = letters_idx / len(text_letters) if text_letters else 0
                approx_start = int(ratio * len(text))
                return approx_start, approx_start + len(excerpt), 'first_last_10'
        
        # Strategy 7: Half matching for shorter excerpts (same as robust_find_improved)
        if 10 <= len(excerpt_letters) < 20:
            excerpt_first_half = excerpt_letters[:len(excerpt_letters)//2]
            excerpt_second_half = excerpt_letters[len(excerpt_letters)//2:]
            if len(excerpt_first_half) >= 5 and len(excerpt_second_half) >= 5:
                if excerpt_first_half in text_letters and excerpt_second_half in text_letters:
                    letters_idx = text_letters.find(excerpt_first_half)
                    ratio = letters_idx / len(text_letters) if text_letters else 0
                    approx_start = int(ratio * len(text))
                    return approx_start, approx_start + len(excerpt), 'first_half'
        
        return None, None, None

    def _map_normalized_to_original(self, original: str, normalized: str, norm_start: int, norm_length: int) -> tuple:
        """Map a position in normalized text back to original text."""
        # This is an approximation - we'll search around the estimated area
        if len(normalized) == 0:
            return None, None
        
        # Estimate the ratio
        ratio_start = norm_start / len(normalized)
        ratio_end = (norm_start + norm_length) / len(normalized)
        
        # Estimate positions in original text
        orig_start_est = int(ratio_start * len(original))
        orig_end_est = int(ratio_end * len(original))
        
        # Expand search window
        window_size = max(50, norm_length * 2)
        search_start = max(0, orig_start_est - window_size)
        search_end = min(len(original), orig_end_est + window_size)
        
        # Try to find the best match in this window
        search_text = original[search_start:search_end]
        excerpt_to_find = normalized[norm_start:norm_start + norm_length]
        
        # Simple substring search in the window
        for i in range(len(search_text) - len(excerpt_to_find) + 1):
            window = search_text[i:i + len(excerpt_to_find)]
            if normalize_text_aggressive(window).lower() == excerpt_to_find.lower():
                return search_start + i, search_start + i + len(window)
        
        # Fallback: return estimated positions
        return max(0, orig_start_est), min(len(original), orig_end_est)

    def _expand_fallback_match(self, text: str, start_pos: int, end_pos: int, original_excerpt: str) -> tuple:
        """Expand a fallback match to include proper word boundaries and context."""
        # Find word boundaries around the match
        new_start = self._find_word_start(text, start_pos)
        new_end = self._find_word_end(text, end_pos)
        
        # Add some context words for better snippet quality
        words_before = self._get_n_words_before(text, new_start, 3)
        words_after = self._get_n_words_after(text, new_end, 3)
        
        # Calculate final boundaries
        final_start = max(0, new_start - len(words_before))
        final_end = min(len(text), new_end + len(words_after))
        
        return final_start, final_end

    def _find_word_start(self, text: str, pos: int) -> int:
        """Find the start of the word containing the given position."""
        if pos <= 0:
            return 0
        # Move backwards to find word boundary
        while pos > 0 and text[pos-1].isalnum():
            pos -= 1
        return pos

    def _find_word_end(self, text: str, pos: int) -> int:
        """Find the end of the word containing the given position."""
        if pos >= len(text):
            return len(text)
        # Move forwards to find word boundary
        while pos < len(text) and text[pos].isalnum():
            pos += 1
        return pos

    def _get_n_words_before(self, text: str, pos: int, n: int) -> str:
        """Get n words before the given position."""
        if pos <= 0:
            return ""
        
        # Look backwards from position to find word boundaries
        before_text = text[:pos]
        words = before_text.split()
        
        if len(words) <= n:
            return before_text
        else:
            return " ".join(words[-n:]) + " "

    def _get_n_words_after(self, text: str, pos: int, n: int) -> str:
        """Get n words after the given position."""
        if pos >= len(text):
            return ""
        
        # Look forwards from position to find word boundaries
        after_text = text[pos:]
        words = after_text.split()
        
        if len(words) <= n:
            return after_text
        else:
            return " " + " ".join(words[:n])

    def consolidate_snippets(self, original_text: str, chunk_results: List[dict], category: str, debug_print: bool = False, chunk_map: Optional[Dict[int, str]] = None) -> List[str]:
        """
        For each chunk, match excerpts in the chunk text first, then in the full text if not found.
        Tracks hit rate stats but doesn't print immediately.
        """
        all_excerpts = []
        chunk_indices = []
        for i, chunk_result in enumerate(chunk_results):
            if category in chunk_result:
                if isinstance(chunk_result[category], list):
                    for item in chunk_result[category]:
                        if isinstance(item, dict):
                            beginning = item.get("beginning excerpt", "")
                            ending = item.get("ending excerpt", "")
                            if beginning:  
                                all_excerpts.append((beginning, ending))
                                chunk_indices.append(i)
        
        found = 0
        snippets = []
        failed = []
        begin_fail_count = 0
        end_fail_count = 0
        
        # Also track strict matching stats for comparison
        strict_matches = 0
        
        for idx, (beginning, ending) in enumerate(all_excerpts):
            chunk_idx = chunk_indices[idx] if idx < len(chunk_indices) else 0
            chunk_text = chunk_map.get(chunk_idx) if chunk_map else None
            snippet = None
            
            # Track strict matching for comparison (before fallbacks)
            strict_begin = strict_find(original_text, beginning)
            strict_end = strict_find(original_text, ending) if ending and ending.strip() else True
            if strict_begin and strict_end:
                strict_matches += 1
            
            # Try in chunk first
            if chunk_text:
                snippet = self.find_snippet_in_text(chunk_text, beginning, ending)
                if debug_print and snippet:
                    print(f"[DEBUG] Found in chunk: '{beginning[:50]}...'")
            
            # If not found, try in full text
            if not snippet:
                snippet = self.find_snippet_in_text(original_text, beginning, ending)
                if debug_print and snippet:
                    print(f"[DEBUG] Found in full text: '{beginning[:50]}...'")
                elif debug_print:
                    print(f"[DEBUG] FAILED to find: '{beginning[:50]}...'")
                    # Show what letters-only matching looks like
                    letters_begin = letters_only(beginning)
                    letters_text = letters_only(original_text)
                    print(f"[DEBUG] Letters-only excerpt: '{letters_begin[:50]}...'")
                    print(f"[DEBUG] Letters-only contains: {letters_begin in letters_text}")
                    
                    # Now diagnose WHY the snippet extraction failed
                    begin_match = robust_find_improved(original_text, beginning)
                    end_match = robust_find_improved(original_text, ending) if ending else True
                    
                    print(f"[DEBUG] Failure analysis for '{beginning[:30]}...':")
                    print(f"[DEBUG]   Begin match: {begin_match is not None}")
                    print(f"[DEBUG]   End match: {end_match is not None} (ending: '{ending[:20]}...' if ending else 'None')")
            
            # Track overall snippet success AND failure reasons
            if snippet:
                if snippet not in snippets:
                    snippets.append(snippet)
                    found += 1
            else:
                # NOW track the actual failure reasons using STRICT matching (not the permissive fallback method)
                # Use direct text matching to see what actually failed
                begin_match = strict_find(original_text, beginning)

                # For ending, we need to distinguish between "no ending provided" vs "ending provided but failed"
                if ending and ending.strip():  # Only count as end failure if there was actually ending text
                    end_match = strict_find(original_text, ending)
                    if not end_match:
                        end_fail_count += 1
                else:
                    # No ending text provided, so this is purely a begin failure
                    pass

                if not begin_match:
                    begin_fail_count += 1
                
                failed.append((beginning, ending))
        
        # Track stats with detailed failure reasons
        total = len(all_excerpts)
        if category not in self.hit_rate_stats:
            self.hit_rate_stats[category] = {
                'found': 0, 'total': 0, 'failed_examples': [],
                'begin_failures': 0, 'end_failures': 0, 'strict_matches': 0
            }
        self.hit_rate_stats[category]['found'] += found
        self.hit_rate_stats[category]['total'] += total
        self.hit_rate_stats[category]['begin_failures'] += begin_fail_count
        self.hit_rate_stats[category]['end_failures'] += end_fail_count
        self.hit_rate_stats[category]['strict_matches'] += strict_matches
        if failed and len(self.hit_rate_stats[category]['failed_examples']) < 3:
            self.hit_rate_stats[category]['failed_examples'].extend(failed[:2])
        
        if debug_print:
            print(f"[DEBUG] Category '{category}': {found}/{total} matched ({100.0*found/total if total else 0:.1f}%)")
            print(f"[DEBUG] Strict matches (before fallbacks): {strict_matches}/{total} ({100.0*strict_matches/total if total else 0:.1f}%)")
            print(f"[DEBUG] Begin failures: {begin_fail_count}, End failures: {end_fail_count}")
            print(f"[DEBUG] Accounted for: {found + len(failed)}/{total} ({100.0*(found + len(failed))/total if total else 0:.1f}%)")
        
        return snippets

    def print_final_hit_rates(self):
        """Print aggregated hit rate statistics with detailed failure analysis."""
        print("\n" + "="*80)
        print("FINAL MATCHING STATISTICS")
        print("="*80)
        
        total_found = 0
        total_excerpts = 0
        total_begin_failures = 0
        total_end_failures = 0
        total_strict_matches = 0
        
        for category in sorted(self.hit_rate_stats.keys()):
            stats = self.hit_rate_stats[category]
            found = stats['found']
            total = stats['total']  
            begin_fail = stats.get('begin_failures', 0)
            end_fail = stats.get('end_failures', 0)
            strict_match = stats.get('strict_matches', 0)
            hit_rate = 100.0 * found / total if total else 0.0
            strict_rate = 100.0 * strict_match / total if total else 0.0
            
            # Calculate failure percentages
            begin_fail_pct = 100.0 * begin_fail / total if total else 0.0
            end_fail_pct = 100.0 * end_fail / total if total else 0.0
            
            print(f"{category:25s}: {found:3d}/{total:3d} ({hit_rate:4.1f}%) | Strict: {strict_match:3d} ({strict_rate:4.1f}%) | Begin fails: {begin_fail:2d} ({begin_fail_pct:4.1f}%) | End fails: {end_fail:2d} ({end_fail_pct:4.1f}%)")
            
            total_found += found
            total_excerpts += total
            total_begin_failures += begin_fail
            total_end_failures += end_fail
            total_strict_matches += strict_match
        
        overall_rate = 100.0 * total_found / total_excerpts if total_excerpts else 0.0
        overall_strict_rate = 100.0 * total_strict_matches / total_excerpts if total_excerpts else 0.0
        overall_begin_fail_rate = 100.0 * total_begin_failures / total_excerpts if total_excerpts else 0.0
        overall_end_fail_rate = 100.0 * total_end_failures / total_excerpts if total_excerpts else 0.0
        
        print("-" * 80)
        print(f"{'OVERALL':25s}: {total_found:3d}/{total_excerpts:3d} ({overall_rate:4.1f}%) | Strict: {total_strict_matches:3d} ({overall_strict_rate:4.1f}%) | Begin fails: {total_begin_failures:2d} ({overall_begin_fail_rate:4.1f}%) | End fails: {total_end_failures:2d} ({overall_end_fail_rate:4.1f}%)")
        print("="*80)
        
        # Show accounting verification
        accounted = total_found + (total_excerpts - total_found)
        print(f"ACCOUNTING CHECK: {accounted}/{total_excerpts} excerpts accounted for ({100.0*accounted/total_excerpts if total_excerpts else 0:.1f}%)")
        if total_begin_failures + total_end_failures == 0 and total_found < total_excerpts:
            print("⚠️  WARNING: No failures recorded but some excerpts are missing - accounting error detected!")

    async def codify(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        categories: Optional[Dict[str, str]] = None,
        user_instructions: str = "",
        max_words_per_call: int = 1000,
        max_categories_per_call: int = 8,
        additional_instructions: str = "",
        n_parallels: int = 400,
        model: str = "gpt-4o-mini",
        save_dir: str,
        file_name: str = "coding_results.csv",
        reset_files: bool = False,
        debug_print: bool = False,
        use_dummy: bool = False,
    ) -> pd.DataFrame:
        """
        Process all texts in the dataframe, coding passages according to categories.

        Args:
            df: Input dataframe
            column_name: Column containing text to code
            categories: Dict mapping category names to their definitions (optional)
            user_instructions: Instructions for dynamic category discovery when categories is None
            max_words_per_call: Maximum words per API call (default 1000)
            max_categories_per_call: Maximum categories per API call (default 8)
            additional_instructions: Additional instructions for the prompt
            n_parallels: Number of parallel API calls
            model: Model to use for coding
            save_dir: Directory for saving results
            file_name: Name of the CSV file for raw model responses
            reset_files: Whether to reset existing files
            debug_print: Whether to print debug information
            use_dummy: Whether to use dummy responses for testing

        Returns:
            Enhanced dataframe with columns for each category (if categories provided)
            or with 'coded_passages' column containing full category dict (if dynamic)
        """
        os.makedirs(save_dir, exist_ok=True)

        df_proc = df.reset_index(drop=True).copy()
        
        # Determine if we're in dynamic category mode
        dynamic_mode = categories is None
        
        # Create category batches for processing
        if dynamic_mode:
            # In dynamic mode, we have only one "batch" per text chunk
            category_batches = [None]  # Single batch for dynamic processing
        else:
            # Split categories into batches of max_categories_per_call
            category_keys = list(categories.keys())
            category_batches = [
                category_keys[i : i + max_categories_per_call] 
                for i in range(0, len(category_keys), max_categories_per_call)
            ]
        
        # Build prompts for all text chunks and category batches
        template = self.teleprompter.env.get_template("codify_prompt.jinja2")
        prompts: List[str] = []
        identifiers: List[str] = []
        text_index_to_chunks: Dict[int, List[int]] = {}  # Maps original text index to chunk indices

        chunk_idx = 0
        chunk_map = {}
        for text_idx, row in df_proc.iterrows():
            text = str(row[column_name])
            chunks = self.chunk_by_words(text, max_words_per_call)

            chunk_indices = []
            for chunk in chunks:
                # Process each category batch for this chunk
                for batch_idx, category_batch in enumerate(category_batches):
                    if dynamic_mode:
                        # Dynamic mode: use user_instructions
                        prompt = template.render(
                            text=chunk,
                            categories=None,
                            user_instructions=user_instructions,
                            additional_instructions=additional_instructions,
                        )
                        batch_suffix = ""
                    else:
                        # Static mode: use subset of categories
                        batch_categories = {k: categories[k] for k in category_batch}
                        prompt = template.render(
                            text=chunk,
                            categories=batch_categories,
                            user_instructions="",
                            additional_instructions=additional_instructions,
                        )
                        batch_suffix = f"_batch_{batch_idx}"

                    prompts.append(prompt)
                    identifiers.append(f"text_{text_idx}_chunk_{chunk_idx}{batch_suffix}")
                    chunk_indices.append(chunk_idx)
                    chunk_map[chunk_idx] = chunk
                    chunk_idx += 1

            text_index_to_chunks[text_idx] = chunk_indices
        
        if debug_print and prompts:
            print(f"\n[DEBUG] First prompt:\n{prompts[0][:500]}...\n")
            print(f"[DEBUG] Total chunks to process: {len(prompts)}")
        
        # Process all chunks - let the model handle JSON structure naturally
        expected_schema = None
        
        batch_df = await get_all_responses(
            prompts=prompts,
            identifiers=identifiers,
            n_parallels=n_parallels,
            save_path=os.path.join(save_dir, file_name),
            reset_files=reset_files,
            use_dummy=use_dummy,
            json_mode=True,
            expected_schema=expected_schema,
            model=model,
            timeout=300,  # This will be forwarded to get_response via **kwargs
            print_example_prompt=True,
        )
        
        # Group results by original text index and batch
        text_to_results: Dict[int, List[dict]] = {}
        for ident, resp in zip(batch_df["Identifier"], batch_df["Response"]):
            # Parse identifier: text_X_chunk_Y[_batch_Z]
            parts = ident.split("_")
            text_idx = int(parts[1])
            
            if debug_print:
                print(f"[DEBUG] {ident}: resp type={type(resp)}")
                if isinstance(resp, list):
                    print(f"[DEBUG] resp is list with {len(resp)} elements")
                    if resp:
                        print(f"[DEBUG] first element type: {type(resp[0])}")
                        if isinstance(resp[0], str):
                            print(f"[DEBUG] first element content: {resp[0][:200]}...")
            
            # Handle the response structure - resp is already deserialized from JSON
            if isinstance(resp, list) and resp:
                main = resp[0]  # Get the first response
            else:
                main = resp
            
            # Parse the JSON string
            parsed = self.parse_json(main) or {}
            
            if debug_print:
                if not parsed:
                    print(f"[DEBUG] Failed to parse response for {ident}")
                else:
                    print(f"[DEBUG] Successfully parsed response with keys: {list(parsed.keys())}")
            
            if text_idx not in text_to_results:
                text_to_results[text_idx] = []
            text_to_results[text_idx].append(parsed)
        
        # Consolidate results for each text
        if dynamic_mode:
            # Dynamic mode: create single column with all discovered categories
            df_proc["coded_passages"] = None
            
            # Add progress bar for overall text processing
            text_iterator = df_proc.iterrows()
            if len(df_proc) > 1:
                text_iterator = tqdm(text_iterator, total=len(df_proc), 
                                   desc="Processing texts", leave=True)
            
            for text_idx, row in text_iterator:
                original_text = str(row[column_name])
                chunk_results = text_to_results.get(text_idx, [])
                
                # Merge all categories from all chunks and batches
                all_categories = {}
                for chunk_result in chunk_results:
                    for category, category_data in chunk_result.items():
                        if category not in all_categories:
                            all_categories[category] = []
                        # Extend with items from this chunk
                        if isinstance(category_data, list):
                            all_categories[category].extend(category_data)
                
                # Convert to actual snippets
                final_coded_passages = {}
                for category in all_categories.keys():
                    snippets = self.consolidate_snippets(original_text, chunk_results, category, debug_print=debug_print, chunk_map=chunk_map)
                    if snippets:  # Only include categories that have snippets
                        final_coded_passages[category] = snippets
                
                df_proc.at[text_idx, "coded_passages"] = final_coded_passages
                
                if debug_print:
                    total_snippets = sum(len(snippets) for snippets in final_coded_passages.values())
                    print(f"[DEBUG] Text {text_idx}: {len(final_coded_passages)} categories, {total_snippets} total snippets")
        else:
            # Static mode: create column for each predefined category
            for category in categories.keys():
                df_proc[category] = None
            
            # Add progress bar for overall text processing
            text_iterator = df_proc.iterrows()
            if len(df_proc) > 1:
                text_iterator = tqdm(text_iterator, total=len(df_proc), 
                                   desc="Processing texts", leave=True)
            
            for text_idx, row in text_iterator:
                original_text = str(row[column_name])
                chunk_results = text_to_results.get(text_idx, [])
                
                # For each category, consolidate snippets from all chunks and batches
                for category in categories.keys():
                    snippets = self.consolidate_snippets(original_text, chunk_results, category, debug_print=debug_print, chunk_map=chunk_map)
                    df_proc.at[text_idx, category] = snippets
                    
                    if debug_print:
                        print(f"[DEBUG] Text {text_idx}, Category '{category}': {len(snippets)} snippets found")
        
        # Save final results
        df_proc.to_csv(os.path.join(save_dir, "coded_passages.csv"), index=False)
        
        if debug_print:
            print(f"\n[DEBUG] Processing complete. Results saved to: {save_dir}")
            if dynamic_mode:
                all_categories = set()
                for coded_passages in df_proc["coded_passages"]:
                    if coded_passages:
                        all_categories.update(coded_passages.keys())
                
                for category in all_categories:
                    total_snippets = sum(
                        len(coded_passages.get(category, [])) 
                        for coded_passages in df_proc["coded_passages"] if coded_passages
                    )
                    print(f"[DEBUG] {category}: {total_snippets} total snippets found")
            else:
                for category in categories.keys():
                    total_snippets = sum(len(snippets) for snippets in df_proc[category] if snippets)
                    print(f"[DEBUG] {category}: {total_snippets} total snippets found")
        
        # At the very end, before returning:
        self.print_final_hit_rates()
        return df_proc 