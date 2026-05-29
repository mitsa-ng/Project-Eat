"""
nlp_engine.py — Analysis Module (v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sends the full essay text to Phi-3 (via Ollama JSON mode) for proofreading.

Key improvements over v1:
  • Stricter system prompt with explicit examples — extracts all errors, not 1
  • Chunked processing for long essays (avoids Phi-3 context overflow)
  • 3-tier fault-tolerant JSON parser handles wrapped / embedded / line-by-line
  • Deduplication of repeated errors across chunks
  • analyse_text() convenience method for plain strings

Output format:
    [
      {"original": "intresting",   "type": "spelling", "suggestion": "interesting"},
      {"original": "eats",         "type": "grammar",  "suggestion": "eat"},
      {"original": "smells like a trash", "type": "semantic", "suggestion": "smells like trash"},
      ...
    ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import logging
import re
import requests
from typing import Union

logger = logging.getLogger(__name__)

# ── Ollama / Phi-3 config ─────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL_NAME   = "phi3"
TEMPERATURE  = 0.0
MAX_TOKENS   = 4096
CHUNK_WORDS  = 200     # split essays longer than this many words into chunks

# ── Prompts ───────────────────────────────────────────────────────────────────
# NOTE: "format":"json" is intentionally NOT used — it causes Phi-3 to collapse
# all errors into one object.  Plain-text mode with a strict few-shot prompt
# produces correct per-error objects for small models.

SYSTEM_PROMPT = """You are an English essay proofreader. Output a JSON array of errors only.

OUTPUT FORMAT — one object per error, exactly like this:
[
{"original":"recieved","type":"spelling","suggestion":"received"},
{"original":"she go","type":"grammar","suggestion":"she goes"},
{"original":"more better","type":"grammar","suggestion":"better"},
{"original":"the future is bright and colorful","type":"semantic","suggestion":"the future is bright"}
]

RULES:
- Output ONLY the JSON array. No words before or after it.
- "original" MUST be the exact text as written in the essay. Scan the essay carefully.
- NEVER output errors that are not actually present in the essay text.
- Each error is 1-4 words. Never full sentences.
- type = "spelling", "grammar", or "semantic".
- If no errors exist, output exactly: []"""

class NLPEngine:
    """
    Wraps Ollama Phi-3 for essay proofreading.

    Usage:
        engine = NLPEngine()
        errors = engine.analyse(ocr_words)      # pass OCREngine output dict
        errors = engine.analyse_text("She go…") # or a plain string
    """

    def __init__(self, model: str = MODEL_NAME, ollama_url: str = OLLAMA_URL):
        self.model      = model
        self.ollama_url = ollama_url
        logger.info(f"NLPEngine ready  model={self.model}  url={self.ollama_url}")

    # ── text preparation ──────────────────────────────────────────────────────

    @staticmethod
    def _words_to_text(ocr_words: Union[list, dict]) -> str:
        """Flatten OCREngine output → plain text string."""
        if isinstance(ocr_words, dict):
            flat = []
            for pg in sorted(ocr_words.keys()):
                flat.extend(ocr_words[pg])
            ocr_words = flat
        return " ".join(w["text"] for w in ocr_words if w.get("text", "").strip())

    @staticmethod
    def _chunk_text(text: str, max_words: int = CHUNK_WORDS) -> list:
        """
        Split text into overlapping sentence-boundary chunks so Phi-3 can
        analyse each piece without losing context across sentence borders.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks, current, current_words = [], [], 0

        for sent in sentences:
            wc = len(sent.split())
            if current_words + wc > max_words and current:
                chunks.append(" ".join(current))
                # keep last sentence as overlap context
                current      = [current[-1], sent] if current else [sent]
                current_words = len(" ".join(current).split())
            else:
                current.append(sent)
                current_words += wc

        if current:
            chunks.append(" ".join(current))

        return chunks or [text]

    # ── Ollama call ───────────────────────────────────────────────────────────

    def _call_ollama(self, text: str, chunk_index: int = 0, total_chunks: int = 0) -> str:
        """POST to Ollama /api/generate, return raw response string."""
        import time as _time
        word_count = len(text.split())
        logger.info(
            f"  Calling Ollama chunk {chunk_index}/{total_chunks} "
            f"({word_count} words, model={self.model}) …"
        )
        payload = {
            "model":   self.model,
            "prompt":  text,
            "system":  SYSTEM_PROMPT,
            "stream":  False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
            },
        }
        t0 = _time.time()
        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=600)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot reach Ollama.  Start it first:  ollama serve\n"
                "Then pull the model:                   ollama pull phi3"
            )
        except requests.exceptions.ReadTimeout:
            elapsed = _time.time() - t0
            logger.error(
                f"  Ollama read timeout after {elapsed:.0f}s "
                f"(chunk {chunk_index}/{total_chunks}, {word_count} words).\n"
                f"  Try: restart Ollama with 'ollama serve' or use a smaller model."
            )
            raise

        elapsed = _time.time() - t0
        data = resp.json()
        raw = data.get("response", "")
        logger.info(f"  Ollama responded in {elapsed:.1f}s  ({len(raw)} chars)")
        return raw

    # ── fault-tolerant JSON parser ────────────────────────────────────────────

    @staticmethod
    def _parse(raw: str) -> list:
        """
        Extract a JSON array from the model response.
        Handles: pure array, dict wrapper, JSON embedded in prose, line-by-line.
        """
        raw = raw.strip()
        if not raw:
            return []

        # Tier 1: direct parse
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for key in ("errors", "results", "corrections", "issues", "data"):
                    if isinstance(parsed.get(key), list):
                        return parsed[key]
                if "original" in parsed:
                    return [parsed]
                # flatten all list values
                out = []
                for v in parsed.values():
                    if isinstance(v, list):
                        out.extend(v)
                return out
        except json.JSONDecodeError:
            pass

        # Tier 2: extract outermost [...] block
        m = re.search(r'\[.+\]', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

        # Tier 3: line-by-line objects
        items = []
        for line in raw.splitlines():
            line = line.strip().rstrip(",")
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "original" in obj:
                        items.append(obj)
                except json.JSONDecodeError:
                    continue
        if items:
            return items

        # Tier 4: Phi-3 collapsed all errors into one object with list-as-string values
        # e.g. {"original": "[\'intresting\',\'peoples\']", "suggestion": "[\'interesting\',\'people\']"}
        try:
            collapsed = json.loads(raw)
            if isinstance(collapsed, dict):
                def _to_list(val):
                    if isinstance(val, list):
                        return [str(v).strip() for v in val]
                    inner = re.findall(r"[\'\"](.*?)[\'\"]", str(val))
                    return inner
                originals   = _to_list(collapsed.get("original",   []))
                types_      = _to_list(collapsed.get("type",       []))
                suggestions = _to_list(collapsed.get("suggestion", []))
                if originals:
                    result = []
                    for idx, orig in enumerate(originals):
                        result.append({
                            "original":   orig,
                            "type":       types_[idx]       if idx < len(types_)       else "grammar",
                            "suggestion": suggestions[idx]  if idx < len(suggestions)  else "",
                        })
                    logger.debug(f"  Tier 4 recovered {len(result)} error(s) from collapsed object")
                    return result
        except Exception:
            pass

        logger.warning("Could not parse model response — skipping chunk.")
        logger.debug(f"  Raw: {raw[:400]}")
        return []


    @staticmethod
    def _validate(errors: list) -> list:
        """
        Normalise and deduplicate error entries.

        Phi-3 may return items in several shapes:
          dict  {"original": "x", "type": "spelling", "suggestion": "y"}  <- ideal
          list  ["x", "spelling", "y"]   (3 elements)
          list  ["x", "y"]               (2 elements, type inferred)
          list  [{...}, {...}]            (nested dicts  - flatten first)
        All shapes are coerced to the canonical dict form.
        """
        valid_types = {"spelling", "grammar", "semantic"}
        seen, out   = set(), []

        # Expand top-level nested lists into a flat sequence
        flat_errors = []
        for item in errors:
            if isinstance(item, list):
                if item and isinstance(item[0], dict):
                    flat_errors.extend(item)   # list-of-dicts -> flatten
                else:
                    flat_errors.append(item)   # list-of-strings -> keep
            else:
                flat_errors.append(item)

        for item in flat_errors:
            # coerce list -> dict
            if isinstance(item, list):
                strings = [str(el).strip() for el in item if el is not None]
                if len(strings) >= 3:
                    item = {"original": strings[0], "type": strings[1], "suggestion": strings[2]}
                elif len(strings) == 2:
                    item = {"original": strings[0], "type": "grammar", "suggestion": strings[1]}
                else:
                    continue

            if not isinstance(item, dict):
                continue

            original   = str(item.get("original",   "") or "").strip()
            error_type = str(item.get("type",        "") or "").strip().lower()
            suggestion = str(item.get("suggestion",  "") or "").strip()

            if not original:
                continue
            # Reject sentences masquerading as errors (model hallucination):
            # a real error token is at most 6 words long
            if len(original.split()) > 6:
                logger.debug(f"  Skipping over-long error: {original!r}")
                continue
            if error_type not in valid_types:
                error_type = "grammar"
            if original in seen:
                continue
            seen.add(original)

            out.append({
                "original":   original,
                "type":       error_type,
                "suggestion": suggestion,
            })
        return out

    @staticmethod
    def _filter_useless(errors: list) -> list:
        """Remove errors where suggestion is identical to original (no-op)."""
        kept = []
        for e in errors:
            orig = e.get("original", "").strip()
            sugg = e.get("suggestion", "").strip()
            if orig.lower() == sugg.lower():
                logger.debug(f"  Filtered no-op: {orig!r} -> {sugg!r}")
                continue
            kept.append(e)
        if len(kept) < len(errors):
            logger.info(f"  Filtered {len(errors) - len(kept)} no-op(s)")
        return kept

    @staticmethod
    def _filter_hallucinations(errors: list, text: str) -> list:
        """
        Remove errors whose 'original' text does NOT appear verbatim (case-insensitive)
        in the essay text.  Phi-3 often emits errors from its few-shot prompt examples
        that don't exist in the actual essay — this filter catches those.
        """
        if not errors:
            return []
        text_lower = text.lower()
        kept = []
        for e in errors:
            orig = e.get("original", "").strip()
            if not orig:
                continue
            if orig.lower() in text_lower:
                kept.append(e)
            else:
                logger.debug(f"  Filtered hallucination: {orig!r} not in essay text")
        if len(kept) < len(errors):
            logger.info(f"  Filtered {len(errors) - len(kept)} hallucination(s)")
        return kept

    # ── public API ────────────────────────────────────────────────────────────

    def analyse_text(self, text: str) -> list:
        """
        Proofread a plain text string.
        Long essays are chunked; results are merged and deduplicated.
        """
        text = text.strip()
        if not text:
            return []

        chunks  = self._chunk_text(text)
        all_err = []

        for i, chunk in enumerate(chunks):
            logger.info(
                f"  Analysing chunk {i+1}/{len(chunks)}  "
                f"({len(chunk.split())} words) …"
            )
            raw    = self._call_ollama(chunk, chunk_index=i + 1, total_chunks=len(chunks))
            errors = self._parse(raw)
            all_err.extend(errors)
            logger.info(f"  Chunk {i+1}/{len(chunks)}: {len(errors)} error(s)")

        final = self._validate(all_err)
        final = self._filter_hallucinations(final, text)
        final = self._filter_useless(final)
        logger.info(f"Analysis complete: {len(final)} error(s)")
        return final

    def analyse(self, ocr_words: Union[list, dict]) -> list:
        """
        Proofread from OCREngine output.
        Converts word dicts to plain text then calls analyse_text().
        """
        text = self._words_to_text(ocr_words)
        if not text.strip():
            logger.warning("Empty OCR text — nothing to analyse.")
            return []
        word_count = len(text.split())
        logger.info(f"Analysing {word_count} words from OCR output …")
        return self.analyse_text(text)


# ── CLI smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s  %(message)s")

    engine = NLPEngine()

    # Built-in test (matches the essay visible in the screenshot)
    test_essay = (
        "Living ."
    )

    print("─── Essay ────────────────────────────────────────")
    print(test_essay[:200], "…")
    print("\n─── Errors ───────────────────────────────────────")

    errors = engine.analyse_text(test_essay)
    for e in errors:
        print(f"  [{e['type']:8s}]  {e['original']!r:30s} → {e['suggestion']!r}")

    if len(sys.argv) > 1:
        import json as _json
        with open(sys.argv[1], encoding="utf-8") as f:
            ocr = {int(k): v for k, v in _json.load(f).items()}
        errors = engine.analyse(ocr)
        out = sys.argv[1].replace("_ocr.json", "_errors.json")
        with open(out, "w", encoding="utf-8") as f:
            _json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"\nSaved → {out}")