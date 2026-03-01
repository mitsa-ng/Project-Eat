"""
annotator.py — Visualization Module (v2 — sidebar tags)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tag placement: SIDEBAR model — tags NEVER overlap body text.

Layout concept
──────────────
  ┌──────────────────────────────────┬─────────────────┐
  │                                  │                  │
  │   Essay text (untouched)         │  [SPE] interest  │←─── leader line
  │   error word highlighted ──────→ │  [GRA] eat       │←───
  │   another error ───────────────→ │  [SEM] trash     │←───
  │                                  │                  │
  └──────────────────────────────────┴─────────────────┘
        TEXT ZONE                        SIDEBAR

Steps per page:
  1.  Collect all (span_rect, error) pairs.
  2.  Sort by span vertical centre (top to bottom).
  3.  Reserve a right-side sidebar of width SIDEBAR_W.
  4.  Pack tags top-to-bottom in the sidebar with collision avoidance.
      If a tag's ideal Y (aligned to its span) would overlap the previous
      tag, push it down — no tag ever overlaps another.
  5.  Draw a dashed horizontal leader line from the span's right edge to
      the tag's left edge at the tag's vertical centre.

Color scheme
────────────
    spelling  → red    (1.0,  0.267, 0.267)
    grammar   → orange (1.0,  0.549, 0.0)
    semantic  → blue   (0.118, 0.565, 1.0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import fitz
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── visual constants ──────────────────────────────────────────────────────────
COLORS = {
    "spelling": (1.0,  0.267, 0.267),
    "grammar":  (1.0,  0.549, 0.0),
    "semantic": (0.118, 0.565, 1.0),
}

HIGHLIGHT_OPACITY = 0.22
UNDERLINE_WIDTH   = 1.2
BORDER_WIDTH      = 0.5

SIDEBAR_W         = 72     # reduced from 110 — fits "[SPE] interesting" at 6pt
SIDEBAR_GUTTER    = 4
SIDEBAR_PAD_TOP   = 4
TAG_H             = 9
TAG_GAP           = 2
TAG_FONT          = "helv"
TAG_FONT_SIZE     = 6.0    # reduced from 7.0 to fit narrower sidebar
TAG_PAD_X         = 3

LEADER_WIDTH      = 0.5
LEADER_DASH       = [2, 1]

LEGEND_W          = 62     # fits inside 72pt sidebar
LEGEND_H          = 9
LEGEND_GAP        = 2
LEGEND_MARGIN_R   = 4
LEGEND_MARGIN_T   = 4


class Annotator:
    """
    Sidebar annotation engine — tags are placed in a reserved right column
    so they can NEVER overlap the essay body text.

    Usage:
        ann = Annotator()
        ann.annotate_pdf("essay.pdf", ocr_words, errors, "essay_annotated.pdf")
    """

    # ── geometry / matching helpers ───────────────────────────────────────────

    @staticmethod
    def _clean(token: str) -> str:
        return re.sub(r"^[^\w]+|[^\w]+$", "", token).lower()

    @staticmethod
    def _box_to_rect(box: list) -> fitz.Rect:
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        return fitz.Rect(min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _group_by_line(rects: list, tol: float = 4.0) -> list:
        """
        Merge consecutive word rects that sit on the same line into one rect.
        This turns per-token rects back into per-line-segment rects so that
        a multi-word error spanning one line produces ONE tag, not N tags.
        Words on different lines (wrapped phrase) produce one rect per line.
        """
        if not rects:
            return []
        groups, cur = [], [rects[0]]
        for r in rects[1:]:
            # Same line = vertical centres within tol pts of each other
            prev_cy = (cur[-1].y0 + cur[-1].y1) / 2
            this_cy = (r.y0 + r.y1) / 2
            if abs(this_cy - prev_cy) <= tol:
                cur.append(r)
            else:
                merged = cur[0]
                for x in cur[1:]:
                    merged |= x
                groups.append(merged)
                cur = [r]
        merged = cur[0]
        for x in cur[1:]:
            merged |= x
        groups.append(merged)
        return groups

    def _find_spans_ocr(self, target: str, page_words: list) -> list:
        """
        Find all occurrences of target in the word list.
        Returns one fitz.Rect per LINE SEGMENT of each match — so a
        multi-word phrase on one line → 1 rect; a phrase that wraps across
        two lines → 2 rects (one per line).  Never one-rect-per-token.
        """
        target_tokens = [self._clean(t) for t in target.split() if t.strip()]
        if not target_tokens:
            return []
        n       = len(target_tokens)
        cleaned = [self._clean(w["text"]) for w in page_words]
        all_rects, i = [], 0
        while i <= len(cleaned) - n:
            if cleaned[i : i + n] == target_tokens:
                word_rects = [self._box_to_rect(page_words[i + j]["box"])
                              for j in range(n)]
                all_rects.extend(self._group_by_line(word_rects))
                i += n
            else:
                i += 1
        return all_rects

    @staticmethod
    def _find_spans_search(target: str, page: fitz.Page) -> list:
        return page.search_for(target, quads=False)

    def _find_spans(self, target: str, page: fitz.Page, page_words: list) -> list:
        spans = self._find_spans_ocr(target, page_words)
        return spans if spans else self._find_spans_search(target, page)

    # ── highlight / underline ─────────────────────────────────────────────────

    @staticmethod
    def _draw_highlight(shape: fitz.Shape, rect: fitz.Rect, color: tuple):
        shape.draw_rect(rect)
        shape.finish(
            fill         = color,
            fill_opacity = HIGHLIGHT_OPACITY,
            color        = color,
            width        = BORDER_WIDTH,
        )

    @staticmethod
    def _draw_underline(shape: fitz.Shape, rect: fitz.Rect, color: tuple):
        shape.draw_line(
            fitz.Point(rect.x0, rect.y1 + 1),
            fitz.Point(rect.x1, rect.y1 + 1),
        )
        shape.finish(color=color, width=UNDERLINE_WIDTH)

    # ── sidebar tag + leader line ─────────────────────────────────────────────

    @staticmethod
    def _draw_tag_in_sidebar(page: fitz.Page,
                              tag_rect: fitz.Rect,
                              span_rect: fitz.Rect,
                              label: str,
                              color: tuple):
        sh = page.new_shape()
        sh.draw_rect(tag_rect)
        sh.finish(fill=color, fill_opacity=0.90, color=color, width=0)
        sh.commit()

        page.insert_text(
            fitz.Point(tag_rect.x0 + TAG_PAD_X, tag_rect.y1 - 2.5),
            label,
            fontname = TAG_FONT,
            fontsize = TAG_FONT_SIZE,
            color    = (1, 1, 1),
        )

    # ── legend ────────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_legend(page: fitz.Page):
        items = [
            ("Spelling",  COLORS["spelling"]),
            ("Grammar",   COLORS["grammar"]),
            ("Semantic",  COLORS["semantic"]),
        ]
        pw   = page.rect.width
        sx0  = pw - SIDEBAR_W
        x0   = sx0 + (SIDEBAR_W - LEGEND_W) / 2
        x1   = x0 + LEGEND_W

        sh = page.new_shape()
        for i, (_, color) in enumerate(items):
            y0 = LEGEND_MARGIN_T + i * (LEGEND_H + LEGEND_GAP)
            r  = fitz.Rect(x0, y0, x1, y0 + LEGEND_H)
            sh.draw_rect(r)
            sh.finish(fill=color, fill_opacity=0.80, color=color, width=0)
        sh.commit()

        for i, (label, _) in enumerate(items):
            y0 = LEGEND_MARGIN_T + i * (LEGEND_H + LEGEND_GAP)
            page.insert_text(
                fitz.Point(x0 + 4, y0 + LEGEND_H - 3),
                f"■ {label}",
                fontname = TAG_FONT,
                fontsize = TAG_FONT_SIZE,
                color    = (1, 1, 1),
            )

    # ── sidebar separator line ────────────────────────────────────────────────

    @staticmethod
    def _draw_sidebar_divider(page: fitz.Page):
        pw  = page.rect.width
        ph  = page.rect.height
        sx0 = pw - SIDEBAR_W - SIDEBAR_GUTTER / 2
        sh  = page.new_shape()
        sh.draw_line(fitz.Point(sx0, 0), fitz.Point(sx0, ph))
        sh.finish(color=(0.85, 0.85, 0.85), width=0.4)
        sh.commit()

    # ── per-page annotation ───────────────────────────────────────────────────

    def _annotate_page(self, page: fitz.Page,
                       page_words: list, errors: list) -> int:
        pw = page.rect.width
        ph = page.rect.height

        sidebar_x0 = pw - SIDEBAR_W
        sidebar_x1 = pw - 2

        legend_bottom = (LEGEND_MARGIN_T +
                         len(COLORS) * (LEGEND_H + LEGEND_GAP) + 6)
        next_tag_y    = max(legend_bottom, SIDEBAR_PAD_TOP)

        # Step 1: collect (span_rect, error_type, color, label) tuples.
        # Use a seen-set keyed on rounded rect coords to avoid duplicate tags
        # when the NLP returns the same error multiple times.
        pairs    = []
        seen_pos = set()   # (x0, y0, x1, y1) rounded to 1 pt

        for error in errors:
            original   = error.get("original",   "")
            error_type = error.get("type",        "grammar")
            suggestion = error.get("suggestion",  "")
            color      = COLORS.get(error_type, COLORS["grammar"])

            spans = self._find_spans(original, page, page_words)
            if not spans:
                logger.debug(f"  No match: {original!r}")
                continue

            max_chars        = int((SIDEBAR_W - TAG_PAD_X * 2) / (TAG_FONT_SIZE * 0.52))
            suggestion_short = suggestion[:max(0, max_chars - 6)]
            label            = f"[{error_type[:3].upper()}] {suggestion_short}"

            for span in spans:
                key = (round(span.x0), round(span.y0),
                       round(span.x1), round(span.y1))
                if key in seen_pos:
                    continue      # skip exact duplicate position
                seen_pos.add(key)
                pairs.append((span, error_type, color, label))

        if not pairs:
            return 0

        # Step 2: sort top → bottom
        pairs.sort(key=lambda p: (p[0].y0 + p[0].y1) / 2)

        # Step 3: draw highlights + underlines
        hl_shape = page.new_shape()
        for (span, _, color, _) in pairs:
            self._draw_highlight(hl_shape, span, color)
            self._draw_underline(hl_shape, span, color)
        hl_shape.commit()

        # Step 4: assign sidebar Y positions + draw tags
        count = 0
        for (span, error_type, color, label) in pairs:
            span_cy  = (span.y0 + span.y1) / 2
            ideal_y0 = span_cy - TAG_H / 2
            tag_y0   = max(ideal_y0, next_tag_y)
            tag_y1   = tag_y0 + TAG_H

            if tag_y1 > ph - 2:
                tag_y0 = ph - TAG_H - 2
                tag_y1 = ph - 2

            tag_rect = fitz.Rect(sidebar_x0 + 2, tag_y0,
                                 sidebar_x1,      tag_y1)

            self._draw_tag_in_sidebar(page, tag_rect, span, label, color)
            next_tag_y = tag_y1 + TAG_GAP
            count += 1

        logger.debug(f"  Page {page.number+1}: {count} tag(s) in sidebar")
        return count

    # ── public API ────────────────────────────────────────────────────────────

    def annotate_pdf(
        self,
        input_pdf:  str,
        ocr_words:  dict,
        errors:     list,
        output_pdf: Optional[str] = None,
    ) -> str:
        if output_pdf is None:
            output_pdf = input_pdf.replace(".pdf", "_annotated.pdf")

        doc, total = fitz.open(input_pdf), 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            logger.info(f"Annotating page {page_num + 1}/{len(doc)} …")
            self._draw_sidebar_divider(page)
            self._draw_legend(page)
            total += self._annotate_page(page, ocr_words.get(page_num, []), errors)

        doc.save(output_pdf, garbage=4, deflate=True)
        doc.close()
        logger.info(f"Saved → {output_pdf}  ({total} annotation(s))")
        return output_pdf


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json as _json
    logging.basicConfig(level=logging.DEBUG,
                        format="%(levelname)-7s %(name)s  %(message)s")
    if len(sys.argv) < 4:
        print("Usage: python annotator.py essay.pdf ocr.json errors.json [out.pdf]")
        sys.exit(1)
    with open(sys.argv[2], encoding="utf-8") as f:
        ocr = {int(k): v for k, v in _json.load(f).items()}
    with open(sys.argv[3], encoding="utf-8") as f:
        err = _json.load(f)
    out = Annotator().annotate_pdf(
        sys.argv[1], ocr, err,
        sys.argv[4] if len(sys.argv) > 4 else None
    )
    print(f"Done → {out}")