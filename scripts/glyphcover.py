"""Shared glyph-coverage contract for the reportlab renderers (R280; precedent: console_safety.py).

The disease this closes
----------------------
reportlab answers "this font has no glyph for that codepoint" by silently emitting .notdef
(glyph 0): the page grows a full-width hole and the text layer reads U+0000, with no warning
anywhere. R280 measured 22 PDFs / 394 such holes across this repo (prereg X-2 / X-3), e.g. the
minus sign in "1−1/K" and in "−21.12°" simply does not appear on the page.

The contract
------------
Every character handed to the renderer is assigned a font that demonstrably carries it (cmaps
are read from the font files, not from a hand-written list), and a character no available font
can draw is a HARD ERROR naming (char, base font, count) - the same "缺图=硬报错，禁静默降级"
rule R254 applies to figures. Routing and the pre-render gate read the same coverage table, so
the gate cannot drift away from the router.

mix(text, base="Song") is a drop-in replacement for r272_render.mix: same signature, same
<b>/<super> placeholder handling, same "Latin sits in Times" convention. It only ever adds a
<span> where the alternative is a hole.
"""
import os
import re

HOUSE = {
    "Song": ("C:/Windows/Fonts/simsun.ttc", 0),
    "Hei": ("C:/Windows/Fonts/simhei.ttf", None),
    "Fang": ("C:/Windows/Fonts/simfang.ttf", None),
    "Times": ("C:/Windows/Fonts/times.ttf", None),
    "Times-Bold": ("C:/Windows/Fonts/timesbd.ttf", None),
    # Math carries the five codepoints no house font has (⇒ ⟨ ⟩ ⌊ ⌋). Cambria is the same serif
    # class as Times, so the 宋体 + Times New Roman convention of the report still holds.
    "Math": ("C:/Windows/Fonts/cambria.ttc", 0),
}

# carrier chain tried, in order, after the font the style already selected
FALLBACK = ("Times", "Math")

LATIN_RUN = re.compile(r"([A-Za-z0-9 !-/\:-@\[-`{-~°±×÷→∈⌊⌋√≈≠≤≥⟨⟩—–…'\"§¶†‡"
                       r"\.\,\;\:\(\)\[\]<>=%+\-/]+)")
_MARKS = ("\x01", "\x02", "\x03", "\x04")          # <b> </b> <super> </super> placeholders
_COV = {}
DEFECTS = {}


def coverage(font):
    """Codepoints a font can draw, read from that font file's own cmap (cached)."""
    if font not in _COV:
        if font not in HOUSE:
            raise KeyError("unknown font: " + font)
        path, idx = HOUSE[font]
        if not os.path.exists(path):
            raise SystemExit("FATAL: font file missing, coverage contract unverifiable: " + path)
        from fontTools import ttLib
        f = ttLib.TTFont(path, fontNumber=idx if idx is not None else 0, lazy=True)
        _COV[font] = set(f.getBestCmap().keys())
    return _COV[font]


def register_all(pdfmetrics, TTFont):
    """Register the five house fonts + the math carrier, refusing to start if one is absent."""
    for name, (path, idx) in HOUSE.items():
        if not os.path.exists(path):
            raise SystemExit("FATAL: font file missing, cannot honour 宋体+Times: " + path)
        pdfmetrics.registerFont(TTFont(name, path) if idx is None
                                else TTFont(name, path, subfontIndex=idx))
    pdfmetrics.registerFontFamily("Song", normal="Song", bold="Hei", italic="Song", boldItalic="Hei")
    pdfmetrics.registerFontFamily("Times", normal="Times", bold="Times-Bold",
                                  italic="Times", boldItalic="Times-Bold")
    pdfmetrics.registerFontFamily("Math", normal="Math", bold="Math", italic="Math", boldItalic="Math")


def carrier(ch, target):
    """Font that will actually draw ch, given the style/run picked `target`; None = unroutable."""
    cp = ord(ch)
    for cand in (target,) + FALLBACK:
        if cp in coverage(cand):
            return cand
    DEFECTS[(ch, target)] = DEFECTS.get((ch, target), 0) + 1
    return None


def route(text, target):
    """[(font, chunk)] - coalesced runs; font None marks a pass-through placeholder mark."""
    out, buf, buf_font = [], "", None

    def flush():
        if buf:
            out.append((buf_font, buf))

    for ch in text:
        if ch in _MARKS:                      # never wrap a tag placeholder, never break it in
            flush()
            out.append((None, ch))
            buf, buf_font = "", None
            continue
        f = target if ord(ch) in coverage(target) else carrier(ch, target)
        f = f or target                        # keep the string whole; assert_clean() refuses
        if f != buf_font:
            flush()
            buf, buf_font = ch, f
        else:
            buf += ch
    flush()
    return out


def mix(text, base="Song"):
    """Coverage-aware sibling of r272_render.mix (same signature, same conventions)."""
    text = (text.replace("&", "&amp;").replace("<b>", "\x01").replace("</b>", "\x02")
            .replace("<sup>", "\x03").replace("</sup>", "\x04"))
    text = re.sub(r"<(?!/?(?:b|sup)\b)[^>]+>", "", text)
    parts = []
    for chunk in LATIN_RUN.split(text):
        if not chunk:
            continue
        target = "Times" if re.search(r"[A-Za-z0-9]", chunk) else base
        for font, sub in route(chunk, target):
            parts.append(sub if font is None or font == base
                         else '<font face="%s">%s</font>' % (font, sub))
    out = "".join(parts)
    return (out.replace("\x01", "<b>").replace("\x02", "</b>")
            .replace("\x03", "<super>").replace("\x04", "</super>"))


def drain_defects():
    bad = dict(DEFECTS)
    DEFECTS.clear()
    return bad


def assert_clean(what="document"):
    """Pre-render gate: never draw a hole."""
    bad = drain_defects()
    if bad:
        lines = "".join("   U+%04X %r  base=%-11s x%d\n" % (ord(c), c, b, n)
                        for (c, b), n in sorted(bad.items(), key=lambda kv: -kv[1]))
        raise SystemExit("FATAL: glyph coverage contract violated while laying out %s - no "
                         "available font draws these codepoints; refusing to emit .notdef holes "
                         "(add a font or change the character):\n%s" % (what, lines))


# --------------------------------------------------------------------- artifact-side gate
def scan_pdf(path):
    """Text-layer truth of a rendered PDF, read from the artifact, never from the renderer."""
    from pypdf import PdfReader
    r = PdfReader(path)
    nul = fffd = img = 0
    for p in r.pages:
        t = p.extract_text() or ""
        nul += t.count(chr(0))
        fffd += t.count(chr(0xFFFD))
        res = p.get("/Resources")
        if res is None:
            continue
        xo = res.get_object().get("/XObject")
        if xo is None:
            continue
        for k, v in xo.get_object().items():
            try:
                if v.get_object().get("/Subtype") == "/Image":
                    img += 1
            except Exception:
                pass
    return {"path": path.replace("\\", "/"), "pages": len(r.pages), "image_xobjects": img,
            "nul": nul, "fffd": fffd, "bytes": os.path.getsize(path)}


def audit_source(text):
    """Census of non-ASCII codepoints in a manuscript and where each one would be drawn."""
    cjk = re.compile(r"[　-〿一-豈-﫺＀-￯]")
    seen = {}
    for ch in text:
        if ord(ch) < 0x80 or cjk.match(ch) or ch in "\r\n\t":
            continue
        seen[ch] = seen.get(ch, 0) + 1
    out = []
    for ch, n in sorted(seen.items(), key=lambda kv: -kv[1]):
        row = {"char": ch, "cp": ord(ch), "n": n}
        for b in ("Song", "Hei", "Times"):
            row["as_" + b] = carrier(ch, b) or "NOT_ROUTABLE"
        out.append(row)
    drain_defects()                     # audit only; do not poison a later assert_clean()
    return out
