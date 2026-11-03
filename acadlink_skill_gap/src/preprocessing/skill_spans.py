"""Deterministic skill-mention span finder over raw text.

Builds a token-stream index from canonical skill labels (canonical name +
aliases, both raw and abbreviation-expanded variants) and locates mentions of
those labels in free text as *contiguous token runs*, returning the original
character span of each mention.

This is the deterministic part of the weak-labeling pipeline (Layer 1/2):
no embeddings, no fuzzy — only exact token-sequence matches. Anything found
here is high-precision but low-recall evidence; recall is intentionally the
job of the ontology matcher (SkillMatcher) with its fuzzy/embedding stages.

Span mapping: text is tokenized with original character positions; each
streamed token may be abbreviation-expanded to several tokens, but all
expanded tokens remember the character span of the originating raw token.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from src.paths import PROCESSED_DIR
from src.preprocessing.text_norm import ABBREVIATIONS, normalize_text

TOKEN_RE = re.compile(r"[^a-z0-9]+", re.IGNORECASE)
# single-token patterns shorter than this are skipped (avoids 'c', 'go', 'r' noise);
# intention is that tech abbreviations like 'sql' or 'etl' still qualify at len 3.
MIN_SINGLE_TOKEN_LEN = 3

# abbreviation keys (normalized) usable for stream expansion; multi-word keys
# matched as contiguous runs of up to MAX_ABBR_WORDS raw tokens
MAX_ABBR_WORDS = 3
_ABBR_EXPANSIONS: list[tuple[list[str], list[str]]] = []
for abbr, full in ABBREVIATIONS.items():
    aw = normalize_text(abbr, expand_abbrev=False).split()
    if 1 <= len(aw) <= MAX_ABBR_WORDS:
        fw = normalize_text(full, expand_abbrev=False).split()
        if fw and fw != aw:
            _ABBR_EXPANSIONS.append((aw, fw))
_ABBR_EXPANSIONS.sort(key=lambda e: -len(e[0]))


@dataclass
class TokenStream:
    """Streamed tokens of one text.

    tokens: expanded token strings (lowercase)
    starts / ends: original character offsets of each expanded token
    raw_pos: index of the originating raw token (for ordering/debug)
    """
    tokens: list[str] = field(default_factory=list)
    starts: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)


def tokenize_stream(text: str, expand_abbrev: bool = True) -> TokenStream:
    """Tokenize raw text (NFKC, lowercased) preserving original offsets, and
    optionally expand abbreviation tokens in place."""
    raw_tokens: list[tuple[str, int, int]] = []  # (lower, start, end)
    for m in re.finditer(r"[a-z0-9]+", text.replace("\n", " ").replace("\t", " "), re.IGNORECASE):
        t = m.group(0).lower()
        if t:
            raw_tokens.append((t, m.start(), m.end()))
    if not expand_abbrev:
        return TokenStream([t for t, _, _ in raw_tokens],
                           [s for _, s, _ in raw_tokens],
                           [e for _, _, e in raw_tokens])
    stream = TokenStream()
    i = 0
    n = len(raw_tokens)
    while i < n:
        words = [raw_tokens[i][0]]
        matched = None
        window_max = min(MAX_ABBR_WORDS, n - i)
        for w in range(window_max, 0, -1):
            phrases = [raw_tokens[i + k][0] for k in range(w)]
            for abbr, full in _ABBR_EXPANSIONS:
                if phrases == abbr:
                    matched = (w, full)
                    break
            if matched:
                break
        if matched:
            w, full = matched
            start = raw_tokens[i][1]
            end = raw_tokens[i + w - 1][2]
            n_orig = len(stream.tokens)
            stream.tokens.extend(full)
            stream.starts.extend([start] * len(full))
            stream.ends.extend([end] * len(full))
            i += w
        else:
            t, s, e = raw_tokens[i]
            stream.tokens.append(t)
            stream.starts.append(s)
            stream.ends.append(e)
            i += 1
    return stream


class SkillSpanFinder:
    """Find contiguous-token mentions of canonical skills in raw text."""

    def __init__(self, canonical: pd.DataFrame | None = None,
                 min_single_token_len: int = MIN_SINGLE_TOKEN_LEN):
        self.canonical = canonical if canonical is not None else pd.read_parquet(
            PROCESSED_DIR / "canonical_skills.parquet")
        self.min_single = min_single_token_len
        self._pattern2skill: dict[str, list[tuple[str, str, str]]] = {}
        # pattern (space-joined expanded tokens) -> [(skill_id, label, label_kind)]
        self._token2patterns: dict[str, set[str]] = {}
        self._build()

    def _add(self, label: str, skill_id: str, kind: str) -> None:
        for text in (normalize_text(label, expand_abbrev=False),
                     normalize_text(label, expand_abbrev=True)):
            toks = text.split()
            if not toks:
                continue
            if len(toks) == 1 and len(toks[0]) < self.min_single:
                continue
            key = " ".join(toks)
            if len(self._pattern2skill.get(key, [])) == 0:
                for t in set(toks):
                    self._token2patterns.setdefault(t, set()).add(key)
            self._pattern2skill.setdefault(key, []).append((skill_id, label, kind))

    def _build(self) -> None:
        for rec in self.canonical.itertuples(index=False):
            sid = rec.skill_id
            aliases = getattr(rec, "aliases", None)
            if aliases is None or (hasattr(aliases, "__len__") and len(aliases) == 0):
                aliases = []
            elif isinstance(aliases, str):
                aliases = [a for a in aliases.split("\n") if a.strip()]
            elif not isinstance(aliases, list):
                aliases = list(aliases)
            self._add(str(rec.canonical_name), sid, "canonical")
            for a in aliases:
                if isinstance(a, str) and a.strip():
                    self._add(a.strip(), sid, "alias")

    def find(self, text: str) -> list[dict]:
        """Return mentions: [{skill_id, label, label_kind, span_start, span_end,
        span_text, pattern_len, confidence}] ordered by (label_kind_rank, -len,
        span_start). Contiguous token-run match only."""
        if not isinstance(text, str) or not text.strip():
            return []
        stream = tokenize_stream(text, expand_abbrev=True)
        toks, starts, ends = stream.tokens, stream.starts, stream.ends
        pos_by_token: dict[str, list[int]] = {}
        for idx, t in enumerate(toks):
            pos_by_token.setdefault(t, []).append(idx)
        # candidate skills via token->pattern index
        cand_patterns: set[str] = set()
        for t in set(toks):
            cand_patterns |= self._token2patterns.get(t, set())
        found: dict[str, dict] = {}
        for pat in cand_patterns:
            pat_toks = pat.split()
            first_positions = pos_by_token.get(pat_toks[0])
            if not first_positions:
                continue
            for p0 in first_positions:
                if p0 + len(pat_toks) > len(toks):
                    continue
                ok = True
                for k in range(1, len(pat_toks)):
                    if toks[p0 + k] != pat_toks[k]:
                        ok = False
                        break
                if not ok:
                    continue
                s_off = starts[p0]
                e_off = ends[p0 + len(pat_toks) - 1]
                span_text = text[s_off:e_off]
                for skill_id, label, kind in self._pattern2skill[pat]:
                    best = found.get(skill_id)
                    rank = 0 if kind == "canonical" else 1
                    if best is None or (rank, -len(pat_toks), -s_off) < (
                            best["kind_rank"], -best["pattern_len"], -best["span_start"]):
                        found[skill_id] = {
                            "skill_id": skill_id,
                            "label": label,
                            "label_kind": kind,
                            "kind_rank": rank,
                            "span_start": s_off,
                            "span_end": e_off,
                            "span_text": span_text,
                            "pattern_len": len(pat_toks),
                            "confidence": 0.97 if kind == "canonical" else 0.93,
                        }
        out = list(found.values())
        out.sort(key=lambda d: (d["kind_rank"], -d["pattern_len"], d["span_start"]))
        for d in out:
            d.pop("kind_rank")
        return out