"""
Prosody-aware chunking of a block of text for TTS.

Every chunk is synthesised by a separate TTS call, so each chunk boundary is a place where
the listener can hear the "glue". Boundaries must therefore sit where a narrator would
breathe anyway, and chunks must be long enough to carry a natural intonation contour but
short enough for the model to stay stable.

Algorithm, per paragraph (text between SML tags):
- candidate cut points are scored by syntactic strength: end of sentence > ; : >
  comma before a connector (et, mais, qui, lorsque…) > comma > connector without comma >
  plain space (last resort)
- a dynamic programme picks the cut points that minimise
  sum(chunk_penalty + length_cost(len) + boundary_cost(strength)) under the hard length
  limit, which packs short sentences together, keeps long sentences whole when they fit,
  and balances the pieces of a sentence that must be split
- paragraphs are never merged: a paragraph break is a dialogue turn or a narrative reset,
  both need their own silence and a fresh intonation (tiny replies are protected by the
  engine's duration QA instead)
SML tags stay attached to the end of the chunk they follow, so pauses keep their position.
"""

import regex as re

STRENGTH_SENTENCE = 5.0
STRENGTH_STRONG = 3.0       # ; :
STRENGTH_CONNECTOR = 2.0    # , + connector
STRENGTH_COMMA = 1.5
STRENGTH_BARE_CONNECTOR = 1.0
STRENGTH_SPACE = 0.0

BOUNDARY_COST = {
    STRENGTH_SENTENCE: 0.0,
    STRENGTH_STRONG: 2.0,
    STRENGTH_CONNECTOR: 4.0,
    STRENGTH_COMMA: 6.0,
    STRENGTH_BARE_CONNECTOR: 10.0,
    STRENGTH_SPACE: 30.0,
}
CHUNK_PENALTY = 2.0
TINY_CHUNK = 25

OPENERS = '«"“‘(\\[—–-'
CLOSERS = '»"”’)\\]'
# "? demanda-t-il" is an incise, not a sentence end; after a period a lowercase word still
# starts a sentence ("…de seize. seize est capable…", numbers spelled out)
_SENTENCE_END = re.compile(
    rf'[.!?…]+[{CLOSERS}]*(?=\s+(?:[\p{{Lu}}\p{{N}}{OPENERS}]|$))'
    rf'|\.[{CLOSERS}]*(?=\s+\p{{Ll}})'
    rf'|[.!?…]+[{CLOSERS}]*$'
)
_STRONG = re.compile(r'[;:](?=\s)')
_COMMA = re.compile(r',(?=\s)')
_INITIAL = re.compile(r'(?:^|\s)\p{Lu}\.$')


class NaturalChunker:

    def __init__(self, hard_max:int, clause_words:list[str], sml_pattern:re.Pattern, pack_sentences:bool=True):
        # pack_sentences=False keeps every sentence in its own chunk (only long sentences
        # are split): pauses between sentences are then fully controlled by the engine
        self.pack_sentences = pack_sentences
        self.hard_max = max(40, int(hard_max))
        self.ideal_min = int(self.hard_max * 0.40)
        self.ideal_max = int(self.hard_max * 0.82)
        self.sml_pattern = sml_pattern
        words = sorted({w.strip().lower() for w in clause_words if w.strip()}, key=len, reverse=True)
        alt = '|'.join(re.escape(w) for w in words) if words else r'(?!x)x'
        self.connector_after_comma = re.compile(rf',\s+(?=(?:{alt})\b)', re.IGNORECASE)
        self.bare_connector = re.compile(rf'(?<=\p{{L}})\s+(?=(?:{alt})\b)', re.IGNORECASE)

    # ---------- paragraphs ----------

    def _paragraphs(self, text:str)->tuple[list[str], list[dict]]:
        """Split into [{'text', 'after': [sml...]}]; returns (leading_sml, paragraphs)."""
        leading:list[str] = []
        paragraphs:list[dict] = []
        pos = 0
        for m in self.sml_pattern.finditer(text):
            chunk = text[pos:m.start()]
            if chunk.strip():
                paragraphs.append({'text': ' '.join(chunk.split()), 'after': []})
            tag = m.group(0)
            if paragraphs:
                paragraphs[-1]['after'].append(tag)
            else:
                leading.append(tag)
            pos = m.end()
        tail = text[pos:]
        if tail.strip():
            paragraphs.append({'text': ' '.join(tail.split()), 'after': []})
        return leading, paragraphs

    # ---------- cut points ----------

    def _cut_points(self, text:str)->dict[int, float]:
        points:dict[int, float] = {}

        def add(pos:int, strength:float)->None:
            # position = index where the next chunk starts (after the whitespace)
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if 0 < pos < len(text) and points.get(pos, -1) < strength:
                points[pos] = strength

        for m in re.finditer(r'\s+', text):
            add(m.end(), STRENGTH_SPACE)
        for m in self.bare_connector.finditer(text):
            add(m.end(), STRENGTH_BARE_CONNECTOR)
        for m in _COMMA.finditer(text):
            add(m.end(), STRENGTH_COMMA)
        for m in self.connector_after_comma.finditer(text):
            add(m.end(), STRENGTH_CONNECTOR)
        for m in _STRONG.finditer(text):
            add(m.end(), STRENGTH_STRONG)
        for m in _SENTENCE_END.finditer(text):
            # "J. B. Priestley": an initial is not a sentence end
            if _INITIAL.search(text[:m.end()]):
                continue
            add(m.end(), STRENGTH_SENTENCE)
        return points

    def _length_cost(self, n:int)->float:
        cost = 0.0
        if n < self.ideal_min:
            cost += 8.0 * (self.ideal_min - n) / self.ideal_min
        if n < TINY_CHUNK:
            cost += 20.0
        if n > self.ideal_max:
            cost += 6.0 * (n - self.ideal_max) / max(1, self.hard_max - self.ideal_max)
        return cost

    def _split_paragraph(self, text:str)->list[str]:
        points = self._cut_points(text)
        sentence_ends = sorted(p for p, st in points.items() if st == STRENGTH_SENTENCE)
        if len(text) <= self.ideal_max and (self.pack_sentences or not sentence_ends):
            return [text]
        positions = [0] + sorted(points) + [len(text)]
        strength = {len(text): STRENGTH_SENTENCE}
        strength.update(points)
        inf = float('inf')
        best = [inf] * len(positions)
        back = [-1] * len(positions)
        best[0] = 0.0
        for i in range(1, len(positions)):
            end = positions[i]
            b_cost = BOUNDARY_COST[strength[end]]
            for j in range(i - 1, -1, -1):
                start = positions[j]
                n = len(text[start:end].strip())
                if n > self.hard_max and j < i - 1:
                    break
                if best[j] == inf:
                    continue
                if not self.pack_sentences and any(start < p < end for p in sentence_ends):
                    break
                cost = best[j] + CHUNK_PENALTY + self._length_cost(n) + b_cost
                if cost < best[i]:
                    best[i] = cost
                    back[i] = j
        chunks = []
        i = len(positions) - 1
        while i > 0:
            j = back[i]
            chunks.append(text[positions[j]:positions[i]].strip())
            i = j
        return [c for c in reversed(chunks) if c]

    # ---------- public ----------

    def chunk(self, text:str)->list[str]:
        leading, paragraphs = self._paragraphs(text)
        result:list[str] = []
        for p in paragraphs:
            pieces = self._split_paragraph(p['text'])
            if not pieces:
                continue
            pieces[-1] = pieces[-1] + ''.join(p['after'])
            result.extend(pieces)
        if leading:
            if result:
                result[0] = ''.join(leading) + result[0]
        return result
