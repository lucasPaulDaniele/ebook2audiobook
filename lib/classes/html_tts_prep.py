"""
Clean an XHTML chapter before it is flattened to text for TTS.

Everything here works on the markup, where the structure is still known:
- notes and page markers are not part of the narration (noterefs, footnotes, pagebreaks)
- inline formatting must not split words ("E<small>NTRÉE</small>" -> "ENTRÉE")
- superscript abbreviations are spoken words ("n<sup>o</sup> 5" -> "numéro 5", "13<sup>e</sup>" -> "treizième")
- all-caps passages (letters, inscriptions) are read as normal text, they would otherwise
  hit acronym/roman-numeral conversions ("DIX" -> 509)
- list bullets and leading dialogue dashes are typography, not speech
- every heading, list item and paragraph ends with a punctuation so the TTS closes its
  intonation and the chunker sees a real boundary
"""

import regex as re

from bs4 import BeautifulSoup, NavigableString, Tag
from num2words2 import num2words

INLINE_TAGS = {
    'a', 'abbr', 'acronym', 'b', 'bdi', 'bdo', 'big', 'cite', 'code', 'del', 'dfn', 'em', 'font',
    'i', 'ins', 'kbd', 'label', 'mark', 'q', 's', 'samp', 'small', 'span', 'strike', 'strong',
    'sub', 'sup', 'time', 'tt', 'u', 'var'
}
BLOCK_TAGS = {'p', 'li', 'dd', 'dt', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'td', 'th', 'caption', 'figcaption', 'div'}
HEADING_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
SKIP_EPUB_TYPES = {
    'noteref', 'annoref', 'footnote', 'footnotes', 'endnote', 'endnotes', 'rearnote', 'rearnotes',
    'pagebreak', 'page-list', 'biblioref', 'glossref'
}
SKIP_ROLES = {'doc-noteref', 'doc-footnote', 'doc-endnote', 'doc-endnotes', 'doc-pagebreak', 'doc-backlink', 'doc-pagelist'}
END_PUNCT = '.!?…:;,'
CLOSERS = '»"”’\')]›'
BULLETS = '.•·▪■◦‣*●○-–—'
DIALOGUE_DASHES = '—–―-'

ROMAN_RE = re.compile(r'^[IVXLCDM]+$')
ROMAN_VALUES = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

# superscript abbreviations spoken as words, keyed by language
SUP_WORDS = {
    'fra': {
        ('n', 'o'): 'numéro', ('n', 'os'): 'numéros', ('N', 'o'): 'Numéro', ('N', 'os'): 'Numéros',
        ('M', 'me'): 'Madame', ('M', 'mes'): 'Mesdames', ('M', 'lle'): 'Mademoiselle', ('M', 'lles'): 'Mesdemoiselles',
        ('M', 'e'): 'Maître', ('M', 'gr'): 'Monseigneur', ('D', 'r'): 'Docteur', ('P', 'r'): 'Professeur',
        ('S', 't'): 'Saint', ('S', 'te'): 'Sainte', ('C', 'ie'): 'Compagnie',
    },
}
FR_ORDINAL_SUFFIXES = {'e', 'è', 'ème', 'eme', 'es', 'èmes', 'er', 're', 'ère', 'ers', 'res', 'ères', 'nd', 'nde', 'nds', 'ndes', 'd', 'de'}
EN_ORDINAL_SUFFIXES = {'st', 'nd', 'rd', 'th'}


def _tokens(value:str|list|None)->set[str]:
    if not value:
        return set()
    if isinstance(value, list):
        value = ' '.join(value)
    return set(value.lower().split())


def _is_note_or_marker(tag:Tag)->bool:
    if _tokens(tag.get('epub:type')) & SKIP_EPUB_TYPES:
        return True
    if _tokens(tag.get('role')) & SKIP_ROLES:
        return True
    if tag.name == 'aside' and 'note' in ' '.join(_tokens(tag.get('class'))):
        return True
    return False


def _is_link_marker(tag:Tag)->bool:
    # <sup><a href="#fn1">1</a></sup> or <a href="#n3">[3]</a> : a note call without semantics
    if tag.name not in ('a', 'sup'):
        return False
    text = tag.get_text('').strip()
    if not text or not re.fullmatch(r'[\[\(]?[\d*†‡§]{1,3}[\]\)]?', text):
        return False
    link = tag if tag.name == 'a' else tag.find('a')
    return link is not None and '#' in (link.get('href') or '')


def _roman_to_int(s:str)->int|None:
    if not ROMAN_RE.match(s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        val = ROMAN_VALUES[ch]
        total = total - val if val < prev else total + val
        prev = max(prev, val)
    return total if 0 < total < 4000 else None


def _ordinal(n:int, suffix:str, lang:str)->str|None:
    try:
        if lang == 'fra':
            if suffix in ('er', 'ers'):
                word = 'premier' if n == 1 else num2words(n, to='ordinal', lang='fr')
            elif suffix in ('re', 'ère', 'res', 'ères'):
                word = 'première' if n == 1 else num2words(n, to='ordinal', lang='fr')
            elif suffix in ('nd', 'nds'):
                word = 'second'
            elif suffix in ('nde', 'ndes'):
                word = 'seconde'
            else:
                word = num2words(n, to='ordinal', lang='fr')
            if suffix.endswith('s') and not word.endswith('s'):
                word += 's'
            return word
        if lang == 'eng':
            return num2words(n, to='ordinal', lang='en')
    except Exception:
        return None
    return None


INLINE_ORDINAL_RE = {
    # "1er", "2e", "XIXe" written without superscript: the digits/letters split of the
    # normaliser would otherwise read "1er" as "un er"
    'fra': re.compile(r'(?<![\p{L}\p{N}])(\d{1,4})(ères|ère|ers|er|res|re|èmes|ème|emes|eme|ndes|nde|nds|nd|es|e)(?![\p{L}\p{N}])'),
    'eng': re.compile(r'(?<![\p{L}\p{N}])(\d{1,4})(st|nd|rd|th)(?![\p{L}\p{N}])', re.IGNORECASE),
}


def speak_inline_ordinals(text:str, lang:str)->str:
    pattern = INLINE_ORDINAL_RE.get(lang)
    if pattern is None:
        return text

    def _sub(m):
        word = _ordinal(int(m.group(1)), m.group(2).lower(), lang)
        return word if word else m.group(0)

    return pattern.sub(_sub, text)


def _speak_superscript(sup:Tag, lang:str)->bool:
    """Replace a superscript abbreviation by its spoken form. Returns True when handled."""
    suffix = sup.get_text('').strip()
    prev = sup.previous_sibling
    if not suffix or not isinstance(prev, NavigableString):
        return False
    prev_text = str(prev)
    # n° / M^me / D^r …
    m = re.search(r'(?<![\p{L}])(\p{L}+)$', prev_text)
    if m and (m.group(1), suffix) in SUP_WORDS.get(lang, {}):
        word = SUP_WORDS[lang][(m.group(1), suffix)]
        prev.replace_with(prev_text[:m.start(1)] + word)
        sup.replace_with('')
        return True
    # 13^e, 1^er, 2^nd, XIX^e
    suffixes = FR_ORDINAL_SUFFIXES if lang == 'fra' else EN_ORDINAL_SUFFIXES if lang == 'eng' else set()
    if suffix in suffixes:
        m = re.search(r'(?<![\p{L}\p{N}])(\d{1,6}|[IVXLCDM]{1,8})$', prev_text)
        if m:
            token = m.group(1)
            n = int(token) if token.isdigit() else _roman_to_int(token)
            word = _ordinal(n, suffix, lang) if n else None
            if word:
                prev.replace_with(prev_text[:m.start(1)] + word)
                sup.replace_with('')
                return True
    return False


def _direct_text(tag:Tag)->str:
    return ''.join(str(c) for c in tag.children if isinstance(c, NavigableString))


def _is_text_block(tag:Tag)->bool:
    if tag.name not in BLOCK_TAGS:
        return False
    # a container whose text lives in nested blocks is handled through its children
    if tag.find(list(BLOCK_TAGS)) is not None and not _direct_text(tag).strip():
        return False
    return bool(tag.get_text('').strip())


def _mostly_uppercase(text:str)->bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 8:
        return False
    upper = sum(1 for c in letters if c.isupper())
    words = re.findall(r'\p{L}{3,}', text)
    return upper / len(letters) > 0.8 and len(words) >= 2


def _sentence_case(block:Tag)->None:
    capitalize = True
    for node in list(block.descendants):
        if not isinstance(node, NavigableString):
            continue
        out = []
        for ch in str(node).lower():
            if capitalize and ch.isalpha():
                out.append(ch.upper())
                capitalize = False
            else:
                out.append(ch)
            if ch in '.!?…':
                capitalize = True
        node.replace_with(''.join(out))


def _first_string(block:Tag)->NavigableString|None:
    for node in block.descendants:
        if isinstance(node, NavigableString) and str(node).strip():
            return node
    return None


def _last_string(block:Tag)->NavigableString|None:
    last = None
    for node in block.descendants:
        if isinstance(node, NavigableString) and str(node).strip():
            last = node
    return last


def _strip_leading_typography(block:Tag)->None:
    node = _first_string(block)
    if node is None:
        return
    text = str(node)
    stripped = text.lstrip()
    # bullet or dialogue dash followed by a space: "• item", "— Bonjour", "<b>.</b> 1 lampe"
    m = re.match(rf'^[{re.escape(BULLETS)}―]+(?:\s| )+', stripped)
    if m and len(stripped) > m.end():
        node.replace_with(stripped[m.end():])
    elif stripped and stripped in BULLETS + '―':
        node.replace_with('')


def _ensure_final_punctuation(block:Tag, punct:str)->None:
    node = _last_string(block)
    if node is None:
        return
    text = str(node)
    core = text.rstrip().rstrip(CLOSERS).rstrip()
    if not core or core[-1] in END_PUNCT:
        return
    if not any(c.isalnum() for c in core):
        return
    pos = len(core)
    node.replace_with(text[:pos] + punct + text[pos:])


SUBTITLE_CLASS_RE = re.compile(r'(?i)sous-?titre|subtitle|subhead|titre|title|-num\b')
SUBTITLE_MAX_CHARS = 160


def _promote_subtitles(body:Tag)->None:
    """A styled paragraph right after a heading is part of the title ("DOUX CHAGRINS" /
    "Jadis, il y a fort longtemps…", "Autre temps, autre lieu :" / "interlude i"): it is read
    as a heading, with the heading pause after it, not as the first line of the text."""
    for p in list(body.find_all('p')):
        text = p.get_text(' ', strip=True)
        if not text or len(text) > SUBTITLE_MAX_CHARS:
            continue
        if not any(SUBTITLE_CLASS_RE.search(c) for c in _tokens(p.get('class'))):
            continue
        prev = p.find_previous_sibling(True)
        while prev is not None and not prev.get_text(strip=True):
            prev = prev.find_previous_sibling(True)
        if prev is not None and prev.name in HEADING_TAGS:
            p.name = 'h3' if prev.name in ('h1', 'h2') else prev.name


def prepare_html_for_tts(soup:BeautifulSoup, lang:str)->None:
    body = soup.body or soup
    _promote_subtitles(body)
    # 1) remove notes, note calls and page markers
    for tag in list(body.find_all(True)):
        if tag.parent is None:
            continue
        if _is_note_or_marker(tag) or _is_link_marker(tag):
            tag.decompose()
    # 2) speak superscript abbreviations, then unwrap inline formatting
    for sup in list(body.find_all('sup')):
        if sup.parent is not None:
            _speak_superscript(sup, lang)
    for tag in list(body.find_all(list(INLINE_TAGS))):
        if tag.parent is not None:
            tag.unwrap()
    body.smooth()
    # 3) per text block typography and closing punctuation
    for block in list(body.find_all(list(BLOCK_TAGS))):
        if not _is_text_block(block):
            continue
        text = block.get_text('')
        if _mostly_uppercase(text):
            _sentence_case(block)
        _strip_leading_typography(block)
        if block.name in HEADING_TAGS:
            _ensure_final_punctuation(block, '.')
        elif block.name == 'li' or (block.parent is not None and block.parent.name == 'li'):
            # list items are read as one enumeration: "; " between items, "." after the last one
            item = block if block.name == 'li' else block.parent
            siblings = item.parent.find_all('li', recursive=False) if item.parent else []
            is_last = not siblings or siblings[-1] is item
            _ensure_final_punctuation(block, '.' if is_last else ';')
        else:
            _ensure_final_punctuation(block, '.')
    body.smooth()
