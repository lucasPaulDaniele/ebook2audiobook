"""
Split EPUB spine documents that bundle several chapters into one fragment per chapter.

Many publishers ship a whole "part" of a book in a single XHTML file, each chapter only
being marked by a repeated heading (e.g. <h3 class="int_niv1"> inside <div class="niv1">).
Converting such a file as a single block produces one huge chapter without navigation.

split_document() detects the repeated chapter heading, cuts the document right before
each occurrence and returns standalone XHTML strings (head + body attributes preserved).
merge_title_only() glues title-only pages ("Part 1 / Title") to the chapter that follows
so they do not become a 2-second audio chapter made of a lonely short sentence.
"""

import copy
import re

from bs4 import BeautifulSoup, NavigableString, Tag

HEADING_RE = re.compile(r'^h[1-6]$')

# a document is only split when the resulting chapters are long enough to be ones,
# this keeps short sub-sections of a real chapter (1.1, 1.2…) together
MIN_SECTION_CHARS = 300
MIN_AVG_SECTION_CHARS = 1500
# a page made only of headings and shorter than this is a part/title page
TITLE_ONLY_MAX_CHARS = 120

EXCLUDED_TYPES = {
    'frontmatter', 'backmatter', 'toc', 'titlepage', 'colophon',
    'acknowledgments', 'dedication', 'glossary', 'index',
    'appendix', 'bibliography', 'copyright-page', 'landmark',
    # epigraphs before the story: the audiobook starts where the narration starts
    'preamble', 'epigraph'
}


def _parse(html:str|bytes)->BeautifulSoup:
    if isinstance(html, bytes):
        html = html.decode('utf-8')
    return BeautifulSoup(html, 'html.parser')


def _text_len(node:Tag)->int:
    return len(' '.join(node.get_text(' ').split()))


def _letters_len(node:Tag)->int:
    return sum(1 for c in node.get_text('') if not c.isspace())


def _heading_level(tag:Tag)->int:
    return int(tag.name[1])


def _prev_tag(node:Tag)->Tag|None:
    sib = node.previous_sibling
    while sib is not None:
        if isinstance(sib, Tag):
            return sib
        if str(sib).strip():
            return None
        sib = sib.previous_sibling
    return None


def epub_type_of(soup:BeautifulSoup)->str:
    body = soup.body
    if not body:
        return ''
    # both levels count: publishers often type the body "bodymatter" and the section
    # inside it "preamble", "dedication"…
    epub_type = (body.get('epub:type') or '').lower()
    section = body.find('section')
    if section:
        epub_type = f"{epub_type} {(section.get('epub:type') or '').lower()}".strip()
    return epub_type


def is_navigation_doc(soup:BeautifulSoup)->bool:
    # EPUB 3 nav documents (toc, page-list, landmarks), also once calibre split them in pieces
    body = soup.body
    if body is None:
        return False
    navs = body.find_all('nav')
    if not navs:
        return False
    total = _letters_len(body)
    in_nav = sum(_letters_len(n) for n in navs)
    return total == 0 or in_nav / total > 0.8


# untyped front/back matter recognised by its heading (many French EPUBs carry no epub:type)
EXCLUDED_HEADINGS = {
    'remerciements', 'acknowledgments', 'acknowledgements', 'du meme auteur', 'du même auteur',
    'des memes auteurs', 'des mêmes auteurs', 'table des matieres', 'table des matières', 'sommaire',
    'contents', 'table of contents', 'copyright', 'credits', 'crédits', 'also by', 'by the same author',
}
COLOPHON_MARKERS = ('isbn', 'titre original', 'dépôt légal', 'depot legal', 'tous droits', 'all rights reserved',
                    'éditeur original', 'printed in', 'achevé d’imprimer', "achevé d'imprimer", 'directeurs de collection')


def _first_heading(body:Tag)->str:
    h = body.find(re.compile(r'^h[1-6]$'))
    return ' '.join(h.get_text(' ').split()).strip(' .:').lower() if h else ''


def is_colophon(soup:BeautifulSoup)->bool:
    # copyright / credits page: short, carries a © and publishing vocabulary
    body = soup.body
    if body is None:
        return False
    text = body.get_text(' ').lower()
    letters = _letters_len(body)
    return 0 < letters < 3000 and '©' in text and sum(m in text for m in COLOPHON_MARKERS) >= 1


def is_excluded(soup:BeautifulSoup)->bool:
    epub_type = epub_type_of(soup)
    if any(part in epub_type for part in EXCLUDED_TYPES) or is_navigation_doc(soup):
        return True
    body = soup.body
    if body is not None and _first_heading(body) in EXCLUDED_HEADINGS:
        return True
    return is_colophon(soup)


def heading_text(tag:Tag)->str:
    # get_text('') keeps small-caps runs glued: E<small>NTRÉE</small> -> ENTRÉE
    return ' '.join(tag.get_text('').replace('\xa0', ' ').split())


def _chapter_headings(body:Tag)->list[Tag]:
    groups:dict[tuple, list[Tag]] = {}
    all_levels = []
    for h in body.find_all(HEADING_RE):
        if not h.get_text(strip=True):
            continue
        all_levels.append(_heading_level(h))
        prev = _prev_tag(h)
        if prev is not None and HEADING_RE.match(prev.name or ''):
            # subtitle stacked under a title, never a chapter start
            continue
        key = (h.name, tuple(h.get('class') or ()))
        groups.setdefault(key, []).append(h)
    candidates = [(k, v) for k, v in groups.items() if len(v) >= 2]
    if not candidates:
        return []
    candidates.sort(key=lambda kv: (_heading_level(kv[1][0]), -len(kv[1])))
    heads = candidates[0][1]
    level = _heading_level(heads[0])
    # a single higher-level heading means we are inside one chapter: keep it whole
    if any(lv < level for lv in all_levels):
        return []
    return heads


def _lowest_common_ancestor(nodes:list[Tag])->Tag|None:
    paths = [list(reversed(list(n.parents))) for n in nodes]
    lca = None
    for level in zip(*paths):
        if all(x is level[0] for x in level):
            lca = level[0]
        else:
            break
    return lca


def _child_of(ancestor:Tag, node:Tag)->Tag:
    while node.parent is not ancestor:
        node = node.parent
    return node


def _index_path(root:Tag, node:Tag)->list[int]:
    path = []
    while node is not root:
        path.append(node.parent.contents.index(node))
        node = node.parent
    return list(reversed(path))


def _follow(root:Tag, path:list[int])->Tag:
    node = root
    for i in path:
        node = node.contents[i]
    return node


def split_document(html:str|bytes)->list[tuple[str, str|None]]:
    """
    Return [(xhtml, title)] — a single item with title None when the document
    does not hold several chapters.
    """
    soup = _parse(html)
    body = soup.body
    original = html.decode('utf-8') if isinstance(html, bytes) else html
    if body is None or is_excluded(soup):
        return [(original, None)]
    heads = _chapter_headings(body)
    if len(heads) < 2:
        return [(original, None)]
    lca = _lowest_common_ancestor(heads)
    if lca is None:
        return [(original, None)]
    anchors = [_child_of(lca, h) for h in heads]
    if len({id(a) for a in anchors}) != len(anchors):
        return [(original, None)]
    starts = [lca.contents.index(a) for a in anchors]
    # sections must be real chapters, otherwise leave the document untouched
    sizes = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lca.contents)
        sizes.append(sum(_text_len(c) if isinstance(c, Tag) else len(str(c).strip()) for c in lca.contents[start:end]))
    if min(sizes) < MIN_SECTION_CHARS or sum(sizes) / len(sizes) < MIN_AVG_SECTION_CHARS:
        return [(original, None)]
    lca_path = _index_path(soup, lca)
    body_path = _index_path(soup, body)
    results = []
    n = len(starts)
    for k in range(n):
        piece = copy.copy(soup)
        node = _follow(piece, lca_path)
        lo = 0 if k == 0 else starts[k]
        hi = starts[k + 1] if k + 1 < n else len(node.contents)
        for child in list(node.contents[hi:]) + list(node.contents[:lo]):
            child.extract()
        # content outside the chapters container: prefix goes to the first chapter,
        # suffix (footnotes…) to the last one
        cur = node
        piece_body = _follow(piece, body_path)
        while cur is not piece_body:
            parent = cur.parent
            idx = parent.contents.index(cur)
            if k > 0:
                for sib in list(parent.contents[:idx]):
                    sib.extract()
            if k < n - 1:
                for sib in list(parent.contents[parent.contents.index(cur) + 1:]):
                    sib.extract()
            cur = parent
        results.append((str(piece), heading_text(heads[k])))
    return results


def is_title_only(html:str|bytes)->bool:
    soup = _parse(html)
    body = soup.body
    if body is None or is_excluded(soup):
        return False
    total = _letters_len(body)
    if total == 0 or total > TITLE_ONLY_MAX_CHARS:
        return False
    in_headings = sum(_letters_len(h) for h in body.find_all(HEADING_RE))
    return in_headings == total


def title_only_text(html:str|bytes)->str:
    soup = _parse(html)
    parts = [heading_text(h) for h in soup.body.find_all(HEADING_RE) if h.get_text(strip=True)]
    return ' : '.join(parts)


def merge_titles_into(titles:list[str], target_html:str|bytes)->str:
    """
    Prepend the titles of title-only pages to the chapter that follows, one heading per
    page ("Partie 1 : Piranèse", then the chapter title), so the TTS reads each of them
    as a single natural utterance instead of a lonely short chapter.
    """
    soup = _parse(target_html)
    body = soup.body
    # insert inside the outermost wrapper so section/epub:type checks still see the content
    container = body
    while True:
        children = [c for c in container.children if isinstance(c, Tag)]
        if len(children) == 1 and children[0].name in ('section', 'div', 'article'):
            container = children[0]
            continue
        break
    for title in reversed(titles):
        new_heading = soup.new_tag('h2')
        new_heading['class'] = 'merged_title'
        new_heading.string = title
        container.insert(0, new_heading)
    return str(soup)


def _has_media(soup:BeautifulSoup)->bool:
    return soup.body is not None and soup.body.find(['img', 'image', 'svg', 'table']) is not None


def split_spine(docs:list[tuple[str, str|bytes]])->list[tuple[str, str, str|None]]:
    """
    docs: [(name, xhtml)] in reading order.
    Returns [(name, xhtml, title)] with multi-chapter documents split and runs of
    title-only pages (part title, chapter title…) merged into the following chapter.
    Empty pages caught inside such a run are dropped.
    """
    expanded:list[tuple[str, str, str|None]] = []
    for name, html in docs:
        pieces = split_document(html)
        if len(pieces) == 1:
            html_str = pieces[0][0]
            expanded.append((name, html_str, None))
        else:
            for i, (piece, title) in enumerate(pieces):
                expanded.append((f'{name}#{i + 1}', piece, title))
    merged:list[tuple[str, str, str|None]] = []
    pending:list[tuple[str, str, str|None]] = []

    def _flush()->None:
        merged.extend(pending)
        pending.clear()

    for name, html, title in expanded:
        soup = _parse(html)
        if soup.body is None or is_excluded(soup):
            _flush()
            merged.append((name, html, title))
            continue
        if is_title_only(html):
            pending.append((name, html, title))
            continue
        if _letters_len(soup.body) == 0:
            if pending and not _has_media(soup):
                continue
            _flush()
            merged.append((name, html, title))
            continue
        if pending:
            titles = [title_only_text(h) for _, h, _ in pending]
            full_title = ' — '.join(titles + ([title] if title else []))
            merged.append((name, merge_titles_into(titles, html), full_title))
            pending.clear()
            continue
        merged.append((name, html, title))
    _flush()
    return merged
