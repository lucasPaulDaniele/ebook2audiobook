"""
Rewrite an EPUB so that every chapter lives in its own spine document.

Usage:
    python tools/split_epub_chapters.py input.epub [output.epub]

Documents bundling several chapters (one repeated heading per chapter) are split,
title-only pages ("Part 1 / Title") are merged into the chapter that follows, and the
EPUB 3 nav + NCX tables of contents are rebuilt to list the resulting chapters.
"""

import os
import posixpath
import sys
import zipfile

from lxml import etree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.classes.epub_sections import split_spine

OPF_NS = 'http://www.idpf.org/2007/opf'
NCX_NS = 'http://www.daisy.org/z3986/2005/ncx/'
XHTML_NS = 'http://www.w3.org/1999/xhtml'
EPUB_NS = 'http://www.idpf.org/2007/ops'
NS = {'o': OPF_NS, 'n': NCX_NS, 'x': XHTML_NS, 'e': EPUB_NS}


def _xhtml(html:str)->bytes:
    if not html.lstrip().startswith('<?xml'):
        html = '<?xml version="1.0" encoding="utf-8"?>\n' + html
    return html.encode('utf-8')


def _parse_xml(data:bytes)->etree._Element:
    # XHTML files often use HTML named entities that a strict XML parser rejects
    return etree.fromstring(data.replace(b'&nbsp;', b'&#160;'))


def _rebuild_nav(nav_bytes:bytes, toc:list[tuple[str, str]], nav_dir:str)->bytes:
    root = _parse_xml(nav_bytes)
    navs = root.findall('.//x:nav', NS)
    target = next((n for n in navs if n.get(f'{{{EPUB_NS}}}type') == 'toc'), navs[0] if navs else None)
    if target is None:
        return nav_bytes
    for ol in target.findall('x:ol', NS):
        target.remove(ol)
    ol = etree.SubElement(target, f'{{{XHTML_NS}}}ol')
    for href, title in toc:
        li = etree.SubElement(ol, f'{{{XHTML_NS}}}li')
        a = etree.SubElement(li, f'{{{XHTML_NS}}}a')
        a.set('href', posixpath.relpath(href, nav_dir) if nav_dir else href)
        a.text = title
    return etree.tostring(root, xml_declaration=True, encoding='utf-8')


def _rebuild_ncx(ncx_bytes:bytes, toc:list[tuple[str, str]], ncx_dir:str)->bytes:
    root = _parse_xml(ncx_bytes)
    nav_map = root.find('n:navMap', NS)
    if nav_map is None:
        return ncx_bytes
    for np in nav_map.findall('n:navPoint', NS):
        nav_map.remove(np)
    for order, (href, title) in enumerate(toc, start=1):
        np = etree.SubElement(nav_map, f'{{{NCX_NS}}}navPoint', id=f'navPoint-{order}', playOrder=str(order))
        label = etree.SubElement(np, f'{{{NCX_NS}}}navLabel')
        etree.SubElement(label, f'{{{NCX_NS}}}text').text = title
        etree.SubElement(np, f'{{{NCX_NS}}}content', src=posixpath.relpath(href, ncx_dir) if ncx_dir else href)
    return etree.tostring(root, xml_declaration=True, encoding='utf-8')


def split_epub(src:str, dst:str)->int:
    with zipfile.ZipFile(src) as zin:
        container = etree.fromstring(zin.read('META-INF/container.xml'))
        opf_path = container.find('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile').get('full-path')
        opf_dir = posixpath.dirname(opf_path)
        opf = etree.fromstring(zin.read(opf_path))
        manifest = opf.find('o:manifest', NS)
        spine = opf.find('o:spine', NS)
        items = {i.get('id'): i for i in manifest.findall('o:item', NS)}

        def full(href:str)->str:
            return posixpath.normpath(posixpath.join(opf_dir, href)) if opf_dir else href

        itemrefs = spine.findall('o:itemref', NS)
        docs = []
        for ref in itemrefs:
            item = items[ref.get('idref')]
            docs.append((ref.get('idref'), zin.read(full(item.get('href')))))
        pieces = split_spine(docs)

        new_files:dict[str, bytes] = {}
        replaced_ids = set()
        new_spine = []
        toc = []
        for name, html, title in pieces:
            base_id, _, num = name.partition('#')
            item = items[base_id]
            href = item.get('href')
            if num or html != docs[[d[0] for d in docs].index(base_id)][1].decode('utf-8', errors='ignore'):
                replaced_ids.add(base_id)
                stem, ext = posixpath.splitext(href)
                new_href = f'{stem}_{int(num):02d}{ext}' if num else f'{stem}_merged{ext}'
                new_id = f'{base_id}_{int(num):02d}' if num else f'{base_id}_merged'
                new_files[full(new_href)] = _xhtml(html)
                etree.SubElement(manifest, f'{{{OPF_NS}}}item', id=new_id, href=new_href, attrib={'media-type': 'application/xhtml+xml'})
                new_spine.append(new_id)
                if title:
                    toc.append((full(new_href), title))
            else:
                new_spine.append(base_id)
        # title-only pages merged into the next chapter disappear from the spine
        kept = set(new_spine)
        for ref in itemrefs:
            if ref.get('idref') not in kept:
                replaced_ids.add(ref.get('idref'))
        for ref in itemrefs:
            spine.remove(ref)
        for idref in new_spine:
            etree.SubElement(spine, f'{{{OPF_NS}}}itemref', idref=idref)
        dropped_files = set()
        for item_id in replaced_ids:
            item = items[item_id]
            manifest.remove(item)
            dropped_files.add(full(item.get('href')))

        nav_item = next((i for i in manifest.findall('o:item', NS) if 'nav' in (i.get('properties') or '').split()), None)
        ncx_item = next((i for i in manifest.findall('o:item', NS) if i.get('media-type') == 'application/x-dtbncx+xml'), None)

        with zipfile.ZipFile(dst, 'w') as zout:
            zout.writestr(zipfile.ZipInfo('mimetype'), 'application/epub+zip', compress_type=zipfile.ZIP_STORED)
            for info in zin.infolist():
                n = info.filename
                if n == 'mimetype' or n in dropped_files:
                    continue
                data = zin.read(n)
                if n == opf_path:
                    data = etree.tostring(opf, xml_declaration=True, encoding='utf-8')
                elif nav_item is not None and n == full(nav_item.get('href')) and toc:
                    data = _rebuild_nav(data, toc, posixpath.dirname(n))
                elif ncx_item is not None and n == full(ncx_item.get('href')) and toc:
                    data = _rebuild_ncx(data, toc, posixpath.dirname(n))
                zout.writestr(n, data, compress_type=zipfile.ZIP_DEFLATED)
            for n, data in new_files.items():
                zout.writestr(n, data, compress_type=zipfile.ZIP_DEFLATED)
    return len(toc)


def main()->None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + '_chapitres.epub'
    count = split_epub(src, dst)
    print(f'{count} chapters written to {dst}')


if __name__ == '__main__':
    main()
