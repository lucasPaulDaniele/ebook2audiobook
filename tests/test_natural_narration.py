import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bs4 import BeautifulSoup

from lib.conf_lang import default_clause_split_words
from lib.conf_models import SML_TAG_PATTERN
from lib.classes.natural_chunker import NaturalChunker
from lib.classes.html_tts_prep import prepare_html_for_tts
from lib.classes.epub_sections import split_spine, is_navigation_doc


def _chunker(pack:bool=True)->NaturalChunker:
    return NaturalChunker(245, default_clause_split_words, SML_TAG_PATTERN, pack_sentences=pack)


def _text(html:str)->str:
    soup = BeautifulSoup(html, 'html.parser')
    prepare_html_for_tts(soup, 'fra')
    return ' '.join(soup.get_text(' ').split())


# ---------- chunking ----------

def test_whole_sentences_are_kept_and_short_ones_packed():
    text = ("Quand la Lune se leva dans la Troisième Salle Nord, je me rendis dans le Neuvième Vestibule pour assister "
            "à la jonction des trois Marées. C'est un phénomène qui survient seulement tous les huit ans.[break]")
    chunks = _chunker().chunk(text)
    assert len(chunks) == 1
    assert chunks[0].endswith('huit ans.[break]')


def test_long_sentence_is_split_on_a_comma_never_mid_clause():
    text = ("Je me faufilai dans la Niche de la Femme à la Ruche, puis attendis d'entendre les Flots gronder dans les "
            "Salles Inférieures et de sentir les Murs vibrer sous la force de ce qui était imminent, tandis que la "
            "Marée des Salles Nord se précipitait dans l'Escalier du milieu en remplissant le Vestibule d'une explosion "
            "de Mousse scintillante et blanche comme neige.")
    chunks = _chunker().chunk(text)
    assert len(chunks) == 2
    assert all(len(c) <= 245 for c in chunks)
    assert chunks[0].endswith(',')
    assert not any(c.endswith('ce qui') for c in chunks)


def test_paragraphs_and_dialogue_turns_are_never_merged():
    text = "Statues.[pause][break]Oui, répondis-je.[break]Comprends-tu?[break]Bien sûr.[break]"
    chunks = _chunker().chunk(text)
    assert chunks == ['Statues.[pause][break]', 'Oui, répondis-je.[break]', 'Comprends-tu?[break]', 'Bien sûr.[break]']


def test_incise_after_question_is_not_a_sentence_end_but_lowercase_after_period_is():
    chunker = _chunker(pack=False)
    assert chunker.chunk("Tu n'as vu personne d'autre dans le labyrinthe, n'est-ce pas? demanda-t-il.") == \
        ["Tu n'as vu personne d'autre dans le labyrinthe, n'est-ce pas? demanda-t-il."]
    assert chunker.chunk("Je pensais à la lettre de seize. seize est capable de tout.") == \
        ['Je pensais à la lettre de seize.', 'seize est capable de tout.']


def test_no_packing_mode_keeps_one_sentence_per_chunk():
    text = "Cela lui rappela les offensives aériennes au Vietnam. Le bruit. La danse hésitante des appareils surchargés."
    chunks = _chunker(pack=False).chunk(text)
    assert chunks == ['Cela lui rappela les offensives aériennes au Vietnam.', 'Le bruit.', 'La danse hésitante des appareils surchargés.']


def test_initials_are_not_sentence_ends():
    text = ("Une série de pièces de J. B. Priestley des années trente, que l'Autre m'avait décrite avec un enthousiasme "
            "étonnant lors de notre dernière rencontre dans la Deuxième Salle Sud-Ouest, juste avant la Marée.")
    assert not any(c.endswith('J.') or c.endswith('B.') for c in _chunker().chunk(text))


# ---------- markup preparation ----------

def test_small_caps_and_inline_tags_do_not_split_words():
    html = '<body><h3>E<small>NTRÉE POUR LE</small> P<small>REMIER</small> J<small>OUR DU</small> C<small>INQUIÈME</small> M<small>OIS</small></h3></body>'
    assert _text(html) == 'Entrée pour le premier jour du cinquième mois.'


def test_note_calls_notes_and_page_markers_are_removed():
    html = ('<body><p>La longue cuillère du diable<a epub:type="noteref" href="#n1">1</a> : Lawrence'
            '<span epub:type="pagebreak" title="12"></span> Arne-Sayles.</p>'
            '<aside epub:type="footnote" id="n1"><p>1. Allusion à une maxime.</p></aside></body>')
    assert _text(html) == 'La longue cuillère du diable : Lawrence Arne-Sayles.'


def test_superscripts_are_spoken():
    assert _text('<body><p>Le n<sup>o</sup> 5 est étiqueté.</p></body>') == 'Le numéro 5 est étiqueté.'
    assert _text('<body><p>La 13<sup>e</sup> pièce.</p></body>') == 'La treizième pièce.'
    assert _text('<body><p>Le 1<sup>er</sup> décembre.</p></body>') == 'Le premier décembre.'
    assert _text('<body><p>Au XIX<sup>e</sup> siècle.</p></body>') == 'Au dix-neuvième siècle.'


def test_all_caps_letters_become_sentence_case():
    html = '<body><p>R<small>ETOURNEZ DANS VOS</small> S<small>ALLES</small>, 16, <small>ET RÉFLÉCHISSEZ À VOTRE MÉCHANCETÉ</small>.</p></body>'
    assert _text(html) == 'Retournez dans vos salles, 16, et réfléchissez à votre méchanceté.'


def test_bullets_dialogue_dashes_and_missing_punctuation():
    html = ('<body><p><b>.</b> 1 lampe torche</p><p>— Je fais de mon mieux, répondit l’Autre.</p>'
            '<ol><li><p>Vaincre la Mort</p></li><li><p>Voler dans les Airs</p></li></ol></body>')
    assert _text(html) == '1 lampe torche. Je fais de mon mieux, répondit l’Autre. Vaincre la Mort; Voler dans les Airs.'


# ---------- chapters ----------

def _page(body:str, epub_type:str='bodymatter')->str:
    return f'<html><head><title>x</title></head><body epub:type="{epub_type}">{body}</body></html>'


def test_multi_chapter_document_is_split_and_part_title_merged():
    para = '<p>' + 'Le Monde est vaste. ' * 100 + '</p>'
    entries = ''.join(f'<div class="niv1"><h3 class="t">Titre {i}</h3><h3 class="st">Sous-titre</h3>{para}</div>' for i in range(3))
    docs = [('p1', _page('<section><h1>Partie 1</h1><h1>Piranèse</h1></section>')),
            ('p1chap1', _page(f'<section><div class="dev">{entries}</div></section>'))]
    out = split_spine(docs)
    assert len(out) == 3
    assert out[0][2] == 'Partie 1 : Piranèse — Titre 0'
    assert 'Partie 1 : Piranèse' in out[0][1]
    assert [t for _, _, t in out[1:]] == ['Titre 1', 'Titre 2']


def test_sub_sections_of_a_real_chapter_are_not_split():
    para = '<p>' + 'Texte. ' * 400 + '</p>'
    doc = _page(f'<h1>Chapitre 3</h1><h2 class="s">1</h2>{para}<h2 class="s">2</h2>{para}')
    assert len(split_spine([('c3', doc)])) == 1


def test_navigation_documents_are_detected():
    soup = BeautifulSoup(_page('<nav epub:type="page-list"><h2>Pagination</h2><ol><li>1</li><li>2</li></ol></nav>'), 'html.parser')
    assert is_navigation_doc(soup)
