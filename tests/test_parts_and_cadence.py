import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bs4 import BeautifulSoup

from lib.classes import cadence
from lib.classes.asr_verifier import same_transcription
from lib.classes.html_tts_prep import prepare_html_for_tts


# ---------- parts of the book ----------

def test_parts_are_whole_chapters_of_about_equal_size():
    from lib.core import split_in_parts
    sizes = [10, 40, 25, 25, 50, 10, 30, 60, 20, 30]
    parts = split_in_parts(sizes, 5)
    assert len(parts) == 5
    # every chapter once, in order, runs of consecutive chapters
    assert [n for p in parts for n in p] == list(range(1, len(sizes) + 1))
    totals = [sum(sizes[n - 1] for n in p) for p in parts]
    assert max(totals) - min(totals) <= 40


def test_parts_never_more_than_chapters_and_one_part_is_the_book():
    from lib.core import split_in_parts
    assert split_in_parts([5, 5], 5) == [[1], [2]]
    assert split_in_parts([3, 4, 5], 1) == [[1, 2, 3]]
    assert split_in_parts([3, 4, 5], 0) == [[1, 2, 3]]


# ---------- chapter names ----------

def test_untitled_chapter_gets_a_short_name():
    from lib.core import chapter_title_of
    sentences = ["Zachary Ezra Rawlins est assis par terre dans son placard, porte fermée, et il lit.[break]", 'Suite.']
    title = chapter_title_of(sentences, 5)
    assert title.startswith('Chapitre 5 — Zachary Ezra Rawlins est')
    assert title.endswith('…')
    assert len(title) < 60


def test_heading_and_subtitle_make_the_chapter_name():
    from lib.core import chapter_title_of
    sentences = ['Doux chagrins.[pause]', 'Jadis, il y a fort longtemps…[pause]', "Il y a un pirate dans la cave."]
    assert chapter_title_of(sentences, 1) == 'Doux chagrins — Jadis, il y a fort longtemps…'
    sentences = ['Partie un: Piranèse.[pause][break]', 'Statues.[pause][break]', 'Texte.']
    assert chapter_title_of(sentences, 1) == 'Partie un: Piranèse — Statues'
    assert chapter_title_of(['Une conversation.[pause]', 'Texte.'], 3) == 'Une conversation'
    sentences = ['Doux chagrins.[pause][break]', '… le Temps tomba amoureux du Destin.[pause][break]', 'Texte.[break][pause]']
    assert chapter_title_of(sentences, 2) == 'Doux chagrins — Le Temps tomba amoureux du Destin'
    # the pause closing the chapter does not make its last line a subtitle
    sentences = ['La Porte dans la Lune.[pause][break]', 'Le fils de la voyante se tient devant six portes,[break][pause]']
    assert chapter_title_of(sentences, 7) == 'La Porte dans la Lune'


def test_subtitle_paragraph_after_heading_is_read_as_a_heading():
    html = ('<body><h2 class="_5-niveau6">DOUX CHAGRINS</h2>'
            '<p class="_5-niveau6-soustitre">Jadis, il y a fort longtemps…</p>'
            '<p class="_1-texte"> </p><p class="_1-texte">Il y a un pirate dans la cave.</p></body>')
    soup = BeautifulSoup(html, 'html.parser')
    prepare_html_for_tts(soup, 'fra')
    headings = [h.get_text(strip=True) for h in soup.find_all(['h2', 'h3'])]
    assert headings == ['Doux chagrins.', 'Jadis, il y a fort longtemps…']
    assert soup.find('p', string=lambda s: s and 'pirate' in s) is not None


def test_ordinary_paragraph_is_not_promoted():
    html = '<body><p class="_1-texte">Janvier 2015</p><p class="_1-texte">Il y a un livre.</p></body>'
    soup = BeautifulSoup(html, 'html.parser')
    prepare_html_for_tts(soup, 'fra')
    assert soup.find(['h2', 'h3']) is None


# ---------- final intonation ----------

def test_cadence_targets_are_titles_and_last_chunk_but_not_questions():
    sentences = ['Doux chagrins.[pause]', 'Jadis…[pause]', 'Texte du milieu.', 'Fin du chapitre.[break]', '[break]']
    assert cadence.cadence_targets(sentences) == {0: 'titre', 1: 'titre', 3: 'fin'}
    assert cadence.cadence_targets(['Titre.[pause][break]', 'Fin.[break][pause]']) == {0: 'titre', 1: 'fin'}
    assert cadence.cadence_targets(['Titre ?[pause]', 'Vraiment ?']) == {}
    assert cadence.cadence_targets(['Il dit : « Viens ! »']) == {}
    # an unfinished fragment keeps its suspended tune
    assert cadence.cadence_targets(['La Porte dans la Lune.[pause]', 'Le fils de la voyante se tient devant six portes,']) == {0: 'titre'}


def test_falling_criterion():
    assert cadence.falls({'fall_st': -1.0, 'end_st': -1.5})
    assert not cadence.falls({'fall_st': -0.9, 'end_st': -3.0})
    assert not cadence.falls({'fall_st': -3.0, 'end_st': -1.4})
    assert not cadence.falls(None)
    assert not cadence.falls({'fall_st': None, 'end_st': None})


def test_choice_keeps_whisper_first_then_the_most_falling():
    flat_ok = {'cer': 0.0, 'cadence': {'fall_st': 0.5, 'end_st': 0.0}}
    falls_ok = {'cer': 0.03, 'cadence': {'fall_st': -2.0, 'end_st': -2.0}}
    falls_bad = {'cer': 0.20, 'cadence': {'fall_st': -9.0, 'end_st': -9.0}}
    assert cadence.choose_take([flat_ok, falls_ok, falls_bad], 0.04) is falls_ok
    # nothing under 4 %: the lowest CER, ties going to the most falling
    a = {'cer': 0.167, 'cadence': {'fall_st': -0.5, 'end_st': 0.0}}
    b = {'cer': 0.167, 'cadence': {'fall_st': -8.0, 'end_st': -7.0}}
    c = {'cer': 0.25, 'cadence': {'fall_st': -11.0, 'end_st': -10.0}}
    assert cadence.choose_take([a, b, c], 0.04) is b
    # a take meeting the criterion beats a deeper sum that does not
    d = {'cer': 0.0, 'cadence': {'fall_st': -2.8, 'end_st': -0.7}}
    e = {'cer': 0.0, 'cadence': {'fall_st': -1.7, 'end_st': -1.7}}
    assert cadence.choose_take([d, e], 0.04) is e


def test_lowercase_roman_after_a_chapter_word():
    from lib.core import roman2number
    assert roman2number('interlude ii.[pause][break]') == 'interlude 2.[pause][break]'
    assert roman2number('Livre IV') == 'Livre 4'
    assert roman2number('Il lit un livre. Il dit mi-temps.') == 'Il lit un livre. Il dit mi-temps.'
    assert roman2number('la partie de cartes') == 'la partie de cartes'


def test_babbling_tail_is_detected():
    from lib.classes.asr_verifier import has_extra_tail
    assert has_extra_tail('Il y a un pirate dans la cave.', "Il y a un pirate dans la cave d'Helto.")
    assert has_extra_tail("l'allégresse du soir.", "l'allégresse du soir-prompo.")
    assert not has_extra_tail('Il y a un pirate dans la cave.', 'Il y a un pirate dans la cave.')
    assert not has_extra_tail('Ils n\'en parlent pas.', 'Il n\'en parle pas.')


def test_identical_transcriptions_mean_whisper_is_wrong():
    assert same_transcription(['Il eut seize ans.', 'il eut seize ans', 'Il eut seize ans !'])
    assert not same_transcription(['Il eut seize ans.', 'Il eut 16 ans.'])
    assert not same_transcription(['Une seule.'])
