import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lib.core as core


def _norm(text:str)->str:
    return core.normalize_text(text, 'fra', 'fr', 'xtts')


def _numbers(text:str)->str:
    return core.math2words(text, 'fra', 'fr', 'xtts', True)


def test_quoted_words_get_no_pauses():
    assert _norm('Je crus discerner le mot « minotaure » gravé.') == 'Je crus discerner le mot minotaure gravé.'


def test_question_survives_punctuation_runs_and_dangling_commas_are_removed():
    assert _norm('Quelles Étoiles sont visibles?. Je ne sais pas.') == 'Quelles Étoiles sont visibles? Je ne sais pas.'
    assert _norm('quelques heures plus tôt (encore un Nuage).') == 'quelques heures plus tôt, encore un Nuage.'


def test_abbreviation_keeps_the_sentence_end():
    assert _norm('des vertèbres, etc. La boîte à biscuits.') == 'des vertèbres, et cetera. La boîte à biscuits.'
    assert _norm('Le Dr Ketterley arriva.') == 'Le Docteur Ketterley arriva.'
    assert _norm('O.K. reprit-il.') == 'Okay reprit-il.'


def test_ellipsis_is_kept():
    assert _norm('Euh... je ne sais pas.') == 'Euh… je ne sais pas.'


def test_french_decimal_comma_and_ranges():
    assert _numbers('Je mesure 1,83 mètre.') == 'Je mesure un virgule quatre-vingt-trois mètre.'
    assert _numbers('pages 19-32 et 40-44, voir') == 'pages dix-neuf à trente-deux et quarante à quarante-quatre, voir'
    assert _numbers('les numéros 1, 2 et 3') == 'les numéros un, deux et trois'


def test_closing_parenthesis_after_number_is_not_a_list_marker():
    assert ':' not in _numbers('le numéro 10) a commencé')
    assert _numbers('1) Vaincre la Mort') == 'un : Vaincre la Mort'


def test_elided_article_is_not_a_roman_numeral():
    assert core.roman2number("cela fait partie d'une série") == "cela fait partie d'une série"
    assert core.roman2number('Partie IV') == 'Partie 4'
