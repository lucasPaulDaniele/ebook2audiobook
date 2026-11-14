import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lib.core as core


def test_hyphenated_words_stay_glued():
    assert core.foreign2latin('Peut-être, dit-elle.', 'fra') == 'Peut-être, dit-elle.'
    assert core.foreign2latin('Okay reprit-il.', 'fra') == 'Okay reprit-il.'


def test_spaced_dialogue_dash_keeps_its_space():
    assert core.foreign2latin('- Bonjour, dit-il.', 'fra') == '- Bonjour, dit-il.'


def test_punctuation_is_followed_by_a_space():
    assert core.foreign2latin("Bonjour,monde. L'homme", 'fra') == "Bonjour, monde. L'homme"
