import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.classes.tts_engines.xtts import XTTS


def test_the_final_full_stop_is_not_given_to_the_model():
    assert XTTS._model_text('Il y a un pirate dans la cave.') == 'Il y a un pirate dans la cave'
    assert XTTS._model_text('Le bruit. La danse hésitante des appareils.') == 'Le bruit. La danse hésitante des appareils'


def test_other_sentence_ends_keep_their_tune():
    for text in ('Vraiment ?', 'Viens !', 'Il hésita…', 'Il hésita...', 'Sans ponctuation'):
        assert XTTS._model_text(text) == text
