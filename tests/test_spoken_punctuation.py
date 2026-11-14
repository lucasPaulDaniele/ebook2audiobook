import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.classes.asr_verifier import (CONFIRM_CER, SCORE_VERSION, SPOKEN_PUNCT_PENALTY, cer, has_extra_tail, score,
                                      spoken_punctuation, stored_score)


# transcriptions heard by Whisper on "La Mer sans Étoiles"
CACHOT = 'La cave pourrait très bien être un cachot.'
CACHOT_HEARD = 'La cave pourrait très bien être un cachot point.'
LONG = ("Il traverse le hall en direction de l'ascenseur, toujours avec le chat dans son sillage, puis il change "
        "d'avis, fait demi-tour au milieu du couloir et regagne lentement sa chambre.")
LONG_HEARD = ("Il traverse le hall en direction de l'ascenseur, toujours avec le chat dans son sillage, puis il change "
              "d'avis, fait demi-tour au milieu du couloir et regagne lentement sa chambre point.")


def test_a_point_heard_but_not_written_is_counted():
    assert spoken_punctuation(CACHOT, CACHOT_HEARD) == 1
    assert spoken_punctuation('Le ressac.', 'Le ressac-point.') == 1
    assert spoken_punctuation('Fin.', 'Fin point point.') == 2
    assert spoken_punctuation(CACHOT, CACHOT) == 0


def test_a_point_of_the_text_is_not_a_defect():
    text = 'Il est sûr sur ce point que la porte est fermée.'
    assert spoken_punctuation(text, text) == 0
    assert spoken_punctuation('Les minuscules points et virgules.', 'Les minuscules points et virgules.') == 0


def test_a_long_sentence_with_a_spoken_point_is_now_above_the_threshold():
    # the CER alone lets it through: the extra word is a few % of a long sentence
    assert cer(LONG, LONG_HEARD) < CONFIRM_CER
    assert score(LONG, LONG_HEARD) > CONFIRM_CER
    assert score(LONG, LONG) == cer(LONG, LONG)


def test_a_take_without_the_point_beats_one_with_it():
    clean = score(CACHOT, 'La cave pourrait très bien être un cachot.')
    with_point = score(CACHOT, CACHOT_HEARD)
    assert clean < with_point
    # the babbling tail cut applies to it
    assert has_extra_tail(CACHOT, CACHOT_HEARD)


def test_old_results_get_the_penalty_new_ones_already_have_it():
    old = {'cer': cer(LONG, LONG_HEARD), 'asr': LONG_HEARD}
    assert abs(stored_score(LONG, old) - (old['cer'] + SPOKEN_PUNCT_PENALTY)) < 1e-9
    new = {'cer': round(score(LONG, LONG_HEARD), 4), 'asr': LONG_HEARD, 'v': SCORE_VERSION}
    assert stored_score(LONG, new) == new['cer']


def test_any_word_heard_after_the_end_of_the_text_is_a_parasite():
    # the extra "point" is often heard as another word, always after the last one
    assert spoken_punctuation('les mains toujours attachées à la chaise.', 'les mains toujours attachées à la chaise peint.') == 1
    assert spoken_punctuation('La Ballade de Simon et Eleanor.', 'La balade de Simon et Eleanor Plain') == 1
    assert spoken_punctuation('Il y a un pirate dans la cave.', 'Il y a un pirate dans la cave.') == 0


def test_chunks_of_a_corrected_part_to_check_for_the_new_control(tmp_path):
    import json
    from lib.classes.asr_verifier import AsrVerifier
    v = AsrVerifier({'process_dir': str(tmp_path), 'language_iso1': 'fr'}, str(tmp_path))
    chunks = [{'j': 0, 'wav': 'a.flac', 'text': CACHOT}, {'j': 1, 'wav': 'b.flac', 'text': CACHOT},
              {'j': 2, 'wav': 'c.flac', 'text': CACHOT}]
    with open(v.queue_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'block_id': 'B1', 'ch_num': 4, 'chunks': chunks}) + '\n')
        f.write(json.dumps({'block_id': 'B2', 'ch_num': 5, 'chunks': [{'j': 0, 'wav': 'd.flac', 'text': CACHOT}]}) + '\n')
    results = [
        {'wav': 'a.flac', 'model': 'small', 'cer': 0.1, 'asr': CACHOT_HEARD},
        # repaired since: the latest turbo transcription no longer hears it
        {'wav': 'b.flac', 'model': 'small', 'cer': 0.1, 'asr': CACHOT_HEARD},
        {'wav': 'b.flac', 'model': 'turbo', 'cer': 0.0, 'asr': CACHOT},
        {'wav': 'c.flac', 'model': 'small', 'cer': 0.0, 'asr': CACHOT},
        {'wav': 'd.flac', 'model': 'small', 'cer': 0.1, 'asr': CACHOT_HEARD},
    ]
    with open(v.results_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    assert v.spoken_punct_chunks({'B1'}) == {(4, 0)}
