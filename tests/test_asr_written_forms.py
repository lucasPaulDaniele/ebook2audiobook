import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.classes.asr_verifier import cer, has_extra_tail


def test_whisper_written_forms_are_read_like_the_prepared_text():
    pairs = [
        ("Envoyez-nous tout ce qui nécessite d'être lavé, monsieur Martin.", "Envoyez-nous tout ce qui nécessite d'être lavé, M. Martin."),
        ('Une épée de huit à dix centimètres de long.', 'Une épée de 8 à 10 cm de long.'),
        ('Il est sûr à quatre-vingt-quatorze pour cent.', 'Il est sûr à 94%.'),
        ('La Seizième Personne.', 'La 16e personne.'),
        ('la Cent Quatre-Vingt-Douzième Salle Ouest', 'la 192e salle ouest'),
        ('Entrée pour le vingt-huit novembre deux mille dix-huit.', 'Entrée pour le 28 novembre 2018.'),
        ('de deux virgule cinq à quatre virgule cinq kilogrammes', 'de 2,5 à 4,5 kg'),
        ('Il a mille livres.', 'Il a 1 000 livres.'),
        ('Cauchemar numéro quatre-vingt-trois:', 'Cauchemar numéro 83.'),
    ]
    for text, heard in pairs:
        assert cer(text, heard) == 0.0, (text, heard, cer(text, heard))


def test_a_tail_is_found_whatever_the_way_whisper_writes_numbers():
    assert not has_extra_tail('Janvier deux mille quinze.', 'Janvier 2015.')
    assert has_extra_tail('Janvier deux mille quinze.', 'Janvier 2015, point.')
