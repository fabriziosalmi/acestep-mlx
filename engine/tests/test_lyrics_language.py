"""lyrics_language on verses written for this test, the N9 lyrics, and texts it must not name."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bench", "n9"))

from acestep_mlx.prompt_rules import lyrics_language  # noqa: E402
from run_n9 import LYRICS_EN, LYRICS_IT  # noqa: E402

VERSES = {
    "it": ["[Verse]\nCammino da solo per le strade della città\nnon so dove andare ma non mi fermo più\n"
           "ogni finestra accesa è una storia che non so\ne tu sei ancora qui nel mio cuore",
           "Questa notte è nostra, come il mare\nnoi siamo il vento che non torna più\n"
           "dimmi cosa resta della luce\nperché ti cerco anche nel buio"],
    "en": ["[Verse]\nI walk alone through the empty town\nthe lights are on but nobody's home\n"
           "you said that we would never fall\nnow all I have is this old song",
           "We are the fire in the cold night air\nyour hand in mine, we don't care\n"
           "this is the moment, this is our time\nwhen the music plays it will be fine"],
    "es": ["[Verso]\nCamino solo por la ciudad\nlas luces brillan pero no hay nadie más\n"
           "el viento sopla y yo no sé\nsi esta noche todo va a volver",
           "Pero tú estás aquí, muy cerca de mí\nlos días pasan y no hay nada que decir\n"
           "es la canción de nuestro amor\nque suena siempre en el corazón"],
    "fr": ["[Couplet]\nJe marche seul dans les rues de la ville\nles lumières sont allumées mais tout est calme\n"
           "tu étais là, je ne sais pas pourquoi\nmais cette nuit est pour nous",
           "Nous sommes le vent qui passe sur la mer\nvous ne savez pas ce que je cherche ici\n"
           "toujours avec toi dans mon cœur\nquand la musique est finie"],
    "de": ["[Strophe]\nIch gehe allein durch die Stadt\ndie Lichter sind an, aber niemand ist da\n"
           "du hast gesagt, wir sind für immer\nund jetzt bin ich noch hier",
           "Wir sind der Wind in der kalten Nacht\nmein Herz schlägt auf dem Weg zu dir\n"
           "wenn die Musik spielt, sind wir nicht allein\nund das ist unser Lied"],
    "pt": ["[Verso]\nEu caminho sozinho pela cidade\nas luzes estão acesas mas não há ninguém\n"
           "você disse que isso era para sempre\nagora tudo é só uma canção",
           "Não sei o que esse vento quer dizer\nminha alma é muito maior que o mar\n"
           "aqui na noite eu canto essa canção\num coração que não quer parar"],
}


class LyricsLanguageTests(unittest.TestCase):
    """Names the language only when it clearly leads."""

    def test_every_verse_gets_its_language(self):
        """Two verses per language, six languages."""
        for lang, verses in VERSES.items():
            for verse in verses:
                self.assertEqual(lyrics_language(verse), lang, msg=verse[:40])

    def test_the_n9_lyrics(self):
        """The lyrics the LM misnamed in N9 ("ca", "unknown")."""
        self.assertEqual(lyrics_language(LYRICS_IT), "it")
        self.assertEqual(lyrics_language(LYRICS_EN), "en")

    def test_short_mixed_or_wordless_texts_get_none(self):
        """No clear leader: the LM's value stays."""
        for text in ("la la la\nla la la", "[Chorus]\noh oh oh yeah", "amore my love\nsempre forever",
                     "Marco, Giulia, Anna, Luca", ""):
            self.assertIsNone(lyrics_language(text), msg=text)

    def test_two_languages_without_a_clear_leader_get_none(self):
        """One Italian line and one English line: 4 hits against 5, no twice-as-many leader."""
        self.assertIsNone(lyrics_language("io ti amo e non ti lascio\nI love you and I never go"))

    def test_section_tags_do_not_count(self):
        """English words inside [tags] are not lyrics."""
        self.assertIsNone(lyrics_language("[Verse of the night and the day]\nla la"))


if __name__ == "__main__":
    unittest.main()
