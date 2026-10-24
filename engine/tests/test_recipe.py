"""Unit tests for the prompt rules and the recipe (no models)."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from acestep_mlx.prompt_rules import explicit_bpm, explicit_key, wants_vocals  # noqa: E402
from acestep_mlx.recipe import (  # noqa: E402
    PRESETS, Options, parse_options, plan_request, song_folder,
)

MINI, LAPTOP = PRESETS["mini"], PRESETS["laptop"]


class VocalsTests(unittest.TestCase):
    """A prompt is instrumental unless it names a voice."""

    def test_no_voice_named_is_instrumental(self):
        """Genre-only prompts, in both languages, ask for no voice."""
        for p in ("freetekno rave, 180 bpm, acid 303", "una ballata folk con chitarra",
                  "dark techno in A minor", "una canzone triste"):
            self.assertFalse(wants_vocals(p), p)

    def test_voice_words_ask_for_vocals(self):
        """Italian and English vocal words, any case."""
        for p in ("ballata con voce maschile calda", "Pop song with a female SINGER",
                  "rap italiano", "coro gospel", "techno with vocals", "brano cantato",
                  "techno vocals, female"):
            self.assertTrue(wants_vocals(p), p)

    def test_negations_win(self):
        """"senza voce" and "no vocals" contain a vocal word but mean the opposite."""
        for p in ("techno senza voce", "house, no vocals", "ambient without voice",
                  "pezzo strumentale con coro di synth"):
            self.assertFalse(wants_vocals(p), p)

    def test_words_inside_other_words_do_not_count(self):
        """Word boundaries: "vocoder", "raptor", "testosterone" are not voices."""
        for p in ("vocoder bassline", "raptor techno", "testosterone gabber", "dark trap beat"):
            self.assertFalse(wants_vocals(p), p)


class BpmAndKeyTests(unittest.TestCase):
    """Explicit BPM and key are read only when unambiguous and valid."""

    def test_bpm(self):
        """One value in range is read; none, two or out-of-range are not."""
        self.assertEqual(explicit_bpm("freetekno 180 bpm"), 180)
        self.assertEqual(explicit_bpm("house a 124BPM, poi 124 bpm"), 124)
        self.assertEqual(explicit_bpm("128.5 bpm"), 128)
        for p in ("fast techno", "120 bpm into 140 bpm", "900 bpm", "1180 bpm", "the 90s"):
            self.assertIsNone(explicit_bpm(p), p)

    def test_english_keys(self):
        """Letter plus optional accidental plus mode, written in upstream's form."""
        self.assertEqual(explicit_key("dark techno in A minor"), "A minor")
        self.assertEqual(explicit_key("in F# Major"), "F# major")
        self.assertEqual(explicit_key("Bbm acid"), "Bb minor")
        self.assertEqual(explicit_key("E♭ maj"), "Eb major")
        self.assertEqual(explicit_key("C sharp minor"), "C# minor")

    def test_italian_keys(self):
        """Solfège names map to letters."""
        self.assertEqual(explicit_key("ballata in la minore"), "A minor")
        self.assertEqual(explicit_key("Sol maggiore"), "G major")
        self.assertEqual(explicit_key("fa diesis minore"), "F# minor")

    def test_no_or_ambiguous_key(self):
        """Ordinary words and two different keys give nothing."""
        for p in ("I am happy", "Drum and bass", "A minor to C major", "la musica minore"):
            self.assertIsNone(explicit_key(p), p)


class OptionsTests(unittest.TestCase):
    """parse_options accepts the documented options and nothing else."""

    def test_defaults(self):
        """No options means the app's defaults."""
        self.assertEqual(parse_options(None, MINI), Options())

    def test_rejects_bad_options(self):
        """Unknown keys, wrong types, out-of-range durations, bad formats and names."""
        for raw in ({"tempo": 1}, {"literal": "yes"}, {"vocals": 1}, {"duration": "60"},
                    {"duration": True}, {"duration": 5}, {"duration": 241}, {"format": "flac"},
                    {"like": "../../etc"}, {"like": "x"}, []):
            with self.assertRaises(ValueError, msg=raw):
                parse_options(raw, MINI)

    def test_lyrics_are_trimmed_and_checked(self):
        """Own lyrics: stripped; blank, too long, with vocals off or with like refused."""
        self.assertEqual(parse_options({"lyrics": "  [Verse]\nCiao  \n"}, MINI).lyrics, "[Verse]\nCiao")
        for raw in ({"lyrics": " \n "}, {"lyrics": 3}, {"lyrics": "x" * 4001},
                    {"lyrics": "la", "vocals": False},
                    {"lyrics": "la", "like": "20261023-194021-11111111"}):
            with self.assertRaises(ValueError, msg=raw):
                parse_options(raw, MINI)
        self.assertEqual(len(parse_options({"lyrics": "x" * 4000}, MINI).lyrics), 4000)

    def test_duration_cap_follows_the_preset(self):
        """60 s is allowed on mini and refused on laptop."""
        self.assertEqual(parse_options({"duration": 60}, MINI).duration, 60)
        with self.assertRaises(ValueError):
            parse_options({"duration": 60}, LAPTOP)


class PlanTests(unittest.TestCase):
    """plan_request turns prompt and options into the decided defaults."""

    def test_default_plan(self):
        """Instrumental, LM chooses BPM and key, preset duration, raw float WAV."""
        plan = plan_request("dark techno", Options(), MINI)
        self.assertEqual((plan.instrumental, plan.bpm, plan.keyscale, plan.duration),
                         (True, None, None, 30.0))
        self.assertEqual((plan.format, plan.literal), ("wav", False))
        self.assertEqual(plan.sources, {"vocals": "prompt", "bpm": "lm", "keyscale": "lm"})

    def test_prompt_values_and_options(self):
        """Explicit values bind; options override the prompt's vocal reading."""
        plan = plan_request("techno 140 bpm in A minor with vocals",
                            Options(vocals=False, duration=60, format="mp3", literal=True), MINI)
        self.assertEqual((plan.instrumental, plan.bpm, plan.keyscale), (True, 140, "A minor"))
        self.assertEqual((plan.duration, plan.format), (60.0, "mp3"))
        self.assertEqual(plan.sources["vocals"], "option")


    def test_own_lyrics_mean_vocals_whatever_the_prompt_says(self):
        """Lyrics win over a prompt that names no voice; the plan carries them."""
        plan = plan_request("dark techno, 140 bpm", Options(lyrics="[Verse]\nNotte"), MINI)
        self.assertFalse(plan.instrumental)
        self.assertEqual(plan.lyrics, "[Verse]\nNotte")
        self.assertEqual(plan.sources["vocals"], "lyrics")
        self.assertEqual(plan.sources["lyrics"], "user")
        self.assertTrue(plan.literal)
        self.assertFalse(plan_request("dark techno", Options(), MINI).literal)
        self.assertIsNone(plan_request("dark techno", Options(), MINI).lyrics)


class SongFolderTests(unittest.TestCase):
    """song_folder only resolves real songs inside the output directory."""

    def test_resolves_existing_song_and_refuses_others(self):
        """A valid name with song.json resolves; missing or outside folders do not."""
        with tempfile.TemporaryDirectory() as root:
            name = "20261023-200509-01234567"
            os.makedirs(os.path.join(root, name))
            with open(os.path.join(root, name, "song.json"), "w") as fh:
                json.dump({}, fh)
            self.assertEqual(song_folder(root, name), os.path.realpath(os.path.join(root, name)))
            for bad in ("20261023-200509-deadbeef", "../x", "20261023-200509-0123/../.."):
                with self.assertRaises(ValueError, msg=bad):
                    song_folder(root, bad)

    def test_symlink_out_of_the_songs_dir_is_refused(self):
        """A well-named link to a song folder elsewhere does not resolve."""
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            with open(os.path.join(outside, "song.json"), "w") as fh:
                json.dump({}, fh)
            name = "20261023-200509-01234567"
            os.symlink(outside, os.path.join(root, name))
            with self.assertRaises(ValueError):
                song_folder(root, name)


if __name__ == "__main__":
    unittest.main()
