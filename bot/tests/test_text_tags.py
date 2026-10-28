import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.text_tags import TagStreamStripper, strip_angle_tags, strip_tags


class StripTagsTest(unittest.TestCase):
    def test_square_tags(self):
        self.assertEqual(strip_tags("[confused] Постойте. [short pause] Кто вы?"), "Постойте. Кто вы?")

    def test_angle_tags_never_survive(self):
        self.assertEqual(strip_tags("<кричит> Что? <пауза>Нет."), " Что? Нет.")

    def test_sidecar_tokens(self):
        self.assertEqual(strip_tags("<|emotion:anger|> Нет."), "Нет.")

    def test_plain_text_untouched(self):
        self.assertEqual(strip_tags("Какой ещё код? Я 2 < 3."), "Какой ещё код? Я 2 < 3.")

    def test_angle_only_keeps_square(self):
        self.assertEqual(strip_angle_tags("<кричит>[annoyed] Нет."), "[annoyed] Нет.")


class StreamStripperTest(unittest.TestCase):
    def run_stream(self, chunks):
        stripper = TagStreamStripper()
        # Frames carry their own spaces; HTML collapses doubled ones, so compare words.
        return " ".join("".join(stripper.feed(c) for c in chunks).split())

    def test_tag_split_across_frames(self):
        self.assertEqual(self.run_stream(["Так", " [short", " pause]", " Кто", " вы?"]), "Так Кто вы?")

    def test_tag_split_inside_word(self):
        self.assertEqual(self.run_stream(["[sig", "hs] Ну", " ладно."]), "Ну ладно.")

    def test_single_frame_tag(self):
        self.assertEqual(self.run_stream(["[annoyed] Нет.", " Совсем."]), "Нет. Совсем.")

    def test_unclosed_bracket_is_released(self):
        stripper = TagStreamStripper()
        out = stripper.feed("[" + "а" * 60)
        self.assertEqual(out, "[" + "а" * 60)

    def test_reset_drops_pending(self):
        stripper = TagStreamStripper()
        self.assertEqual(stripper.feed("[sig"), "")
        stripper.reset()
        self.assertEqual(stripper.feed("Алло."), "Алло.")


if __name__ == "__main__":
    unittest.main()
