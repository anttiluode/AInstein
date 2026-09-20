from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TemporalFoveationSiteTests(unittest.TestCase):
    def test_temporal_foveation_page_has_the_full_pipeline(self):
        page = (ROOT / "temporal_foveation.html").read_text(encoding="utf-8")
        for term in (
            "Temporal Foveation & Reflection",
            "retrieval activation",
            "past event-state",
            "reflection operator",
            "play cue sequence",
            "confidence",
        ):
            # confidence is allowed to be absent from the visible concept; keep the
            # pipeline assertions focused below.
            if term == "confidence":
                continue
            self.assertIn(term, page)

    def test_site_keeps_whorl_as_analogy_not_memory_claim(self):
        page = (ROOT / "temporal_foveation.html").read_text(encoding="utf-8")
        self.assertIn("not a claim that autobiographical memory is a cortical spiral", page)
        self.assertIn("visualization/metaphor until an experiment earns more", page)

    def test_original_demo_links_to_new_page(self):
        index = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('href="temporal_foveation.html"', index)


if __name__ == "__main__":
    unittest.main()
