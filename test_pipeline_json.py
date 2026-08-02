import tempfile
import unittest
import zlib
from pathlib import Path

import qbank_pipeline as qp
from qbank_pipeline import _dedupe_tables, _normalize_solution_payload, looks_truncated_solution, parse_gemini_json_array


def _write_test_pdf(path, texts, images):
    """Build a tiny single-page PDF (612x792) with Helvetica text at
    (x, y) -- y in PDF user space, origin bottom-left -- and one red image
    XObject per (obj_num, name, x, y). Pure-python; no poppler needed."""
    objects = {}
    stream_parts = []
    for i, (t, x, y, sz) in enumerate(texts):
        stream_parts.append(f"BT /F1 {sz} Tf {x} {y} Td ({t}) Tj ET".encode("latin-1"))
    w, h = 20, 10
    img_data = zlib.compress(b"\xff\x00\x00" * (w * h))
    xobjs = {}
    for obj_num, name, x, y in images:
        objects[obj_num] = (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
                            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
                            f"/Length {len(img_data)} >>\nstream\n{img_data.decode('latin-1')}\nendstream")
        xobjs[name] = f"{obj_num} 0 R"
        stream_parts.append(f"q {w} 0 0 {h} {x} {y} cm /{name} Do Q".encode("latin-1"))
    objects[5] = "<< /Length %d >>\nstream\n%s\nendstream" % (
        sum(len(p) + 1 for p in stream_parts), b"\n".join(stream_parts).decode("latin-1"))
    objects[1] = "<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
    xres = " ".join(f"/{n} {r}" for n, r in xobjs.items())
    objects[3] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                  f"/Resources << /Font << /F1 4 0 R >> /XObject << {xres} >> >> /Contents 5 0 R >>")
    objects[4] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    out = b"%PDF-1.4\n"
    offsets = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objects[num].encode("latin-1") + b"\nendobj\n"
    xref_pos = len(out)
    n = len(objects) + 1
    out += f"xref\n0 {n}\n".encode() + b"0000000000 65535 f \n"
    for num in sorted(objects):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    path.write_bytes(out)


class SolutionFigureMappingTests(unittest.TestCase):
    """Regression tests for the solutions-page figure mapping fix
    (user report: a 7-figure solutions page collapsed into 2 solutions
    because a single decoded header swallowed every image on the page)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._old_assets = qp.ASSETS_DIR
        qp.ASSETS_DIR = self.tmp / "assets"
        self.subj_dir = qp.ASSETS_DIR / "questions" / "PSY"
        self.subj_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        qp.ASSETS_DIR = self._old_assets

    def _claim(self, pdf, oids, recs, page=1):
        rels = []
        for oid in oids:
            fname = f"PSY-p{page}-{oid}.webp"
            (self.subj_dir / fname).write_bytes(b"x" * 3000)  # > MIN_IMAGE_BYTES
            rels.append(f"PSY/{fname}")
        image_files_by_q = {}
        leftover = qp.claim_page_images(rels, pdf, page, "PSY", 1, recs, image_files_by_q)
        return leftover, image_files_by_q

    def test_headers_located_with_positions_top_first(self):
        pdf = self.tmp / "solutions.pdf"
        _write_test_pdf(pdf, [
            ("Solution to Question 3:", 72, 700, 12),
            ("Explanation text for q3", 72, 660, 10),
            ("Solution to Question 7:", 72, 550, 12),
            ("Solution to Question 2:", 72, 400, 12),
        ], [])
        self.assertEqual(qp.solution_headers_on_page(pdf, 1, {2: {}, 3: {}, 7: {}}),
                         [(3, 700.0), (7, 550.0), (2, 400.0)])
        # a header whose q_no is not in the chapter is ignored
        self.assertEqual(qp.solution_headers_on_page(pdf, 1, {3: {}}), [(3, 700.0)])

    def test_each_figure_maps_to_its_own_solution_block(self):
        # 3 headers at y=700/550/400, one figure under each (y=640/490/340)
        pdf = self.tmp / "three_blocks.pdf"
        _write_test_pdf(pdf, [
            ("Solution to Question 3:", 72, 700, 12),
            ("Solution to Question 7:", 72, 550, 12),
            ("Solution to Question 2:", 72, 400, 12),
        ], [(6, "Im6", 300, 640), (7, "Im7", 300, 490), (8, "Im8", 300, 340)])
        recs = {2: {"has_figure_in_solution": True},
                3: {"has_figure_in_solution": True},
                7: {"has_figure_in_solution": True}}
        leftover, owned = self._claim(pdf, [6, 7, 8], recs)
        self.assertEqual(leftover, [])
        # each figure lands on the solution whose header is drawn above it
        self.assertEqual(owned[3]["solution"], ["PSY/PSY-001-003_SOL_01.webp"])
        self.assertEqual(owned[7]["solution"], ["PSY/PSY-001-007_SOL_01.webp"])
        self.assertEqual(owned[2]["solution"], ["PSY/PSY-001-002_SOL_01.webp"])

    def test_under_detected_headers_no_longer_swallow_the_page(self):
        # ONE decoded header but FIVE figures below it (the old code dumped
        # all five onto that one solution) -> cap at MAX_SOLUTION_IMAGES,
        # the rest stay unclaimed for the model/manual pass.
        pdf = self.tmp / "one_block.pdf"
        _write_test_pdf(pdf, [
            ("Solution to Question 3:", 72, 700, 12),
            ("Explanation text for q3", 72, 660, 10),
        ], [(6, "Im6", 300, 640), (7, "Im7", 300, 600), (8, "Im8", 300, 560),
            (9, "Im9", 300, 520), (10, "Im10", 300, 480)])
        leftover, owned = self._claim(pdf, [6, 7, 8, 9, 10], {3: {"has_figure_in_solution": True}})
        self.assertEqual(len(owned[3]["solution"]), qp.MAX_SOLUTION_IMAGES)
        self.assertEqual(len(leftover), 3)

    def test_figure_above_all_headers_is_not_guessed(self):
        # figure drawn ABOVE the only header -> no deterministic owner
        pdf = self.tmp / "above_header.pdf"
        _write_test_pdf(pdf, [
            ("Solution to Question 5:", 72, 400, 12),
        ], [(6, "Im6", 300, 500)])
        leftover, owned = self._claim(pdf, [6], {5: {"has_figure_in_solution": True}})
        self.assertEqual(leftover, ["PSY/PSY-p1-6.webp"])
        self.assertEqual((owned.get(5) or {}).get("solution") or [], [])

    def test_no_headers_means_no_auto_claim(self):
        pdf = self.tmp / "no_headers.pdf"
        _write_test_pdf(pdf, [("Plain text with no headers", 72, 700, 12)],
                        [(6, "Im6", 300, 500)])
        leftover, owned = self._claim(pdf, [6], {5: {"has_figure_in_solution": True}})
        self.assertIn("PSY/PSY-p1-6.webp", leftover)
        self.assertEqual((owned.get(5) or {}).get("solution") or [], [])


class RetryForeignFragmentGuardTests(unittest.TestCase):
    """Wrong-owner guard for targeted-retry solution continuations
    (external-audit 2026-08-02: q16's truncated re-ask returned q17's
    solution and the old code APPENDED it, blending two solutions)."""

    def setUp(self):
        self.rec = {"q_no": 16, "options": {"A": "alpha", "B": "beta",
                                            "C": "gamma", "D": "delta"},
                    "solution_text": "q16's own partial explanation"}
        self.chapter = {15: {"solution_text": "q15's solution text"},
                        16: self.rec,
                        17: {"solution_text": "q17's completely different "
                                              "explanation of q17's topic"}}

    def test_genuine_continuation_is_kept(self):
        frag = "and here the continuation continues without overlap"
        self.assertIsNone(qp._solution_fragment_foreign(frag, 16, self.rec, self.chapter))

    def test_foreign_option_line_head_blocked(self):
        # owner has no 'Option D' explanation topic matching this line
        frag = "Option D: the exact wording of some other question's option"
        self.assertIsNotNone(qp._solution_fragment_foreign(frag, 16, self.rec, self.chapter))

    def test_embedded_solution_header_for_another_question_blocked(self):
        frag = "text...\nSolution to Question 17: q17's completely different explanation"
        self.assertIsNotNone(qp._solution_fragment_foreign(frag, 16, self.rec, self.chapter))

    def test_own_header_not_blocked(self):
        frag = "text...\nSolution to Question 16: continued"
        self.assertIsNone(qp._solution_fragment_foreign(frag, 16, self.rec, self.chapter))

    def test_first_line_verbatim_in_sibling_blocked(self):
        # the retry fragment restates q17's solution -- its first line IS a
        # verbatim line of q17's solution (sibling-donor proof)
        self.chapter[17]["solution_text"] = ("q17's completely different explanation of q17's "
                                             "topic with lots more detail and then even more")
        frag = ("q17's completely different explanation of q17's topic with lots more "
                "detail and then even more\nand the fragment continues here")
        self.assertIsNotNone(qp._solution_fragment_foreign(frag, 16, self.rec, self.chapter))


class OrphanForeignGuardTests(unittest.TestCase):
    """recover_orphans rule-3 append must not glue a neighbour's solution
    onto a partial owner (audit foreign-tail candidates: 006-014, 011-017,
    011-026, 012-002, 014-015, 022-008)."""

    def test_foreign_fragment_blocked_and_kept_for_review(self):
        recs = {
            16: {"q_no": 16, "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                 "question_text": "q16 stem", "correct_option": "B",
                 "solution_text": "q16's own partial explanation", "tables": []},
            17: {"q_no": 17, "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                 "question_text": "q17 stem", "correct_option": "C",
                 "solution_text": "q17's completely different explanation of q17's "
                                  "topic with lots more detail and then even more",
                 "tables": []},
        }
        frag = ("q17's completely different explanation of q17's topic with lots more "
                "detail and then even more\nand the fragment continues here")
        orphans = [{"chapter_id": "PSY-016", "batch_start": 0, "pdf_pages": [1],
                    "new_pages": [1], "carry_q_no": None, "cut_part": None,
                    "last_qn_in_batch": 16,
                    "item": {"q_no": None, "question_text": None, "options": None,
                             "correct_option": None, "solution_text": frag,
                             "tables": [], "has_figure_in_question": False,
                             "has_figure_in_solution": False}}]
        stats = {"orphans_recovered": 0, "foreign_fragments_blocked": 0,
                 "carry_merges": 0, "chapter_id": "PSY-016"}
        remaining = qp.recover_orphans(orphans, recs, "PSY", 16, stats)
        # the fragment must NOT be appended to q16
        self.assertNotIn("q17's completely", recs[16]["solution_text"])
        self.assertIn("q17's completely", recs[17]["solution_text"])
        # and must be kept for review with a blocked reason
        self.assertEqual(len(remaining), 1)
        self.assertIn("blocked_reason", remaining[0])
        self.assertIn("foreign", remaining[0]["blocked_reason"])
        self.assertEqual(stats["foreign_fragments_blocked"], 1)

    def test_genuine_continuation_still_appends(self):
        recs = {
            16: {"q_no": 16, "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                 "question_text": "q16 stem", "correct_option": "B",
                 "solution_text": "q16's own partial explanation", "tables": []},
        }
        frag = "and here the genuine continuation continues without any overlap"
        orphans = [{"chapter_id": "PSY-016", "batch_start": 0, "pdf_pages": [1],
                    "new_pages": [1], "carry_q_no": None, "cut_part": None,
                    "last_qn_in_batch": 16,
                    "item": {"q_no": None, "solution_text": frag, "tables": [],
                             "question_text": None, "options": None,
                             "correct_option": None}}]
        stats = {"orphans_recovered": 0, "foreign_fragments_blocked": 0,
                 "carry_merges": 0, "chapter_id": "PSY-016"}
        remaining = qp.recover_orphans(orphans, recs, "PSY", 16, stats)
        self.assertIn("genuine continuation", recs[16]["solution_text"])
        self.assertEqual(remaining, [])
        self.assertEqual(stats["orphans_recovered"], 1)


class GeminiJsonParserTests(unittest.TestCase):
    def test_parses_one_array(self):
        self.assertEqual(parse_gemini_json_array('[{"q_no": 1}]'), [{"q_no": 1}])

    def test_recovers_adjacent_arrays(self):
        response = '[{"q_no": 1}]\n[{"q_no": 2}]'
        self.assertEqual(
            parse_gemini_json_array(response),
            [{"q_no": 1}, {"q_no": 2}],
        )

    def test_recovers_newline_delimited_objects(self):
        response = '{"q_no": 1}\n{"q_no": 2}'
        self.assertEqual(
            parse_gemini_json_array(response),
            [{"q_no": 1}, {"q_no": 2}],
        )

    def test_recovers_json_prefixed_by_model_prose(self):
        self.assertEqual(
            parse_gemini_json_array('Here is the requested data:\n```json\n[{"q_no": 1}]\n```'),
            [{"q_no": 1}],
        )

    def test_table_makes_dangling_lead_in_complete(self):
        self.assertFalse(looks_truncated_solution("Stages are:", has_tables=True))
        self.assertTrue(looks_truncated_solution("Stages are:", has_tables=False))

    def test_missing_terminal_period_is_not_a_truncation_signal(self):
        self.assertFalse(looks_truncated_solution("This is a complete source sentence"))
        self.assertFalse(looks_truncated_solution("Complete OCR sentence "))

    def test_table_dedupe_prefers_full_overlap_capture_in_any_order(self):
        partial = {"markdown": "| Stage | Goal |\n|---|---|\n| One | Trust |\n| Two | Autonomy |"}
        full = {"markdown": "| Stage | Goal |\n|---|---|\n| One | Trust |\n| Two | Autonomy |\n| Three | Initiative |"}
        self.assertEqual(_dedupe_tables([partial, full]), [full])
        self.assertEqual(_dedupe_tables([full, partial]), [full])

    def test_inline_table_is_moved_out_of_solution_prose(self):
        text = "Explanation:\n| Phase | Result |\n|---|---|\n| Oral | Fixation |\nEnd."
        clean, tables = _normalize_solution_payload(text, [], 7)
        self.assertEqual(clean, "Explanation:\nEnd.")
        self.assertEqual(len(tables), 1)

    def test_plain_bullet_solution_and_words_are_unchanged(self):
        text = "• First clinical finding\n• the classic example of this is Stockholm syndrome"
        clean, tables = _normalize_solution_payload(text, [], 24)
        self.assertEqual(clean, text)
        self.assertEqual(tables, [])
        self.assertIn(" is Stockholm", clean)

    def test_same_source_table_is_kept_for_different_questions(self):
        table = {"markdown": "| Phase | Result |\n|---|---|\n| Oral | Fixation |"}
        # Dedupe is deliberately record-local: q7 and q8 may both print it.
        self.assertEqual(_dedupe_tables([table]), [table])
        self.assertEqual(_dedupe_tables([table]), [table])

    def test_rejects_non_json_tail(self):
        with self.assertRaises(ValueError):
            parse_gemini_json_array('[{"q_no": 1}] explanation')


if __name__ == "__main__":
    unittest.main()
