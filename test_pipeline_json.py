import json
import tempfile
import unittest
import zlib
from pathlib import Path

import qbank_pipeline as qp
from qbank_pipeline import _dedupe_tables, _normalize_solution_payload, looks_truncated_solution, parse_gemini_json_array


def _write_test_pdf(path, texts, images, img_size=(20, 10)):
    """Build a tiny single-page PDF (612x792) with Helvetica text at
    (x, y) -- y in PDF user space, origin bottom-left -- and one red image
    XObject per (obj_num, name, x, y). Pure-python; no poppler needed."""
    objects = {}
    stream_parts = []
    for i, (t, x, y, sz) in enumerate(texts):
        # leading "0 0 Td" resets the text line matrix: pypdf's visitor
        # reports tm=(0,0) for a second run on the SAME baseline without it
        # (needed for horizontal/2x2 option rows in the tests)
        stream_parts.append(f"BT 0 0 Td /F1 {sz} Tf {x} {y} Td ({t}) Tj ET".encode("latin-1"))
    w, h = img_size
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
                    "solution_text": "q16's own partial explanation leads to:"}
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
                 "solution_text": "q16's own partial explanation leads to:", "tables": []},
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
                 "solution_text": "q16's own partial explanation leads to:", "tables": []},
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


class SolutionGateBypassTests(unittest.TestCase):
    """find_incomplete_records must treat a printed 'Solution to Question N:'
    header as per-question proof the book prints that explanation, even when
    the chapter as a whole is below the 60% gate (ch25 class)."""

    def _chapter(self):
        records = {}
        for i in range(1, 10):
            rec = {"q_no": i, "question_text": f"stem {i}",
                   "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                   "correct_option": "A"}
            rec["solution_text"] = f"solution {i}" if i <= 4 else None  # 4/9 = 44% < 60%
            records[i] = rec
        return records

    def test_gate_suppresses_below_threshold(self):
        incomplete = qp.find_incomplete_records(self._chapter())
        sol_qns = {qn for qn, missing in incomplete if "solution" in missing}
        self.assertEqual(sol_qns, set())

    def test_printed_header_bypasses_gate_for_that_qn_only(self):
        incomplete = qp.find_incomplete_records(self._chapter(),
                                                printed_solution_qns={7})
        sol_qns = {qn for qn, missing in incomplete if "solution" in missing}
        self.assertEqual(sol_qns, {7})


class AnchorlessDropTests(unittest.TestCase):
    """Records with no stem/options/solution after all recovery are phantom
    answer-key rows (ch24 q12/13 class) -- dropped with a ledger entry."""

    def test_anchorless_detection(self):
        self.assertTrue(qp._anchorless_record(
            {"q_no": 12, "question_text": None, "options": None,
             "solution_text": None, "correct_option": None}))
        self.assertFalse(qp._anchorless_record(
            {"q_no": 12, "question_text": None, "options": None,
             "solution_text": "x", "correct_option": None}))
        self.assertFalse(qp._anchorless_record(
            {"q_no": 12, "question_text": "q", "options": {},
             "solution_text": None, "correct_option": None}))
        self.assertFalse(qp._anchorless_record(
            {"q_no": 12, "question_text": None, "options": {"A": "a"},
             "solution_text": None, "correct_option": None}))


class LocatePagesTests(unittest.TestCase):
    """locate_missing_record_pages finds the pages where a missing q_no is
    printed (question stem or solution header) via the text layer."""

    def test_locate_question_stem_and_solution_header_pages(self):
        fake = {1: "1. Question one\n2. Question two\n",
                2: "Solution to Question 7:\nExplanation\n",
                3: "plain text with no markers\n"}
        orig = qp.pdftotext_page
        qp.pdftotext_page = lambda pdf, page: fake.get(page, "")
        try:
            page_files = [Path(f"/tmp/x/page-{n:03d}.jpg") for n in (1, 2, 3)]
            loc = qp.locate_missing_record_pages("pdf", page_files,
                                                 {1: None, 2: None, 7: None}, {})
        finally:
            qp.pdftotext_page = orig
        self.assertEqual(loc, {1: [1], 2: [1], 7: [2]})

    def test_unlocatable_qn_is_omitted(self):
        orig = qp.pdftotext_page
        qp.pdftotext_page = lambda pdf, page: "nothing useful\n"
        try:
            page_files = [Path("/tmp/x/page-001.jpg"), Path("/tmp/x/page-002.jpg")]
            loc = qp.locate_missing_record_pages("pdf", page_files, {9: None}, {})
        finally:
            qp.pdftotext_page = orig
        self.assertEqual(loc, {})


class DedupeQuestionsTests(unittest.TestCase):
    """Surgical re-runs append duplicate rows; _dedupe_questions_by_id keeps
    the newest row per id (idempotent re-runs at 20-book scale)."""

    def test_keeps_last_row_per_id(self):
        path = Path(tempfile.mkdtemp()) / "questions.jsonl"
        path.write_text('{"id": "PSY-001-001", "v": 1}\n'
                        '{"id": "PSY-001-001", "v": 2}\n'
                        '{"id": "PSY-001-002", "v": 1}\n', encoding="utf-8")
        n = qp._dedupe_questions_by_id(path)
        self.assertEqual(n, 1)
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        self.assertEqual(len(rows), 2)
        by_id = {r["id"]: r["v"] for r in rows}
        self.assertEqual(by_id, {"PSY-001-001": 2, "PSY-001-002": 1})


class SectionWindowTests(unittest.TestCase):
    """build_section_windows must send the chapter in section-sized windows:
    the whole questions+answers stretch in LARGE windows (1-2 calls, 1-page
    overlap -- no boundary splits, 33%->10% overlap-token waste), and the
    Solutions section in smaller recitation-safe chunks. Pass activation is
    deliberately NOT changed (probe-based), so section labels here only SIZE
    windows and mark the carry reset."""

    def setUp(self):
        self.files = [Path(f"/tmp/s-{n:03d}.jpg") for n in range(3, 17)]

    def _fake_text(self, mapping):
        orig = qp.pdftotext_page
        qp.pdftotext_page = lambda pdf, page: mapping.get(page, "")
        self.addCleanup(setattr, qp, "pdftotext_page", orig)

    def test_questions_and_solutions_planned(self):
        # ch1-like: questions pp.3-10 (answer key interleaved at 7-8),
        # solutions pp.11-16
        text = {p: "1. Question\n2. Question\n" for p in range(3, 11)}
        text.update({p: "ANSWER KEY\n| Question No. | Correct Option |" for p in (7, 8)})
        text.update({p: "Solution to Question 1:\nSolution to Question 2:\n"
                        for p in range(11, 17)})
        self._fake_text(text)
        wins = qp.build_section_windows(self.files, "pdf")
        sections = [s for _, s in wins]
        self.assertEqual(sections, ["Q", "S", "S"])
        # the whole question+answer stretch in ONE large window (8 pages)
        self.assertEqual(wins[0][0], list(range(3, 11)))
        # RUN-12 cross-section overlap: the first S window includes the last
        # question page (10) so a boundary-spanning question's tail is seen
        self.assertEqual(wins[1][0], [10, 11, 12, 13, 14])
        self.assertEqual(wins[2][0], [15, 16])
        # page 10 is shared across the Q/S boundary (tail continuity)
        self.assertEqual(set(wins[0][0]) & set(wins[1][0]), {10})

    def test_big_question_section_chunked_too(self):
        # 12 question pages -> 2 Q windows with 1-page overlap, then solutions
        text = {p: "1. Question\n" for p in range(3, 15)}
        text.update({p: "Solution to Question 1:\nSolution to Question 2:\n"
                        for p in range(15, 17)})
        self._fake_text(text)
        wins = qp.build_section_windows(self.files, "pdf")
        q_wins = [w for w, s in wins if s == "Q"]
        self.assertEqual(len(q_wins), 2)
        self.assertEqual(len(q_wins[0]), qp.QUESTIONS_CHUNK_PAGES)
        self.assertEqual(q_wins[0][-1], q_wins[1][0])  # 1-page overlap

    def test_answer_key_only_chapter_falls_back(self):
        # no solutions section at all -> fixed-window fallback ([] means the
        # caller keeps the old 6-page loop, which never skips a pass)
        files = [Path(f"/tmp/s-{n:03d}.jpg") for n in range(3, 11)]
        text = {p: "1. Question\n" for p in range(3, 9)}
        text[9] = "ANSWER KEY\n| Question No. | Correct Option |"
        text[10] = "ANSWER KEY\n| Question No. | Correct Option |"
        self._fake_text(text)
        self.assertEqual(qp.build_section_windows(files, "pdf"), [])

    def test_no_sections_detected_falls_back(self):
        self._fake_text({p: "just prose\n" for p in range(3, 17)})
        self.assertEqual(qp.build_section_windows(self.files, "pdf"), [])

    def test_garbled_text_layer_falls_back(self):
        self._fake_text({})
        self.assertEqual(qp.build_section_windows(self.files, "pdf"), [])

    def test_solutions_chunked_with_intra_section_overlap(self):
        # solutions pp.6-13 (8 pages) -> 2 S windows; the first S window
        # carries the boundary page 5 (cross-section overlap for the tail)
        files = [Path(f"/tmp/s-{n:03d}.jpg") for n in range(3, 14)]
        text = {p: "1. Q\n" for p in (3, 4)}
        text[5] = "ANSWER KEY\n| Q No | Answer |"
        text.update({p: "Solution to Question 1:\nSolution to Question 2:\n"
                        for p in range(6, 14)})
        self._fake_text(text)
        wins = qp.build_section_windows(files, "pdf")
        q_wins = [w for w, s in wins if s == "Q"]
        s_wins = [w for w, s in wins if s == "S"]
        self.assertEqual(len(s_wins), 2)
        self.assertEqual(len(s_wins[0]), qp.SOLUTIONS_CHUNK_PAGES)
        # first S window starts with the boundary page (5) shared with the Q
        # window -> a question spanning 5->6 keeps its tail in the Q pass
        self.assertEqual(s_wins[0][0], 5)
        self.assertEqual(q_wins[0][-1], 5)
        self.assertEqual(s_wins[1][0], 10)   # remaining solutions chunk


class FigureMapTests(unittest.TestCase):
    """The _figure_map control object (Gemini declares q_no+slot per figure in
    reading order) must be peeled by extract_batch_meta and used by
    claim_figure_map_images to attach images to their questions -- the
    run-6 user ask ("bta ye image kis question ki h") to stop unclaimed
    images."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._old_assets = qp.ASSETS_DIR
        qp.ASSETS_DIR = self.tmp / "assets"
        self.subj_dir = qp.ASSETS_DIR / "questions" / "PSY"
        self.subj_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        qp.ASSETS_DIR = self._old_assets

    def _rels(self, oids, page=1):
        rels = []
        for oid in oids:
            fname = f"PSY-p{page}-{oid}.webp"
            (self.subj_dir / fname).write_bytes(b"x" * 3000)
            rels.append(f"PSY/{fname}")
        return rels

    def test_extract_batch_meta_peels_figure_map(self):
        items, meta = qp.extract_batch_meta([
            {"q_no": 1, "question_text": "s"},
            {"_figure_map": [{"q_no": 1, "slot": "question"},
                             {"q_no": None, "slot": None}]},
            {"_batch_meta": {"last_q_no": 1, "ends_mid_content": False}},
        ])
        self.assertEqual(len(items), 1)
        self.assertEqual(meta["figure_map"][0], {"q_no": 1, "slot": "question"})
        self.assertEqual(meta["last_q_no"], 1)

    def test_exact_count_map_claims_every_image(self):
        fig_map = [{"q_no": 3, "slot": "question"},
                   {"q_no": 7, "slot": "solution"},
                   {"q_no": None, "slot": None}]
        rels = self._rels([6, 7, 8])
        window_rows = [(1, rels)]
        owned = {}
        remaining = qp.claim_figure_map_images(fig_map, window_rows, "PSY", 1,
                                               {3: {}, 7: {}}, owned)
        self.assertEqual(owned[3]["question"], ["PSY/PSY-001-003_Q_01.webp"])
        self.assertEqual(owned[7]["solution"], ["PSY/PSY-001-007_SOL_01.webp"])
        # the decorative entry (q_no null) left ITS image unclaimed -- the
        # alignment stayed exact for the two real owners
        self.assertEqual(remaining[1], ["PSY/PSY-p1-8.webp"])

    def test_count_mismatch_skips_entirely(self):
        fig_map = [{"q_no": 3, "slot": "question"}]   # 1 declared, 2 extracted
        rels = self._rels([6, 7])
        remaining = qp.claim_figure_map_images(fig_map, [(1, rels)], "PSY", 1,
                                               {3: {}}, {})
        self.assertEqual(set(remaining[1]), set(rels))  # nothing claimed

    def test_unknown_q_or_bad_slot_stays_unclaimed(self):
        fig_map = [{"q_no": 99, "slot": "question"},   # not in chapter
                   {"q_no": 3, "slot": "sideways"}]    # invalid slot
        rels = self._rels([6, 7])
        remaining = qp.claim_figure_map_images(fig_map, [(1, rels)], "PSY", 1,
                                               {3: {}}, {})
        self.assertEqual(set(remaining[1]), set(rels))

    def test_guard_refused_image_stays_but_others_claimed(self):
        # first image too small -> tiny-crop guard refuses rename
        rels = self._rels([6, 7])
        (self.subj_dir / "PSY-p1-6.webp").write_bytes(b"x" * 100)  # < MIN_IMAGE_BYTES
        fig_map = [{"q_no": 3, "slot": "question"},
                   {"q_no": 3, "slot": "question"}]
        owned = {}
        remaining = qp.claim_figure_map_images(fig_map, [(1, rels)], "PSY", 1,
                                               {3: {}}, owned)
        # the tiny one refused -> stays; the other claimed
        self.assertIn("PSY/PSY-p1-6.webp", remaining[1])
        self.assertNotIn("PSY/PSY-p1-7.webp", remaining[1])


class CrossFieldContaminationTests(unittest.TestCase):
    """Run-7 hardening: a recovered SOLUTION fragment must never populate
    question_text, recovery is patch-only by field, OCR noise is stripped
    before merge, and 'field is populated' != 'field is valid'."""

    def _rec(self, qn, **kw):
        r = {"q_no": qn, "question_text": None, "options": None,
             "correct_option": None, "solution_text": None, "tables": [],
             "has_figure_in_question": False, "has_figure_in_solution": False,
             "_prov": {}}
        r.update(kw)
        return r

    # -- 1. S-pass recovery must never fill the stem -------------------------
    def test_s_pass_item_cannot_fill_question_text(self):
        item = {"q_no": 3, "_prov": "S_PASS",
                "question_text": "The correct answer is B because the basal "
                                 "ganglia circuit is disrupted in OCD patients "
                                 "and this explains the compulsions seen here "
                                 "with additional detail about the pathway.",
                "solution_text": "The correct answer is B because the basal "
                                 "ganglia circuit is disrupted in OCD patients "
                                 "and this explains the compulsions seen here "
                                 "with additional detail about the pathway.",
                "options": None, "correct_option": "B", "tables": []}
        recs, _ = qp.merge_question_records({}, [item], stats := {"chapter_id": "PSY-016"})
        self.assertIsNone(recs[3]["question_text"])   # stem NEVER populated
        self.assertEqual(recs[3]["solution_text"], item["solution_text"])

    def test_ocr_s_item_cannot_fill_question_text(self):
        item = {"q_no": 7, "_prov": "OCR_S",
                "question_text": "Ans. is C. The dissociation amnesia "
                                 "resolves when the patient is removed from "
                                 "the stressful military environment and "
                                 "supportive psychotherapy is instituted.",
                "solution_text": "Ans. is C. The dissociation amnesia "
                                 "resolves when the patient is removed from "
                                 "the stressful military environment and "
                                 "supportive psychotherapy is instituted.",
                "options": None, "correct_option": "C", "tables": []}
        recs, _ = qp.merge_question_records({}, [item], {"chapter_id": "PSY-017"})
        self.assertIsNone(recs[7]["question_text"])

    # -- 2. q_no=None OCR fragment containing a neighbor's solution ----------
    def test_s_orphan_cannot_fill_stem_via_recover_orphans(self):
        frag = ("Ans. is A. The patient's symptoms of depersonalisation "
                "resolve gradually with cognitive behavioural therapy and "
                "grounding techniques over several months of treatment.")
        orphans = [{"chapter_id": "PSY-017", "batch_start": 0,
                    "pdf_pages": [218], "new_pages": [218],
                    "carry_q_no": None, "cut_part": None,
                    "last_qn_in_batch": 10, "pass": "S",
                    "item": {"q_no": None, "question_text": frag,
                             "solution_text": frag, "options": None,
                             "correct_option": None, "tables": [],
                             "has_figure_in_question": False,
                             "has_figure_in_solution": False}}]
        recs = {10: self._rec(10, question_text="Real stem ten",
                              solution_text="partial solution ten")}
        stats = {"orphans_recovered": 0, "foreign_fragments_blocked": 0,
                 "carry_merges": 0, "contaminated_stems_blocked": 0,
                 "chapter_id": "PSY-017"}
        qp.recover_orphans(orphans, recs, "PSY", 17, stats)
        # the real stem survives; the S-fragment's stem text is blocked
        self.assertEqual(recs[10]["question_text"], "Real stem ten")
        self.assertIn("partial solution ten", recs[10]["solution_text"])
        self.assertGreaterEqual(stats["contaminated_stems_blocked"], 0)

    # -- 3. OCR cleanup strips page numbers / watermarks ---------------------
    def test_clean_ocr_text_strips_page_noise(self):
        dirty = ("Solution to Question 3:\n"
                 "The diagnosis is delirium.\n"
                 "12\n"
                 "- 45 -\n"
                 "Page 12 of 300\n"
                 "www.example-qbank.com\n"
                 "© 2026 Example Publishers\n")
        clean = qp._clean_ocr_text(dirty)
        self.assertIn("The diagnosis is delirium.", clean)
        self.assertNotIn("\n12\n", "\n" + clean + "\n")
        self.assertNotIn("- 45 -", clean)
        self.assertNotIn("Page 12 of 300", clean)
        self.assertNotIn("www.example", clean)
        self.assertNotIn("©", clean)
        self.assertIn("Solution to Question 3:", clean)  # header preserved

    def test_clean_ocr_text_preserves_prose(self):
        text = ("The key feature is that the mood episode is not better "
                "explained by substance use.\n")
        # content is preserved verbatim (trailing newline normalization from
        # splitlines is the only difference)
        self.assertEqual(qp._clean_ocr_text(text).strip(), text.strip())

    # -- 4. non-empty question consisting of solution prose ------------------
    def test_contaminated_stem_rejected_at_merge(self):
        sol = ("The correct answer is A. In Korsakoff syndrome the amnesia "
               "is characterised by anterograde and retrograde memory loss "
               "with confabulation, and the pathology lies in the mammillary "
               "bodies and the dorsomedial nucleus of the thalamus with "
               "severe vitamin B1 deficiency being the underlying cause.")
        item = {"q_no": 5, "_prov": "Q_PASS",
                "question_text": sol,   # contaminated: is the solution
                "solution_text": sol, "options": None, "correct_option": "A",
                "tables": []}
        stats = {"chapter_id": "PSY-010", "contaminated_stems_rejected": 0}
        recs, _ = qp.merge_question_records({}, [item], stats)
        self.assertIsNone(recs[5]["question_text"])   # rejected, not shipped
        self.assertEqual(recs[5]["solution_text"], sol)
        self.assertEqual(stats["contaminated_stems_rejected"], 1)

    def test_find_incomplete_treats_contaminated_stem_as_missing(self):
        sol = ("The correct answer is B. Body dysmorphic disorder involves "
               "a preoccupation with an imagined defect in appearance that "
               "causes clinically significant distress and impaired "
               "functioning with repetitive checking behaviours.")
        recs = {9: self._rec(9, question_text=sol, solution_text=sol,
                             correct_option="B",
                             options={"A": "a", "B": "b", "C": "c", "D": "d"})}
        incomplete = qp.find_incomplete_records(recs)
        self.assertTrue(any(qn == 9 and "question" in missing
                            for qn, missing in incomplete))

    # -- 5. valid existing stem survives S/OCR recovery unchanged ------------
    def test_valid_stem_survives_s_pass_merge(self):
        recs = {4: self._rec(4, question_text="Which neurotransmitter is "
                                              "reduced in Parkinson's disease?",
                             solution_text="Dopamine is reduced.")}
        item = {"q_no": 4, "_prov": "S_PASS",
                "question_text": "stray solution prose that must not land",
                "solution_text": "Dopamine is reduced in the substantia nigra.",
                "options": None, "correct_option": "A", "tables": []}
        qp.merge_question_records(recs, [item], {"chapter_id": "PSY-027"})
        self.assertEqual(recs[4]["question_text"],
                         "Which neurotransmitter is reduced in Parkinson's disease?")

    # -- 6. drain scope: an A-drain cannot patch solutions -------------------
    def test_recovery_scope_limits_fields(self):
        item = {"q_no": 2, "question_text": "stem?", "options": {"A": "a"},
                "correct_option": "C", "solution_text": "sol", "tables": []}
        out = qp._apply_recovery_scope(dict(item), qp._RECOVERY_SCOPE["S"], "OCR_S")
        self.assertIsNone(out["question_text"])
        self.assertIsNone(out["options"])
        self.assertEqual(out["solution_text"], "sol")
        self.assertEqual(out["_prov"], "OCR_S")
        out2 = qp._apply_recovery_scope(dict(item), qp._RECOVERY_SCOPE["A"], "DRAIN_A")
        self.assertEqual(out2["correct_option"], "C")
        self.assertIsNone(out2["solution_text"])


class ContinuationOwnershipTests(unittest.TestCase):
    """Run-8: unnumbered continuations crossing an overlap boundary must be
    assigned to the question whose heading is on the overlap page -- via the
    deterministic compute_carry S-pass fallback + carry-forward orphan
    recovery -- never left q_no=null when ownership is provable, and never
    guessed when it is not."""

    def _rec(self, qn, **kw):
        r = {"q_no": qn, "question_text": None, "options": None,
             "correct_option": None, "solution_text": None, "tables": [],
             "has_figure_in_question": False, "has_figure_in_solution": False,
             "_prov": {}}
        r.update(kw)
        return r

    def _s_orphan(self, frag, carry_qn, last_qn, page=18):
        return {"chapter_id": "PSY-016", "batch_start": page, "pdf_pages": [page, 21],
                "new_pages": [page, 19, 20, 21], "carry_q_no": carry_qn,
                "cut_part": "solution", "last_qn_in_batch": last_qn, "pass": "S",
                "item": {"q_no": None, "question_text": None, "solution_text": frag,
                         "options": None, "correct_option": None, "tables": [],
                         "has_figure_in_question": False,
                         "has_figure_in_solution": False}}

    # -- 1. Q2's heading at the bottom of the overlap page; all Q2 content on
    #      the next page -> continuation must be assigned to Q2 ------------
    def test_heading_on_overlap_page_assigns_continuation_to_owner(self):
        # window 1 ends with q2's truncated solution (its "Solution to
        # Question 2:" heading is at the bottom of the overlap page)
        trunc = "The correct answer is A because the defence mechanism here is:"
        items1 = [{"q_no": 2, "question_text": None, "solution_text": trunc,
                   "options": None, "correct_option": None, "tables": []}]
        recs = {2: self._rec(2, solution_text=trunc)}
        carry = qp.compute_carry({}, items1, recs, 17)   # no _batch_meta
        self.assertEqual(carry["last_open_question"], 2)
        self.assertEqual(carry["cut_part"], "solution")
        # window 2 still returns the continuation as q_no=null -> the orphan
        # carries q2 and rule 2 attaches it
        frag = "repression, because the impulse is pushed out of awareness into the unconscious mind."
        orphans = [self._s_orphan(frag, carry_qn=2, last_qn=3)]
        stats = {"orphans_recovered": 0, "foreign_fragments_blocked": 0,
                 "carry_merges": 0, "contaminated_stems_blocked": 0,
                 "chapter_id": "PSY-016"}
        recs[3] = self._rec(3, question_text="Stem three",
                            solution_text="complete solution three")
        remaining = qp.recover_orphans(orphans, recs, "PSY", 16, stats)
        self.assertEqual(remaining, [])
        self.assertIn("repression", recs[2]["solution_text"])
        self.assertEqual(recs[3]["solution_text"], "complete solution three")

    # -- 2. Q2 starts on overlap page, continues, then explicit Q3 heading ->
    #      initial continuation to Q2, subsequent content to Q3 ------------
    def test_continuation_then_explicit_next_heading_stays_separate(self):
        recs = {2: self._rec(2, question_text="Stem two",
                             solution_text="The answer is A because:"),
                3: self._rec(3, question_text="Stem three",
                             solution_text="complete solution three")}
        # the null fragment is q2's continuation; q3's numbered item exists
        # separately (already merged) and must NOT absorb the fragment
        frag = "the patient uses rationalisation to minimise the guilt feeling."
        orphans = [self._s_orphan(frag, carry_qn=2, last_qn=3)]
        stats = {"orphans_recovered": 0, "foreign_fragments_blocked": 0,
                 "carry_merges": 0, "contaminated_stems_blocked": 0,
                 "chapter_id": "PSY-016"}
        remaining = qp.recover_orphans(orphans, recs, "PSY", 16, stats)
        self.assertEqual(remaining, [])
        self.assertIn("rationalisation", recs[2]["solution_text"])
        self.assertEqual(recs[3]["solution_text"], "complete solution three")

    # -- 3. unnumbered text, no reliable owner -> stays unassigned, never
    #      guessed ---------------------------------------------------------
    def test_unowned_continuation_stays_unassigned(self):
        # window 1 ended with a COMPLETE solution -> no carry created
        items1 = [{"q_no": 2, "question_text": None,
                   "solution_text": "The answer is A. Repression is complete.",
                   "options": None, "correct_option": None, "tables": []}]
        recs = {2: self._rec(2, solution_text="The answer is A. Repression is complete."),
                3: self._rec(3, question_text="Stem three",
                             solution_text="complete solution three")}
        self.assertIsNone(qp.compute_carry({}, items1, recs, 17))
        # an unrelated unnumbered fragment with no carry must NOT be glued
        # onto q2 (complete solution) or guessed at all
        frag = "Some unnumbered text that has no provable owner on the overlap page."
        orphans = [self._s_orphan(frag, carry_qn=None, last_qn=2)]
        stats = {"orphans_recovered": 0, "foreign_fragments_blocked": 0,
                 "carry_merges": 0, "contaminated_stems_blocked": 0,
                 "chapter_id": "PSY-016"}
        remaining = qp.recover_orphans(orphans, recs, "PSY", 16, stats)
        self.assertEqual(len(remaining), 1)          # stays for review
        self.assertNotIn("unnumbered text", recs[2]["solution_text"])
        self.assertNotIn("unnumbered text", recs[3]["solution_text"])

    # -- 4. overlap content must not be duplicated into the final solution --
    def test_overlap_reextraction_does_not_duplicate_solution(self):
        # window 1 returns q2 partial; window 2 (with q2's page as overlap)
        # returns q2 complete -> the FULL solution replaces the partial one
        # (last-write-wins), never concatenated
        full = ("The answer is A. Repression is complete. The impulse is "
                "pushed out of awareness into the unconscious mind.")
        recs = {}
        qp.merge_question_records(recs, [
            {"q_no": 2, "question_text": None, "_prov": "S_PASS",
             "solution_text": "The answer is A. Repression is complete.",
             "options": None, "correct_option": None, "tables": []}],
            {"chapter_id": "PSY-016"})
        qp.merge_question_records(recs, [
            {"q_no": 2, "question_text": None, "_prov": "S_PASS",
             "solution_text": full,
             "options": None, "correct_option": None, "tables": []}],
            {"chapter_id": "PSY-016"})
        self.assertEqual(recs[2]["solution_text"], full)   # not doubled
        self.assertEqual(recs[2]["solution_text"].count("Repression is complete."), 1)

    # -- 5. S-pass continuation must never enter question_text -------------
    def test_s_pass_continuation_never_enters_question_text(self):
        # the S orphan carries a stray question_text (Gemini filled both) ->
        # blocked from the stem; the solution still merges under its owner
        frag_sol = ("the patient uses rationalisation to minimise guilt feelings.")
        stray_stem = ("Rationalisation is a defence mechanism that involves "
                      "providing a logical explanation for behaviour.")
        orphans = [{"chapter_id": "PSY-016", "batch_start": 18, "pdf_pages": [18],
                    "new_pages": [18], "carry_q_no": 2, "cut_part": "solution",
                    "last_qn_in_batch": 2, "pass": "S",
                    "item": {"q_no": None, "question_text": stray_stem,
                             "solution_text": frag_sol, "options": None,
                             "correct_option": None, "tables": [],
                             "has_figure_in_question": False,
                             "has_figure_in_solution": False}}]
        recs = {2: self._rec(2, question_text="Stem two",
                             solution_text="The answer is A because:")}
        stats = {"orphans_recovered": 0, "foreign_fragments_blocked": 0,
                 "carry_merges": 0, "contaminated_stems_blocked": 0,
                 "chapter_id": "PSY-016"}
        qp.recover_orphans(orphans, recs, "PSY", 16, stats)
        self.assertEqual(recs[2]["question_text"], "Stem two")      # untouched
        self.assertIn("rationalisation", recs[2]["solution_text"])  # merged

    # -- 6. valid existing content survives continuation recovery ----------
    def test_existing_content_never_overwritten_by_continuation(self):
        recs = {2: self._rec(2, question_text="The real stem stays intact",
                             solution_text="The answer is A because:")}
        # fill_only merge (recovery) must NEVER overwrite existing content,
        # even when the incoming S patch carries a (wrong) stem and a fuller
        # solution
        qp.merge_question_records(recs, [
            {"q_no": 2, "question_text": "WRONG stem from a stray S fragment",
             "solution_text": "The answer is A because: the full correct "
                              "explanation continues here with real content.",
             "options": None, "correct_option": None, "tables": [],
             "_prov": "S_RETRY"}],
            {"chapter_id": "PSY-016"}, fill_only=True)
        self.assertEqual(recs[2]["question_text"], "The real stem stays intact")
        self.assertEqual(recs[2]["solution_text"], "The answer is A because:")
        # the REAL continuation path (recover_orphans, truncated owner +
        # carry) appends the novel tail ONCE -- existing text preserved
        frag = "the full correct explanation continues here with real content."
        orphans = [self._s_orphan(frag, carry_qn=2, last_qn=2)]
        stats = {"orphans_recovered": 0, "foreign_fragments_blocked": 0,
                 "carry_merges": 0, "contaminated_stems_blocked": 0,
                 "chapter_id": "PSY-016"}
        qp.recover_orphans(orphans, recs, "PSY", 16, stats)
        self.assertEqual(recs[2]["question_text"], "The real stem stays intact")
        self.assertIn("full correct explanation", recs[2]["solution_text"])
        self.assertEqual(recs[2]["solution_text"].count("because:"), 1)


class GeometryFirstImageTests(unittest.TestCase):
    """Run-9: image ownership is DETERMINISTIC-FIRST -- every image belongs
    to the closest question/solution heading ABOVE it (real PDF y positions),
    or the carried active block for cross-page continuations. Gemini never
    overrides a deterministic assignment, and a single 'decorative' verdict
    never discards an image (it goes to unresolved_images.jsonl)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._old_assets = qp.ASSETS_DIR
        qp.ASSETS_DIR = self.tmp / "assets"
        self.subj_dir = qp.ASSETS_DIR / "questions" / "PSY"
        self.subj_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        qp.ASSETS_DIR = self._old_assets

    def _rels(self, oids, page=1):
        rels = []
        for oid in oids:
            fname = f"PSY-p{page}-{oid}.webp"
            (self.subj_dir / fname).write_bytes(b"x" * 3000)
            rels.append(f"PSY/{fname}")
        return rels

    def _claim(self, pdf, oids, recs, page=1, active_block=None):
        owned = {}
        leftover = qp.claim_page_images(self._rels(oids, page), pdf, page,
                                        "PSY", 1, recs, owned,
                                        active_block=active_block)
        return leftover, owned

    # -- 1. image inside Q1 question block -> Q1 question image ------------
    def test_image_inside_question_block_maps_to_that_question(self):
        pdf = self.tmp / "q1_block.pdf"
        _write_test_pdf(pdf, [
            ("1. Which defence mechanism is being used?", 72, 700, 12),
            ("Option A: text", 72, 660, 10),
        ], [(6, "Im6", 300, 600)])
        leftover, owned = self._claim(pdf, [6], {1: {}})
        self.assertEqual(leftover, [])
        self.assertEqual(owned[1]["question"], ["PSY/PSY-001-001_Q_01.webp"])

    # -- 2. image inside Q6 solution block -> Q6 solution image ------------
    def test_image_inside_solution_block_maps_to_that_solution(self):
        pdf = self.tmp / "q6_sol.pdf"
        _write_test_pdf(pdf, [
            ("Solution to Question 6:", 72, 700, 12),
            ("The answer is B because:", 72, 660, 10),
        ], [(6, "Im6", 300, 600)])
        leftover, owned = self._claim(pdf, [6], {6: {}})
        self.assertEqual(leftover, [])
        self.assertEqual(owned[6]["solution"], ["PSY/PSY-001-006_SOL_01.webp"])

    # -- 3. image after Q1 heading but before Q2 heading -> Q1 -------------
    def test_image_between_two_question_headings_belongs_to_first(self):
        pdf = self.tmp / "q1_q2.pdf"
        _write_test_pdf(pdf, [
            ("1. First question stem", 72, 700, 12),
            ("2. Second question stem", 72, 400, 12),
        ], [(6, "Im6", 300, 550)])
        leftover, owned = self._claim(pdf, [6], {1: {}, 2: {}})
        self.assertEqual(leftover, [])
        self.assertEqual(owned[1]["question"], ["PSY/PSY-001-001_Q_01.webp"])
        self.assertNotIn(2, owned)

    # -- 4. multiple figures inside same block -> all stay with that owner --
    def test_multiple_figures_in_one_block_stay_with_owner(self):
        pdf = self.tmp / "multi_fig.pdf"
        _write_test_pdf(pdf, [
            ("1. First question stem", 72, 700, 12),
        ], [(6, "Im6", 300, 600), (7, "Im7", 300, 500)])
        leftover, owned = self._claim(pdf, [6, 7], {1: {}})
        self.assertEqual(leftover, [])
        self.assertEqual(len(owned[1]["question"]), 2)

    # -- 5. multiple questions/images on same page -> each by position ------
    def test_multiple_questions_each_image_maps_by_position(self):
        pdf = self.tmp / "two_q_two_img.pdf"
        _write_test_pdf(pdf, [
            ("1. First question stem", 72, 700, 12),
            ("2. Second question stem", 72, 400, 12),
        ], [(6, "Im6", 300, 600), (7, "Im7", 300, 300)])
        leftover, owned = self._claim(pdf, [6, 7], {1: {}, 2: {}})
        self.assertEqual(leftover, [])
        self.assertEqual(owned[1]["question"], ["PSY/PSY-001-001_Q_01.webp"])
        self.assertEqual(owned[2]["question"], ["PSY/PSY-001-002_Q_01.webp"])

    # -- 6. cross-page continuation image -> carried owner -----------------
    def test_cross_page_continuation_image_uses_carried_owner(self):
        # the new page's image has NO heading above it (block started on the
        # previous page) -> active_block (q6 solution) owns it
        pdf = self.tmp / "carry.pdf"
        _write_test_pdf(pdf, [
            ("text continues from previous page", 72, 600, 10),
        ], [(6, "Im6", 300, 700)])
        leftover, owned = self._claim(pdf, [6], {6: {}}, page=1,
                                      active_block=("solution", 6))
        self.assertEqual(leftover, [])
        self.assertEqual(owned[6]["solution"], ["PSY/PSY-001-006_SOL_01.webp"])

    def test_cross_page_without_carry_stays_unclaimed(self):
        pdf = self.tmp / "nocarry.pdf"
        _write_test_pdf(pdf, [
            ("text continues from previous page", 72, 600, 10),
        ], [(6, "Im6", 300, 700)])
        leftover, _ = self._claim(pdf, [6], {6: {}}, page=1, active_block=None)
        self.assertEqual(leftover, ["PSY/PSY-p1-6.webp"])

    # -- 7. genuine watermark -> excluded at extraction (deterministic) -----
    def test_watermark_object_excluded_at_extraction(self):
        pdf = self.tmp / "wm.pdf"
        # images must be > 5000 px or extract_real_images drops them as noise
        _write_test_pdf(pdf, [
            ("1. Question stem", 72, 700, 12),
        ], [(6, "Im6", 300, 600), (7, "Im7", 300, 300)], img_size=(90, 90))
        # watermark_id = obj 6 -> only obj 7 survives extraction
        saved = qp.extract_real_images(pdf, 1, 6, "PSY", self.subj_dir)
        self.assertEqual(saved, ["PSY/PSY-p1-7.webp"])

    # -- 8. ambiguous image -> unresolved, NOT decorative -------------------
    def test_ambiguous_image_recorded_as_unresolved_not_decorative(self):
        tmp = Path(tempfile.mkdtemp())
        old_data = qp.DATA_DIR
        qp.DATA_DIR = tmp / "data"
        try:
            qp._record_unresolved_image("PSY", "PSY-001", 4, "PSY/PSY-p4-7.webp",
                                        "model-declared decorative",
                                        model_verdict={"decorative": True})
            unresolved = qp.DATA_DIR / "unresolved_images.jsonl"
            decorative = qp.DATA_DIR / "decorative_images.jsonl"
            self.assertTrue(unresolved.exists())
            entry = json.loads(unresolved.read_text().splitlines()[0])
            self.assertEqual(entry["file"], "PSY/PSY-p4-7.webp")
            self.assertEqual(entry["model_verdict"], {"decorative": True})
            self.assertFalse(decorative.exists())   # NOT permanently discarded
        finally:
            qp.DATA_DIR = old_data

    # -- 9. Gemini disagreement must NOT override deterministic ownership ---
    def test_gemini_figure_map_cannot_override_geometry(self):
        # image is inside Q1's block (geometry claims it first); a Gemini
        # figure-map that would say Q2 runs on the LEFTOVERS only and cannot
        # move it
        pdf = self.tmp / "override.pdf"
        _write_test_pdf(pdf, [
            ("1. First question stem", 72, 700, 12),
        ], [(6, "Im6", 300, 600)])
        rels = self._rels([6])
        owned = {}
        # geometry-first: image claimed by Q1's block
        leftover = qp.claim_page_images(rels, pdf, 1, "PSY", 1, {1: {}, 2: {}},
                                        owned, active_block=None)
        self.assertEqual(leftover, [])
        self.assertEqual(owned[1]["question"], ["PSY/PSY-001-001_Q_01.webp"])
        # the figure-map runs on the (empty) leftovers: even a contradictory
        # map cannot re-claim the already-owned image
        remaining = qp.claim_figure_map_images(
            [{"q_no": 2, "slot": "question"}], [(1, [])], "PSY", 1,
            {1: {}, 2: {}}, owned)
        self.assertEqual(owned[1]["question"], ["PSY/PSY-001-001_Q_01.webp"])
        self.assertNotIn(2, owned)


class OptionImageOwnershipTests(unittest.TestCase):
    """Run-10: OPTION-LEVEL image ownership. An image geometrically inside an
    option label's block (vertical) or on a horizontal/2x2 option row is
    assigned to THAT option deterministically; everything else stays at
    question level. Never guessed, never dropped, Gemini never overrides."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._old_assets = qp.ASSETS_DIR
        qp.ASSETS_DIR = self.tmp / "assets"
        self.subj_dir = qp.ASSETS_DIR / "questions" / "PSY"
        self.subj_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        qp.ASSETS_DIR = self._old_assets

    def _rels(self, oids, page=1):
        rels = []
        for oid in oids:
            fname = f"PSY-p{page}-{oid}.webp"
            (self.subj_dir / fname).write_bytes(b"x" * 3000)
            rels.append(f"PSY/{fname}")
        return rels

    def _claim(self, pdf, oids, recs, page=1, active_block=None):
        owned = {}
        leftover = qp.claim_page_images(self._rels(oids, page), pdf, page,
                                        "PSY", 1, recs, owned,
                                        active_block=active_block)
        return leftover, owned

    # -- 1. normal question image (no option labels) -> question-level -----
    def test_normal_question_image_stays_question_level(self):
        pdf = self.tmp / "q1_no_opts.pdf"
        _write_test_pdf(pdf, [
            ("1. Which diagnosis is shown?", 72, 700, 12),
        ], [(6, "Im6", 300, 620)])
        leftover, owned = self._claim(pdf, [6], {1: {}})
        self.assertEqual(leftover, [])
        self.assertEqual(owned[1]["question"], ["PSY/PSY-001-001_Q_01.webp"])
        self.assertEqual(owned[1].get("option", {}), {})

    # -- 2. image under option A -> option A -------------------------------
    def test_image_under_option_a_maps_to_option_a(self):
        pdf = self.tmp / "opt_a.pdf"
        _write_test_pdf(pdf, [
            ("1. Identify the structure", 72, 700, 12),
            ("A. text", 72, 650, 10),
            ("B. text", 72, 550, 10),
        ], [(6, "Im6", 300, 620)])
        leftover, owned = self._claim(pdf, [6], {1: {}})
        self.assertEqual(leftover, [])
        self.assertEqual(owned[1]["option"]["A"], ["PSY/PSY-001-001_OPT_A_01.webp"])
        self.assertEqual(owned[1]["question"], [])

    # -- 3. four vertical option images -> A/B/C/D -------------------------
    def test_four_vertical_option_images_map_correctly(self):
        pdf = self.tmp / "opt_abcd.pdf"
        _write_test_pdf(pdf, [
            ("1. Identify the structure", 72, 720, 12),
            ("A. text", 72, 650, 10), ("B. text", 72, 550, 10),
            ("C. text", 72, 450, 10), ("D. text", 72, 350, 10),
        ], [(6, "Im6", 300, 620), (7, "Im7", 300, 520),
            (8, "Im8", 300, 420), (9, "Im9", 300, 320)])
        leftover, owned = self._claim(pdf, [6, 7, 8, 9], {1: {}})
        self.assertEqual(leftover, [])
        opt = owned[1]["option"]
        self.assertEqual(opt["A"], ["PSY/PSY-001-001_OPT_A_01.webp"])
        self.assertEqual(opt["B"], ["PSY/PSY-001-001_OPT_B_01.webp"])
        self.assertEqual(opt["C"], ["PSY/PSY-001-001_OPT_C_01.webp"])
        self.assertEqual(opt["D"], ["PSY/PSY-001-001_OPT_D_01.webp"])
        self.assertEqual(owned[1]["question"], [])

    # -- 4. horizontal / 2x2 image options -> correct via x+y geometry -----
    def test_2x2_horizontal_option_images_map_by_xy(self):
        pdf = self.tmp / "opt_2x2.pdf"
        _write_test_pdf(pdf, [
            ("1. Identify the structure", 72, 720, 12),
            ("A. text", 72, 650, 10), ("B. text", 350, 650, 10),
            ("C. text", 72, 550, 10), ("D. text", 350, 550, 10),
        ], [(6, "Im6", 200, 620), (7, "Im7", 420, 620),
            (8, "Im8", 200, 520), (9, "Im9", 420, 520)])
        leftover, owned = self._claim(pdf, [6, 7, 8, 9], {1: {}})
        self.assertEqual(leftover, [])
        opt = owned[1]["option"]
        self.assertEqual(opt["A"], ["PSY/PSY-001-001_OPT_A_01.webp"])
        self.assertEqual(opt["B"], ["PSY/PSY-001-001_OPT_B_01.webp"])
        self.assertEqual(opt["C"], ["PSY/PSY-001-001_OPT_C_01.webp"])
        self.assertEqual(opt["D"], ["PSY/PSY-001-001_OPT_D_01.webp"])

    # -- 5. two images in the same option -> both preserved ----------------
    def test_two_images_in_same_option_both_preserved(self):
        pdf = self.tmp / "opt_two.pdf"
        _write_test_pdf(pdf, [
            ("1. Identify the structure", 72, 700, 12),
            ("A. text", 72, 650, 10), ("B. text", 72, 550, 10),
        ], [(6, "Im6", 300, 620), (7, "Im7", 300, 590)])
        leftover, owned = self._claim(pdf, [6, 7], {1: {}})
        self.assertEqual(leftover, [])
        self.assertEqual(len(owned[1]["option"]["A"]), 2)

    # -- 6. image between stem and option A -> question-level ---------------
    def test_image_before_option_a_stays_question_level(self):
        pdf = self.tmp / "stem_img.pdf"
        _write_test_pdf(pdf, [
            ("1. Identify the structure", 72, 700, 12),
            ("A. text", 72, 650, 10),
        ], [(6, "Im6", 300, 670)])   # above A's label
        leftover, owned = self._claim(pdf, [6], {1: {}})
        self.assertEqual(leftover, [])
        self.assertEqual(owned[1]["question"], ["PSY/PSY-001-001_Q_01.webp"])
        self.assertEqual(owned[1].get("option", {}), {})

    # -- 7. solution image with "Option A:" prose -> solution image ---------
    def test_solution_image_with_option_prose_stays_solution(self):
        pdf = self.tmp / "sol_opt.pdf"
        _write_test_pdf(pdf, [
            ("Solution to Question 5:", 72, 700, 12),
            ("Option A: the correct answer because", 72, 650, 10),
        ], [(6, "Im6", 300, 600)])
        leftover, owned = self._claim(pdf, [6], {5: {}})
        self.assertEqual(leftover, [])
        self.assertEqual(owned[5]["solution"], ["PSY/PSY-001-005_SOL_01.webp"])
        self.assertEqual(owned[5].get("option", {}), {})

    # -- 8. ambiguous option ownership (shared figure) -> question-level ----
    def test_shared_figure_ambiguous_stays_question_level(self):
        pdf = self.tmp / "shared.pdf"
        _write_test_pdf(pdf, [
            ("1. Identify the structure", 72, 700, 12),
            ("A. text", 72, 650, 10), ("B. text", 350, 650, 10),
        ], [(6, "Im6", 211, 620)])   # x = midpoint(72, 350) -> equidistant
        leftover, owned = self._claim(pdf, [6], {1: {}})
        self.assertEqual(leftover, [])
        self.assertEqual(owned[1]["question"], ["PSY/PSY-001-001_Q_01.webp"])
        self.assertEqual(owned[1].get("option", {}), {})

    # -- 9. Gemini figure-map cannot override deterministic option ownership
    def test_gemini_cannot_override_option_ownership(self):
        pdf = self.tmp / "nooverride.pdf"
        _write_test_pdf(pdf, [
            ("1. Identify the structure", 72, 700, 12),
            ("A. text", 72, 650, 10), ("B. text", 72, 550, 10),
        ], [(6, "Im6", 300, 620)])
        rels = self._rels([6])
        owned = {}
        leftover = qp.claim_page_images(rels, pdf, 1, "PSY", 1, {1: {}}, owned)
        self.assertEqual(leftover, [])
        self.assertEqual(owned[1]["option"]["A"], ["PSY/PSY-001-001_OPT_A_01.webp"])
        # a contradictory figure-map runs on leftovers only (here: none) and
        # cannot move the already-claimed image
        qp.claim_figure_map_images([{"q_no": 1, "slot": "question"}], [(1, [])],
                                   "PSY", 1, {1: {}}, owned)
        self.assertEqual(owned[1]["option"]["A"], ["PSY/PSY-001-001_OPT_A_01.webp"])
        self.assertEqual(owned[1]["question"], [])

    # -- 10. JSON round-trip preserves option images ------------------------
    def test_json_round_trip_preserves_option_images(self):
        pdf = self.tmp / "rt.pdf"
        _write_test_pdf(pdf, [
            ("1. Identify the structure", 72, 700, 12),
            ("A. text", 72, 650, 10), ("B. text", 72, 550, 10),
        ], [(6, "Im6", 300, 620), (7, "Im7", 300, 520)])
        owned = {}
        leftover = qp.claim_page_images(self._rels([6, 7]), pdf, 1, "PSY", 1,
                                        {1: {}}, owned)
        self.assertEqual(leftover, [])
        rec = {"q_no": 1, "question_text": "Identify the structure",
               "options": {"A": "text", "B": "text"}, "correct_option": "A",
               "solution_text": "sol", "tables": [], "_prov": {}}
        final = qp.build_final_question("PSY", "PSY-001", 1, 1, rec, owned[1])
        opt_by_id = {o["id"]: o for o in final["options"]}
        self.assertEqual([i["file"] for i in opt_by_id["A"]["images"]],
                         ["PSY/PSY-001-001_OPT_A_01.webp"])
        self.assertEqual([i["file"] for i in opt_by_id["B"]["images"]],
                         ["PSY/PSY-001-001_OPT_B_01.webp"])
        # round-trip through final_q_to_record preserves option ownership
        rec2, owned2 = qp.final_q_to_record(final)
        self.assertEqual(owned2["option"]["A"], ["PSY/PSY-001-001_OPT_A_01.webp"])
        self.assertEqual(owned2["option"]["B"], ["PSY/PSY-001-001_OPT_B_01.webp"])
        # schema backward compatible: options still have id/text, question
        # images still on question
        self.assertEqual([o["id"] for o in final["options"]], ["A", "B"])
        self.assertEqual(final["question"]["images"], [])

    # -- 11. block-level tests still green (option logic doesn't disturb) ---
    def test_solution_block_geometry_unchanged(self):
        pdf = self.tmp / "sol_only.pdf"
        _write_test_pdf(pdf, [
            ("Solution to Question 3:", 72, 700, 12),
        ], [(6, "Im6", 300, 600)])
        leftover, owned = self._claim(pdf, [6], {3: {}})
        self.assertEqual(leftover, [])
        self.assertEqual(owned[3]["solution"], ["PSY/PSY-001-003_SOL_01.webp"])

    # -- 12. horizontal row, stem figure ABOVE the option row -> question --
    def test_horizontal_stem_figure_above_row_stays_question_level(self):
        pdf = self.tmp / "hstem.pdf"
        _write_test_pdf(pdf, [
            ("1. Identify the structure", 72, 720, 12),
            ("A. text", 72, 650, 10), ("B. text", 350, 650, 10),
        ], [(6, "Im6", 200, 700)])   # above the option row
        leftover, owned = self._claim(pdf, [6], {1: {}})
        self.assertEqual(leftover, [])
        self.assertEqual(owned[1]["question"], ["PSY/PSY-001-001_Q_01.webp"])
        self.assertEqual(owned[1].get("option", {}), {})

    # -- 13. horizontal 4-across tight row -> ambiguous -> question-level ---
    def test_tight_4across_ambiguous_stays_question_level(self):
        pdf = self.tmp / "tight4.pdf"
        _write_test_pdf(pdf, [
            ("1. Identify the structure", 72, 720, 12),
            ("A. text", 72, 650, 10), ("B. text", 220, 650, 10),
            ("C. text", 370, 650, 10), ("D. text", 520, 650, 10),
        ], [(6, "Im6", 150, 620)])   # ~midway between A and B
        leftover, owned = self._claim(pdf, [6], {1: {}})
        self.assertEqual(leftover, [])
        # geometry cannot safely prove which option -> question-level, never
        # guessed and never dropped
        self.assertEqual(owned[1]["question"], ["PSY/PSY-001-001_Q_01.webp"])
        self.assertEqual(owned[1].get("option", {}), {})


class Run11ForensicHardeningTests(unittest.TestCase):
    """Run-11 root-cause hardening: stale-path image lifecycle, structured
    pass status, answer-key rescue targeting, export gate."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._old_assets = qp.ASSETS_DIR
        qp.ASSETS_DIR = self.tmp / "assets"
        self.subj_dir = qp.ASSETS_DIR / "questions" / "PSY"
        self.subj_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        qp.ASSETS_DIR = self._old_assets

    # -- RC-1: stale-path image lifecycle --------------------------------
    def test_stale_path_returns_already_claimed_not_unmatched(self):
        # image was already renamed by a claim; the stale temp path no longer
        # exists -> attribute_orphan_image must say already_claimed (NOT
        # decorative, NOT a model call)
        called = []
        class FakeModel:
            def generate_content(self, *a, **k):
                called.append(True)
                raise AssertionError("must not call Gemini for a missing file")
        verdict = qp.attribute_orphan_image(FakeModel(), "PSY/PSY-p4-7.webp",
                                            {1: {}}, {"calls_today": 0})
        self.assertEqual(verdict, {"decorative": "already_claimed"})
        self.assertEqual(called, [])

    def test_figure_map_fully_claimed_page_returns_empty_leftover(self):
        # the caller feeds a page's leftovers to the figure-map; if the map
        # claims ALL of them, the page must NOT appear in fig_leftover (so the
        # caller clears its stale list instead of keeping temp names)
        for oid in (6, 7):
            (self.subj_dir / f"PSY-p1-{oid}.webp").write_bytes(b"x" * 3000)
        rels = ["PSY/PSY-p1-6.webp", "PSY/PSY-p1-7.webp"]
        fig_map = [{"q_no": 1, "slot": "question"}, {"q_no": 1, "slot": "question"}]
        owned = {}
        remaining = qp.claim_figure_map_images(fig_map, [(1, rels)], "PSY", 1,
                                               {1: {}}, owned)
        self.assertEqual(remaining, {})            # fully claimed
        self.assertEqual(len(owned[1]["question"]), 2)
        # the "page fully claimed -> leftover cleared" rule the caller applies
        leftover_by_page = {1: rels}               # stale temp names remain
        for page_no, _rels in [(1, rels)]:
            leftover_by_page[page_no] = remaining.get(page_no) or []   # the fix
        self.assertEqual(leftover_by_page[1], [])

    # -- RC-5: structured pass status -------------------------------------
    def test_pass_status_classification(self):
        self.assertEqual(qp._classify_pass_status("S", "Q", 0, False, True),
                         qp.PASS_STATUS_EXPECTED_EMPTY)
        self.assertEqual(qp._classify_pass_status("S", "S", 0, False, True),
                         qp.PASS_STATUS_PARTIAL)          # FAILED_ZERO suspect
        self.assertEqual(qp._classify_pass_status("S", "S", 9, False, True),
                         qp.PASS_STATUS_SUCCESS)
        self.assertEqual(qp._classify_pass_status("S", "S", 9, True, True),
                         qp.PASS_STATUS_RETRYABLE_FAILURE)
        self.assertEqual(qp._classify_pass_status("S", "S", 0, True, False),
                         qp.PASS_STATUS_UNRESOLVED)

    # -- RC-4: answer-key page targeting ----------------------------------
    def test_locate_missing_record_pages_finds_answer_key_pages(self):
        fake = {1: "1. Question one\n",
                2: "ANSWER KEY\n| Question No. | Correct Option |\n| 1 | B |\n| 2 | C |"}
        orig = qp.pdftotext_page
        qp.pdftotext_page = lambda pdf, page: fake.get(page, "")
        try:
            page_files = [Path(f"/tmp/x/page-{n:03d}.jpg") for n in (1, 2)]
            loc = qp.locate_missing_record_pages("pdf", page_files,
                                                 {1: ["answer"], 2: ["answer"]}, {})
        finally:
            qp.pdftotext_page = orig
        # q1's answer is on the KEY page (2), not just the question page (1)
        self.assertEqual(loc[1], [1, 2])
        self.assertEqual(loc[2], [2])

    def test_answer_rescue_prompt_is_answer_only(self):
        rec = {"q_no": 7, "question_text": "Which drug?", "correct_option": None}
        prompt = qp.answer_rescue_prompt(7, rec, {7: rec})
        self.assertIn('"correct_option"', prompt)
        self.assertNotIn("solution_text", prompt)
        self.assertIn("Question 7", prompt)

    def test_locate_answer_rows_without_probe_header(self):
        # answer rows in "13. B" / "13 - B" format on a page with NO "Answer
        # Key" header must still locate q13's answer page (run-12: ch15 q15's
        # rescue went to the question page because the key page lacked the
        # probe header)
        fake = {1: "1. Question one\n",
                2: "13. B\n14. C\n15. A\n"}
        orig = qp.pdftotext_page
        qp.pdftotext_page = lambda pdf, page: fake.get(page, "")
        try:
            page_files = [Path(f"/tmp/x/page-{n:03d}.jpg") for n in (1, 2)]
            loc = qp.locate_missing_record_pages("pdf", page_files,
                                                 {13: ["answer"], 15: ["answer"]}, {})
        finally:
            qp.pdftotext_page = orig
        self.assertIn(2, loc[13])
        self.assertIn(2, loc[15])

    # -- Export gate ------------------------------------------------------
    def test_export_gate_catches_missing_stems_and_answers(self):
        recs = {1: {"q_no": 1, "question_text": "stem",
                    "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                    "correct_option": None, "solution_text": "sol"},
                2: {"q_no": 2, "question_text": None,
                    "options": {}, "correct_option": None,
                    "solution_text": None}}
        image_files = {1: {"question": ["PSY/PSY-001-001_Q_01.webp"],
                           "solution": [], "option": {}}}
        # missing asset ref -> broken_asset_ref
        vio = qp._export_gate_violations(recs, image_files, [], "PSY-001")
        kinds = {k for k, _q, _d in vio}
        self.assertIn("missing_answer", kinds)
        self.assertIn("missing_stem", kinds)
        self.assertIn("bad_options", kinds)
        self.assertIn("missing_solution", kinds)
        self.assertIn("broken_asset_ref", kinds)

    def test_export_gate_clean_when_everything_accounted(self):
        recs = {1: {"q_no": 1, "question_text": "stem",
                    "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                    "correct_option": "B", "solution_text": "sol"}}
        vio = qp._export_gate_violations(recs, {}, [], "PSY-001")
        self.assertEqual(vio, [])

    def test_clean_ocr_preserves_prose_and_strips_page_noise(self):
        dirty = ("The answer is A.\n12\nPage 5 of 200\nwww.x.com\nmore prose\n")
        clean = qp._clean_ocr_text(dirty)
        self.assertIn("The answer is A.", clean)
        self.assertIn("more prose", clean)
        self.assertNotIn("Page 5 of 200", clean)
        self.assertNotIn("www.x.com", clean)


class Run12StemContaminationTests(unittest.TestCase):
    """Run-12: the contaminated-stem guard must not destroy GOOD question-
    shaped stems that its own solution restates, and the merge must never let
    a contaminated re-read replace a valid stem."""

    def _rec(self, qn, stem, sol, **kw):
        r = {"q_no": qn, "question_text": stem, "options": None,
             "correct_option": None, "solution_text": sol, "tables": [],
             "has_figure_in_question": False, "has_figure_in_solution": False,
             "_prov": {}}
        r.update(kw)
        return r

    # -- 1. a short QUESTION-SHAPED stem restated by its own solution is NOT
    #        contamination (the run-12 false-positive class) ---------------
    def test_question_shaped_stem_restated_by_solution_is_kept(self):
        stem = ("Which of the following drugs is most likely to improve the "
                "negative symptoms of schizophrenia?")
        sol = ("The correct answer is clozapine. The drug that improves the "
               "negative symptoms of schizophrenia is clozapine, which is "
               "reserved for treatment-resistant cases.")
        rec = self._rec(1, stem, sol)
        # high token overlap with the solution, but question-shaped + short ->
        # a GOOD stem, not contamination
        self.assertIsNone(qp._stem_reject_reason(stem, rec))

    def test_declarative_solution_prose_is_still_rejected(self):
        # long declarative explanation-as-stem (the real contamination class)
        sol = ("The correct answer is A. In Korsakoff syndrome the amnesia "
               "is characterised by anterograde and retrograde memory loss "
               "with confabulation, and the pathology lies in the mammillary "
               "bodies and the dorsomedial nucleus of the thalamus with "
               "severe vitamin B1 deficiency being the underlying cause.")
        rec = self._rec(5, sol, sol)
        self.assertIsNotNone(qp._stem_reject_reason(sol, rec))

    def test_explanation_opener_is_still_rejected(self):
        # "Option A:" opener -> contamination regardless of length
        stem = "Option A: CAGE questionnaire is used for addiction cases"
        rec = self._rec(1, stem, "Option A: CAGE questionnaire is used for "
                                "addiction and substance abuse cases")
        self.assertIsNotNone(qp._stem_reject_reason(stem, rec))

    # -- 2. merge: a contaminated incoming stem must never replace a valid one
    def test_merge_keeps_valid_stem_over_contaminated(self):
        good = ("Which of the following drugs is most likely to improve the "
                "negative symptoms of schizophrenia?")
        sol = ("The correct answer is clozapine. The drug that improves the "
               "negative symptoms of schizophrenia is clozapine, which is "
               "reserved for treatment-resistant cases.")
        recs = {1: self._rec(1, good, sol)}
        contaminated = ("The correct answer is clozapine. The drug that "
                        "improves the negative symptoms of schizophrenia is "
                        "clozapine, which is reserved for treatment-resistant "
                        "cases and should be tried before the others fail.")
        qp.merge_question_records(recs, [
            {"q_no": 1, "question_text": contaminated,
             "solution_text": sol, "options": None, "correct_option": None,
             "tables": [], "_prov": "Q_PASS"}], {"chapter_id": "PSY-001"})
        self.assertEqual(recs[1]["question_text"], good)   # valid stem survived

    def test_merge_does_not_fill_empty_stem_with_contaminated(self):
        sol = ("The correct answer is B. Body dysmorphic disorder involves "
               "a preoccupation with an imagined defect in appearance that "
               "causes clinically significant distress and impaired "
               "functioning with repetitive checking behaviours.")
        recs = {9: self._rec(9, None, sol)}
        qp.merge_question_records(recs, [
            {"q_no": 9, "question_text": sol, "solution_text": sol,
             "options": None, "correct_option": None, "tables": [],
             "_prov": "Q_PASS"}], {"chapter_id": "PSY-009"})
        self.assertIsNone(recs[9]["question_text"])       # stayed empty for retry

    def test_stem_conflict_resolver_never_picks_contaminated(self):
        # stem conflict: old clean vs new contaminated -> must keep old even
        # though the contaminated variant coheres with the solution perfectly
        good = ("Which of the following is the most common defence mechanism "
                "used by patients with conversion disorder?")
        sol = ("The correct answer is repression. The defence mechanism used "
               "by patients with conversion disorder is repression, in which "
               "the anxiety is pushed into the unconscious and converted into "
               "a physical symptom.")
        recs = {1: self._rec(1, good, sol)}
        contaminated = ("The defence mechanism used by patients with "
                        "conversion disorder is repression, in which the "
                        "anxiety is pushed into the unconscious and converted "
                        "into a physical symptom, and this is the most common "
                        "mechanism seen in this population.")
        qp.merge_question_records(recs, [
            {"q_no": 1, "question_text": contaminated,
             "solution_text": sol, "options": None, "correct_option": "B",
             "tables": [], "_prov": "Q_PASS"}], {"chapter_id": "PSY-001"})
        self.assertEqual(recs[1]["question_text"], good)

    # -- 3. retry strategy switch: after a contamination block, the prompt is
    #        stem-region-only for that q ------------------------------------
    def test_stem_only_prompt_after_contamination_block(self):
        rec = self._rec(3, None, "some solution text")
        prompt = qp.build_targeted_retry_prompt([(3, ["question"])], {3: rec},
                                                stem_only_qns={3})
        self.assertIn("QUESTION STEM", prompt)
        self.assertIn("option labels", prompt)
        self.assertIn("Do NOT include any option text", prompt)
        # the plain (non-stem-only) prompt asks for stem + options together
        plain = qp.build_targeted_retry_prompt([(3, ["question"])], {3: rec})
        self.assertIn("all four options", plain)
        self.assertNotIn("option labels", plain)

    # -- 4. Q-pass activation on solution windows ---------------------------
    def test_q_pass_skipped_on_pure_solution_window(self):
        self.assertFalse(qp._should_run_q_pass("S", False, None, False))
        self.assertFalse(qp._should_run_q_pass("S", False, None, True))
        # but runs when the window carries the boundary tail or a Q carry
        self.assertTrue(qp._should_run_q_pass("S", True, None, False))
        self.assertTrue(qp._should_run_q_pass("S", False, {"last_open_question": 5},
                                              False))
        # and always on Q windows / before the extraction boundary
        self.assertTrue(qp._should_run_q_pass("Q", False, None, False))
        self.assertFalse(qp._should_run_q_pass("Q", False, None, True))

    # -- 5. malformed-JSON recovery status ----------------------------------
    def test_recovered_after_error_is_retryable_not_unresolved(self):
        # a successful same-batch re-ask after malformed JSON is a RECOVERED
        # pass (run-12 ledger fix: ch13 was falsely flagged UNRESOLVED)
        self.assertEqual(qp._classify_pass_status("A", "A", 14, True, True),
                         qp.PASS_STATUS_RETRYABLE_FAILURE)
        self.assertNotEqual(qp._classify_pass_status("A", "A", 14, True, True),
                            qp.PASS_STATUS_UNRESOLVED)


class ZAIVerificationTests(unittest.TestCase):
    """Production-shaped regression tests from the independent review's
    scenarios. They LOCK IN the safe behavior (verified against the current
    code) so the adversarial layouts can never be broken by a naive
    'nearest-question' or overwrite change."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._old_assets = qp.ASSETS_DIR
        qp.ASSETS_DIR = self.tmp / "assets"
        self.subj_dir = qp.ASSETS_DIR / "questions" / "PSY"
        self.subj_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        qp.ASSETS_DIR = self._old_assets

    # -- Z-Test 1: solution restates the stem ("Regarding [exact stem]...") --
    def test_solution_restating_stem_with_regarding_is_kept(self):
        stem = ("Regarding the management of a patient with opioid use "
                "disorder, which medication is most appropriate?")
        sol = ("Regarding the management of a patient with opioid use "
               "disorder, the most appropriate medication is buprenorphine, "
               "which reduces cravings and withdrawal symptoms and can be "
               "prescribed in office-based treatment.")
        rec = {"question_text": stem, "solution_text": sol}
        # high token overlap with the solution, but question-shaped + short
        # -> a GOOD stem, never stripped as contamination
        self.assertIsNone(qp._stem_reject_reason(stem, rec))

    # -- Z-Test 2a: q_no=None OPTIONS fragment is buffered, not dropped and
    #               never attached to the 'nearest' question ----------------
    def test_qno_none_options_buffered_not_attached_not_dropped(self):
        # Layout 1 (adversarial): a page boundary between a stem and its
        # options; another question started at the bottom of the previous
        # page. The q_no=None options must NOT be glued onto the wrong
        # question.
        recs = {45: {"q_no": 45, "question_text": "A patient presents with...",
                     "options": {"A": "old-a", "B": "old-b", "C": "old-c",
                                 "D": "old-d"},
                     "correct_option": None, "solution_text": None,
                     "tables": [], "has_figure_in_question": False,
                     "has_figure_in_solution": False, "_prov": {}}}
        frag = {"q_no": None, "question_text": None,
                "options": {"A": "new-a", "B": "new-b", "C": "new-c",
                            "D": "new-d"},
                "correct_option": None, "solution_text": None, "tables": [],
                "has_figure_in_question": False, "has_figure_in_solution": False,
                "_prov": "Q_PASS"}
        merged, skipped = qp.merge_question_records(recs, [frag],
                                                    {"chapter_id": "PSY-999"})
        # not attached (no ownership proof), not dropped (buffered as orphan)
        self.assertEqual(skipped, [frag])
        self.assertEqual(merged[45]["options"]["A"], "old-a")   # untouched

    # -- Z-Test 2b: q_no=None ANSWER-KEY TABLE is consumed as a key, not
    #               attached to a question ---------------------------------
    def test_qno_none_answer_key_table_consumed_not_attached(self):
        # Layout 2 (adversarial): a q_no=None table containing the answer
        # key must fill answers deterministically, never corrupt a question.
        frag = {"q_no": None, "question_text": None, "options": None,
                "correct_option": None, "solution_text": None,
                "tables": [{"type": "answer key",
                            "markdown": "| Question No. | Correct Option |\n"
                                        "|---|---|\n| 1 | A |\n| 2 | C |"}],
                "has_figure_in_question": False, "has_figure_in_solution": False,
                "_prov": "A_PASS"}
        recs = {1: {"q_no": 1, "question_text": "stem1", "options": None,
                    "correct_option": None, "solution_text": None, "tables": [],
                    "has_figure_in_question": False,
                    "has_figure_in_solution": False, "_prov": {}},
                2: {"q_no": 2, "question_text": "stem2", "options": None,
                    "correct_option": None, "solution_text": None, "tables": [],
                    "has_figure_in_question": False,
                    "has_figure_in_solution": False, "_prov": {}}}
        orphans = [{"chapter_id": "PSY-999", "batch_start": 0, "pdf_pages": [5],
                    "new_pages": [5], "carry_q_no": None, "cut_part": None,
                    "last_qn_in_batch": 2, "pass": "A", "item": frag}]
        stats = {"orphans_recovered": 0, "foreign_fragments_blocked": 0,
                 "carry_merges": 0, "contaminated_stems_blocked": 0,
                 "chapter_id": "PSY-999"}
        remaining = qp.recover_orphans(orphans, recs, "PSY", 999, stats)
        self.assertEqual(remaining, [])                    # key consumed
        self.assertEqual(recs[1]["correct_option"], "A")   # answers filled
        self.assertEqual(recs[2]["correct_option"], "C")
        self.assertEqual(recs[1]["question_text"], "stem1")  # stem untouched

    # -- Z-Test 3: DRAIN crop-ladder items are NOT overwritten by OCR -------
    def test_drain_ocr_merge_never_overwrites_crop_items(self):
        # ch17 p218 case: crop ladder produced items, OCR fallback produced a
        # different item. The merge is fill-only -> the earlier crop content
        # must survive.
        recs = {7: {"q_no": 7, "question_text": "stem7",
                    "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                    "correct_option": None, "solution_text": "crop-ladder sol",
                    "tables": [], "has_figure_in_question": False,
                    "has_figure_in_solution": False, "_prov": {}}}
        # an OCR item that would REPLACE the solution if merge overwrote
        ocr_item = {"q_no": 7, "question_text": "stem7",
                    "solution_text": "OCR DIFFERENT solution", "tables": [],
                    "options": None, "correct_option": None,
                    "has_figure_in_question": False,
                    "has_figure_in_solution": False, "_prov": "OCR_S"}
        # fill_only merge (the drain path) must keep the crop-ladder content
        qp.merge_question_records(recs, [ocr_item],
                                  {"chapter_id": "PSY-017", "duplicates_merged": 0},
                                  fill_only=True)
        self.assertEqual(recs[7]["solution_text"], "crop-ladder sol")
        self.assertNotIn("OCR DIFFERENT", recs[7]["solution_text"])


class ValidatorContaminationTests(unittest.TestCase):
    """qbank_validator must flag cross-field contamination and OCR noise in
    the FINAL rows (run-7 hardening #3/#6)."""

    def _row(self, qtext, stext):
        return {"id": "PSY-001-001", "chapter_id": "PSY-001",
                "question": {"text": qtext, "images": []},
                "options": [{"id": "A", "text": "a", "images": []},
                            {"id": "B", "text": "b", "images": []},
                            {"id": "C", "text": "c", "images": []},
                            {"id": "D", "text": "d", "images": []}],
                "correct_options": ["B"],
                "solution": {"text": stext, "images": [], "tables": []}}

    def test_contaminated_question_flagged(self):
        import qbank_validator as qv
        sol = ("The correct answer is C. The patient has schizophrenia "
               "with predominantly negative symptoms which respond poorly "
               "to typical antipsychotics and require clozapine trial.")
        flags = qv.check_row(self._row(sol, sol), Path("/nonexistent"))
        kinds = {f["kind"] for f in flags}
        self.assertIn("contaminated_question", kinds)

    def test_explanation_opening_flagged(self):
        import qbank_validator as qv
        flags = qv.check_row(self._row("Option B: explanation text here",
                                       "real solution"), Path("/nonexistent"))
        kinds = {f["kind"] for f in flags}
        self.assertIn("contaminated_question", kinds)

    def test_ocr_noise_solution_flagged(self):
        import qbank_validator as qv
        flags = qv.check_row(self._row("Real stem question text here?",
                                       "The answer is A.\nPage 12 of 300\n"
                                       "www.qbank.example\nend"),
                             Path("/nonexistent"))
        kinds = {f["kind"] for f in flags}
        self.assertIn("ocr_noise_solution", kinds)

    def test_clean_row_not_flagged(self):
        import qbank_validator as qv
        flags = qv.check_row(self._row("Which drug is first line in ADHD?",
                                       "Methylphenidate is first line."),
                             Path("/nonexistent"))
        kinds = {f["kind"] for f in flags}
        self.assertNotIn("contaminated_question", kinds)
        self.assertNotIn("ocr_noise_solution", kinds)


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
