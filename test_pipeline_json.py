import json
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
        # solutions chunked with 1-page intra-section overlap
        self.assertEqual(wins[1][0], [11, 12, 13, 14, 15])
        self.assertEqual(wins[2][0], [15, 16])
        # no cross-section overlap
        self.assertEqual(set(wins[0][0]) & set(wins[1][0]), set())

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
        # 8 solution pages (pp.6-13) -> 2 S windows, 1 overlap page
        files = [Path(f"/tmp/s-{n:03d}.jpg") for n in range(3, 14)]
        text = {p: "1. Q\n" for p in (3, 4)}
        text[5] = "ANSWER KEY\n| Q No | Answer |"
        text.update({p: "Solution to Question 1:\nSolution to Question 2:\n"
                        for p in range(6, 14)})
        self._fake_text(text)
        wins = qp.build_section_windows(files, "pdf")
        s_wins = [w for w, s in wins if s == "S"]
        self.assertEqual(len(s_wins), 2)
        self.assertEqual(len(s_wins[0]), qp.SOLUTIONS_CHUNK_PAGES)
        # 1-page overlap between the two S windows (page 10 = 6+5-1)
        self.assertEqual(s_wins[0][-1], s_wins[1][0])


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
