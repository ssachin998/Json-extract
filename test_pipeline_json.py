import unittest

from qbank_pipeline import parse_gemini_json_array


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

    def test_rejects_non_json_tail(self):
        with self.assertRaises(ValueError):
            parse_gemini_json_array('[{"q_no": 1}] explanation')


if __name__ == "__main__":
    unittest.main()
