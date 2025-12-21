import unittest
from bailing.utils import remove_think_tags

class TestResponseFilter(unittest.TestCase):
    def test_remove_think_tags_basic(self):
        text = "<think>This is a thought.</think>Hello world!"
        expected = "Hello world!"
        self.assertEqual(remove_think_tags(text), expected)

    def test_remove_think_tags_multiline(self):
        text = "<think>\nThinking about life...\nAnd everything.\n</think>\nHow are you?"
        expected = "\nHow are you?"
        self.assertEqual(remove_think_tags(text), expected)

    def test_remove_think_tags_no_tags(self):
        text = "Hello world!"
        expected = "Hello world!"
        self.assertEqual(remove_think_tags(text), expected)

    def test_remove_think_tags_unclosed(self):
        # 现在应该移除未闭合标签及其后面的内容
        text = "<think>I am thinking... Hello?"
        expected = ""
        self.assertEqual(remove_think_tags(text), expected)

    def test_remove_think_tags_multiple(self):
        text = "<think>T1</think>Part 1<think>T2</think>Part 2"
        expected = "Part 1Part 2"
        self.assertEqual(remove_think_tags(text), expected)

    def test_remove_think_tags_mixed(self):
        text = "Part 1<think>T1</think>Part 2<think>T2"
        expected = "Part 1Part 2"
        self.assertEqual(remove_think_tags(text), expected)

if __name__ == '__main__':
    unittest.main()
