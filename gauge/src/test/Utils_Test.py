import unittest
from src.utils import Utils


class MyTestCase(unittest.TestCase):
    def test_something(self):
        print('>>> ', Utils.generate_img_title())
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
