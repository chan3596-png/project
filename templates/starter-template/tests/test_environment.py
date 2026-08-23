import unittest
import sys

class TestEnvironment(unittest.TestCase):
    def test_python_version(self):
        self.assertTrue(sys.version_info >= (3, 11), "Python version must be >= 3.11")

if __name__ == '__main__':
    unittest.main()
