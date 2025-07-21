import unittest
import numpy as np
from e8leech.utils.fp16 import to_fp16, from_fp16

class TestFP16(unittest.TestCase):

    def test_fp16_conversion(self):
        """
        Tests the fp16 conversion functions.
        """
        arr = np.random.rand(10)
        arr_fp16 = to_fp16(arr)
        self.assertEqual(arr_fp16.dtype, np.float16)
        arr_fp32 = from_fp16(arr_fp16)
        self.assertEqual(arr_fp32.dtype, np.float32)
        self.assertTrue(np.allclose(arr, arr_fp32, atol=1e-3))

if __name__ == '__main__':
    unittest.main()
