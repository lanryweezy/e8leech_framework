# import unittest
# import numpy as np
# from e8leech.utils.fp16 import to_fp16, from_fp16
# from e8leech.utils.serialization import to_json, from_json

# class TestUtils(unittest.TestCase):

#     def test_fp16_conversion(self):
#         """
#         Tests the fp16 conversion functions.
#         """
#         arr = np.random.rand(10)
#         arr_fp16 = to_fp16(arr)
#         self.assertEqual(arr_fp16.dtype, np.float16)
#         arr_fp32 = from_fp16(arr_fp16)
#         self.assertEqual(arr_fp32.dtype, np.float32)
#         self.assertTrue(np.allclose(arr, arr_fp32, atol=1e-3))

#     def test_serialization(self):
#         """
#         Tests the serialization functions.
#         """
#         data = {
#             'a': np.array([1, 2, 3]),
#             'b': 'hello',
#             'c': 123,
#             'd': 45.6
#         }
#         s = to_json(data)
#         data2 = from_json(s)
#         self.assertTrue(np.allclose(data['a'], data2['a']))
#         self.assertEqual(data['b'], data2['b'])
#         self.assertEqual(data['c'], data2['c'])
#         self.assertAlmostEqual(data['d'], data2['d'])

# if __name__ == '__main__':
#     unittest.main()
