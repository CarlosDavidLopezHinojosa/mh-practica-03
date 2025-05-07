# from occ import _mean
# import numpy as np
# import time

# def test_mean():
#     # Test with a numpy array
#     data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
#     expected_mean = np.mean(data)
#     assert _mean(data) == expected_mean, f"Expected {_mean(data)} to be {expected_mean}"

# def compare_performance():
#     sizes = [10, 100, 1000, 10000, 100000, 1_000_000, 10_000_000 ]
#     for size in sizes:
#         data = np.random.rand(size).astype(np.float64)

#         start = time.time()
#         custom_mean = _mean(data)
#         custom_time = time.time() - start
#         # Measure performance of numpy.mean
#         start = time.time()
#         np_mean = np.mean(data)
#         np_time = time.time() - start

#         # Measure performance of _mean
        

#         print(f"Size: {size}")
#         print(f"numpy.mean: {np_time:.6f}s, Result: {np_mean}")
#         print(f"_mean: {custom_time:.6f}s, Result: {custom_mean}")
#         print("-" * 40)

# if __name__ == "__main__":
#     test_mean()
#     print("All tests passed!")
#     compare_performance()