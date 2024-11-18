import time
import numpy as np

# ---
# Benchmarking Single Instruction, Multiple-Data operation
# ---

# creating two large lists
SIZE = 10**8  # 100 million - 100_000_000
list1 = [1.0] * SIZE
list2 = [2.0] * SIZE

start = time.monotonic()
result = [a + b for a, b in zip(list1, list2)]

end = time.monotonic()
print(f"Time without SIMD: {end - start}")

arr1 = np.ones(SIZE)
arr2 = np.ones(SIZE) * 2

start = time.monotonic()

result = arr1 + arr2

end = time.monotonic()

print(f"Time with SIMD: {end - start}")

# On my desktop PC:
# Time without SIMD: 5.986150297000222
# Time with SIMD: 1.995753385000171
