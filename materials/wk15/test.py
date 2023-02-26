# enlarge the recursion limit
import sys
import time
import random
from functools import wraps

sys.setrecursionlimit(100000)


def quick_sort(a):
    if len(a) <= 1:
        return a
    pivot = a[0]
    left = [x for x in a[1:] if x < pivot]
    right = [x for x in a[1:] if x >= pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)


if __name__ == "__main__":
    start = time.time()
    total_len = int(5e6)
    random.seed(0)
    a = [random.randint(0, total_len//10) for _ in range(total_len)]
    result = quick_sort(a)
    print(result[:10])
    total_time = time.time() - start
    print(f"total time: {total_time:.3f}s ({total_time/60:.3f}min)")
    print(f'random state: {random.random():.5f}')
