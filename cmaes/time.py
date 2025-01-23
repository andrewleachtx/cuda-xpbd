import os
import time
import numpy as np

# 1 nano == 1_000_000_000 seconds => 1_000_000 ms
def nano2ms(t: int) -> float:
    return t / 1_000_000

cwd = os.getcwd()
TIME_DIR = os.path.join(cwd, "cmaes/data", "times.txt")

times = []
with open(TIME_DIR, 'r') as fin:
    for line in fin.readlines():
        try:
            time_ns = int(line)
            time_ms = nano2ms(time_ns)
            times.append(time_ms)
        except ValueError:
            print(f"Failed to convert {line} to an integer")
        except:
            print("Err")

time_data = np.array(times)
total  = np.sum(time_data)
avg    = np.mean(time_data)
median = np.median(time_data)

print(f"Total kernel launches: {len(time_data)}")
print(f"Total time (ms): {total}")
print(f"Average time (ms): {avg}")
print(f"Median (ms): {median}")