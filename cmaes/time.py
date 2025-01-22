import os
import time

# 1 nano == 1_000_000_000 seconds => 1_000_000 ms
def nano2ms(t: int) -> float:
    return t / 1_000_000

cwd = os.getcwd()
TIME_DIR = os.path.join(cwd, "cmaes/data", "times.txt")

total_time = 0
with open(TIME_DIR, 'r') as fin:
    for line in fin.readlines():
        try:
            time_ns = int(line)
            time_ms = nano2ms(time_ns)
            total_time += time_ms
        except ValueError:
            print(f"Failed to convert {line} to an integer")
        except:
            print("")