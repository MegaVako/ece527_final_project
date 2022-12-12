import math
import sys

def sigmoid(x):
    return 1 / (1 + math.exp(-x))



for i in range(1024):
    sys.stdout.write(str(sigmoid(-6 + 12  * i/1024)))
    sys.stdout.write(',')
