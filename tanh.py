import math 
import sys

for i in range(1024):
    sys.stdout.write((str)(math.tanh(-4 + 8 * i /1024)))
    sys.stdout.write(',')
