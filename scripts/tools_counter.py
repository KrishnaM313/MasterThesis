from collections import Counter
import math

def getCounterPercentile(percentile: int,counter: Counter):
    total = sum(counter.values())
    current = 0
    for item in counter.items():
        if current/total > percentile/100:
            return item[0]
        current += item[1]

# def getCounterStats(counter: Counter):
#     sum_of_numbers = sum(number*count for number, count in counter.items())
#     count = sum(count for n, count in counter.items())
#     mean = sum_of_numbers / count
#     total_squares = sum(number*number * count for number, count in counter)
#     mean_of_squares = total_squares / count
#     variance = mean_of_squares - mean * mean
#     std_dev = math.sqrt(variance)
#     #Source: https://stackoverflow.com/a/33695469

#     return mean, variance, std_dev