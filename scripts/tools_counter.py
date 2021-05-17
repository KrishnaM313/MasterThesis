from collections import Counter


def getCounterPercentile(percentile: int, counter: Counter):
    total = sum(counter.values())
    current = 0
    for item in counter.items():
        if current/total > percentile/100:
            return item[0]
        current += item[1]
