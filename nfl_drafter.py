
from sklearn.neural_network import MLPClassifier
import csv

# Load Previous Draft Data

test_combines = ['2014', '2015']
test_combine_stats = []
test_labels = []

combine_stats = []
was_drafted = []

with open("nfl_data.csv", "rb") as csvfile:
    nfl_data = csv.reader(csvfile, delimiter=",", quotechar="|")
    is_headers = True
    for row in nfl_data:
        if is_headers is True:
            print "skipping headers"
            is_headers = False
            continue
        if row[0] in test_combines:
            test_combine_stats.append(row[2:16])
            test_labels.append(row[-1])
        else:
            combine_stats.append(row[2:16])
            was_drafted.append(row[-1])

# Normalize our data

# Train our MLP

# Test our draft data against that of the most recent draft

# Predict what next years draft might look like
