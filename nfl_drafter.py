import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
import csv

# Load Previous Combine Data

test_combines = ['2014', '2015']
test_combine_stats = []
test_labels = []

combine_stats = []
was_drafted = []


def convert_row(row):
    print row
    row = row[2:16] + [row[-1]]
    row = [float(x) for x in row if x != '']
    return row


with open("nfl_data.csv", "rb") as csvfile:
    nfl_data = csv.reader(csvfile, delimiter=",", quotechar='"')
    is_headers = True
    # Skip our CSV header
    for row in nfl_data:
        if is_headers is True:
            print row
            is_headers = False
            continue
        if row[0] in test_combines:
            row = convert_row(row)
            test_combine_stats.append(row)
            test_labels.append(row[-1])
        else:
            row = convert_row(row)
            combine_stats.append(row)
            was_drafted.append(row[-1])

# TODO: Normalize our data

# Train our MLP
combine_stats = np.array(combine_stats)
was_drafted = np.array(was_drafted)
print combine_stats
print was_drafted
normalized_stats = normalize(combine_stats, axis=1)
np.random.shuffle(normalized_stats)
print normalized_stats
learn = MLPClassifier(hidden_layer_sizes=(14, ), activation='tanh', solver='sgd')
model = learn.fit(normalized_stats, was_drafted)
print model
test_combine_stats = np.array(test_combine_stats)
print test_combine_stats
output = learn.predict(normalize(test_combine_stats, axis=1))
for x in output:
    if x == -1:
        print x
print '-----------'
print output
# Test our draft data against that of the most recent draft

# Predict what next years draft might look like
