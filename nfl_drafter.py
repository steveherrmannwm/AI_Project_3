import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
import pydotplus
# from sklearn.preprocessing import normalize
import csv

# Load Previous Combine Data
combine_stats = []
was_drafted = []


def convert_row(row_to_change):
    row_to_change = row_to_change[4:]
    row_to_change = [float(x) if x != '' else 0.0 for x in row_to_change]
    return row_to_change

with open("CornerBacks.csv", "rb") as csvfile:
    nfl_data = csv.reader(csvfile, delimiter=",", quotechar='"')
    is_headers = True
    # Skip our CSV header
    for row in nfl_data:
        if is_headers is True:
            is_headers = False
            print "SKIPPED HEADERS"
            continue
        print row
        row = convert_row(row)
        combine_stats.append(row[:-1])
        was_drafted.append(row[-1])


def unified_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

zero_count = 0
for x in combine_stats:
    if x[-1] == 0:
        zero_count += 1

print zero_count
print len(combine_stats) - zero_count

combine_stats = np.array(combine_stats)
was_drafted = np.array(was_drafted)

combine_stats, was_drafted = unified_shuffle(combine_stats, was_drafted)

test_combine_stats = combine_stats[9 * len(combine_stats) / 10:]
test_labels = was_drafted[9 * len(was_drafted)/10:]

combine_stats = combine_stats[:9 * len(combine_stats) / 10]
was_drafted = was_drafted[:9 * len(was_drafted)/10]
test = []
new_labels = []
print "Actual Labels: "

print test_labels
zero_count = 0
for x in test_labels:
    if x == 0:
        zero_count += 1
print zero_count
print len(test_labels) - zero_count
print

print "Starting Decision Tree Classifier"
learn = DecisionTreeClassifier()
model = learn.fit(combine_stats, was_drafted.ravel())
output = learn.predict(test_combine_stats)
dot_data = tree.export_graphviz(learn, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("tree.pdf")
print output

misses = 0
for out, actual in zip(output, test_labels):
    if out != actual:
        misses += 1

print '--------------'
print 'Missed:', misses
print "accuracy: ", 1 - float(misses)/(len(output) * 1.0)


print "Trying SGGClassifer"
learn = SGDClassifier(loss="hinge")
learn.fit(combine_stats, was_drafted.ravel())
output = learn.predict(test_combine_stats)
print output
misses = 0
for out, actual in zip(output, test_labels):
    if out != actual:
        misses += 1
print 'Missed:', misses
print 1 - float(misses)/(len(output) * 1.0)
print '--------------'

learn = MLPClassifier(hidden_layer_sizes=(15, ), activation='tanh', solver='sgd')

print "Trying MLPClassifer"
learn.fit(combine_stats, was_drafted.ravel())
output = learn.predict(test_combine_stats)
print output
misses = 0
print '--------------'
for out, actual in zip(output, test_labels):
    if out != actual:
        misses += 1
print 'Missed:', misses
print len(output)
print 1 - float(misses)/(len(output) * 1.0)
