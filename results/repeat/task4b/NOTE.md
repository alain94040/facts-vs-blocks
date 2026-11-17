# Note on the task 4b null (raised after the real run; method not changed)

The "has a match" rule compares each neuron's best similarity in layer j with the 99th
percentile of best similarities against a copy of j whose every vector has its residual
coordinates shuffled independently. That copy keeps norms but removes any structure shared by
all vectors of a layer (for example a common mean direction of the gate or down vectors), so
almost any real layer beats it:

| Distance | Neuron match fraction (all / no outliers) | Mutual best match | Median best similarity | Null median | Head match fraction | Mean matched OV / QK cosine (null 0.000) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.995 / 0.991 | 0.30 | 0.341 | 0.119 | 0.99 | 0.054 / 0.072 |
| 8 | 0.596 / 0.545 | 0.31 | 0.161 | 0.120 | 0.92-0.94 | - |

Match fractions near 1 coexist with only about 30% mutual best matches (flat over distance)
and head cosines near 0.05. The positive control (100% recovery at 30% noise) shows the method
finds true partners when they exist; it does not show that the null separates "shared unit"
from "layers share a common subspace". This is the same kind of weak-null issue as Phase 2 Task 5.
A stricter null (e.g. the same best-match statistic against a layer of a different model, or
a null that keeps each layer's mean direction) would be a method change, left to the reviewer.
