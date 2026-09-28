# Validation

| check | status | detail |
|---|---|---|
| every sample belongs to a run | PASS | 0 unknown runs |
| every sample has source statistics | PASS | 0 missing |
| every distance array matches the reference | PASS | 0 of 29060 wrong |
| changing weights never changed the topology | PASS | 10 groups checked, 0 broken |
| layered graphs have the depth their construction implies | PASS | 0 off |
| grid depth is at most rows + cols | PASS | 0 off |
| geometric graphs are deep (no spanning tree) | PASS | smallest depth 698 |
| timed phases cover at least 90% of wall time | WARN | 1382 runs below, from gpu-nearfar |
| no other process was on the GPU | PASS | 0 of 11487 GPU runs |
| GPU at full clock before each run | PASS | median 2509 MHz, 0 runs slower |
| results come from committed code | WARN | 165 of 165 runs, code hash c9083752a348 |
| no duplicate samples | PASS | 0 duplicates |
