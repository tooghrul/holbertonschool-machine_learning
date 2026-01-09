#!/usr/bin/env python3
import pandas as pd

arr = [{"First":0.0, "Second": "one"}, 
       {"First":0.5, "Second": "two"},
       {"First": 1.0, "Second": "three"},
       {"First": 1.5, "Second": "four"}]

df = pd.DataFrame(arr, rows = ['A', 'B', 'C', 'D'])

