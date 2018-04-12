#!/usr/bin/env python3

'try grocery'

__author__ = 'lxp'

import pandas as pd
from tgrocery import Grocery
from pandas import Series, DataFrame
import numpy as np

grocery = Grocery('sample')
grocery.train('trainingData.txt', delimiter = ',')
grocery.save()
new = Grocery('sample')
new.load()
print(new.predict('Game of Thrones series 1-3 available for catch-up on NOW TV and Sky '))