import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv("report_mcp_traffic_accidents_2014-01-01_2019-12-31.csv")
dic = dict()
i = 0
def mapper(j):
  global i
  if j not in dic:
    i += 1
    dic[j] = i
  return dic[j]

data['jurisdiction'] = data['jurisdiction'].map(mapper)

class Regress:
  def __init__(self, data):
    self.x = data['deadly_accidents'] + data['serious_accidents'] + data['other_accidents']
    self.y = data['deadly_accidents']
    self.slope, self.intercept, _, _, self.std_error = stats.linregress(self.x, self.y)

regressions = {jur: Regress(data.loc[data['jurisdiction'] == jur]) for jur in dic.values()}

i = 0

for jur in regressions:
    r = regressions[jur]
    ax = plt.subplot(6, 6, i := i + 1)
    ax.scatter(r.x, r.y)
    ax.plot(r.x, [x * r.slope + r.intercept for x in r.x])

plt.show()