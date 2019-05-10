import sys
import json
import matplotlib.pyplot as plt
import numpy as np

filename = "results.json"
	
with open(filename, 'r') as f:
	results = json.load(f)


layers= [5,4,3,2,1]

for k, v in results.items():
	print(k)
	print(v[0])
	plt.plot(layers,np.log(v[0]), label=k)

plt.legend()
plt.show()