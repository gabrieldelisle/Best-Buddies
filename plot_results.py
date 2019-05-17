import sys
import json
import matplotlib.pyplot as plt
import numpy as np

filename = "results_select.json"
	
with open(filename, 'r') as f:
	results = json.load(f)


layers = [1,2,3,4,5]
for k, v in results.items():
	print(k)
	print(v[0])
	plt.plot(layers,v[0],"o--", label="$\\gamma = " + str(k) + "$")


plt.grid(axis ="y",color='gray', linestyle='--', linewidth=1, alpha=0.4)

plt.ylabel("$|BB|$")
plt.xticks(layers)
plt.yscale("log")
plt.legend()
plt.savefig("bestbuddies_through_layers.pdf", bbox_inches="tight")
plt.show()
