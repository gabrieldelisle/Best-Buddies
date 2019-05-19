import shutil
import os
import numpy as np
from numpy.random import shuffle

N_train = 1000
N_val = 100

try:
	shutil.rmtree('parsed_img')
except:
	pass

os.mkdir('parsed_img')



with open("list_category_img.txt", 'r') as f:
	lines = np.array(f.read().split("\n")[2:])


index = np.arange(len(lines))
shuffle(index)
train = index[:N_train]
val = index[N_train:N_train+N_val]


os.mkdir('parsed_img/train')
for i,l in enumerate(lines[train]):
	l = l.split(' ')
	name = l[0]
	label = l[-1]
	try:
		os.mkdir('parsed_img/train/'+label)
	except:
		pass
	shutil.copyfile(name, 'parsed_img/train/'+label+'/' +name.split('/')[-2] +name.split('/')[-1])


os.mkdir('parsed_img/val')
for i,l in enumerate(lines[val]):
	l = l.split(' ')
	name = l[0]
	label = l[-1]
	try:
		os.mkdir('parsed_img/val/'+label)
	except:
		pass
	shutil.copyfile(name, 'parsed_img/val/'+label+'/' +name.split('/')[-2]+ name.split('/')[-1])




try:
	shutil.rmtree('reparsed_img')
except:
	pass

os.mkdir('reparsed_img')

os.mkdir('reparsed_img/val')
os.mkdir('reparsed_img/train')
for i in range(1,4):
	os.mkdir('reparsed_img/val/'+str(i))
for i in range(1,4):
	os.mkdir('reparsed_img/train/'+str(i))

for i in range(1, 51):
	if i<=20:
		label=1
	elif i<=36:
		label=2
	else:
		label=3
	src = 'parsed_img/val/'+str(i)+'/'
	try: 
		for name in os.listdir(src):
			shutil.copy(src + name, 'reparsed_img/val/'+str(label)) 
	except:
		pass
	src = 'parsed_img/train/'+str(i)+'/'
	try: 
		for name in os.listdir(src):
			shutil.copy(src + name, 'reparsed_img/train/'+str(label)) 
	except:
		pass
