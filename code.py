import torchvision.models as models
from torchvision import transforms
from network_info import *
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torch.nn.functional as functional

import sys
import json


# save all the options in this dictionary
options = None


def load_options(argv):
	if len(argv) > 1:
		filename = argv[1]
	else:
		filename = "options.json"
	
	with open(filename, 'r') as f:
		options = json.load(f)
	

	print(" +++ Running options +++\n")
	for key, value in options.items():
		print("\t",key, " : ", value)
	print("\n +++++++++++++++++++++++\n")
	return options



def load(im):
	I = Image.open(im).convert('RGB')
	transform = get_transform(224)
	I = transform(I).unsqueeze(0)
	return I


def get_transform(witdh):
    transform_list = []
    osize = [witdh, witdh]
    transform_list.append(transforms.Scale(osize, Image.BICUBIC))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])]

    return transforms.Compose(transform_list)

def forward_pass(im, net):

	# last output of the layer in the net,
	# initialize to the image
	last_activation = im

	# list of the activations at the 
	# selected layers.
	activations = [im]
	
	# for each layer
	for index , f in enumerate(net.features) :
		last_activation = f.forward(last_activation)
		# if the layer is selected, then append
		# it to the list.
		if index in options["layers"]:
			activations.append(last_activation)
	return activations
    
def normalize(FA,FB):
	mu_A = torch.mean(FA)
	mu_B = torch.mean(FB)
	mu = (mu_A + mu_B)/2
	sigma_A = torch.std(FA)
	sigma_B = torch.std(FB)
	sigma = (sigma_A + sigma_B)/2
	return (FA - mu_A) / sigma_A * sigma + mu, (FB - mu_B) / sigma_B * sigma + mu 


def neighbours(x, size, n) :
	return [x + i 
		for i in range(-(size//2), size//2+1) 
		if x+i >=0 and x+i <n
	]

def correlation_conv(CA, CB, neighbours_size=3, sum_over_neigh= False):
	# number of pixels in the region
	n = CA.shape[2]
	m = CA.shape[3]

	nb = CB.shape[2]
	mb = CB.shape[3]

	# print("CA: ",CA.shape)
	# print("CB: ",CB.shape)


	pad_list = tuple([neighbours_size // 2,neighbours_size // 2,neighbours_size // 2,neighbours_size // 2])
	pad_size = neighbours_size // 2
	# initialize matrix of correlations
	corr = torch.zeros((n*m,nb*mb))

	#print("size of corr matrix :", corr.shape)

	ones_filter = torch.ones((1,1,neighbours_size,neighbours_size))
	
	#print("Using filter of size : ", ones_filter.shape, " padding : ", pad_size)
	# normalize CA and CB

	normA = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (CA,CA)))
	normB = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (CB,CB)))
	
	# normalize puxels here
	CAnorm = CA / normA
	CBnorm = CB / normB
	
	# padding using reflected feature
	CAnorm = functional.pad(CAnorm, pad_list,  'reflect').data
	CBnorm = functional.pad(CBnorm, pad_list,  'reflect').data

	# for each pixel in image A (not starting from the padded region)
	for px in range(pad_size,n + pad_size):
		for py in range(pad_size,m + pad_size):
			
			# find the patch of neighbouring pixels
			ca = CAnorm[:,:,neighbours(px, neighbours_size, n+2*pad_size),:][:,:,:,neighbours(py, neighbours_size, m+2*pad_size)]
			
			## compute correlation
			# here there is no need to flip the image
			# since the convolution is implemented as correlation
			R = functional.conv2d(CBnorm, ca.contiguous(), padding=0).data
			if ca.shape[2] != ca.shape[3]:
				print(px,py,neighbours_size, n, m, pad_size)
				print(neighbours(px, neighbours_size, n+2*pad_size))
				print(neighbours(py, neighbours_size, m+2*pad_size))
				print()
				print(CBnorm.shape, ca.shape)
				print("After convolution ", R.shape)
			# if option is set, sum the correlations in the 
			# neighbouring region
			if sum_over_neigh:
				R = functional.pad(R, pad_list,  'reflect')
				R = functional.conv2d(R, ones_filter, padding=0).data
			
			# reshape to fit the result in the correlation matrix
			R = R.reshape(1, -1)
			R = R.squeeze()
			corr[(px-pad_size) * m + (py - pad_size),:] = R

	return corr



def correlation(CA, CB, size) :
	n = CA.shape[2]

	corr = torch.zeros((n*n,n*n))
	for px in range(n):
		for py in range(n):
			for qx in range(n):
				for qy in range(n):
					ca = CA[:,:,neighbours(px, size, n),:][:,:,:,neighbours(py, size, n)]
					cb = CB[:,:,neighbours(qx, size, n),:][:,:,:,neighbours(qy, size, n)]
					dot = torch.einsum("ijkl,ijuv->kluv", (ca,cb))
					normA = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (ca,ca)))
					normB = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (cb,cb)))
					normA = normA.view(*normA.shape, 1, 1)
					normB = normB.view(1, 1, *normB.shape)
					corr[px * n + py, qx * n + qy] = torch.sum (dot / normA / normB)
	return corr


def bestBuddies(FA, xA, yA, FB, xB, yB, radius):
	na = FA.shape[2]
	ma = FA.shape[3]


	nb = FB.shape[2]
	mb = FB.shape[3]

	CA, CB = (FA, FB)

	#CA, CB = normalize(FA, FB)
	corr = correlation_conv(CA, CB, radius)
	# filter for keeping only significative activations
	normA = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (FA,FA)))
	normB = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (FB,FB)))
	HA = (normA - torch.min(normA)) / (torch.max(normA) - torch.min(normA))
	HB = (normB - torch.min(normB)) / (torch.max(normB) - torch.min(normB))
	gamma = options["threshold"]
	bbA = []
	bbB = []

	#print(
	#	"CA", CA.shape,"\n",
	#	"CB", CB.shape,"\n",
	#	"HA", HA.shape,"\n",
	#	"HB", HB.shape,"\n",
	#	"corr", corr.shape
	#	)



	for p in range(corr.shape[0]):
		q = torch.argmax(corr[p,:]).item()
		if torch.argmax(corr[:,q]).item() == p:
			px, py = p//ma, p%ma
			qx, qy = q//mb, q%mb

			# print("p)",p , "-->", (px,py))
			# print("q)",q , "-->", (qx,qy))

			if HA[px, py] > gamma and HB[qx, qy] > gamma: 
				bbA.append((px + xA, py + yA))
				bbB.append((qx + xB, qy + yB))
	return bbA, bbB


radius_list=[4,4,6,6] #use radius_list[l-2] for l=5to2
l_layer=[5,10,19,28,37] #use l_layer[l-1] for l=5to1

def pyramid_search(FA_list, FB_list):

	# number of layers
	L = len(FA_list) 

	R = [[FA_list[-1], 0, 0, FB_list[-1], 0, 0]]
	finalA = []
	finalB = []

	for l in range(L-1,0,-1) :
		print("### Searching at level : ", l)
		new_R=[]
		
		# get the activations at
		# the leyer before
		FA = FA_list[l-1]
		FB = FB_list[l-1]

		# for every region at the current level
		# find the best buddies
		for regions in R:
			bbA, bbB = bestBuddies(*regions, radius=options["patch_radius"][l-1])
			#print(bbA, bbB)
			
			# if in the first layer
			# then save the best buddies
			if l==1:
				finalA += bbA
				finalB += bbB

			# if not in the last layer compute the 
			# regions in the above layer

			if l > 1:
				for k in range(len(bbA)):
					px, py = bbA[k]
					qx, qy = bbB[k]
					na = FA.shape[2]
					ma = FA.shape[3]
					nb = FB.shape[2]
					mb = FB.shape[3]
					r = radius_list[l-2]+1

					# the normalization is done 
					# patch wise
					R1 = FA[:,:,neighbours(2*px, r, na),:][:,:,:,neighbours(2*py, r, ma)]
					R2 = FB[:,:,neighbours(2*qx, r, nb),:][:,:,:,neighbours(2*qy, r, mb)]
					
					# if the regions are too small ignore them
					if R1.shape[2] < 2 or R1.shape[3] < 2 or R2.shape[2] < 2 or R2.shape[3] < 2: 
						continue

					R1, R2 = normalize(R1, R2)
					# print("for ",(px,py), " , ",(qx,qy))
					# print(R1.shape)
					# print(R2.shape)
					new_R.append(( R1, 2 * px - (r//2), 2 * py - (r//2),
						R2, 2 * qx - (r//2), 2 * qy - (r//2)))
				
					#new_R.append(( R1, 2 * px, 2 * py,
					#	R2, 2 * qx, 2 * qy))
				R = new_R

	return finalA, finalB

def display(im, points):
	points = np.array([[u for u in v] for v in points])
	points = KMeans(n_clusters=options["clusters"], random_state=0).fit(points).cluster_centers_
	r = 3
	n = im.shape[0]
	print(im.shape)
	for px,py in points:
		for u in neighbours(int(px), r, n) :
			for v in neighbours(int(py), r, n) :
				print(u,v)
				im[round(u),round(v)] = np.array([255,0,0])

	plt.imshow(im)
	plt.show()
	plt.close()



if __name__ == "__main__" :

	# assign the global variable option with the
	# options contained in the option file
	options = load_options(sys.argv)


	# load the model
	VGG19 = models.vgg19(pretrained=True)

	# print some informations about the network

	nameA = "pietro.png"
	nameB = "gab.png"

	imA = load(nameA)
	FA = forward_pass(imA, VGG19)

	imB = load(nameB)
	FB = forward_pass(imB, VGG19)


	# print sizes of intermediate "images"
	net_info(VGG19, imA)
	intermediate_shapes(FA)


	pointsA, pointsB = pyramid_search(FA, FB)
	imA = np.array(Image.open(nameA).convert('RGB'))

	imB = np.array(Image.open(nameB).convert('RGB'))

	display(imA, pointsA)
	display(imB, pointsB)







# print(bestBuddies(FA[-1], FB[-1]))



