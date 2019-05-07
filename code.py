import torchvision.models as models
from torchvision import transforms
from network_info import *
from sklearn.cluster import KMeans

import numpy as np
from numpy.linalg import pinv
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

def normalize_style(imA, imB):
	mu_A = np.mean(imA, axis=(0,1)).reshape(1,1,-1)
	mu_B = np.mean(imB, axis=(0,1)).reshape(1,1,-1)
	mu = (mu_A + mu_B)/2
	sigma_A = np.std(imA, axis=(0,1)).reshape(1,1,-1)
	sigma_B = np.std(imB, axis=(0,1)).reshape(1,1,-1)
	sigma = (sigma_A + sigma_B)/2
	return (imA - mu_A) / sigma_A * sigma + mu, (imB - mu_B) / sigma_B * sigma + mu 


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


def bestBuddies(FA,FB,FA_e, xA, yA, FB_e, xB, yB, radius, norm=True):
	px1, py1, px2, py2 = FA_e
	qx1, qy1, qx2, qy2 = FB_e
	
	# extract regions
	fA = FA[:,:,px1:px2, :][:,:,:, py1:py2]
	fB = FB[:,:,qx1:qx2, :][:,:,:, qy1:qy2]

	na = fA.shape[2]
	ma = fA.shape[3]

	if norm:
		CA, CB = normalize(fA, fB)
	else:
		CA, CB = (fA, fB)

	nb = fB.shape[2]
	mb = fB.shape[3]

	#CA, CB = normalize(fA, FB)
	corr = correlation_conv(CA, CB, radius)
	# filter for keeping only significative activations
	normA = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (fA,fA)))
	normB = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (fB,fB)))
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

	A_extremes = (0,0, FA_list[-1].shape[2], FA_list[-1].shape[3])
	B_extremes = (0,0, FB_list[-1].shape[2], FB_list[-1].shape[3])
	R = [[A_extremes, 0, 0, B_extremes , 0, 0]]
	finalA = []
	finalB = []

	for l in range(L-1,0,-1) :
		print("\n### Searching at level :", l)
		new_R=[]
		
		# get the activations at
		# the leyer before
		FA = FA_list[l-1]
		FB = FB_list[l-1]


		tot_regions = len(R)
		# for every region at the current level
		# find the best buddies
		for idx, regions in enumerate(R):
			bbA, bbB = bestBuddies(FA_list[l], FB_list[l], *regions, radius=options["patch_radius"][l-1], norm=(l!=L-1))
			#print(bbA, bbB)
			progress(idx + 1, tot_regions)
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
					R1_Nx = neighbours(2*px, r, na)
					R1_Ny = neighbours(2*py, r, ma)


					R2_Nx = neighbours(2*qx, r, nb)
					R2_Ny = neighbours(2*qy, r, mb)


					# if the regions are too small ignore them
					if len(R1_Nx) < 2 or len(R2_Nx) < 2 or len(R1_Ny) < 2 or len(R2_Ny) < 2 :
						continue
					else:
						R1_extremes = (R1_Nx[0],R1_Ny[0], R1_Nx[-1] + 1, R1_Ny[-1] + 1) 
						R2_extremes = (R2_Nx[0],R2_Ny[0], R2_Nx[-1] + 1, R2_Ny[-1] + 1) 
					# print("for ",(px,py), " , ",(qx,qy))
					# print(R1.shape)
					# print(R2.shape)
					Xa = max(0,2 * px - (r//2))
					Ya = max(0,2 * py - (r//2))
					Xb = max(0,2 * qx - (r//2))
					Yb = max(0,2 * qy - (r//2))
					new_R.append(( R1_extremes, Xa, Ya,
						R2_extremes,Xb , Yb))
				
					#new_R.append(( R1, 2 * px, 2 * py,
					#	R2, 2 * qx, 2 * qy))
				R = new_R

	print("\n\nNumber of BB found :", len(finalA))
	return np.array([[u for u in v] for v in finalA]), np.array([[u for u in v] for v in finalB])


def display(im, points):
	centers = KMeans(n_clusters=min(options["clusters"], len(points)), random_state=0).fit(points).cluster_centers_
	r = 3
	n = im.shape[0]
	for px,py in centers:
		for u in neighbours(int(px), r, n) :
			for v in neighbours(int(py), r, n) :
				im[round(u),round(v)] = np.array([255,0,0])

	return im

def merge(imA, pointsA, imB, pointsB):
	#merge A on B
	X = np.array([[u for u in v]+[1] for v in pointsA])


	print(np.sum((pointsA - pointsB)**2))
	W = pinv(X.T.dot(X)).dot(X.T).dot(pointsB)

	print(np.sum((X.dot(W) - pointsB)**2))
	print(W)

	def func(i,j):
		x = np.array([[i,j,1]])
		return x.dot(W)

	for i in range(imB.shape[0]):
		for j in range(imB.shape[1]):
			u,v = func(i,j)[0]
			u = int(round(u))
			v = int(round(v))
			if u>=0 and u<imB.shape[0] and v>=0 and v<imB.shape[1]:
				imA[i,j] = (imA[i,j] + imB[u,v])/2

	plt.imshow(imA)
	plt.show()



if __name__ == "__main__" :

	# assign the global variable option with the
	# options contained in the option file
	options = load_options(sys.argv)


	# load the model
	VGG19 = models.vgg19(pretrained=True)

	# print some informations about the network

	nameA = "original_A.png"
	nameB = "original_B.png"

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

	# displays the centers on the images
	plt.subplot(1, 2, 1)
	plt.imshow(display(imA, pointsA))
	plt.subplot(1, 2, 2)
	plt.imshow(display(imB, pointsB)) 
	plt.show()
	plt.close()

	imA = np.array(Image.open(nameA).convert('RGB'))
	imB = np.array(Image.open(nameB).convert('RGB'))
	imA = imA.astype(float)/255
	imB = imB.astype(float)/255
	#imA, imB = normalize_style(imA, imB)

	merge(imA, pointsA, imB, pointsB)







# print(bestBuddies(FA[-1], FB[-1]))



