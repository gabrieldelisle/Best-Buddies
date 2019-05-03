import torchvision.models as models
from torchvision import transforms

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

import sys
import json

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

def forward_pass(im, net):
	activations = [im]
	for f in net.features :
		activations.append(f.forward(activations[-1]))
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
	pad_list = tuple([neighbours_size // 2,neighbours_size // 2,neighbours_size // 2,neighbours_size // 2])
	pad_size = neighbours_size // 2
	# initialize matrix of correlations
	corr = torch.zeros((n*n,n*n))
	ones_filter = torch.ones((1,1,neighbours_size,neighbours_size))
	print(ones_filter.shape)
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
		for py in range(pad_size,n + pad_size):
			
			# find the patch of neighbouring pixels
			ca = CAnorm[:,:,neighbours(px, neighbours_size, n+pad_size+1),:][:,:,:,neighbours(py, neighbours_size, n+pad_size+1)]
			## compute correlation
			# here there is no need to flip the image
			# since the convolution is implemented as correlation
			R = functional.conv2d(CBnorm, ca.contiguous(), padding=0).data
			
			# if option is set, sum the correlations in the 
			# neighbouring region
			if sum_over_neigh:
				R = functional.pad(R, pad_list,  'reflect')
				R = functional.conv2d(R, ones_filter, padding=0).data
			
			# reshape to fit the result in the correlation matrix
			R = R.reshape(1, -1)
			R = R.squeeze()
			corr[(px-pad_size) * n + (py - pad_size),:] = R

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


def bestBuddies(FA, xA, yA, FB, xB, yB):
	n = FA.shape[2]

	CA, CB = normalize(FA, FB)
	corr = correlation(CA, CB, 3)
	# filter for keeping only significative activations
	normA = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (FA,FA)))
	normB = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (FB,FB)))
	HA = (normA - torch.min(normA)) / (torch.max(normA) - torch.min(normA))
	HB = (normB - torch.min(normB)) / (torch.max(normB) - torch.min(normB))
	gamma = 0#.05
	bbA = []
	bbB = []

	
	for p in range(corr.shape[0]):
		q = torch.argmax(corr[p,:]).item()
		if torch.argmax(corr[:,q]).item() == p:
			px, py = p//n, p%n
			qx, qy = q//n, q%n
			if HA[px, py] > gamma and HB[qx, qy] > gamma: 
				bbA.append((px + xA, py + yA))
				bbB.append((qx + xB, qy + yB))
	return bbA, bbB


radius_list=[4,4,6,6] #use radius_list[l-2] for l=5to2
l_layer=[5,10,19,28,37] #use l_layer[l-1] for l=5to1

def pyramid_search(FA, FB):
	R = [[FA[-1], 0, 0, FB[-1], 0, 0]]
	finalA = []
	finalB = []

	for l in range(4,-1,-1) :
		print(l)
		new_R=[]
		
		for regions in R:
			bbA, bbB = bestBuddies(*regions)
			print(bbA, bbB)
			if l==0:
				finalA += bbA
				finalB += bbB

			for k in range(len(bbA)):
				px, py = bbA[k]
				qx, qy = bbB[k]
				n = FA[l_layer[l-1]].shape[2]
				r = radius_list[l-2]+1

				new_R.append((
					FA[l_layer[l-1]][:,:,neighbours(2*px, r, n),:][:,:,:,neighbours(2*py, r, n)],
					2 * px - r//2,
					2 * py - r//2,
					FB[l_layer[l-1]][:,:,neighbours(2*qx, r, n),:][:,:,:,neighbours(2*qy, r, n)],
					2 * qx - r//2,
					2 * qy - r//2,
				))
		R = new_R
	return finalA, finalB


def display(im, points):
	r = 5
	n = im.shape[0]
	print(im.shape)
	im = im.view(3,224,224)
	for px,py in points:
		im[:,neighbours(px, r, n),:][:,:,neighbours(py, r, n)] = 0
	im = im.numpy()
	im = im.transpose([2,1,0])
	print(im.shape)
		
	plt.imshow(im)
	plt.show()
	plt.close()

if __name__ == "__main__" :

	options = load_options(sys.argv)

	VGG19 = models.vgg19(pretrained=True)

	imA = load("original_A.png")
	FA = forward_pass(imA, VGG19)

	imB = load("original_B.png")
	FB = forward_pass(imB, VGG19)


	pointsA, pointsB = pyramid_search(FA, FB)

	display(imA, pointsA)
	display(imB, pointsB)







# print(bestBuddies(FA[-1], FB[-1]))



