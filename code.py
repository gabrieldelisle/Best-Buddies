import torchvision.models as models
from torchvision import transforms

import cv2
import numpy as np
import torch



def load(im):
    I = cv2.imread(im)
    I = I.astype(float)/255
    I = I.transpose([2, 0, 1])
    I = torch.from_numpy(I)
    I = I.float()
    I = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(I)
    I = I.unsqueeze(0)
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


def bestBuddies(FA, FB):
	n = FA.shape[2]

	CA, CB = normalize(FA, FB)
	corr = correlation(CA, CB, 3)
	# filter for keeping only significative activations
	normA = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (FA,FA)))
	normB = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (FB,FB)))
	HA = (normA - torch.min(normA)) / (torch.max(normA) - torch.min(normA))
	HB = (normB - torch.min(normB)) / (torch.max(normB) - torch.min(normB))
	gamma = 0.05

	bbA = []
	bbB = []

	
	for p in range(corr.shape[0]):
		q = torch.argmax(corr[p,:]).item()
		if torch.argmax(corr[:,q]).item() == p:
			px, py = p//n, p%n
			qx, qy = q//n, q%n
			if HA[px, py] > gamma and HB[qx, qy] > gamma: 
				bbA.append((px, py))
				bbB.append((qx, qy))
	return bbA, bbB


radius_list=[4,4,6,6] #use radius_list[l-2] for l=5to2
l_layer=[5,10,19,28,37] #use l_layer[l-1] for l=5to1

def next_layer(R,l):
	for regions in R:
		print(regions[0].shape)
		new_R=[]
		bbA, bbB = bestBuddies(*regions)
		print(bbA, bbB)
		for k in range(len(bbA)):
			new_R.append([])
			px, py = bbA[k]
			qx, qy = bbB[k]
			new_R[k].append(FA[l_layer[l-1]][:,:,neighbours(2*px, radius_list[l-2]+1, FA[l_layer[l-1]].shape[2]),:][:,:,:,neighbours(2*py, radius_list[l-2]+1, FA[l_layer[l-1]].shape[2])])
			new_R[k].append(FB[l_layer[l-1]][:,:,neighbours(2*qx, radius_list[l-2]+1, FB[l_layer[l-1]].shape[2]),:][:,:,:,neighbours(2*qy, radius_list[l-2]+1, FB[l_layer[l-1]].shape[2])])
	return new_R



VGG19 = models.vgg19(pretrained=True)

imA = load("original_A.png")
FA = forward_pass(imA, VGG19)

imB = load("original_B.png")
FB = forward_pass(imB, VGG19)


R = [[FA[-1], FB[-1]]]
for l in range(4,0,-1) :
	print(l)
	R = next_layer(R,l)







# print(bestBuddies(FA[-1], FB[-1]))



