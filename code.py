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


def neighbours(p, size, n) :
	return [p + n * i + j 
		for i in range(-(size//2), size//2+1) 
		for j in range(-(size//2), size//2+1) 
		if p//n+i>=0 and p//n+i<n and p%n+j>=0 and p%n+j<n
	]

def correlation(CA, CB) :
	n = CA.shape[2]
	row = int(np.sqrt(n))

	corr = torch.zeros((n,n))
	for p in range(n):
		for q in range(n) :
			ca = CA[:,:,neighbours(p, 3, row)]
			cb = CB[:,:,neighbours(q, 3, row)]
			dot = torch.einsum("ijk,ijl->kl", (ca,cb))
			normA = torch.sqrt(torch.einsum("ijk,ijk->k", (ca,ca))).view(-1,1)
			normB = torch.sqrt(torch.einsum("ijk,ijk->k", (cb,cb))).view(1,-1)
	
			corr[p,q] = torch.sum(dot / normA / normB)
	return corr


def bestBuddies(FA, FB):
	FA = FA.view(FA.shape[:2]+(-1,))
	FB = FB.view(FB.shape[:2]+(-1,))
	CA, CB = normalize(FA, FB)
	corr = correlation(CA, CB)
	print(corr)

	# filter for keeping only significative activations
	normA = torch.sqrt(torch.einsum("ijk,ijk->k", (FA,FA)))
	normB = torch.sqrt(torch.einsum("ijk,ijk->k", (FB,FB)))
	HA = (normA - torch.min(normA)) / (torch.max(normA) - torch.min(normA))
	HB = (normB - torch.min(normB)) / (torch.max(normB) - torch.min(normB))
	gamma = 0.05

	bbA = []
	bbB = []
	for p in range(CA.shape[2]):
		q = torch.argmax(corr[p]).item()
		if torch.argmax(corr[q]).item() == p:
			print(HA[p], HB[q])
			if HA[p] > gamma and HB[q] > gamma: 
				bbA.append(p)
				bbB.append(q)
	return bbA, bbB


VGG19 = models.vgg19(pretrained=True)

imA = load("original_A.png")
FA = forward_pass(imA, VGG19)

imB = load("original_B.png")
FB = forward_pass(imB, VGG19)



print(bestBuddies(FA[-1], FB[-1]))



