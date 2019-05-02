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

def correlation(CA, CB) :
	n = CA.shape[2]
	print(CA.shape)

	corr = torch.zeros((n*n,n*n))
	for px in range(n):
		for py in range(n):
			for qx in range(n):
				for qy in range(n):
					print(neighbours(px, 3, n))
					ca = CA[:,:,neighbours(px, 3, n),:][:,:,:,neighbours(py, 3, n)]
					cb = CB[:,:,neighbours(qx, 3, n),:][:,:,:,neighbours(qy, 3, n)]
					print(ca.shape, cb.shape)
					dot = torch.einsum("ijkl,ijuv->kluv", (ca,cb))
					normA = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (ca,ca)))
					normB = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (cb,cb)))
					normA = normA.view(*normA.shape, 1, 1)
					normB = normB.view(1, 1, *normB.shape)
					print(px, py, qx, dot / normA / normB)
					corr[px * n + py, qx * n + qy] = torch.sum(dot / normA / normB)
	return corr


def bestBuddies(FA, FB):
	CA, CB = normalize(FA, FB)
	corr = correlation(CA, CB)
	print(corr)

	# filter for keeping only significative activations
	normA = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (FA,FA)))
	normB = torch.sqrt(torch.einsum("ijkl,ijkl->kl", (FB,FB)))
	HA = (normA - torch.min(normA)) / (torch.max(normA) - torch.min(normA))
	HB = (normB - torch.min(normB)) / (torch.max(normB) - torch.min(normB))
	gamma = 0.05

	bbA = []
	bbB = []

	n = corr.shape[0]
	for p in range(CA.shape[2]):
		q = torch.argmax(corr[p,:]).item()
		if torch.argmax(corr[:,q]).item() == p:
			if HA[p] > gamma and HB[q] > gamma: 
				bbA.append((p//n, p%n))
				bbB.append((q//n, q%n))
	return bbA, bbB


VGG19 = models.vgg19(pretrained=True)

imA = load("original_A.png")
FA = forward_pass(imA, VGG19)

imB = load("original_A.png")
FB = forward_pass(imB, VGG19)



print(bestBuddies(FA[-1], FB[-1]))



