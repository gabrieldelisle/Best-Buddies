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
	print(p, size//2)
	return [p + n * i + j 
		for i in range(-(size//2), size//2+1) 
		for j in range(-(size//2), size//2+1) 
		if p//n+i>=0 and p//n+i<n and p%n+j>=0 and p%n+j<n
	]

def correlation(CA, CB) :
	p = 0
	q = 0
	shape = CA.shape
	vector_shape = shape[:2]+(-1,)
	CA = CA.view(vector_shape)
	CB = CB.view(vector_shape)
	n = CA.shape[2]
	side = int(np.sqrt(n))

	corr = torch.zeros((n,n))
	for p in range(n):
		for q in range(n) :
			
			
			print(neighbours(q, 3, side))
			ca = CA[:,:,neighbours(p, 3, side)]
			cb = CB[:,:,neighbours(q, 3, side)]
			dot = torch.einsum("ijk,ijl->kl", (ca,cb))
			normA = torch.sqrt(torch.einsum("ijk,ijk->k", (ca,ca))).view(-1,1)
			normB = torch.sqrt(torch.einsum("ijk,ijk->k", (cb,cb))).view(1,-1)
	
			corr[p,q] = torch.sum(dot / normA / normB)
	return corr


VGG19 = models.vgg19(pretrained=True)

imA = load("original_A.png")
FA = forward_pass(imA, VGG19)

imB = load("original_B.png")
FB = forward_pass(imB, VGG19)

CA, CB = normalize(FA[-1], FB[-1])

print(CA, CB)

print(correlation(CA, CB))



