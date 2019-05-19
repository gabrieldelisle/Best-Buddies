from torchvision import transforms, models, datasets

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.nn import functional, Sequential, Linear, Softmax, CrossEntropyLoss

def load(im):
	I = Image.open(im).convert('RGB')
	transform = get_transform(224)
	I = transform(I).unsqueeze(0)
	return I


def get_transform(witdh):
    transform_list = []
    osize = [witdh, witdh]
    transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])]

    return transforms.Compose(transform_list)

def forward_pass(im, net, last):
	activation = im

	for index , f in enumerate(net.features) :
		if index<=last:
			activation = f.forward(activation)
		
	return activation

def train_model(model, dataloaders, criterion, optimizer, num_epochs):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)
	print(device)
	print(model.features[30].weight[0])
	model.train(True)

	for epoch in range(num_epochs):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(dataloaders['train'], 0):
			# get the inputs
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			k = 10
			running_loss += loss.item()
			if i % k == (k-1):    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / k))
				running_loss = 0.0


		#print accuracy on validation dataset
		correct = 0
		total = 0
		for data in dataloaders_dict["val"]:
			images, labels = data
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

		print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

	print(model.features[30].weight[0])
	torch.save(model, "fine-tuned")


if __name__ == "__main__" :
	# load the model
	print("Loading model...")
	VGG19 = models.vgg19(pretrained=True)

	# #new output
	classifier = list(VGG19.classifier.children())
	print(classifier)
	VGG19.classifier = Sequential(*classifier[:-1])
	VGG19.classifier.add_module('6', Linear(classifier[-1].in_features, 50))
	

	#no training for the first layers
	params_to_update = []
	for i,param in enumerate(VGG19.parameters()):
		if i <= 18:
			param.requires_grad = False
		else:
			param.requires_grad = True
			params_to_update.append(param)
			if "weight" in dir(param): 
				torch.nn.init.xavier_uniform(param.weight)
		print(i)	
	for i,param in enumerate(VGG19.classifier):
		param.requires_grad = True
		params_to_update.append(param)
		if "weight" in dir(param): 
			torch.nn.init.xavier_uniform(param.weight)
		print(i)		#print(param[0])

	print(VGG19)

	# Train and evaluate
	print("Loading data...")
	image_datasets = {x: datasets.ImageFolder('reparsed_img/'+x,  get_transform(224)) for x in ['train', 'val']}
	dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64) for x in ['train', 'val']}

	print("Training...")
	optimizer_ft = torch.optim.Adam(VGG19.parameters())
	criterion = CrossEntropyLoss()
	train_model(VGG19, dataloaders_dict, criterion, optimizer_ft, num_epochs=30)


	# for i,param in enumerate(VGG19.parameters()):
	# 	print(param[0])