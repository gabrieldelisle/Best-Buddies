

def intermediate_shapes(F):
	for l,f in enumerate(F):
		print("at layer ", l, " the size is ", f.shape[2:])


def net_info(net, im):

	print("Starting from image of size ", im.shape[2:])
	last_activation = im

	for l , f in enumerate(net.features) :
		last_activation = f.forward(last_activation)
		print("at layer ", l, " the module is ", f.__class__.__name__, \
						 "\toutput --> ", *last_activation.shape[2:])
