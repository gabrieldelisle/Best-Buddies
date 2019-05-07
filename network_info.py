import sys

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



def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben