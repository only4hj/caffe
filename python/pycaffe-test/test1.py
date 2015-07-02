import time
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILE = '../../examples/images/cat.jpg'

mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')

mean = mean.resize(227, 227)

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

#caffe.set_mode_cpu()
caffe.set_mode_gpu()



input_image = caffe.io.load_image(IMAGE_FILE)
#plt.imshow(input_image)

start_time = time.time()

#prediction = net.predict([input_image], oversample=False)  # 281
prediction = net.predict([input_image], oversample=True)  # 282

end_time = time.time()

print 'prediction shape:', prediction[0].shape
print prediction[0]
plt.plot(prediction[0])
print 'predicted class:', prediction[0].argmax()
print '%.3fs' % (end_time - start_time)

#plt.show()
