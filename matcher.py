import cv2
import numpy as np
import scipy
import scipy.spatial
from imageio import imread
import pickle
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import operator

# Feature extractor
def extract_features(image_path, vector_size=32):
    image = imread(image_path, pilmode="RGB")
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them.
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print ('Error: ', e)
        return None
    return dsc


def batch_extractor(images_path, pickled_db_path="C:\\Users\\Administrator\\Desktop\\flask1\\static\\features2.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    result = {}
    for f in files:
        print ('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)
    # saving all our feature vectors in pickled file
    with open(pickled_db_path, 'wb') as fp:
        pickle.dump(result, fp)

class Matcher(object):
    def __init__(self, pickled_db_path="C:\\Users\\Administrator\\Desktop\\flask1\\static\\features2.pck"):
        with open(pickled_db_path,'rb') as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

    def match(self, image_path, topn=20):
        features = extract_features(image_path)
        # print(features)
        img_distances = self.cos_cdist(features)
        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()

        return nearest_img_paths, img_distances[nearest_ids].tolist()


def show_img(path):
    img = imread(path, pilmode="RGB")
    plt.imshow(img)
    plt.show()


def sort_key(old_dict, reverse=True):
    keys = sorted(old_dict.keys(),reverse=reverse)
    new_dict = OrderedDict()
    for key in keys:
        new_dict[key] = old_dict[key]
    return new_dict


def run(input,model,basedir):
    # model location
    ma = Matcher(model)
    print("input url: ", input)
    names, match = ma.match(input, topn=20)
    res = dict()
    for i in range(19):
        # we got cosine distance, less cosine distance between vectors
        # more they similar, thus we subtruct it from 1 to get match value
        res[str(round((1 - match[i]),4))] =  basedir + names[i][37::]
        # show_img(res[(1 - match[i])])
    # res = sorted(res)
    sorted_res = sort_key(res)
    # sorted_res = dict(sorted(res.items(), key=operator.itemgetter(1), reverse=True))
    return sorted_res

