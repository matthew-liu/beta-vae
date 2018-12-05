import csv
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import crop

'''

202,599 align & cropped face images of 178*218
40 binary attribute labels

In evaluation status,
  "0" -> training, "1" -> validation, "2" -> testing
  
'''

IMAGE_PATH = '../celebA/'


def split_dataset():

    train_im_ids = []
    test_im_ids = []

    with open('data/list_eval_partition.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # header
        for row in reader:
            im_id, category = row
            if category == '2':
                test_im_ids.append(im_id)
            else:
                train_im_ids.append(im_id)

    return train_im_ids, test_im_ids


def get_attributes():

    id_attr = {}
    with open('data/list_attr_celeba.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        attr = next(reader)[1:]
        attributes = {descrb.lower(): idx for idx, descrb in enumerate(attr)}

        for row in reader:
            idx = row[0]
            attr_arr = [int(i) for i in row[1:]]
            id_attr[idx] = attr_arr

    return attributes, id_attr


'''

Available Attributes:

['5_o_clock_shadow', 'arched_eyebrows', 'attractive', 'bags_under_eyes', 'bald',
 'bangs', 'big_lips', 'big_nose', 'black_hair', 'blond_hair', 'blurry', 'brown_hair',
 'bushy_eyebrows', 'chubby', 'double_chin', 'eyeglasses', 'goatee', 'gray_hair',
 'heavy_makeup', 'high_cheekbones', 'male', 'mouth_slightly_open', 'mustache', 'narrow_eyes',
 'no_beard', 'oval_face', 'pale_skin', 'pointy_nose', 'receding_hairline', 'rosy_cheeks',
 'sideburns', 'smiling', 'straight_hair', 'wavy_hair', 'wearing_earrings', 'wearing_hat',
 'wearing_lipstick', 'wearing_necklace', 'wearing_necktie', 'young']
 
'''


def get_attr(attr_map, id_attr_map, attr):

    attr_idx = attr_map[attr]
    im_ids = []
    for im_id in id_attr_map:
        if id_attr_map[im_id][attr_idx] == 1:
            im_ids.append(im_id)
    return im_ids


class ImageLoader(torch.utils.data.Dataset):

    def __init__(self, im_ids):
        self.transform = transforms.Compose([
                        transforms.Resize(64),
                        transforms.ToTensor(),
                    ])
        self.im_ids = im_ids

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):
        im_path = IMAGE_PATH + self.im_ids[idx]
        im = Image.open(im_path)
        im = crop(im, 30, 0, 178, 178)
        data = self.transform(im)

        return data

# import utils
# images = sorted(glob.glob(IMAGE_PATH + '*.jpg'))
#
# resize = transforms.Compose([
#                         transforms.Resize(64),
#                         transforms.ToTensor(),
#                     ])
#
# data = []
# for image in images[10000:10025]:
#     im = Image.open(image)
#     im = crop(im, 30, 0, 178, 178)
#     im_tensor = resize(im)
#     data.append(np.transpose(im_tensor, (1, 2, 0)))
#
# utils.show_images(data, columns=5, max_rows=10)