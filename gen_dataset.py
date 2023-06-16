import os
import pickle
import cv2
import numpy as np

perturbed_image_path = "./results/imagenet/resnet50/gn/"
perturbed_dataset_path = os.path.join(".", 'perturbed_datasets/imagenet/resnet50/gn/numpy')


def get_class(fn):
    classid = int(fn.split("_")[0][1:])
    return classid


files_ext = '.npy'
levels = {'original': 'x' + files_ext,
          0: 'x_adv' + files_ext,
          1: '1_adv' + files_ext,
          2: '2_adv' + files_ext,
          3: '3_adv' + files_ext,
          4: '4_adv' + files_ext,
          5: '5_adv' + files_ext,
          7: '7_adv' + files_ext,
          8: '8_adv' + files_ext,
          9: '9_adv' + files_ext,
          10: '10_adv' + files_ext,
          12: '12_adv' + files_ext,
          16: '16_adv' + files_ext,
          20: '20_adv' + files_ext,
          30: '30_adv' + files_ext,
          40: '40_adv' + files_ext}

for level, ext in levels.items():
    print('Noise level {} is running...'.format(level))
    dataset = []
    # Iterate over the sample folders
    for folder_name in sorted(os.listdir(perturbed_image_path)):
        folder_path = os.path.join(perturbed_image_path, folder_name)

        # Create a list to store the images
        images = []

        # Iterate over the images in the sample folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # Assuming the images are in a supported format (e.g., PNG, JPEG)
            # You can modify this part if you have different image formats
            if file_name.endswith(ext):
                image_id = os.path.basename(folder_path).split('_')[-1]
                classid = get_class(file_name)
                if files_ext in ['.jpg', '.png', '.jpeg']:
                    im = cv2.imread(file_path)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                else:
                    im = np.load(file_path)
                dataset.append([im, classid, image_id])

    if 0 != len(dataset):
        pickle_path = os.path.join(perturbed_dataset_path, 'images_{}.pickle'.format(level))
        if not os.path.exists(os.path.dirname(pickle_path)):
            os.makedirs(os.path.dirname(pickle_path))

        with open(pickle_path, 'wb') as pickle_file:
            pickle.dump(dataset, pickle_file)

        print("\tLevel {} is Done with {} samples, saved @ {}".format(level, len(dataset), pickle_path))
    else:
        print("\tLevel {} is Done with {} samples".format(level, len(dataset)))
