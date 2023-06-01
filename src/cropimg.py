import cv2
import os

patchsize = 64


def crop(filename):

    name_single = filename.split('/')[-1].split('.')[0]
    if name_single == '':
        return
    resultpath = '/data1/zhn/macdata/all_data/deepsvdd/testsuidao/'
    img = cv2.imread(filename)
    halflength = (int)(patchsize / 2)
    for i in range(0, img.shape[0], 16):

        for j in range(0, img.shape[1], 16):
            if i - halflength < 0 or j - halflength < 0 or i + halflength >= img.shape[0] or j + halflength >= img.shape[1]:
                continue
            crop = img[i - halflength: i + halflength,
                       j - halflength: j + halflength]
            name = name_single + str(i) + '_' + str(j) + '.png'
            cv2.imwrite(resultpath + name, crop)


if __name__ == '__main__':
    # filename = "/data1/zhn/macdata/all_data/deepsvdd/moni-0001.png"
    path = "/data1/zhn/macdata/all_data/deepsvdd/suidaolumian/"
    listimg = os.listdir(path)
    for name in listimg:
        print(name)
        filename = os.path.join(path, name)
        crop(filename)
