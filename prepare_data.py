import os
from PIL import Image


def convert_and_write(img_file, txt_f, lbl):
    img_f = open(img_file, "rb")
    img_f.read(16)
    for j in range(784):  # get 784 pixel vals
        pixel = img_f.read(1)
        if len(pixel) == 0:
            pixel = "0"
        val = ord(pixel)
        txt_f.write(str(val) + "\t")
    txt_f.write(str(lbl) + "\n")
    print(img_file + " - Done")


def convert_and_write_1(img_file, txt_f, lbl):
    im = Image.open(img_file, 'r')
    pix_vals = list(im.getdata())
    print("No pixels found ", len(pix_vals))
    img_line = "\t".join([str(pix_val) for pix_val in pix_vals])
    txt_f.write(img_line + "\t" + lbl + "\n")
    print(img_file + " - Done")


def prepare_training_data():
    trainingDir = "/Users/raghavendra.c/PycharmProjects/digit-rec-mnist/data/trainingSet"
    if not os.path.exists(trainingDir):
        return
    list_label_dirs = os.listdir(trainingDir)
    txt_f = open(os.path.join("/Users/raghavendra.c/PycharmProjects/digit-rec-mnist/data", "pixels_1.txt"), "w")
    count = 1
    for label_dir in list_label_dirs:
        label_dir_path = os.path.join(trainingDir, label_dir)
        if os.path.isfile(label_dir_path):
            continue
        list_files = os.listdir(label_dir_path)
        for file in list_files:
            convert_and_write_1(os.path.join(trainingDir, label_dir, file), txt_f, label_dir)
            print("imgs processed: ", count)
            count = count+1
    txt_f.close()


def prepare_test_data():
    txt_f = open(os.path.join("/Users/raghavendra.c/PycharmProjects/digit-rec-mnist/data", "test-pixels.txt"), "w")

    test_dir = "/Users/raghavendra.c/PycharmProjects/digit-rec-mnist/data/testSet"

    test_images = os.listdir(test_dir)
    for file in test_images:
        convert_and_write_1(os.path.join(test_dir, file), txt_f, "-1")
    txt_f.close()


if __name__ == "__main__":
    prepare_training_data()
    # prepare_test_data()
