import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

debug = 2

class colorDecetection:
    def __init__(self, img, net, sample_num, kernel_size=(10, 10)):
        self.sample_num = sample_num
        self.img = img

        result = self.init_model(img, net)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        self.contours = self.get_roi(result)  # 按照 x 轴顺序进行排列的 contours list
        self.masks = self.get_mask(self.contours)  # 按照从左到右，得到多个 mask
        self.bgr_hist_list = [self.cal_bgr_hist(self.img, m) for m in self.masks]
        print()

    def init_model(self, img: np.ndarray, net):

        input = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(500, 500),
                                      mean=(104.00698793, 116.66876762, 122.67891434),
                                      swapRB=False, crop=False)

        net.setInput(input)
        out = net.forward()[0, 0]
        out_r = cv2.resize(out, (img.shape[1], img.shape[0]))
        return out_r

    def get_roi(self, img):
        img_bool = ((img > 0.1) * 1).astype('uint8')
        img_fill = self.fill_hole(img_bool)

        if debug == 2:
            # show_array(img_bool*255)
            show_array(img_fill)

        # 计算轮廓
        filtered_contours = []
        contours, hierarchy = cv2.findContours(img_fill.copy(), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        # 遍历轮廓
        for (i, c) in enumerate(contours):
            # 计算矩形
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
            if ar < 1:
                filtered_contours.append(c)

        # 选出面积 top 的 mask
        top_area_contours = sorted(filtered_contours, key=lambda x: cv2.contourArea(x), reverse=True)[:self.sample_num]

        # 将符合的轮廓从左到右排序
        results = self.sort_contours(top_area_contours)
        assert len(results) >= self.sample_num, 'Missing objects after selection!'
        return results  # 筛选后的 contours

    def fill_hole(self, mask):
        """
        根据所有的 contours 封闭 contour
        :param mask:
        :return:
        """
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        len_contour = len(contours)
        contour_list = []

        for i in range(len_contour):
            drawing = np.zeros_like(mask, np.uint8)  # create a black image
            img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
            contour_list.append(img_contour)

        out = sum(contour_list)
        return out

    def get_mask(self, contours_list):
        masks = []
        for contour in contours_list:
            mask = np.zeros_like(self.img)
            masks.append(cv2.drawContours(mask, [contour], 0, (1, 1, 1), cv2.FILLED))
        masks = [mask[:, :, 0] for mask in masks]

        if debug == 1:
            # show 经过筛选后的 mask
            mask_combine = np.zeros_like(masks[0])
            for m in masks:
                mask_combine = mask_combine + m
            final_mask = Image.fromarray(mask_combine * 255)
            final_mask.show()

        return masks

    def sort_contours(self, contours):
        x_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_contours.append((x, contour))
        x_contours = sorted(x_contours)
        contours = [c[1] for c in x_contours]
        return contours

    @staticmethod
    def cal_sim(stand_hist: list[list[np.ndarray]], target_hist: list[list[np.ndarray]]):
        """
        :param stand_hist: 对照组的 hist list，其长度为对照组中样本数量
        :param target_hist: 检测样本的 hist，长度为测试组中样本的数量
        :return: 返回最小值的索引，样本 0 返回 1
        """
        result = []
        for rgb_hist_list_target in target_hist:
            similarity_list = []
            for rgb_hist_list_stand in stand_hist:
                sim = colorDecetection.cal_his_similarity(rgb_hist_list_stand, rgb_hist_list_target)
                similarity_list.append(sim)
            result.append(similarity_list.index(min(similarity_list)) + 1)
        return result

    @staticmethod
    def cal_bgr_hist(img, mask=None):
        """
        :param img: cv2.imread('xxx.jpg') 得到的 bgr 图像
        :param mask: 是否有 mask, 通道数需要与 img 通道数相等
        :return: bgr 图像的 hist list，按照 b g r 排列
        """
        color = ('b', 'g', 'r')
        bgr_list = []
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], mask, [256], [0, 256])
            bgr_list.append(histr)
            if debug == 3:
                plt.plot(histr, color=col)
                plt.xlim([0, 256])
                plt.show()
        return bgr_list

    @staticmethod
    def cal_his_similarity(hist1: list, hist2: list):
        """
        :param hist1: img1 的 rgb hist list
        :param hist2: img2 的 ....
        :return: 三通道巴氏距离之和求均值，越小相似度越高
        """
        bhattach_result = 0
        for i, c in enumerate(['b', 'g', 'r']):
            h1, h2 = hist1[i], hist2[i]
            BHATTACH_score = cv2.compareHist(h1, h2, method=cv2.HISTCMP_BHATTACHARYYA)
            bhattach_result += BHATTACH_score
        return bhattach_result / 3


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)

        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


def show_array(img: np.ndarray):
    new_im = Image.fromarray(img)
    new_im.show()


def main(img_path, sample_num, target_num):
    img_original = cv2.imread(img_path)
    size = img_original.shape

    # split the original img to two parts
    h = size[0]
    img_stand = img_original[:h // 2]
    img_test = img_original[h // 2:]

    # Load the model.
    cv2.dnn_registerLayer('Crop', CropLayer)
    net = cv2.dnn.readNet(cv2.samples.findFile(r'./pretrained_model/deploy.prototxt'),
                          cv2.samples.findFile(r'pretrained_model/hed_pretrained_bsds.caffemodel'))

    # calculate the histogram of the two parts
    stand_group = colorDecetection(img_stand, net, sample_num)
    target_group = colorDecetection(img_test, net, target_num)

    # 计算颜色相似度
    result_index = colorDecetection.cal_sim(stand_group.bgr_hist_list, target_group.bgr_hist_list)

    print(result_index)


if __name__ == '__main__':
    # img_name = '0BE1B19822EB3F49A817EF993D670BB2.png'  #
    # img_name = '5CAC56C7DABA6CD0213279A6213323FE.png'  #
    # img_name = '5EDCB0038E7C7315990997B80CD90ACF.png'  # 成功
    # img_name = '5F0E7ABFB3DD3C07A7FE83535DD93333.png'  # 成功
    # img_name = '6B8B7C98EE3522AB8036CD239980E62B.png'  # 反光
    # img_name = '7BA24FF89E790157089F4A7536FF10B2.png'  # 成功
    # img_name = '7D7A43EF00E9CD6784032347011A6101.png'  # 成功
    # img_name = '56BF8DF8922FF549443E283E7A3DF92C.png'  # 成功
    # img_name = '81D56E286E6F1603084579CC1970959E.png'  # 成功
    # img_name = '90BB0EA89116246826D5431CDD9ECB14.png'  # 成功
    # img_name = '25677F600ADE66B3B599BAF86C63522F.png'  # 成功
    # img_name = '95875DC45B3D006FABF1F63066E2E280.png'  #
    # img_name = 'B1D77DAA0C1C63DBA67A7730B86B6A99.png'  #
    # img_name = 'D102A54301657C84862B3A2CDD295A4B.png'
    # img_name = 'DC001EE63CDA633989A8E25961F19601.png'
    # img_name = 'EA8BA058FECAF9A463BF33E1AE2CCCF2.png'
    # img_name = 'B1D77DAA0C1C63DBA67A7730B86B6A99.png'
    img_name = '848A3E6B82EDA42451DBF4AE5A60A000.png'
    img_name = '1111_20230402105500.jpg'

    img_path = './data/' + img_name
    main(img_path, sample_num=7, target_num=7)
