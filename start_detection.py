import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from objectDetection import CropLayer


class colorDecetection:
    def __init__(self, img, img_gray, sample_num=8, kernal_size=(20, 20), threshold=160):
        self.sample_num = sample_num
        self.kernal = cv2.getStructuringElement(cv2.MORPH_RECT, kernal_size)
        self.img = img  # cv2.cvtColor(path, cv2.COLOR_BGR2RGB)
        self.imGray = img_gray  # cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        _, self.imBinary = cv2.threshold(self.imGray, threshold, 255, cv2.THRESH_BINARY_INV)
        self.contours = self.get_roi()  # 按照 x 轴顺序进行排列的 contours list
        self.masks = self.get_mask(self.contours)  # 按照从左到右，得到多个 mask
        self.bgr_hist_list = [self.cal_bgr_hist(self.img, m) for m in self.masks]
        print()

    def get_roi(self):
        img = self.imBinary.copy()
        if debug == 2:
            plt.imshow(img)
            plt.show()
        assert len(
            cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]) >= self.sample_num, 'Missing objects, please change the threshold!'

        img = cv2.erode(img, self.kernal, iterations=1)
        img = cv2.dilate(img, self.kernal, iterations=1)
        self.img_opening = img.copy()  # 开运算后的 img，去噪

        # 计算轮廓
        filtered_contours = []
        contours, hierarchy = cv2.findContours(self.img_opening.copy(), cv2.RETR_EXTERNAL,
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

    def get_mask(self, contours_list):
        masks = []
        for contour in contours_list:
            mask = np.zeros_like(self.img)
            masks.append(cv2.drawContours(mask, [contour], 0, (255, 255, 255), cv2.FILLED))
        img_opening = np.repeat(self.img_opening[:, :, np.newaxis], 3, axis=2)  # 需要和开运算后的 mask 进行取交集
        masks = [(np.array(mask, dtype=bool) * img_opening)[:, :, 0] for mask in masks]

        if debug:
            # show 经过筛选后的 mask
            mask_combine = np.zeros_like(masks[0])
            for m in masks:
                mask_combine = mask_combine + m
            plt.imshow(mask_combine)
            plt.show()

        return masks

    def sort_contours(self, contours):
        x_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_contours.append((x, contour))
        x_contours = sorted(x_contours)
        contours = [c[1] for c in x_contours]
        return contours

    def get_img_info(self):
        size = self.image.shape
        self.w = size[1]  # 宽度
        self.h = size[0]  # 高度


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
        if debug:
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


def read_and_split_img(img_path, stand_sample_num=7, test_sample_num=1, threshold=160):
    img_orginal = cv2.imread(img_path)
    imGray = cv2.cvtColor(img_orginal, cv2.COLOR_BGR2GRAY)
    size = img_orginal.shape
    w = size[1]  # 宽度
    h = size[0]  # 高度

    img_stand = img_orginal[h//4:h//2]
    img_gray_stand = imGray[h//4:h//2]

    img_test = img_orginal[h // 4 * 3:]
    img_gray_test = imGray[h // 4 * 3:]

    # ! [Register]
    cv2.dnn_registerLayer('Crop', CropLayer)
    # Load the model.
    net = cv2.dnn.readNet(cv2.samples.findFile(r'./pretrained_model/deploy.prototxt'),
                          cv2.samples.findFile(r'./pretrained_model/hed_pretrained_bsds.caffemodel'))

    stand_group = colorDecetection(img_stand, img_gray_stand, stand_sample_num, threshold=threshold)
    target_group = colorDecetection(img_test, img_gray_test, test_sample_num, threshold=threshold)

    result_index = colorDecetection.cal_sim(stand_group.bgr_hist_list, target_group.bgr_hist_list)
    print(result_index)

def read_and_split_and_detect_img(img_path, stand_sample_num=7, test_sample_num=1, threshold=160):
    img_orginal = cv2.imread(img_path)
    imGray = cv2.cvtColor(img_orginal, cv2.COLOR_BGR2GRAY)
    size = img_orginal.shape
    w = size[1]  # 宽度
    h = size[0]  # 高度

    img_stand = img_orginal[h//4:h//2]
    img_gray_stand = imGray[h//4:h//2]

    img_test = img_orginal[h // 4 * 3:]
    img_gray_test = imGray[h // 4 * 3:]

    stand_group = colorDecetection(img_stand, img_gray_stand, stand_sample_num, threshold=threshold)
    target_group = colorDecetection(img_test, img_gray_test, test_sample_num, threshold=threshold)

    result_index = colorDecetection.cal_sim(stand_group.bgr_hist_list, target_group.bgr_hist_list)
    print(result_index)

def main():
    parser = argparse.ArgumentParser(description='Detect objects and compare the color')
    parser.add_argument('img', type=str, help='The name of image')
    parser.add_argument('num1', type=int, help='The num of sample')
    parser.add_argument('num2', type=int, help='The num of target sample')
    parser.add_argument('--threshold', type=int, default=160,
                        help='The threshold of object detection (usually in 100-200)')
    parser.add_argument('--debug', type=int, default=0, help='The debug mode: 0 1 2')

    args = parser.parse_args()
    img_name = args.img
    num1 = args.num1
    num2 = args.num2
    debug = args.debug
    threshold = args.threshold

    img_path = 'data/' + img_name
    read_and_split_img(img_path, num1, num2, threshold=threshold)


if __name__ == '__main__':
    debug = 0
    # main()
    img_path = r'.\data\5EDCB0038E7C7315990997B80CD90ACF.png'
    read_and_split_img(img_path, 7, 7, threshold=160)

    # img_name = '0BE1B19822EB3F49A817EF993D670BB2.png'  # 半成功，有一些丢失
    # img_name = '5CAC56C7DABA6CD0213279A6213323FE.png'  # 暗色调 failed
    # img_name = '5EDCB0038E7C7315990997B80CD90ACF.png'  # 成功
    # img_name = '5F0E7ABFB3DD3C07A7FE83535DD93333.png'  # 成功
    # img_name = '6B8B7C98EE3522AB8036CD239980E62B.png'  # 反光 failed
    # img_name = '7BA24FF89E790157089F4A7536FF10B2.png'  # 成功
    # img_name = '7D7A43EF00E9CD6784032347011A6101.png'  # 成功
    # img_name = '56BF8DF8922FF549443E283E7A3DF92C.png'  # 成功
    # img_name = '81D56E286E6F1603084579CC1970959E.png'  # 成功
    # img_name = '90BB0EA89116246826D5431CDD9ECB14.png'  # 成功
    # img_name = '25677F600ADE66B3B599BAF86C63522F.png'  # 成功
    # img_name = '95875DC45B3D006FABF1F63066E2E280.png'  # 可以运行，但是结果不对，受到反光干扰
    # img_name = 'B1D77DAA0C1C63DBA67A7730B86B6A99.png' # 可以运行，但是有目标丢失，是因为透光导致
    # img_name = 'D102A54301657C84862B3A2CDD295A4B.png'  # 可以运行，但是有目标丢失
    # img_name = 'DC001EE63CDA633989A8E25961F19601.png'  # 可以运行，但是有部分目标丢失
    # img_name = 'EA8BA058FECAF9A463BF33E1AE2CCCF2.png'  # 成功






