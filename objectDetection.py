import cv2
import argparse

parser = argparse.ArgumentParser(
    description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
# parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera', default=r'C:\Users\Administrator\Desktop\edgeDetection\data\81D56E286E6F1603084579CC1970959E.png')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera', default=r'.\data\5EDCB0038E7C7315990997B80CD90ACF.png')
parser.add_argument('--prototxt', help='Path to deploy.prototxt', default=r'C:\Users\Administrator\Desktop\edgeDetection\deploy.prototxt')
parser.add_argument('--caffemodel', help='Path to hed_pretrained_bsds.caffemodel',
                    default=r'C:\Users\Administrator\Desktop\edgeDetection\hed_pretrained_bsds.caffemodel')
parser.add_argument('--width', help='Resize input image to a specific width', default=500, type=int)
parser.add_argument('--height', help='Resize input image to a specific height', default=500, type=int)
args = parser.parse_args()


# ! [CropLayenr]
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

        # self.ystart = (inputShape[2] - targetShape[2]) / 2
        # self.xstart = (inputShape[3] - targetShape[3]) / 2

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)

        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


if __name__ == '__main__':
    # ! [CropLayer]
    cv2.dnn_registerLayer('Crop', CropLayer)
    # ! [Register]
    cv2.dnn_registerLayer('Crop', CropLayer)
    # ! [Register]

    # Load the model.
    net = cv2.dnn.readNet(cv2.samples.findFile(args.prototxt), cv2.samples.findFile(args.caffemodel))

    kWinName = 'Holistically-Nested Edge Detection'
    cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
    cv2.namedWindow(kWinName, cv2.WINDOW_NORMAL)

    frame = cv2.imread(args.input)

    cv2.imshow('Input', frame)
    # cv.waitKey(0)

    inp = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(args.width, args.height),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
    net.setInput(inp)

    out = net.forward()
    out = out[0, 0]
    out = cv2.resize(out, (frame.shape[1], frame.shape[0]))
    cv2.imshow(kWinName, out)
    cv2.imwrite('result.png', out)
    cv2.waitKey(0)