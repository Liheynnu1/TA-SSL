import os.path
import tifffile as tiff
import cv2
import numpy as np
import matplotlib.pyplot as plt


def Splicing_result(args, out_data, labels):
    """
    拼接成完整的结果图
    """
    h, w = args.dataset_shape[0], args.dataset_shape[1]
    piece = args.image_size
    out_data = out_data.reshape(h, w)
    # out_data = np.transpose(out_data, (1, 0))
    # result = np.zeros((w, h), np.uint8)
    result = out_data
    # for i, img in enumerate(out_data):
    #     index = i
    #     xmin = int(index % h)
    #     ymin = int(index // h)
    #     # print(xmin,ymin)
    #     result[ymin, xmin] = img
    plt.imsave(os.path.join(args.out_path, f'{args.info}.png'), result, cmap='gray')
    # tiff.imwrite(os.path.join(args.out_path, f'{args.info}.tif'), result)
    plt.imsave(os.path.join(args.out_path, f'{args.dataset}_label.png'), labels, cmap='gray')
    tiff.imwrite(os.path.join(args.out_path, f'{args.dataset}_label.tif'), labels)
    # cv2.imwrite(os.path.join(args.out_path, 'result.tif'), result)
    # labels = labels.reshape(h, w)
    # result[labels == 0] = 0

    # tiff.imwrite(os.path.join(args.out_path, f'{args.dataset}_result_IO.tif'), result)
    # cv2.imwrite(os.path.join(args.out_path, f'{args.dataset}_result_CV2.tif'), result)
    return result

def Splicing_result_Salinas(args, out_data, labels, clip_size=224):
    """
    拼接成完整的结果图
    """
    out_data = out_data.reshape(out_data.size//(224*224), 224, 224)
    result = np.zeros((3*clip_size, 1*clip_size), np.uint8)

    for i, img in enumerate(out_data):
        index = i
        ymin = index % 1 * clip_size
        xmin = int(index // 1 * clip_size)
        # print(xmin,ymin)
        result[xmin:xmin+clip_size, ymin:ymin+clip_size] = img
    plt.imsave(os.path.join(args.out_path, 'result.png'), result[:512, :210], cmap='gray')
    tiff.imwrite(os.path.join(args.out_path, 'result.tif'), result[:512, :210])
    # cv2.imwrite(os.path.join(args.out_path, 'result.tif'), result)
    labels = labels.reshape(3*clip_size, 1*clip_size)
    result[labels == 0] = 0

    tiff.imwrite(os.path.join(args.out_path, 'result_IO.tif'), result[:512, :210])
    # cv2.imwrite(os.path.join(args.out_path, 'result_CV2.tif'), result)
    return result

def Splicing_result_paviaU(args, out_data, labels, clip_size=224):
    """
    拼接成完整的结果图
    """
    out_data = out_data.reshape(out_data.size//(224*224), 224, 224)
    result = np.zeros((3*clip_size, 2*clip_size), np.uint8)

    for i, img in enumerate(out_data):
        index = i
        ymin = index % 2 * clip_size
        xmin = int(index // 2 * clip_size)
        # print(xmin,ymin)
        result[xmin:xmin+clip_size, ymin:ymin+clip_size] = img
    plt.imsave(os.path.join(args.out_path, 'result.png'), result[:610, :340], cmap='gray')
    tiff.imwrite(os.path.join(args.out_path, 'result.tif'), result[:610, :340])
    labels = labels.reshape(3*clip_size, 2*clip_size)
    result[labels == 0] = 0

    tiff.imwrite(os.path.join(args.out_path, 'result_IO.tif'), result[:610, :340])
    # cv2.imwrite(os.path.join(args.out_path, 'result_CV2.tif'), result)
    return result

if __name__ == '__main__':
    """
    将数据中的背景类替换掉
    """
    # label_paths = r'C:\Users\Admin\Desktop\GW-main\SVM\data\label\labeal.png'
    # result_paths = r'C:\Users\Admin\Desktop\GW-main\SVM\out\result.png'
    # result = np.array(cv2.imread(result_paths, cv2.IMREAD_GRAYSCALE))
    # labels = np.array(cv2.imread(label_paths, cv2.IMREAD_GRAYSCALE))
    # labels = labels[:1188, :4752]
    # result[labels == 0] = 0
    #
    # cv2.imwrite('result_ch.tif', result)
    # tiff.imwrite('result_IO.tif', result)
    # 灰度转彩色
    import os
    import cv2


    def apply_colormap(input_path, output_path, colormap=cv2.COLORMAP_VIRIDIS):
        # 读取灰度图
        grayscale_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # 应用调色板
        color_image = cv2.applyColorMap(grayscale_image, colormap)
        color_image[grayscale_image==0, 0] = 0
        # 保存彩色图
        cv2.imwrite(output_path, color_image)


    # 处理整个文件夹中的所有PNG图像
    input_folder = "D:\gw\GW-main\SVM\out\PaviaU"
    output_folder = "D:\gw\GW-main\SVM\out\PaviaU\colours"

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 使用默认的JET调色板，你可以根据需要选择其他调色板
            apply_colormap(input_path, output_path, colormap=cv2.COLORMAP_JET)

