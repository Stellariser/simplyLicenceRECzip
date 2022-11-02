import cv2
import os
import numpy as np
import json

# with open('test.json') as f:
#     datas = f.readlines()
#     for data in datas:
#         data = data.strip()
#         annotations = json.loads(data)['annotation']
#         annotation = annotations[0]
#         points = annotation['points']
#         imageWidth = annotation['imageWidth']
#         imageHeight = annotation['imageHeight']
#         for point in points:
#             point[0] *= imageWidth
#             point[1] *= imageHeight
#             P.append([int(point[0]), int(point[1])])


# P = [[20, 10], [10, 27], [20, 44], [40, 44], [50, 27], [40, 10]]
# originalimg = cv2.imread('ch01_00000000000000401 (1).mp454.jpg')
# H, W, _ = originalimg.shape
# img = np.zeros((H, W, 3), np.uint8)
# area1 = np.array(P)
#
# cv2.fillPoly(img, [area1], (255, 255, 255))
# maskindex = np.where(img > 0)
# img[maskindex] = originalimg[maskindex]
# cv2.imshow("", img)
# cv2.imwrite('result.jpg', img)
# cv2.waitKey(0)




# def mouseclick():
#     Start_point = [x, y]
#     points.append(Start_point)
#     cv2.circle(img, tuple(Start_point), 1, (255, 255, 255), 0)
#     cv2.imshow("", img)


# if event == cv2.EVENT_RBUTTONDOWN and flag is False:
#     print('end')
#     points.append(Start_point)
#     # cv2.imshow("", img)

# elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
#     Cur_point = [x, y]
#     # print(points)
#     cv2.line(img, tuple(points[-1]), tuple(Cur_point), (255, 255, 255),2)
#     cv2.imshow("", img)
#     points.append(Cur_point)
# elif event == cv2.EVENT_LBUTTONUP:
#     Cur_point = Start_point
#     cv2.line(img, tuple(points[-1]), tuple(Cur_point), (255, 255, 255))
#     cv2.circle(img, tuple(Cur_point), 1, (255, 255, 255))
#     ret, image, mask, rect = cv2.floodFill(img, mask_img, (x, y), (255, 255, 255), cv2.FLOODFILL_FIXED_RANGE)
#     cv2.imwrite("maskImage.jpg", img)
#     print(np.shape(image))
#     segImg = np.zeros((h, w, 3), np.uint8)
#     src = cv2.bitwise_and(img, image)
#     cv2.imwrite("segImg.jpg", src)
#     cv2.waitKey(0)
#     img = cv2.imread('segImg.jpg')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)  # opencv里面画轮廓是根据白色像素来画的，所以反转一下。
#     ret, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
#     contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
#     cv2.drawContours(copyImg, contours, -1, (0, 0, 255), 3)
#     cv2.imshow('RoiImg', copyImg)  # 只显示零件外轮廓
#     cv2.waitKey(0)
#     cimg = np.zeros_like(img)
#     cimg[:, :, :] = 255
#     cv2.drawContours(cimg, contours, 1, color=(0, 0, 0), thickness=-1)
#     cv2.imshow('maskImg', cimg)  # 将零件区域像素值设为(0, 0, 0)
#     cv2.waitKey(0)
#     final = cv2.bitwise_or(copyImg, cimg)
#     cv2.imshow('finalImg', final)  # 执行或操作后生成想要的图片
#     cv2.waitKey(0)

points = []
flag = True

path = 'ch01_00000000000000401 (1).mp458.jpg'
img = cv2.imread(path)


def on_mouse(event, x, y, flags, param):
    global points, img, Cur_point, Start_point, flag
    # # copyImg = copy.deepcopy(img)
    # h, w = img.shape[:2]
    # mask_img = np.zeros([h + 2, w + 2], dtype=np.uint8)
    if event == cv2.EVENT_LBUTTONDOWN and flag:
        Start_point = [x, y]
        print('start')
        points.append(Start_point)
        cv2.circle(img, tuple(Start_point), 20, (255, 255, 255), -1)
        flag = False
    if event == cv2.EVENT_LBUTTONDOWN and flag is False:
        Cur_point = [x, y]
        print('darwing')
        points.append(Cur_point)
        cv2.line(img, tuple(points[-2]), tuple(Cur_point), (255, 255, 255), 4)
        cv2.circle(img, tuple(Cur_point), 10, (255, 255, 255), -1)

def setROI():

    cv2.namedWindow("amy", 0)
    cv2.resizeWindow("amy", 1600, 900)
    cv2.setMouseCallback("amy", on_mouse)
    while (1):
        cv2.imshow('amy', img)
        # cbf()
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
            #标完点后输入q退出！
    print(points)
    cv2.destroyAllWindows()
    with open('ROI.json','w') as fp:
        json.dump(points, fp)


if __name__ == "__main__":
    jsonfile = 'ROI.json'
    if not os.path.exists(jsonfile):
        with open(jsonfile, 'a') as f:
            f.close()
    if not os.path.getsize(jsonfile) == 0:
        with open(jsonfile, 'r') as f:
                points = json.load(f)
    else:
        setROI(path)
    #没有ROI.json的情况下调用setROI()
    originalimg = cv2.imread(path)
    H, W, _ = originalimg.shape
    img = np.zeros((H, W, 3), np.uint8)
    area1 = np.array(points)
    cv2.fillPoly(img, [area1], (255, 255, 255))
    maskindex = np.where(img > 0)
    img[maskindex] = originalimg[maskindex]

    cv2.namedWindow("result", 0)
    cv2.resizeWindow("result", 1600, 900)
    cv2.imshow("result", img)
    cv2.imwrite('result.jpg', img)
    cv2.waitKey(0)

