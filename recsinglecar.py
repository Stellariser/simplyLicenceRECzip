from paddleocr import PaddleOCR, draw_ocr
import re
import detectout
from detectout import *

all = []

pattern_str = "([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]" \
              "{1}(([A-HJ-Z]{1}[A-HJ-NP-Z0-9]{5})|([A-HJ-Z]{1}(([DF]{1}[A-HJ-NP-Z0-9]{1}[0-9]{4})|([0-9]{5}[DF]" \
              "{1})))|([A-HJ-Z]{1}[A-D0-9]{1}[0-9]{3}警)))|([0-9]{6}使)|((([沪粤川云桂鄂陕蒙藏黑辽渝]{1}A)|鲁B|闽D|蒙E|蒙H)" \
              "[0-9]{4}领)|(WJ[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼·•]{1}[0-9]{4}[TDSHBXJ0-9]{1})" \
              "|([VKHBSLJNGCE]{1}[A-DJ-PR-TVY]{1}[0-9]{5})"
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
def is_car_number(pattern, string):
    if re.findall(pattern, string):
        return True
    else:
        return False
def resultprocess(result):
    a = []
    car = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            for i in line:
                if len(i) == 2:
                    a.append(i[0])
    for k in a:
        k = k.replace('·', '')
        if is_car_number(pattern_str, k):
            car.append(k)
    return car


def RecSingleCar(img):
    result = ocr.ocr(img, cls=True)
    a = resultprocess(result)
    return a




# 显示结果
# from PIL import Image
# result = result[0]
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')