import cv2
import numpy as np


def extractSkin(image):
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_threshold = np.array([50, 50, 50], dtype=np.uint8)
    upper_threshold = np.array([255,173,127], dtype=np.uint8)

    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 2)

    skin = cv2.bitwise_and(img, img, mask=skinMask)

    img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    HSV_mask = cv2.inRange(img_HSV, (10, 10, 107), (17, 255, 255))
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 5), np.uint8))
    HSV_result = cv2.bitwise_not(HSV_mask)

    img_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 75), (255, 180, 135))
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    #cv2.imshow("", YCrCb_result)

    global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    global_result = cv2.bitwise_not(global_mask)
    #cv2.imshow("", global_mask)

    #return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)
    return global_mask

camera = cv2.VideoCapture(2)

while True:
    try:
        (_,image) = camera.read()
        #image[image<255-40]+=10
        cv2.imshow("oo", image)
        #image=cv2.blur(image, (5,5))
        img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        HSV_mask = cv2.inRange(img_HSV, (0, 0, 77), (17, 255, 255))
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 4), np.uint8))
        HSV_result = cv2.bitwise_not(HSV_mask)
        #cv2.imshow("", HSV_result)

        img_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 75), (255, 180, 135))
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        #YCrCb_result = cv2.bitwise_not(YCrCb_mask)
        #cv2.imshow("", YCrCb_result)

        global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
        global_mask = cv2.medianBlur(global_mask, 3)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
        global_result = cv2.bitwise_not(global_mask)
        cv2.imshow("global", global_mask)
        #cv2.imshow('original', image)
        #image = extractSkin(image)
        #cv2.imshow('skin', image)
        #global_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        global_mask = cv2.resize(global_mask, None, fx=0.75, fy=0.75)
        image = cv2.resize(image, None, fx=0.75, fy=0.75)
        image_blur = cv2.GaussianBlur(global_mask, (15, 15), None)
        #cv2.imshow("s",image_blur)
        h,w = global_mask.shape
        image_color = np.zeros((h,w,3), dtype = np.uint8)
        _,image_thresh = cv2.threshold(global_mask, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow('threshold', image_thresh)

        contours, hierarchy = cv2.findContours(image_thresh, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)
        #cv2.drawContours(image_color, [cnt], 0, (255, 0, 0), 3)
        cv2.drawContours(image, [cnt], 0, (255, 0, 0), 3)
        hull = cv2.convexHull(cnt)
        cv2.drawContours(image_color, [hull], 0, (0, 0, 255), 3)
        cv2.drawContours(image, [hull], 0, (0, 0, 255), 3)
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)


        count = 0
        for i in range(len(defects)):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
            if angle <= 60:
                count += 1
                cv2.line(image_color, start, end, (255, 255, 0), 3)
        print('нашли ', count + 1, 'пальцев')
        cv2.imshow('contours', image_color)
    except:
        continue
    keypress = cv2.waitKey(1) & 0xFF

    if keypress == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
