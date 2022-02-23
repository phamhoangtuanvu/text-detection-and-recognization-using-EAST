from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import pytesseract

#cap = cv2.VideoCapture(0)

net = cv2.dnn.readNet("frozen_east_text_detection.pb")


def text_detector(image):
	#hasFrame, image = cap.read()
	orig = image
	(H, W) = image.shape[:2]

	(newW, newH) = (640, 320)

	#xác định tỷ lệ của kích thước hình ảnh ban đầu với kích thước hình ảnh mới
	rW = W / float(newW)
	rH = H / float(newH)

	#thay đổi kích thước hình ảnh và lấy kích thước hình ảnh mới
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	#xác định 2 layer EAST đầu ra
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid", #xác suất đầu ra
		"feature_fusion/concat_3"] #lấy tọa độ bounding box của text

	#tạo một đốm màu từ image và sau đó thực hiện chuyển tiếp mô hình để có được hai tập hợp lớp đầu ra
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)

	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):

		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < 0.5:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	boxes = non_max_suppression(np.array(rects), probs=confidences)
	results = []
	
	for (startX, startY, endX, endY) in boxes:

		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		# draw the bounding box on the image
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)

		# trích xuất text ROI
		roi = orig[startY:endY, startX:endX]

		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)     #orig vẫn đang ở BGR

		thresh = 255 - cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

		thresh = cv2.GaussianBlur(thresh, (3,3), 0)

		# cv2.imshow("Text Detection", thresh)
		# cv2.waitKey(0)
		# set config for Tesseract
		config = "-l eng --oem 1 --psm 7"
		pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

		text = pytesseract.image_to_string(thresh, config=config)
		# lưu bounding box và text tương ứng vào list
		results.append(((startX, startY, endX, endY), text))
	# print(results)
	return orig,results


num_of_test = 40

for index in range(1,num_of_test+1):
	img = cv2.imread('./input/test' + str(index) +'.jpg')

	orig,results = text_detector(img)
		# duyệt qua kết quả
	for ((xmin, ymin, xmax, ymax), text) in results:
		print("{}\n".format(text))
		
		# strip out non-ASCII text so we can draw the text on the image using OpenCV
		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		#output = orig.copy()

		#cv2.rectangle(output, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
		cv2.putText(orig, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, ), 3)

	#cv2.imshow("Text Detection", orig)
	cv2.imwrite('./output/test' + str(index) +'.jpg',orig)
