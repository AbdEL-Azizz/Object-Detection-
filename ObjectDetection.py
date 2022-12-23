import cv2

classNames = []
classFile = 'z.txt'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


configPath = 'x.pbtxt'
weightPath = 'y.pb'

net = cv2.dnn_DetectionModel(weightPath, configPath)

net.setInputSize(320, 230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
cap = cv2.VideoCapture(0)
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
while cap.isOpened():
    ret, frame = cap.read()
    f=cv2.flip(frame,180)
    ClassIndex, confidece, bbox = net.detect(f, confThreshold=0.55)
    if (len(ClassIndex) != 0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if (ClassInd <= 80):
                cv2.rectangle(f, boxes, (255, 0, 0), 2)
                cv2.putText(f, classNames[ClassInd-1], (boxes[0]+10, boxes[1]+40),
                            font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
    cv2.imshow('video', f)
    if cv2.waitKey(2) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
