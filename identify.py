import cv2, os, numpy
hfile = "haarcascade.xml"
datasets = "datasets"
fcascade = cv2.CascadeClassifier(hfile)
print("******Training******")
(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id +=1

(images, labels) = [numpy.array(lis) for lis in [images, labels]]
print(images, labels)                   
(width, height) = (130, 100)

#model = cv2.face.LBPHFaceRecognizer_create() #loading face recognizer
model =  cv2.face.FisherFaceRecognizer_create()

model.train(images, labels)
cam = cv2.VideoCapture(0)
cnt = 0

while True : 
    (_,img) = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = fcascade.detectMultiScale(gray,1.3,4)
    for(x,y,w,h) in faces : 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        prediction = model.predict(face_resize)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if prediction[1]<800:
            cv2.putText(img,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,2,(0, 0, 255),2)
            print (names[prediction[0]])
            cnt=0
        else:
            cnt+=1
            cv2.putText(img,'Unknown',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            if(cnt>100):
                print("Unknown Person")
                cv2.imwrite("unKnown.jpg",img)
                cnt=0
    cv2.imshow("Frame",img)
    key = cv2.waitKey(1) &0xFF
    if(key == ord("q")):
        break
cam.release()
cv2.destroyAllWindows()
        
