import cv2, os
hfile = "haarcascade.xml"
datasets = "datasets"

subdata = str(input("Enter your name : "))

path = os.path.join(datasets,subdata)
if not os.path.isdir(path) :
    os.mkdir(path)
(width,height) = (130,100)
fcascade = cv2.CascadeClassifier(hfile)
cam = cv2.VideoCapture(0)
count = 1
while count < 31 : 
    print(count)
    (_,img) = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = fcascade.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in faces :
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y+h,x:x+w]
        fresize = cv2.resize(face,(width,height))
        cv2.imwrite("%s/%s.png"% (path,count),fresize)
    count += 1

    cv2.imshow("Frame",img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") :
        break
cam.release()
cv2.destroyAllWindows()