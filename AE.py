import cv2 as cv
vid=cv.VideoCapture('video.mp4')
car_cascade=cv.CascadeClassifier('cars.xml')
count=0
while True:
    ret,frames=vid.read()
    width=int(frames.shape[1]*0.55)
    height=int(frames.shape[0]*0.55)
    dimensions=(width,height)
    video=cv.resize(frames,dimensions,interpolation=cv.INTER_AREA)
    gray=cv.cvtColor(video,cv.COLOR_BGR2GRAY)
    height,width=video.shape[0:2]
    video[0:60,0:width]=[255,0,0]
    cv.line(video,(0,height-410),(width,height-410),(0,255,255),2)
    car=car_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=2)
    for x,y,w,h in car:
        carc=int(y+h/2)
        linec=height-410
        if(carc<linec+1 and carc>linec-1):
            count=count+1
        cv.rectangle(video,(x,y),(x+w,y+h),(0,0,255),1)
        cv.putText(video,'vehicle',(x,y-10),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
        cv.putText(video,str(count),(500,50),cv.FONT_HERSHEY_COMPLEX,1.5,(255,255,255),2)
    cv.imshow('video',video)
    if cv.waitKey(33)==27:
        break
#print(height,width)
cv.destroyAllWindows()