from ultralytics import YOLO
import cv2
import math
import datetime
import numpy as np
import threading
from queue import Queue
from pymodbus.client import ModbusTcpClient


red_color = (0,0,255)
green_color = (0,255,0)
thickness = 2

light_yellow = np.array([15, 100, 90])
dark_yellow =np.array([30, 255, 255]) 

dir_output_wheel_present = r'D:\aa_proj\ztest\training\yolo\output\wheel_present'
dir_output_obstruction = r'D:\aa_proj\ztest\training\yolo\output\obstruction'
dir_output_no_trolley = r'D:\aa_proj\ztest\training\yolo\output\dir_output_no_trolley'
model = YOLO("best.pt")
##==============
client = ModbusTcpClient('10.121.42.55' , 502)
##==============



class FrameProcessingThread(threading.Thread):
    def __init__(self, frame):
        threading.Thread.__init__(self)
        self.frame = frame

    def crop_img(self , img):
        pts = np.array([[89,362],[433,350],[955,318],  [955,338],[590,365],[89,385]])
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        croped = img[y:y+h, x:x+w].copy()
        pts = pts - pts.min(axis=0)
        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        return dst

    def percentage_obstruction(self , crop_img11):    #this method return the percentage of yellow color present in an image
        hsv_crop_img2= cv2.cvtColor(crop_img11, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_crop_img2, light_yellow,dark_yellow)
        # cv2.imshow("mask1" , mask1)   
        yellow_pixel_count = cv2.countNonZero(mask1)
        total_pixels = hsv_crop_img2.shape[0] * hsv_crop_img2.shape[1]
        yellow_percentage = (yellow_pixel_count / total_pixels) * 100
        return yellow_percentage
    
    def run(self):
        crop_img1 = self.crop_img(img)
        perc = self.percentage_obstruction(crop_img1)
        if perc < 20:
            print("obstruction")
            cv2.putText(img = img,text = "obstruction" ,org = (90,100),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 2,color = red_color,thickness = 2)
            filename = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            cv2.imwrite(dir_output_obstruction+"\\"+filename+"_demo.jpg",img)
        results = model(img , )
        conf1 = []
        for r in results:
                boxes = r.boxes
                for box in boxes:
                    # cls = int(box.cls[0])
                    conf = math.ceil(box.conf[0] * 100) / 100

                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # w, h = x2 - x1, y2 - y1
                    # x = x1 + (x2 - x1) / 3
                    start_point1 = (x1,y1)
                    end_point1 = (x2,y2)

                    if conf >=0.8 and y1 >335 and y1< 510 : 
                        conf1.append(conf)
                        cv2.rectangle(img,start_point1,end_point1,red_color,thickness)
                        cv2.putText(img = img,text = "wheel: " + str(conf),org = start_point1,fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 0.5,color = (125, 246, 55),thickness = 1)
        print("conf1" +"===" + str(conf1))
        print("percentage_obstruction" +"===" + str(perc))
        if len(conf1) != 0:
            cv2.line(img , (0,335) , (950,335) , red_color,5)
            cv2.line(img , (0,480) , (950,480) , red_color,5)
            cv2.putText(img = img,text = "STOP CONVEYER:" ,org = (50,100),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 2,color = red_color,thickness = 2)
            filename = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            cv2.imwrite(dir_output_wheel_present+"\\"+filename+"_demo.jpg",img)

            # client.connect()
            # client.write_register(60,int(1))
        else:
            cv2.line(img , (0,335) , (950,335) , green_color,5)
            cv2.line(img , (0,480) , (950,480) , green_color,5)
            # filename = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            # cv2.imwrite(dir_output_no_trolley+"\\"+filename+"_demo.jpg",img)

            
            # client.connect()
            # client.write_register(60,int(0))
        output_queue.put(img)

# Define a class for frame displaying thread
class FrameDisplayThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while True:
            # Get the processed frame from the queue
            frame = output_queue.get()
            # Display the frame
            cv2.imshow("Image", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

camip= 'rtsp://admin:tvsm123%23%23%23@10.131.163.217/profile2/media.smp'
cap = cv2.VideoCapture(camip)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

output_queue = Queue()
display_thread = FrameDisplayThread()
display_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame . Exiting ...")
        break
    img = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
    # cv2.imshow("imgg", img)
    # Create a new frame processing thread for each frame
    processing_thread = FrameProcessingThread(img)
    processing_thread.start()
    processing_thread.join()  # Wait for the processing thread to finish



    