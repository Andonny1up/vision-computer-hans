import cv2 as cv
import mediapipe as mpp
import time

class handDetector:
    
    def __init__(self,mode=False,max_hands=2,redefineLms = False,detection_con = 0.5,tracking_confidence = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.redefineLms = redefineLms
        self.detection_con = detection_con
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mpp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,self.max_hands,self.redefineLms,self.detection_con,self.tracking_confidence)
        self.mp_draw = mpp.solutions.drawing_utils
        

    def findHands(self,frame,draw = True):
        imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(frame,hand_lms,self.mp_hands.HAND_CONNECTIONS)
                
        return frame
                # for id, lm in enumerate(hand_lms.landmark):
                #     # print(id,lm)
                #     h, w, c= frame.shape
                #     c_x, c_y = int(lm.x*w), int(lm.y*h)
                #     print(c_x,c_y)
                #     if id == 0:
                #         cv.circle(frame,(c_x,c_y),25,(255,0,255),cv.FILLED)
    
    
def run():
    p_time = 0
    c_time = 0

    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        isTrue, frame = cap.read()
        frame = detector.findHands(frame)
        
        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time
        
        cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,2,(255,56,56),2)
        
        cv.imshow("Life Video", frame)
        cv.waitKey(1)


if __name__ == '__main__':
    run()