import cv2
import mediapipe as mp
import numpy as np
import os
import math
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)
width = 1920
height = 1080
cap.set(3, width)
cap.set(4, height)

# Image that will contain the drawing and then passed to the camera image
imgCanvas = np.zeros((height, width, 3), np.uint8)

# Getting all header images in a list
folderPath = '.\\Header'
myList = os.listdir(folderPath)
#overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #overlayList.append(image)

# Presettings:
#header = overlayList[0]
drawColor = (0, 0, 255)
thickness = 20 # Thickness of the painting
tipIds = [4, 8, 12, 16, 20] # Fingertips indexes
xp, yp = [0, 0] # Coordinates that will keep track of the last position of the index finger

with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 畫每個關節點的編號
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    # 文字的位置要依照翻轉進行調整
                    #flipped_cx = w - cx  # 水平翻轉後，X位置會變成原來的對稱點
                    cv2.putText(image, str(id), (cx+5, cy-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)                
                
                
                
                
                
                
                # Getting all hand points coordinates
                points = []
                for lm in hand_landmarks.landmark:
                    points.append([int(lm.x * width), int(lm.y * height)])

                # Only go through the code when a hand is detected
                if len(points) != 0:
                    x1, y1 = points[8]  # Index finger
                    x2, y2 = points[12] # Middle finger
                    x3, y3 = points[4]  # Thumb
                    x4, y4 = points[20] # Pinky

                    ## Checking which fingers are up
                    fingers = []
                    # Checking the thumb
                    if points[tipIds[0]][0] < points[tipIds[0] - 1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # The rest of the fingers
                    for id in range(1, 5):
                        if points[tipIds[id]][1] < points[tipIds[id] - 2][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    ## Selection Mode - Two fingers are up
                    nonSel = [0, 3, 4] # indexes of the fingers that need to be down in the Selection Mode
                    if (fingers[1] and fingers[2]) and all(fingers[i] == 0 for i in nonSel):
                        xp, yp = [x1, y1]

                        # Selecting the colors and the eraser on the screen
                        if(y1 < 125):
                            if(0 < x1 < width/4):
                                #header = overlayList[0]
                                drawColor = (0, 0, 255)
                            elif(width/4 < x1 < width/4*2):
                                #header = overlayList[1]
                                drawColor = (255, 0, 0)
                            elif(width/4*3 < x1 < width/4*3.5):
                                #header = overlayList[2]
                                drawColor = (0, 255, 0)
                            elif(width/4*3.5 < x1 < width/4*4):
                                #header = overlayList[3]
                                drawColor = (0, 0, 0)

                        cv2.rectangle(image, (x1-10, y1-15), (x2+10, y2+23), drawColor, cv2.FILLED)

                    ## Stand by Mode - Checking when the index and the pinky fingers are open and dont draw
                    nonStand = [0, 2, 3] # indexes of the fingers that need to be down in the Stand Mode
                    if (fingers[1] and fingers[4]) and all(fingers[i] == 0 for i in nonStand):
                        # The line between the index and the pinky indicates the Stand by Mode
                        cv2.line(image, (xp, yp), (x4, y4), drawColor, 5) 
                        xp, yp = [x1, y1]

                    ## Draw Mode - One finger is up
                    nonDraw = [0, 2, 3, 4]
                    if fingers[1] and all(fingers[i] == 0 for i in nonDraw):
                        # The circle in the index finger indicates the Draw Mode
                        cv2.circle(image, (x1, y1), int(thickness/2), drawColor, cv2.FILLED) 
                        if xp==0 and yp==0:
                            xp, yp = [x1, y1]
                        # Draw a line between the current position and the last position of the index finger
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                        # Update the last position
                        xp, yp = [x1, y1]

                    ## Clear the canvas when the hand is closed
                    if all(fingers[i] == 0 for i in range(0, 5)):
                        imgCanvas = np.zeros((height, width, 3), np.uint8)
                        xp, yp = [x1, y1]

                    ## Adjust the thickness of the line using the index finger and thumb
                    selecting = [1, 1, 0, 0, 0] # Selecting the thickness of the line
                    setting = [1, 1, 0, 0, 1]   # Setting the thickness chosen
                    if all(fingers[i] == j for i, j in zip(range(0, 5), selecting)) or all(fingers[i] == j for i, j in zip(range(0, 5), setting)):

                        # Getting the radius of the circle that will represent the thickness of the draw
                        # using the distance between the index finger and the thumb.
                        r = int(math.sqrt((x1-x3)**2 + (y1-y3)**2)/3)
                        
                        # Getting the middle point between these two fingers
                        x0, y0 = [(x1+x3)/2, (y1+y3)/2]
                        
                        # Getting the vector that is orthogonal to the line formed between
                        # these two fingers
                        v1, v2 = [x1 - x3, y1 - y3]
                        v1, v2 = [-v2, v1]

                        # Normalizing it 
                        mod_v = math.sqrt(v1**2 + v2**2)
                        v1, v2 = [v1/mod_v, v2/mod_v]
                        
                        # Draw the circle that represents the draw thickness in (x0, y0) and orthogonaly 
                        # translated c units
                        c = 3 + r
                        x0, y0 = [int(x0 - v1*c), int(y0 - v2*c)]
                        cv2.circle(image, (x0, y0), int(r/2), drawColor, -1)

                        # Setting the thickness chosen when the pinky finger is up
                        if fingers[4]:                        
                            thickness = r
                            cv2.putText(image, 'Check', (x4-25, y4-8), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,0), 1)

                        xp, yp = [x1, y1]

        # Setting the header in the video
        #image[0:125, 0:width] = header
   
        # The image processing to produce the image of the camera with the draw made in imgCanvas
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGRA)
        
        
        image_result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
        #alpha_channel = imgInv[:, :, 3]  # Extract the alpha channel of the mask
        image_result[:, :, 3] = 0  # Apply the mask to the alpha channel of the image_result
        
        imgCanvas = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2BGRA)
        img = cv2.bitwise_and(image_result, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        #print(imgCanvas.shape)
        #img = cv2.flip(img, 1)
        img = cv2.resize(img, (1920, 1080))
        cv2.imshow('MediaPipe Hands', img)
    
        
        
        
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
