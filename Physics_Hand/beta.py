import cv2
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import pygame
from pygame.locals import *

hand = {}
for i in range(0,21):
    hand[i]=[]


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize Pygame
glutInit()
pygame.init()
pygame.display.set_mode((1280, 720), DOUBLEBUF | OPENGL)
gluPerspective(90, (1280 / 720), 0.1, 100.0)
glTranslatef(0.0, 0.0, -5)

# Function to draw a 3D sphere
def draw_sphere(x, y, z, radius=0.5):
    glPushMatrix()
    glColor3f(1.0, 0.0, 0.0)  # Red color for the sphere
    #glScalef(0.9,0.9,2.0)
    glTranslatef(x, y, z)
    glutSolidSphere(radius, 20, 20)  # Adjust resolution as needed
    glPopMatrix()

def draw_ball(x, y, z, radius=1):
    glPushMatrix()
    glColor3f(1.0, 1.0, 0.0)  # Red color for the sphere
    #glScalef(0.9,0.9,2.0)
    glTranslatef(x, y, z)
    glutSolidSphere(radius, 20, 20)  # Adjust resolution as needed
    glPopMatrix()

def normalize(value, base=50):
    value*=base*2
    value-=base
    return value

def draw_palm():
    glColor3f(1.0, 1.0, 1.0)
    glPushMatrix()
    glBegin(GL_QUADS)
    glVertex3fv(hand[0])
    glVertex3fv(hand[3])
    glVertex3fv(hand[5])
    glVertex3fv(hand[17])

    glEnd()
    glPopMatrix()

def draw_spheres_between_landmarks(start, end, num_spheres=10):
    for i in range(num_spheres + 1):
        alpha = i / num_spheres
        sphere_position = [
            start[0] + alpha * (end[0] - start[0]),
            start[1] + alpha * (end[1] - start[1]),
            start[2] + alpha * (end[2] - start[2]),
        ]
        draw_sphere(sphere_position[0], sphere_position[1], sphere_position[2], radius=0.5)

def detect_sphere_collision(x1,y1,z1,x2,y2,z2,threshold=5):
    distance = ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5
    return distance<threshold
def draw_hand_model(h):
    glClearColor(0, 0, 0, 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glDisable(GL_LIGHTING)
    #glLoadIdentity()
    glColor4f(0.7, 0.7, 0.7, 0.7)

    glRotatef(90, 1, 0, 0)
    glTranslatef(0, 0, 10)
    glRectf(10, 10, -10, -10)

    glColor4f(0.8, 0.8, 0.8, 0.4)
    glRotatef(90, 0, 1, 0)
    glTranslatef(10, 0, -10)
    glRectf(10, 10, -10, -10)

    glColor4f(0.8, 0.8, 0.8, 0.2)
    glTranslatef(0, 0, 20)
    glRectf(10, 10, -10, -10)

    glColor4f(0.6, 0.6, 0.6, 0.3)
    glRotatef(90, 1, 0, 0)
    glTranslatef(0, -10, 10)
    glRectf(10, 10, -10, -10)

    draw_ball(0,h,-20)
    for i in range(20):
        cx, cy, cz = hand[i][0], hand[i][1], hand[i][2]
        draw_sphere(cx, cy, cz)
        if(detect_sphere_collision(cx,cy,cz,0,h,-20)):
            return True


    draw_spheres_between_landmarks(hand[3], hand[4])
    draw_spheres_between_landmarks(hand[5], hand[6])
    draw_spheres_between_landmarks(hand[6], hand[7])
    draw_spheres_between_landmarks(hand[7], hand[8])
    draw_spheres_between_landmarks(hand[9], hand[10])
    draw_spheres_between_landmarks(hand[10], hand[11])
    draw_spheres_between_landmarks(hand[11], hand[12])
    draw_spheres_between_landmarks(hand[13], hand[14])
    draw_spheres_between_landmarks(hand[14], hand[15])
    draw_spheres_between_landmarks(hand[15], hand[16])
    draw_spheres_between_landmarks(hand[17], hand[18])
    draw_spheres_between_landmarks(hand[18], hand[19])
    draw_spheres_between_landmarks(hand[19], hand[20])

    draw_palm()



def render_3d_hand():
    
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        h=25
        while cap.isOpened():
            if(h<-25):
                h=25

            success, image = cap.read()
            if not success:
             print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
             continue
            
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            
            image.flags.writeable = False
            results = hands.process(image)
            image_height, image_width, _ = image.shape
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for ids, landmrk in enumerate(hand_landmarks.landmark):
                        cx, cy, cz = normalize(landmrk.x,50), -normalize(landmrk.y, 25), normalize(landmrk.z,20)                        
                        hand[ids]=[cx,cy,cz]
                        print(cx,cy,cz)
                        
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                if(draw_hand_model(h)):
                    h+=2
                h-=1.5
               
            else:
                 glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

# Call the function to start rendering the 3D hand
render_3d_hand()