from __future__ import division
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import time
import math
import sys
import cv2
import mediapipe as mp

from random import uniform, randint

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

TIME_TICK = 0.1
GRAVITY_CONST = 4
GRAVITY_VECTOR = np.array([0, -GRAVITY_CONST, 0])
COLLISON_MARGIN = 0.1
SPACE_LIMITS = [[-200, 200], [-200, 800], [-200, 200]]
DAMPING_FACTOR = 0.25
BALLS = 150
MIN_BALL_SIZE = 20
MAX_BALL_SIZE = 20

# Helper function to turn arrays into parameters the OpenGL bindings can swallow
def vec(*args):
    return (GLfloat * len(args))(*args)

def randomize_balls(count):
    data = []
    while len(data) < count:
        
        ball = {}
        if(len(data)<21):
            ball['size'] = 20
            ball['mass'] = 10000000000
            ball['white']=True
        else:
            ball['size'] = 15
            ball['mass'] = math.pow(ball['size'], 3)
            ball['white']=False
        ball['pos'] = [uniform(-100, 190), uniform(380, 230), uniform(0, 190)]
        ball['init_vel'] = [uniform(-7, 10), uniform(-20, 20), uniform(-10, -25)]
        touching = False
        for b in data:
            pos = np.array(ball['pos'])
            pos_b = np.array(b['pos'])
            if np.linalg.norm(pos - pos_b) < (ball['size'] + b['size']):
                touching = True
        if not touching:
            data.append(ball)
    return data


class PhysicalBody(object): 
    def __init__(self, mass, size, pos, speed):
        self.mass = mass
        self.size = size
        self.pos = np.array(pos)
        self.speed = np.array(speed)
        self.collisions_handled = False

    def is_touching(self, pos, size):
        dist_norm = np.linalg.norm(pos - self.pos)
        return dist_norm < (size + self.size)

    def update_velocity(self):
        self.speed = self.speed + TIME_TICK * GRAVITY_VECTOR

    def get_collisions(self, objects):
        collisions = []
        for n, obj in enumerate(objects):
            if obj != self:
                dist_norm = np.linalg.norm(self.pos - obj.pos)
                if dist_norm < (self.size + obj.size):
                    collisions.append(n)
        return collisions

    def update_pos(self):
        self.pos = self.pos + TIME_TICK * self.speed
        self.handled_collisions = []

    def kick(self):
        self.speed += GRAVITY_VECTOR * uniform(2, 8)

    def resolve_wall_collisions(self, space_limits):
        for idx, limit in enumerate(space_limits):
            if (self.pos[idx] - self.size) < limit[0]:
                self.speed[idx] *= -(1 - DAMPING_FACTOR)
                self.pos[idx] = limit[0] + self.size + COLLISON_MARGIN
            elif (self.pos[idx] + self.size) > limit[1]:
                self.speed[idx] *= -(1 - DAMPING_FACTOR)
                self.pos[idx] = limit[1] - self.size - COLLISON_MARGIN

    def resolve_obj_collisions(self, objects):
        for obj in objects:
            if obj != self:
                if obj.is_touching(self.pos, self.size) and obj.index not in self.handled_collisions:
                    print("collided!")
                    direction = self.pos - obj.pos
                    direction = direction / np.linalg.norm(direction)
                    self_dv = np.dot(self.speed, direction)
                    obj_dv = np.dot(obj.speed, direction)
                    self.speed += (obj_dv - self_dv) * direction * (1 - DAMPING_FACTOR) * (
                            2 * obj.mass / (self.mass + obj.mass))
                    obj.speed += (self_dv - obj_dv) * direction * (1 - DAMPING_FACTOR) * (
                            2 * self.mass / (self.mass + obj.mass))
                    self.pos += COLLISON_MARGIN * 3 * direction
                    self.handled_collisions.append(obj.index)
                    obj.handled_collisions.append(self.index)
    

class Body3D(PhysicalBody):
    index_counter = 0

    def __init__(self, mass, size, pos=[0, 0, 0], speed=[0, 0, 0]):
        self.mass = mass
        self.size = size
        self.pos = np.array(pos)
        self.speed = np.array(speed)
        self.handled_collisions = []
        self.index = Body3D.index_counter
        Body3D.index_counter += 1

class GraphObject(Body3D):
    def __init__(self, mass, size, pos=[0, 0, 0], speed=[0, 0, 0], texture=None, white = False):
        super(GraphObject, self).__init__(mass, size, pos, speed)
        self.rotation = 0
        self.texture = texture
        self.white=white
        if not self.white:
            self.r1 = uniform(0.1, 1)
            self.r2 = uniform(0.1, 1)
            self.g1 = uniform(0.1, 1)
            self.g2 = uniform(0.1, 1)
            self.b1 = uniform(0.1, 1)
            self.b2 = uniform(0.1, 1)
        else:
            self.r1 = 1.0
            self.r2 = 1.0
            self.g1 = 0.5
            self.g2 = 1.0
            self.b1 = 0.5
            self.b2 = 1.0

    def __draw_sphere(self):
        sphere = gluNewQuadric()
        if self.texture:
            glColor4f(1, 1, 1, 1.0)
            gluQuadricTexture(sphere, GLU_TRUE)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(self.texture.target, self.texture.id)
        else:
            glColor4f(self.r1, self.g1, self.b1, 1.0)
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(self.r1, self.g1, self.b1, 1))

        gluSphere(sphere, 1.0, 24, 24)
        gluDeleteQuadric(sphere)

    def draw(self):
        glLoadIdentity()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glScalef(self.size, self.size, self.size)
        self.__draw_sphere()

    def update_rot(self):
        self.rotation += 10 / self.size

class World(object):
    def __init__(self, objects):
        self.ticks = 0
        self.objects = []
        for obj in objects:
            self.objects.append(GraphObject(obj['mass'],
                                            obj['size'],
                                            obj['pos'],
                                            obj['init_vel'] if 'init_vel' in obj else [0.0, 0.0, 0.0], None, obj['white']))

    def tick(self):
        self.ticks += 1
        for obj in self.objects:
            obj.update_velocity()


        for obj in self.objects:
            obj.update_pos()
            obj.update_rot()
            obj.resolve_wall_collisions(SPACE_LIMITS)
            obj.resolve_obj_collisions(self.objects)
            

        return self.objects[2].pos.tolist()

    def draw_axis(self):
        glLineWidth(20.0)

        # X-axis (Red)
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(200, 0, 0)
        
        glEnd()

        # Y-axis (Green)
        glColor3f(0.0, 1.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 200, 0)
        glEnd()

        # Z-axis (Blue)
        glColor3f(0.0, 0.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 200)
        glEnd()

        glLineWidth(1.0)

    def draw(self):
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_LIGHTING)

        self.draw_axis()

        for obj in self.objects:
            obj.draw()

        glDisable(GL_LIGHTING)
        glLoadIdentity()
        glColor4f(0.7, 0.7, 0.7, 0.7)


        # BOX
        glRotatef(90, 1, 0, 0)
        glTranslatef(0, 0, 200)
        glRectf(200, 200, -200, -200)

        glColor4f(0.8, 0.8, 0.8, 0.4)
        glRotatef(90, 0, 1, 0)
        glTranslatef(200, 0, -200)
        glRectf(200, 200, -200, -200)

        glColor4f(0.8, 0.8, 0.8, 0.2)
        glTranslatef(0, 0, 400)
        glRectf(200, 200, -200, -200)

        glColor4f(0.6, 0.6, 0.6, 0.3)
        glRotatef(90, 1, 0, 0)
        glTranslatef(0, -200, 200)
        glRectf(200, 200, -200, -200)

class Camera(object):
    def __init__(self, win, x=0.0, y=0.0, rot=60.0, zoom=1.0):
        self.win = win
        self.x = x
        self.y = y
        self.rot = rot
        self.zoom = zoom
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

    def worldProjection(self, pos=[0, 0, 0]):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        zwr = self.zoom * self.win.get_width() / self.win.get_height()
        gluPerspective(50, self.win.get_width() / self.win.get_height(), 0.1, 10000)
        glTranslatef(0, 0, -700)
        

class App(object):
    def __init__(self, no_balls, fs=False):
        world_data = randomize_balls(BALLS)
        self.world = World(world_data)
        self.win = pygame.display.set_mode((1280, 720), DOUBLEBUF | OPENGL)
        self.camera = Camera(self.win, zoom=150.0)
        self.mp_hands = mp_hands.Hands()
    
    def map_hand_landmarks_to_balls(self, ids,landmrk):
            landmrk_pos = landmrk.x-0.5, \
                        -landmrk.y+0.5, \
                        landmrk.z+0.4
            ####################
            self.world.objects[ids].pos = np.array(landmrk_pos) * 500

    def mainLoop(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            with mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
                
                success, image = cap.read()
                if not success:
                        print("Ignoring empty camera frame.")                    
                    
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                    
                image.flags.writeable = False
                results = hands.process(image)
                image_height, image_width, _ = image.shape
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            for ids, landmrk in enumerate(hand_landmarks.landmark):
                                self.map_hand_landmarks_to_balls(ids, landmrk)
                                for obj in self.world.objects:
                                    obj.update_pos()
                                for obj in self.world.objects:
                                    obj.resolve_wall_collisions(SPACE_LIMITS)
                                    obj.resolve_obj_collisions(self.world.objects)
                            mp_drawing.draw_landmarks(
                                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                    
            pos = self.world.tick()
            self.camera.worldProjection(pos)
            self.world.draw()
            self.world.draw_axis()
            pygame.display.flip()
            cv2.imshow('MediaPipe Hands', image)

if __name__ == "__main__":
    pygame.init()
    app = App(BALLS)
    app.mainLoop()
