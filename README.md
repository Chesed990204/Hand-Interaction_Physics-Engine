# Hand Interaction Physics Engine

A Python prototype for interacting with simulated balls using a webcam-tracked hand. MediaPipe hand landmarks drive a set of spherical proxies, allowing hand movements to influence other balls through collision responses.

**Individual project by Inhyuk Song · Fall 2023**  
Python · OpenCV · MediaPipe · NumPy · Pygame · PyOpenGL

![Webcam hand tracking and the sphere-based physics simulation](https://github.com/Chesed990204/Hand-Interaction_Physics-Engine/assets/45059838/1d5f0e56-4ae1-44d3-843d-0c82521ef18a)

## How it works

1. **Track the hand:** OpenCV captures webcam frames, and MediaPipe estimates hand landmarks.
2. **Map input to the simulation:** 21 landmarks are mapped to the positions of high-mass spheres representing the hand.
3. **Update the physics:** other spheres move under gravity and respond to sphere–sphere and wall collisions, with damping.
4. **Display the interaction:** Pygame and OpenGL render the simulated scene, while an OpenCV window shows the camera image with hand landmarks.

The main demo initializes **150 bodies**, including the 21 hand proxies. Its collision model uses sphere overlap, relative velocity along the collision direction, and mass-weighted velocity changes.

## Requirements

- A webcam and permission for Python to access it.
- A graphical desktop with OpenGL/GLU support.
- Python with NumPy, OpenCV, MediaPipe, Pygame, and PyOpenGL.

The code uses the legacy `mp.solutions.hands` API. A proposed compatibility setup is **Python 3.11 with MediaPipe 0.10.21**. The original environment was not recorded, and this installation recipe has not been tested with a live camera.

## Setup and run

```bash
git clone https://github.com/Chesed990204/Hand-Interaction_Physics-Engine.git
cd Hand-Interaction_Physics-Engine
python -m venv .venv
```

Create the environment using Python 3.11, then activate it:

- macOS/Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`

Install the dependencies:

```bash
python -m pip install "numpy<2" "opencv-contrib-python==4.11.0.86" "mediapipe==0.10.21" pygame PyOpenGL
```

This uses the OpenCV distribution required by MediaPipe; avoid installing another OpenCV distribution into the same environment. See the [MediaPipe 0.10.21 dependencies](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/requirements.txt) and [available Python wheels](https://pypi.org/project/mediapipe/0.10.21/).

Start the main demo:

```bash
python Physics_Hand/main.py
```

Show one hand to the camera and move it to interact with the simulated balls. The application opens a **1280 × 720 OpenGL window** and a **MediaPipe Hands camera window**. Close the Pygame/OpenGL window to exit.

The default camera is `cv2.VideoCapture(0)`. Change that index in the script if your camera uses a different device number.

## Two examples

| File | Purpose |
| --- | --- |
| `Physics_Hand/main.py` | Main multi-body simulation with hand proxies, gravity, wall collisions, and sphere–sphere collision responses. |
| `Physics_Hand/beta.py` | Separate hand-visualization experiment with a palm, interpolated finger geometry, and a single moving ball with a simple collision response. |

Run the alternate example with:

```bash
python Physics_Hand/beta.py
```

`beta.py` additionally calls GLUT functions and requires a native GLUT/freeglut runtime. In that example, **Escape** exits when the OpenCV window has focus. The main demo does not implement this shortcut.

## Adjusting the main simulation

The following values are defined near the top of `main.py`:

| Parameter | Default | Role |
| --- | --- | --- |
| `BALLS` | `150` | Total body count, including 21 hand proxies. Keep this at least 21. |
| `TIME_TICK` | `0.1` | Integration step used by position and velocity updates. |
| `GRAVITY_CONST` | `4` | Downward gravity magnitude. |
| `DAMPING_FACTOR` | `0.25` | Damping applied during collision response. |
| `SPACE_LIMITS` | Defined per axis | Simulation boundaries. |
| `COLLISON_MARGIN` | `0.1` | Separation offset used when resolving collisions. |

Sphere radii are assigned directly in `randomize_balls()`: 20 for hand proxies and 15 for other balls. The existing `MIN_BALL_SIZE` and `MAX_BALL_SIZE` constants are currently unused. Hand-coordinate offsets and scaling are defined in `map_hand_landmarks_to_balls()`.

## Implementation notes

- The landmark mapping uses scaled camera-image coordinates and estimated relative depth; it is not a calibrated metric reconstruction of the hand.
- Both hands would write into the same set of 21 proxies. Use one visible hand for the current demo.
- The simulation uses fixed-step updates and pairwise collision checks. Its speed and response depend on frame processing and body count.
- If camera capture fails, check permissions, device selection, and competing camera applications. The main loop does not safely skip a failed frame before image conversion.

## Author

Designed and implemented by **Inhyuk Song** as an individual project in **Fall 2023**.
