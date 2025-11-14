
#VirtualMouseUsingHandGestures


🖱️ VirtualMouse — Gesture-Controlled Mouse (MediaPipe + OpenCV)
    Control your computer completely hands-free using real-time hand tracking and gestures.
    VirtualMouse converts your webcam feed into smooth, accurate mouse actions — including move, click, drag, and scroll — without any external hardware.


🚀 Features
    Cursor control using index-finger movement
    Left & right click with finger gestures
    Drag & drop via thumb–index pinch
    Scrolling with four-finger gesture
    High FPS performance with threaded capture
    Smoothing + velocity filtering
    Calibration mode for improved accuracy

🧠 Tech Stack
    Python
    MediaPipe Hands
    OpenCV
    NumPy
    PyAutoGUI / Autopy (optional)


    RUN COMMAND ---- > source venv-py311/bin/activate python "Virtual Mouse.py"


🕹️ Gestures & Controls    
    
| Gesture                | Action            |
| ---------------------- | ----------------- |
| Index finger up        | Move cursor       |
| Index + Middle up      | Left click        |
| Thumb + Index + Middle | Right click       |
| Thumb–Index pinch      | Drag (mouse down) |
| Four fingers up        | Scroll            |



🎯 How It Works

Uses MediaPipe Hands for 21 landmark detection, then applies:
Finger-state classification
Distance-based pinch detection
Normalized smoothing
Inertia-based cursor motion
Adaptive drag threshold
Screen interpolation


🧪 Calibration

Press C while running to auto-calibrate hand size & frame margins.



PROJECT STRUCTURE:

📁 Virtual Mouse Using HG
 ├── HandTracking.py
 ├── Virtual Mouse.py
 ├── requirements.txt
 ├── README.md
 ├── LICENSE
 └── .gitignore




THANK YOU!

