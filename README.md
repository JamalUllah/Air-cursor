# 🪄 AirCursor: Dual-Hand AI Mouse Controller

AirCursor is a computer vision-powered Python application that turns your webcam into a touchless mouse. By utilizing MediaPipe's Hand Landmarker API and OpenCV, it tracks both of your hands to provide an intuitive, split-control interface for your computer.

## ✨ Features

*   **Dual-Hand Tracking:** Splits responsibilities between hands to prevent accidental clicks while moving.
*   **Right Hand Navigation:** Controls cursor movement and scrolling.
*   **Left Hand Actions:** Controls left clicks, right clicks, and drag-and-drop holding.
*   **Smooth Cursor Control:** Uses Exponential Moving Average (EMA) smoothing to eliminate jitter.
*   **Non-Blocking GUI:** The camera preview runs in a dedicated thread to ensure zero lag during gesture recognition.
*   **Edge-Friendly:** Adjusted screen margins make it easy to reach the corners of your monitor without moving your hand out of the camera's view.

## ✋ Gesture Controls

### Right Hand (Movement & Scrolling)
| Gesture | Action |
| :--- | :--- |
| **Fist (All fingers curled)** | **IDLE:** Cursor freezes. Perfect for repositioning your hand. |
| **Index Finger Up** | **MOVE:** Cursor follows your index fingertip. |
| **Index + Middle + Ring Up** | **SCROLL:** Move your hand up and down to scroll the page. |

### Left Hand (Clicks & Drags)
| Gesture | Action |
| :--- | :--- |
| **Index + Thumb Snap** | **LEFT CLICK:** Quick pinch and release. |
| **Index + Thumb Hold** | **DRAG / HIGHLIGHT:** Hold the pinch to hold the left mouse button down. |
| **Middle + Thumb Snap** | **RIGHT CLICK:** Quick pinch and release. |

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/AirCursor.git
   cd AirCursor
   ```

2. **Install the required dependencies:**
   Make sure you have Python 3.8+ installed.
   ```bash
   pip install opencv-python mediapipe numpy pyautogui
   ```
   *(Note: The script automatically downloads the required MediaPipe `.task` model on its first run.)*

## 🎮 Usage

Run the script from your terminal:

```bash
python hand_cursor.py
```

A preview window will open showing your webcam feed, the recognized hand landmarks, and a HUD (Heads-Up Display) at the bottom reminding you of the controls. 

*   **To quit:** Press `Q` while focused on the preview window, or use `Ctrl+C` in your terminal.
*   **Fail-safe:** If the mouse ever gets stuck, ram the cursor into any of the four corners of your physical monitor to trigger PyAutoGUI's fail-safe and abort the script.

## ⚙️ Tuning / Customization

You can easily adjust the sensitivity to your liking by modifying the constants at the top of `hand_cursor.py`:

*   `MOVE_SMOOTH`: Controls cursor responsiveness (higher = faster, lower = smoother).
*   `SCROLL_SPEED`: Multiplier for scroll distance.
*   `MARGIN`: Sets the camera dead-zone boundary. Increase this if you have trouble reaching the corners of your screen.
*   `PINCH_DIST`: Distance threshold for registering a click.

## 🛠️ Tech Stack

*   **Python**
*   **MediaPipe** (Tasks API for high-performance ML inference)
*   **OpenCV** (Video capture, drawing, and image manipulation)
*   **PyAutoGUI** (Cross-platform GUI automation)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
