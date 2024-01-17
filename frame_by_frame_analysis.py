import cv2
from matplotlib import pyplot as plt
import math
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from vision_gui import Gui
import time

# Disable spines, ticks, and labels in the plot
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def display_result(image, auto_gui):
    recognition_result = get_result(image)
    if recognition_result.gestures:
        top_gesture = recognition_result.gestures[0][0]
        hand_landmarks = recognition_result.hand_landmarks[0]

        title = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
        dynamic_titlesize = 80  # Increase the value to make the text larger
        annotated_image = image  # Use image instead of frame
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        # add title to the annotated image top center
        cv2.putText(annotated_image, title, (int(annotated_image.shape[1]/2) - int(len(title) * dynamic_titlesize / 4), 50), cv2.FONT_HERSHEY_SIMPLEX, dynamic_titlesize/100, (0, 0, 0), 2)

        # make hand center equal to the average of x and y of landmarks 0, 5 and 17
        hand_center_x = (hand_landmarks[0].x + hand_landmarks[5].x + hand_landmarks[17].x) / 3
        hand_center_y = (hand_landmarks[0].y + hand_landmarks[5].y + hand_landmarks[17].y) / 3

        # create a line whos length is 100 * the x value of the first landmark
        cv2.line(annotated_image, (50, 200), (50 + int(hand_landmarks[0].x * 100), 200), (0, 0, 0), 2)

        # create a line whos length is 100 * the y value of the first landmark
        cv2.line(annotated_image, (50, 250), (50, 250 + int(hand_landmarks[0].y * 100)), (0, 0, 0), 2)

        # auto_gui.check_input(top_gesture.category_name, (hand_center_x, hand_center_y), (annotated_image.shape[1], annotated_image.shape[0]))

        cv2.imshow(f"Annotated Frame", annotated_image)
        cv2.waitKey(1)

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))

def resize_and_show(image):
    h, w, _ = image.shape
    DESIRED_HEIGHT = 300  # Replace with your desired height

    if h <= DESIRED_HEIGHT:
        img = image
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

REDUCE_RESOLUTION = True

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if REDUCE_RESOLUTION:
    frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))

# create gesture recognizer object
# base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
# options = vision.GestureRecognizerOptions(base_options=BaseOptions(model_asset_path='/path/to/model.task'), running_mode=vision.RunningMode.LIVE_STREAM, result_callback=print_result)
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
recognizer = GestureRecognizer.create_from_options(options)

def frame_by_frame_analysis():
    # Open the video capture
    cap = cv2.VideoCapture(0)
    auto_gui = Gui()

    # Check if the video capture is opened successfully
    if not cap.isOpened():
        print("Error opening video capture")
        exit()

    # Set the frame rate to 1 fps
    fps = 40
    delay = int(1000 / fps)

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if REDUCE_RESOLUTION:
            frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
        

        # Check if the frame was read successfully
        if not ret:
            print("Error reading frame")
            break

        # Display the frame
        # cv2.imshow("Frame", frame)

        display_result(frame, auto_gui)
        if cv2.waitKey(1) == 27:
            break


        

        # dispatch_action(frame, auto_gui)


        # Wait for the specified delay and check if the Esc key is pressed

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

def get_result(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    return recognizer.recognize_async(mp_image,  time.time_ns())

def dispatch_action(image, auto_gui):
    recognition_result = get_result(image)

    if recognition_result.gestures:
        top_gesture = recognition_result.gestures[0][0]
        hand_landmarks = recognition_result.hand_landmarks[0]

        title = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
        auto_gui.check_input(top_gesture.category_name, hand_landmarks, (image.shape[1], image.shape[0]))




def display_one_image(image, title, subplot, titlesize=16):
    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)



if __name__ == "__main__":
    frame_by_frame_analysis()