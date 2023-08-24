# Computer-Vision-Essay
My essay on computer vision. Check out the computer vision project
# Project Assignment - Computer Vision

Computer vision is a field of artificial intelligence enabling computers to derive information from an input source, such as a video, photo, or live video feed. It enables computers to replicate the human vision system. The entire computer vision process implicates image acquisition data screening, repetitive analysis, and the identification and extraction of information. This comprehensive digital process enables computer systems to comprehend a diverse range of visual content and operate on accordingly. 

Computer vision tasks include methods of acquiring, processing, analysing and understanding digital information, and extraction of high-dimensional data from the real world in order to produce numerical or symbolic information, often decision.

Computer vision uses neural networks, the same way the brain does. Neural networks are mode up of an input layer, multiple hidden layers, and an output layer. In this case, we are using YOLO (You Look Only Once), which uses a neural network designed to take an input image and produce bounding box predictions along with the associated class probabilities for the objects present in the image. Here is a breakdown of YOLO:

1. **Input Image:** The neural network takes an input image as its input. YOLO resizes the input image to a fixed size, often dividing it into a grid.
2. **Neural Network Architecture:** YOLO uses a convolutional neural network (CNN) architecture to process the input image. This CNN processes the entire image in a single forward pass and generates a set of bounding box predictions and class probabilities simultaneously.
3. **Grid and Anchor Boxes:** YOLO divides the input image into a grid of cells. Each cell is responsible for predicting objects that fall within its boundaries. Within each cell, YOLO predicts bounding boxes (usually multiple per cell) and assigns them to anchor boxes. Anchor boxes are predefined shapes that help the model predict a wide range of object shapes and sizes.
4. **Predictions:** For each anchor box in each cell, YOLO predicts:
    - The coordinates of the bounding box relative to the cell's location.
    - The confidence score, indicating how likely the predicted box contains an object.
    - The class probabilities for each class label. This is often represented as a softmax output, where each class gets a probability score.
5. **Non-Maximum Suppression:** After predictions are made, YOLO applies non-maximum suppression to eliminate duplicate or highly overlapping bounding box predictions. This step ensures that only the most confident and non-overlapping predictions are kept.
6. **Output:** The final output of the neural network is a list of bounding box predictions along with their class probabilities. These predictions are generated for all cells in the grid.
7. **Post-Processing and Visualisation:** The bounding box predictions and class probabilities are then used to draw bounding boxes around detected objects on the input image. The class label associated with each box is determined based on the highest class probability.
8. **Thresholding:** YOLO often applies a confidence threshold to the predicted bounding boxes to filter out low-confidence detections. This helps reduce false positives in the final results.
9. **Usage:** The processed image with bounding boxes and class labels is then used for various applications, such as object detection in autonomous vehicles, surveillance systems, robotics, and more.

The key innovation of YOLO is its ability to perform object detection in a single forward pass of the neural network*, making it extremely efficient for real-time applications. The neural network's architecture is designed to handle both the localisation (bounding box prediction) and classification (class probabilities) tasks in a unified manner.

*Single forward pass: The entire object detection process—from analysing the input image to generating bounding box predictions and class probabilities—is done in a single pass through a neural network. There's no need for separate region proposals or multiple passes.


Computer vision from a live webcam feed. Code below.


Proof that it isn’t always reliable: The breadboard is classified as a chair…

## Code

```python
from ultralytics import YOLO
import cv2
import math 
import tkinter as tk
from PIL import Image, ImageTk

# Create a GUI window
root = tk.Tk()
root.title("Object Detection")

# Create a label to display the webcam feed
label = tk.Label(root)
label.pack()

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

def update_frame():
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100

            # class name
            cls = int(box.cls[0])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            text = classNames[cls]
            color = (255, 242, 0) # Bright bold red color (BGR format)
            bg_color = (255, 0, 255)  # Purple color for background
            thickness = 2        # Increased thickness for boldness

            # Get text size to create a background rectangle
            text_size = cv2.getTextSize(text, font, fontScale, thickness)[0]
            text_width = text_size[0]
            text_height = text_size[1]

            # Calculate the coordinates for the background rectangle
            bg_x1 = x1
            bg_y1 = y1 - text_height - 5
            bg_x2 = x1 + text_width + 10
            bg_y2 = y1 - 3

            # Draw background rectangle and text
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)  # Filled rectangle
            cv2.putText(img, text, (x1 + 5, y1 - 5), font, fontScale, color, thickness)

    # Convert the OpenCV image to a format compatible with Tkinter
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)

    # Update the label with the new frame
    label.img = img_tk
    label.config(image=img_tk)
    label.after(1, update_frame)  # Reduced delay for higher frame rate

# Start updating the frame
update_frame()

# Run the GUI event loop
root.mainloop()

cap.release()
cv2.destroyAllWindows()
```

# What is Computer Vision?

Computer vision is the work of making sense out of visual, audio, or other data inputs for machines. So far with computer vision, we are able to recognise faces, read licence plates, find objects in a given image, summarise the content of a photo, and a lot more.

At first computer vision started quite manually, with researchers trying to come up with to detect certain patterns and images, but with the surge of deep learning, over the last decade, we now have models that can automatically learn these patterns just by looking at the examples.

> “Computer vision is a field of study which enables computers to replicate the human visual system. It’s a subset of artificial intelligence which collects information from digital images or videos and processes them to define the attributes. The entire process involves image acquiring, screening, analysing, identifying and extracting information. This extensive processing helps computers to understand any visual content and act on it accordingly. You can also take up a computer vision course for free to understand the basics under Artificial intelligence domain.” (M. Chatterjee, Nov. 2022)
> 

Computer vision is a branch of computer science that powers machines to see, recognise, and process images. It is a multi-disciplinary field, and is a subfield of artificial intelligence and machine learning. 


# How does Computer Vision Work?

<aside>
👁️ Computer vision primarily uses pattern recognition techniques to self-train and understand data.

</aside>

Computer vision algorithms are trained by using a lot of visual data. The model process these images and finds patterns. “For example, If we send a million pictures of vegetable images to a model to train, it will analyse them and create an Engine (Computer Vision Model) based on patterns that are similar to all vegetables. As a result, Our Model will be able to accurately detect whether a particular image is a Vegetables every time we send it.” (Dharmaraj, Mar. 2022).

> ”While machine learning algorithms were previously used for computer vision applications, now deep learning methods have evolved as a better solution for this domain. For instance, machine learning techniques require a humongous amount of data and active human monitoring in the initial phase monitoring to ensure that the results are as accurate as possible. Deep learning on the other hand, relies on neural networks, and uses examples for problem solving. It self-learns by using labeled data to recognise common patterns in the examples.” (M. Chatterjee, Nov. 2022)
> 


# Computer Vision Platforms

Some pre-trained models developed by companies are:

- **OpenCV**: OpenCV provides both pre-trained models for tasks like object detection, face recognition, and more, as well as the tools to train your own models.
- **TensorFlow**: TensorFlow offers a wide range of pre-trained models for various tasks such as image classification, object detection, and natural language processing. It also supports training your own models.
- **Keras**: Keras, which is now part of TensorFlow, supports using pre-trained models from TensorFlow's Model Zoo, along with the ability to build and train your own models.
- **MATLAB**: MATLAB offers pre-trained models for various tasks like image classification and object detection through its Deep Learning Toolbox.
- **CAFFE**: CAFFE includes a model zoo with pre-trained models for various deep learning tasks, particularly in computer vision.
- **DeepFace**: DeepFace is a library built on top of Keras and TensorFlow, and it's designed for face recognition tasks. It may use pre-trained models or allow training new ones.
- **YOLO**: YOLO is an object detection algorithm that often comes with pre-trained models for real-time object detection.
- **GPUImage**: GPUImage is a library primarily used for GPU-accelerated image and video processing, and it provides tools to work with filters and effects.
- 
**Not Necessarily Pre-trained:**

- **Viso Suite:** This appears to be a collection of tools for visual odometry and SLAM (Simultaneous Localisation and Mapping), which generally involve real-time localisation and mapping of environments using camera sensors. While it may not always use pre-trained models, it relies on sensor data and algorithms.

- **SimpleCV:** SimpleCV is a library aimed at simplifying computer vision tasks but may not always come with pre-trained models. It's designed to be user-friendly and accessible.
- **CUDA:** CUDA is a parallel computing platform and API developed by NVIDIA, primarily used to accelerate computations on NVIDIA GPUs. While it's not a pre-trained model itself, it allows various libraries and frameworks to leverage GPU acceleration for faster computations.


---

## Credits:

[What is computer vision?](https://www.youtube.com/shorts/KmsvNU8zib4)

[Computer Vision Explained in 5 Minutes | AI Explained](https://www.youtube.com/watch?v=puB-4LuRNys)

[What is Computer Vision? Know Computer Vision Basic to Advanced & How Does it Work?](https://www.mygreatlearning.com/blog/what-is-computer-vision-the-basics/)

[What is Computer Vision? & Its Applications](https://medium.com/@draj0718/what-is-computer-vision-its-applications-826c0bbd772b)
