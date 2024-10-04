from collections import deque
import torch
from ultralytics import YOLO
import cv2
from PIL import Image
import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
import base64
from io import BytesIO

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model_pose = YOLO("models\yolov8x-pose.pt").to(device)

# Define the keypoint mapping for all 14 keypoints
keypoint_mapping = {
    'nose': 0, 'r_eye': 1, 'l_eye': 2, 'r_ear': 3, 'l_ear': 4,
    'r_shoulder': 5, 'l_shoulder': 6, 'r_elbow': 7, 'l_elbow': 8,
    'r_wrist': 9, 'l_wrist': 10, 'r_hip': 11, 'l_hip': 12,
    'r_knee': 13, 'l_knee': 14, 'r_ankle': 15, 'l_ankle': 16
}


def get_pose_from_image(image, yolo_model_pose, keypoint_mapping):
    """
    Process pose estimation from the given image using YOLOv8-pose, returning the pose keypoints
    with real pixel coordinates and the bounding box coordinates of the person.
    Only returns results if at least 9 keypoints are non-zero.

    Args:
        image (np.array): Image loaded using OpenCV.
        yolo_model_pose: Loaded YOLOv8 pose model.
        keypoint_mapping: Dictionary that maps keypoint names to indices in the pose result.

    Returns:
        dict or None: A dictionary containing the pose data for all 14 keypoints with real pixel coordinates,
                      as well as the bounding box coordinates. Returns None if less than 9 keypoints are detected.
    """

    # Get image height and width
    image_height, image_width = image.shape[:2]

    # Initialize a dictionary to store all 14 keypoints with [0, 0] as default
    results = {
        'nose': [0, 0], 'r_eye': [0, 0], 'l_eye': [0, 0],
        'r_ear': [0, 0], 'l_ear': [0, 0], 'r_shoulder': [0, 0],
        'l_shoulder': [0, 0], 'r_elbow': [0, 0], 'l_elbow': [0, 0],
        'r_wrist': [0, 0], 'l_wrist': [0, 0], 'r_hip': [0, 0],
        'l_hip': [0, 0], 'r_knee': [0, 0], 'l_knee': [0, 0],
        'r_ankle': [0, 0], 'l_ankle': [0, 0],
        'xmin': 0, 'xmax': 0, 'ymin': 0, 'ymax': 0  # Bounding box coordinates
    }

    # Perform pose estimation with YOLOv8 pose model
    pose_results = yolo_model_pose(image, verbose=False, conf=0.1, imgsz=800)

    for pr in pose_results:
        if pr.keypoints is not None and len(pr.keypoints) > 0:
            # Update the results dictionary with the detected keypoints in pixel coordinates
            for key, idx in keypoint_mapping.items():
                keypoint_list = pr.keypoints.xyn[0][idx].tolist()
                # Convert normalized values to pixel coordinates
                results[key] = [int(keypoint_list[0] * image_width), int(keypoint_list[1] * image_height)]

            # Update bounding box coordinates in pixel values
            bbox = pr.boxes.xyxy[0].tolist()  # Assuming there's only one person per image
            results['xmin'] = int(bbox[0])
            results['xmax'] = int(bbox[2])
            results['ymin'] = int(bbox[1])
            results['ymax'] = int(bbox[3])

    # Check if at least 9 keypoints have non-zero values
    non_zero_count = sum(1 for coord in results.values() if isinstance(coord, list) and coord != [0, 0])

    # Return results only if at least 9 keypoints have non-zero values
    if non_zero_count >= 9:
        return results
    else:
        return None


def process_image(image_path):
    """
    Load an image, perform pose estimation, and print the results.

    Args:
        image_path (str): Path to the image file.
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Get pose estimation results
    pose_result = get_pose_from_image(image, yolo_model_pose, keypoint_mapping)

    # Print the pose result if available
    if pose_result:
       # print("Pose Estimation Results:")
        # for key, value in pose_result.items():
        #     print(f"{key}: {value}")
        return pose_result
    else:
       # print("Pose estimation failed or less than 15 keypoints detected.")
        return "no pose result found...."

# Function to convert image to base64
def convert_to_base64(image):
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

# Node 1: Pose Detection (Simulating process_image function)
def pose_detection(input):
    image_path = input["image_path"]
    # Assuming process_image would output pose points for this image
    pose_result=process_image(image_path)
    return {"output": f"Pose Estimation Results: {pose_result}", "image_path": image_path}
    


def agent2(input_2):
    # LLM Model Setup
    llm = ChatOllama(model="llava-llama3", temperature=0)

    pose_output = input_2["output"]
    image_path = input_2["image_path"]
    
    # Opening and converting image to base64
    pil_image = Image.open(image_path)
    image_b64 = convert_to_base64(pil_image)

    # Function to construct prompt
    def prompt_func(data):
        text = data["text"]
        image = data["image"]

        image_part = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image}",
        }

        content_parts = []

        text_part = {"type": "text", "text": text}

        content_parts.append(image_part)
        content_parts.append(text_part)

        return [HumanMessage(content=content_parts)]

    # Constructing the LangChain pipeline
    chain = prompt_func | llm | StrOutputParser()

    # Create the query using pose result and image
    query_chain = chain.invoke(
        {"text": f"Analyze the image of the sleeping person, focusing on the bounding box coordinates. Locate the person on the bed and visually assess their posture while using the pose data as supplementary information.\n\n"
f"Use the pose result: {pose_output} (pose points and bounding box coordinates). If a pose point is [0, 0], it indicates that point is not visible; rely on other visible pose points.\n\n"
f"Classify the person’s posture into one category:\n"
f"- 'Supine Position' (lying on their back)\n"
f"- 'Prone Position' (lying on their belly)\n"
f"- 'Half Turn Side Lying' (turned more than halfway to the side).\n\n"
f"IMPORTANT NOTE: If the eyes, nose, and one ear pose points are not visible ([0, 0]), but the other ear is present, classify as 'Half Turn Side Lying'. This is a critical rule. Ensure only one posture category is marked as true in the final result.\n\n"
f"Return the result in the following structured format:\n\n"
f"{{\n"
f"  'Supine Position': true/false,\n"
f"  'Prone Position': true/false,\n"
f"  'Half Turn Side Lying': true/false,\n"
f"  'Confidence': {{\n"
f"    'overall_decision': <probability>\n"
f"  }}\n"
f"}}\n\n"
f"Only return the JSON structure result. No additional explanations.",

         "image": image_b64}
    )

    # Return the response, including image path and classification
    return f"Agent Says: This image {image_path}, Classified as: {query_chain}"
