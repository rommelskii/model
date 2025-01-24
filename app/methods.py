from transformers import TrOCRProcessor, VisionEncoderDecoderModel, T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import torch
import google.generativeai as genai
from keybert import KeyBERT

# Initialize TrOCR
trocr_device = torch.device('cpu')
trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', use_fast=True)
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten').to(trocr_device)

kw_model = KeyBERT(model='all-MiniLM-L6-v2')

aqg_model = T5ForConditionalGeneration.from_pretrained("./T5_model")
aqg_tokenizer = T5Tokenizer.from_pretrained("./T5_model")

genai.configure(api_key="AIzaSyA_dUIPPaNxppOHVXHzYaYEl65ytsl63bY")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def process_and_crop_image(image):
    """
    Processes an image to detect skew, correct rotation, and crop text regions.

    Args:
        image (numpy.ndarray): Input image array.

    Returns:
        list: List of cropped image arrays for each detected text region.
    """
    # Edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # Function to calculate a fine-tuned rotation angle
    def fine_tuned_angle(lines):
        angles = []
        if lines is not None:
            for rho, theta in lines[:, 0]:
                # Only consider lines with theta close to horizontal alignment (±20 degrees from 90)
                if np.abs(theta - np.pi / 2) < np.deg2rad(20):
                    angle = (theta - np.pi / 2) * (180 / np.pi)
                    angles.append(angle)
        return np.median(angles) if angles else 0

    # Get the fine-tuned rotation angle
    adjusted_angle = fine_tuned_angle(lines)

    # Rotate the image with the adjusted angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix_adjusted = cv2.getRotationMatrix2D(center, adjusted_angle, 1.0)
    adjusted_rotated_image = cv2.warpAffine(image, rotation_matrix_adjusted, (w, h),
                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite("./debug/adjusted_rotated_image.png", adjusted_rotated_image) # Debug rotated

    # Threshold the image to binary
    _, thresholded_image = cv2.threshold(adjusted_rotated_image, 127, 255, cv2.THRESH_BINARY)

    # Duplicate the thresholded image for cropping basis
    thresholded_for_cropping = thresholded_image.copy()

    # Invert binary image for text isolation
    _, binary = cv2.threshold(thresholded_image, 127, 255, cv2.THRESH_BINARY_INV)

    # Apply dilation to isolate text mass
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    dilated = cv2.dilate(binary, kernel, iterations=3)
    cv2.imwrite("./debug/dilated.png", dilated) # Debug dilated

    # Find contours of the text mass
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from top to bottom based on their y-coordinate
    sorted_contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[1])

    # Extract cropped regions
    cropped_regions = []
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = thresholded_for_cropping[y:y + h, x:x + w]

        # Apply less intense morphological closing to bridge character gaps
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cropped_closed = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, closing_kernel)

        # Convert single-channel binary image to 3-channel
        cropped_regions.append(cv2.merge([cropped_closed, cropped_closed, cropped_closed]))

    return cropped_regions

def batch_infer_trocr(cropped_images):
    """
    Performs batched inference on cropped images using the TrOCR model.

    Args:
        cropped_images (list): List of cropped image arrays.

    Returns:
    """


    def extract_response(response):
      """Extracts the text response from a protos.GenerateContentResponse object.

      Args:
        response: The protos.GenerateContentResponse object.

      Returns:
        The extracted text response as a string.
      """
      candidate = response.candidates[0]
      full_text = ""
      for part in candidate.content.parts:
        full_text += part.text
      return full_text
    
    recognized_texts = []
    for cropped_image in tqdm(cropped_images, desc="Processing images"):
        # Convert image to PIL format
        pil_image = Image.fromarray(cropped_image)

        # Preprocess the image
        inputs = trocr_processor(pil_image, return_tensors="pt").to(trocr_device)

        # Perform inference
        outputs = trocr_model.generate(**inputs)

        # Decode the output
        recognized_text = trocr_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        recognized_texts.append(recognized_text)

    # Join all recognized texts into a single string
    combined_text = " ".join(recognized_texts)


    # Correct the combined text using Gemini
    gemini_response = gemini_model.generate_content(f"The following text is a collection of OCR extracted keywords possibly mispelled or incorrect. Based on the keywords, try to create a recreated paragraph based on the information you know regarding the keywords. If you think there is a mistake on the words or text, please correct it based on what you know. Note that there may be mispelled acronyms and such.: {combined_text}. Do not reference anything diagrammatical nor provide any filler analysis words on your end. Also, ignore the part in the text that contains a lot of 0's")

    corrected_text = extract_response(gemini_response)
    print(combined_text)

    return corrected_text

def aqg(text):
    """
    Generates questions from a given text using the AQG model.

    Args:
        text (str): Input text for generating questions.

    Returns:
        list: List of generated questions.
    """
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 5), stop_words=None, top_n=10)
    keyword_array = [keyword for keyword, score in keywords]
    questions = []

    for k in keyword_array:
        inputs = f"Answer: {k} | Context: {text}"

        input_ids = aqg_tokenizer.encode(inputs, return_tensors="pt", max_length=512, truncation=True).to('cpu')
        outputs = aqg_model.generate(input_ids, max_length=128, do_sample=True, top_p=0.98, temperature=0.9, early_stopping=True)
        decoded = aqg_tokenizer.decode(outputs[0], skip_special_tokens=True)

        questions.append(f"{decoded} hint: {k}")

    return questions