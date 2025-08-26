import cv2
import os
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Disable symlink warning for Hugging Face models
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load pre-trained CLIP model and processor
model_name = "openai/clip-vit-base-patch16"
try:
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to preprocess images
def preprocess_image(image):
    """Resize and enhance the image for better face detection."""
    resized_image = cv2.resize(image, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)  # Reduce noise
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)  # Improve contrast
    return resized_image, gray_image

# Function to extract faces from an image
def extract_faces_from_image(file_name, output_folder):
    """Extract faces from a single image and save them to the output folder."""
    alg = r"C:\Users\sujat\OneDrive\Desktop\DIP\haarcascade_frontalface_default.xml"
    haar_cascade = cv2.CascadeClassifier(alg)

    img = cv2.imread(file_name)
    if img is None:
        print(f"Error: Unable to load image {file_name}.")
        return []

    # Preprocess the image
    resized_img, gray_img = preprocess_image(img)

    # Adjust Haar Cascade parameters for better detection
    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    os.makedirs(output_folder, exist_ok=True)
    face_files = []

    for i, (x, y, w, h) in enumerate(faces):
        cropped_image = resized_img[y:y + h, x:x + w]
        target_file_name = os.path.join(output_folder, f'face_{i}.jpg')
        cv2.imwrite(target_file_name, cropped_image)
        face_files.append(target_file_name)
        print(f"Saved: {target_file_name}")

    return face_files

# Function to calculate image embeddings
def calculate_image_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))  # Resize to a consistent size
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.get_image_features(**inputs)
        # Normalize the embedding for better comparison
        embedding = outputs.detach().numpy()[0]
        return embedding / np.linalg.norm(embedding)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to precompute dataset embeddings
def precompute_dataset_embeddings(dataset_folder):
    """Precompute and store embeddings for all images in the dataset folder."""
    embeddings = {}
    for file_name in os.listdir(dataset_folder):
        dataset_image_path = os.path.join(dataset_folder, file_name)
        embedding = calculate_image_embedding(dataset_image_path)
        if embedding is not None:
            embeddings[file_name] = embedding
    return embeddings

# Function to compare embeddings
def find_matching_faces(face_embedding, dataset_embeddings, threshold=0.85):
    """Compare a face embedding with all precomputed dataset embeddings."""
    best_match = None
    best_similarity = -1

    for file_name, dataset_embedding in dataset_embeddings.items():
        # Calculate cosine similarity
        similarity = np.dot(face_embedding, dataset_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = file_name

    # Use a stricter threshold for similarity
    if best_similarity > threshold:
        return best_match
    return None

# Main function
def main():
    input_image = input("Enter the path to the input image: ").strip()
    dataset_folder = input("Enter the path to the dataset folder: ").strip()
    output_folder = "extracted-faces"

    if not os.path.isfile(input_image):
        print("Error: The specified input image does not exist.")
        return

    if not os.path.isdir(dataset_folder):
        print("Error: The specified dataset folder does not exist.")
        return

    # Step 1: Precompute dataset embeddings
    print("Precomputing dataset embeddings...")
    dataset_embeddings = precompute_dataset_embeddings(dataset_folder)

    # Step 2: Extract faces from the input image
    face_files = extract_faces_from_image(input_image, output_folder)

    # Step 3: Compare each extracted face with the dataset folder
    for face_file in face_files:
        face_embedding = calculate_image_embedding(face_file)
        if face_embedding is not None:
            # Adjust threshold dynamically based on input image quality
            matching_file = find_matching_faces(face_embedding, dataset_embeddings, threshold=0.85)
            if matching_file:
                print(f"Match found: {matching_file}")
            else:
                print("No match found")

    print("Processing completed.")

if __name__ == "__main__":
    main()