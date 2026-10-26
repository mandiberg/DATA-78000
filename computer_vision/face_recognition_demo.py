
# call our imports, including face_recognition
from deepface import DeepFace
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# load the LFW dataset
from sklearn.datasets import fetch_lfw_people

# go get your people
# higher the min number, the less people will be returned
min_faces_per_person = 40
# Load LFW people with a minimum number of images per person
lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=0.4)

# Display some information about the dataset
print("Number of images in the dataset:", lfw_people.data.shape[0])
print("Number of people in the dataset:", len(np.unique(lfw_people.target)))
print("Number of features in each image:", lfw_people.data.shape[1])

def display_image(image, title="Image"):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def resize(image,window_height = 500):
    aspect_ratio = float(image.shape[1])/float(image.shape[0])
    window_width = window_height/aspect_ratio
    image = cv2.resize(image, (int(window_height),int(window_width)))
    return image

def ensure_rgb(image):
    """DeepFace expects RGB images with three channels; LFW images are often grayscale."""
    if len(image.shape) == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    return image

# Display the first face in the dataset using the dataset's actual image shape.
display_face = False

if display_face:
    first_face = lfw_people.images[0]
    plt.imshow(first_face, cmap='gray')
    plt.title(lfw_people.target_names[lfw_people.target[0]])
    plt.axis('off')
    plt.show()

target_name = "Arnold Schwarzenegger"
target_index = np.where(lfw_people.target_names == target_name)[0][0]
print(f"Index of {target_name} in target_names: {target_index}")

# Get the indices of all images of the target person
target_indices = np.where(lfw_people.target == target_index)[0]
print(f"Indices of all images of {target_name}: {target_indices}")

# display the first image of the target person
first_target_face = ensure_rgb(lfw_people.images[target_indices[0]])
# display_image(first_target_face, f"First image of {target_name}")

# pick a set of images that mix the target person and other people
# take the first num_target_images images of the target person and num_target_images images of other people
num_target_images = 5
# Get the first num_target_images images of the target person
target_faces = np.stack([ensure_rgb(face) for face in lfw_people.images[target_indices[:num_target_images]]])

# Get num_target_images images of other people
other_indices = np.where(lfw_people.target != target_index)[0]
other_faces = np.stack([ensure_rgb(face) for face in lfw_people.images[other_indices[:num_target_images]]])

# add the target faces and other faces together, keeping track of who each face belongs to
joined_data = [(face, target_name) for face in target_faces]
other_labels = [lfw_people.target_names[lfw_people.target[idx]] for idx in other_indices[:num_target_images]]
joined_data += list(zip(other_faces, other_labels))
# and put them in a random order
np.random.shuffle(joined_data)
joined_faces = np.stack([item[0] for item in joined_data])
joined_labels = [item[1] for item in joined_data]

model_list = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "Dlib", "ArcFace"]
model_index = 0  # You can change this index to use a different model from the model_list

# use deepface to encode the reference face
reference_face_encoding = DeepFace.represent(
    first_target_face,
    model_name=model_list[model_index],
    enforce_detection=False,
)[0]["embedding"]

# encode the joined_faces using deepface
joined_faces_encodings = [
    DeepFace.represent(face, model_name=model_list[model_index], enforce_detection=False)[0]["embedding"]
    for face in joined_faces
]

# compare the reference face encoding with each of the joined_faces_encodings
results = []
for encoding in joined_faces_encodings:
    # Compare the target face embedding to each candidate face embedding.
    # The cosine similarity is a measure of the angle between the two vectors.
    # A value of 1.0 means the vectors are identical, while 0.0 means they are orthogonal.
    # to compute the cosine similarity, we can use the dot product of the two vectors divided by the product of their magnitudes.
    # dot product of the two vectors means multiplying each corresponding element of the two vectors and summing the results.
    # The magnitude of a vector is the square root of the sum of the squares of its elements.
    # The sum of the squares of the elements of a vector is also known as the L2 norm or Euclidean norm.
    # more on norms here: https://www.geeksforgeeks.org/maths/vector-norms/
    similarity = np.dot(reference_face_encoding, encoding) / (
        np.linalg.norm(reference_face_encoding) * np.linalg.norm(encoding)
    )
    results.append(similarity)

# set a threshold for similarity to determine if the faces match
threshold = 0.375  # .375 is an ok default threshold for cosine similarity in DeepFace

# display the results in a grid
num_images = len(joined_faces)
plt.figure(figsize=(15, 5))
for i in range(num_images):
    ax = plt.subplot(2, num_images // 2, i + 1)
    ax.imshow(joined_faces[i], cmap='gray')
    ax.set_title(f"Similarity: {results[i]:.2f}\nMatch: {'Yes' if results[i] > threshold else 'No'}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(joined_labels[i], labelpad=10)
plt.tight_layout()
plt.show()
