## Libraries Used

### 1. OpenCV (cv2)

- **Purpose**: OpenCV (Open Source Computer Vision) is a powerful open-source computer vision and machine learning library.
- **Usage**: Used for image and video processing tasks, such as reading and manipulating images, capturing and processing video frames, image filtering, and computer vision applications.

### 2. Django Rest Framework

- **Purpose**: Django Rest Framework (DRF) is a toolkit for building Web APIs in Django.
- **Components**:
  - **FileUploadParser**: Handles file uploads in API views.
  - **Video, Video1**: Django models representing database tables for video-related data.
  - **VideoSerializer, VideoSerializer1, VideoSerializer2**: Serializers that convert complex data types (e.g., models) to JSON for API communication.
- **Usage**: Creates RESTful APIs for handling video data, including uploading, retrieving, and serializing video-related information.

### 3. pathlib

- **Purpose**: The `pathlib` module provides an object-oriented interface for working with filesystem paths.
- **Usage**: Used for handling file paths and directories in a clean and platform-independent way.

### 4. PIL (Python Imaging Library)

- **Purpose**: PIL is a library for opening, manipulating, and saving various image file formats.
- **Usage**: Used for image processing tasks, especially if the application involves working with images.

### 5. gridfs

- **Purpose**: `gridfs` is a specification for storing and retrieving large files in MongoDB.
- **Usage**: Interacts with MongoDB to store and retrieve video files efficiently.

### 6. NumPy

- **Purpose**: NumPy is a powerful library for numerical operations in Python.
- **Usage**: Used for numerical operations, especially if there is image or video data involved.


###

Certainly! Below is a brief introduction and a README.md for the two API views provided:

### 1. VideoUploadViewFrames API

**Introduction:**
The `VideoUploadViewFrames` API is designed to handle video uploads, extract frames, and perform feature extraction on those frames. It then compares the extracted features with pre-existing features stored in a MongoDB collection. The API utilizes OpenCV for video processing, Django Rest Framework for API handling, and MongoDB (GridFS) for efficient storage and retrieval of video files and features.

**Usage:**
- Endpoint: `/VideoUploadViewFrames/`
- Method: POST

**Functionality:**
1. Accepts a video file upload with additional metadata (category, product, brand).
2. Processes the video, extracts frames, and performs feature extraction on each frame.
3. Compares the extracted features with pre-existing features in the MongoDB collection.
4. Returns a JSON response containing information about matching images and their probabilities.

### 2. LogoImageUploadView API

**Introduction:**
The `LogoImageUploadView` API is responsible for handling the upload of logo images in an administrative context. It saves the uploaded images, extracts features, and stores them in MongoDB (GridFS). This API is equipped with features for checking if features for an image already exist, and it returns appropriate responses based on the existence of features.

**Usage:**
- Endpoint: `/logo-upload-image/`
- Method: POST

**Functionality:**
1. Accepts an image file upload with additional metadata (category, product, brand, flag).
2. Saves the uploaded image and extracts features from it.
3. Checks if features for the image already exist in the MongoDB collection.
4. If features do not exist, the image is stored in GridFS, and features are stored in MongoDB.
5. Returns a JSON response indicating the success of the image and feature upload.

### README.md

```markdown
# Video Processing and Image Upload APIs

This project includes two APIs for video processing and image uploads. The APIs leverage OpenCV for video processing, Django Rest Framework for API handling, and MongoDB (GridFS) for efficient storage and retrieval of video files and features.

## 1. VideoUploadViewFrames API

### Usage

- Endpoint: `/VideoUploadViewFrames/`
- Method: POST

### Functionality

1. Accepts a video file upload with additional metadata (category, product, brand).
2. Processes the video, extracts frames, and performs feature extraction on each frame.
3. Compares the extracted features with pre-existing features in the MongoDB collection.
4. Returns a JSON response containing information about matching images and their probabilities.

## 2. LogoImageUploadView API

### Usage

- Endpoint: `/logo-upload-image/`
- Method: POST

### Functionality

1. Accepts an image file upload with additional metadata (category, product, brand, flag).
2. Saves the uploaded image and extracts features from it.
3. Checks if features for the image already exist in the MongoDB collection.
4. If features do not exist, the image is stored in GridFS, and features are stored in MongoDB.
5. Returns a JSON response indicating the success of the image and feature upload.
