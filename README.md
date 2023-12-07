```markdown
# Dowell LogoScan API's

## Postman Documentation
[Read the documentation on Postman](https://documenter.getpostman.com/view/29895764/2s9YkgC4qX)

## Video Upload and Logo Image Upload API

This API provides endpoints for uploading videos and logo images, extracting features, and performing image recognition.

### Video Upload Endpoint

#### Upload Video and Perform Image Recognition

- **Endpoint:**
  ```markdown
  POST https://uxlivinglab100110.pythonanywhere.com/VideoUploadViewFrames/
  ```

- **Payload:**
  - `api_key`: Your Dowell API Key
  - `category`: Category of the video
  - `brand`: Brand of the video
  - `product`: Product of the video
  - `video`: Video file (MP4 format, size < 30MB, duration <= 10.1 seconds)

- **Example (Postman):**
  1. Open Postman and set the request type to `POST`.
  2. Enter the endpoint URL: `https://uxlivinglab100110.pythonanywhere.com/VideoUploadViewFrames/`.
  3. In the `Body` tab, select `form-data`.
  4. Add the following key-value pairs:
     - `api_key`: Your Dowell API Key
     - `category`: YourCategory
     - `brand`: YourBrand
     - `product`: YourProduct
     - `video`: [Select your video file]

- **Example (Python - using `requests` library):**
  ```python
  import requests

  url = "https://uxlivinglab100110.pythonanywhere.com/VideoUploadViewFrames/"
  files = {'video': open(r'C:\Users\user\Downloads\video.mp4', 'rb')}
  data = {
      'api_key': 'Your Dowell API Key',
      'category': 'YourCategory',
      'brand': 'YourBrand',
      'product': 'YourProduct'
  }

  response = requests.post(url, files=files, data=data)
  print(response.json())
  ```

### Logo Image Upload Endpoint

#### Upload Logo Image and Extract Features

- **Endpoint:**
  ```markdown
  POST https://uxlivinglab100110.pythonanywhere.com/logo-upload-image/
  ```

- **Payload:**
  - `api_key`: Your Dowell API Key
  - `category`: Category of the image
  - `brand`: Brand of the image
  - `product`: Product of the image
  - `flag`: Flag for identification
  - `image`: Logo image file

- **Example (Postman):**
  1. Open Postman and set the request type to `POST`.
  2. Enter the endpoint URL: `https://uxlivinglab100110.pythonanywhere.com/logo-upload-image/`.
  3. In the `Body` tab, select `form-data`.
  4. Add the following key-value pairs:
     - `api_key`: Your Dowell API Key
     - `category`: YourCategory
     - `brand`: YourBrand
     - `product`: YourProduct
     - `flag`: YourFlag
     - `image`: [Select your logo image file]

- **Example (Python - using `requests` library):**
  ```python
  import requests

  url = "https://uxlivinglab100110.pythonanywhere.com/logo-upload-image/"
  files = {'image': open(r'c:\Users\Downloads\wallpaper.jpg', 'rb')}
  data = {
      'api_key': 'Your Dowell API Key',
      'category': 'YourCategory',
      'brand': 'YourBrand',
      'product': 'YourProduct',
      'flag': 'YourFlag'
  }

  response = requests.post(url, files=files, data=data)
  print(response.json())
  ```

Remember to replace placeholders like `Your Dowell API Key`, `YourCategory`, `YourBrand`, `YourProduct`, and others with your actual values.
```