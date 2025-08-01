# Nepali License Plate Recognition System (NLPR)
# Team Members
- **Astha Thapa** (221707)   
- **Sneha Mahat** (221742)   
- **Sonu Giri** (221743)  
- **Lamin Tamang** (221720) 

## Introduction
Automatic Nepali License Plate Recognition (NLPR) is a system capable of extracting license plate characters written in the Nepali Devanagari script. This project addresses the gap in existing recognition systems that primarily support Latin characters, making it challenging for traffic monitoring and vehicle identification in Nepal. Our solution combines YOLOv8 for license plate detection with a custom OCR engine using CRNN (CNN + LSTM) for Nepali character recognition.

## Objectives
1. **Vehicle Detection**: Detect vehicles from input images/videos using YOLOv8.
2. **License Plate Extraction**: Accurately localize and crop license plate regions.
3. **OCR for Nepali Script**: Recognize Devanagari characters.
4. **System Integration**: End-to-end pipeline with a user-friendly interface.

## Key Features
- **YOLOv8-Based Detection**: Real-time vehicle and license plate detection.
- **CRNN-OCR Engine**: Custom CNN + LSTM model optimized for Nepali characters.
- **Preprocessing Pipeline**: Image enhancement and noise reduction for better accuracy.
- **Web Interface**: React.js frontend with FastAPI backend for easy uploads and results.

## Methodology
### System Architecture
1. **Detection Phase**:  
   - YOLOv8 detects vehicles and extracts license plate bounding boxes.
   - Cropped license plates are preprocessed (grayscale, normalization, resizing).

2. **OCR Phase**:  
   - CRNN model processes the plate image:
     - **CNN Layers**: Extract spatial features.
     - **LSTM Layers**: Handle character sequences.
     - **CTC Loss**: Aligns predictions with variable-length labels.
       
3.  **Dataset**:
- **YOLOv8 Training**: 6,000 annotated images (buses, cars, bikes) split into train/val/test sets.
- **OCR Training**: 580 high-quality Nepali license plate images with Devanagari labels.

4. **System Integration**:
   - **Backend API** (FastAPI):
     - Single `/recognize` endpoint
     - Accepts image uploads
     - Returns JSON with detected text
   - **Frontend** (React):
     - Simple file upload form
     - Displays recognition results
     - Shows original image with bounding boxes
   - **Communication**:
     - HTTP POST requests
     - FormData for image transfer

# Installation

1. Clone the repository:
   ```bash
    https://github.com/im-sonu-giri/-Nepali_License_Plate_Recognition.git
    cd nlpr-system

2. Backend Setup 
   ```bash
    cd backend
    pip install -r requirements.txt
    uvicorn main:app --reload

3. Frontend Setup 
   ```bash  
    cd frontend
    npm install

4. OCR Setup
   ```bash
   cd ocr
   pip install -r requirements.txt
   python predict.py \
   --model weights/best_model.pt \
   --image cropped_plate.jpg \
   --device cpu  # or 'cuda' for GPU
## Images
## Tools & Technologies
| Category       | Tools/Libraries                          |
|----------------|------------------------------------------|
| **Frontend**   | React.js                                 |
| **Backend**    | FastAPI, Python                          |
| **ML Models**  | YOLOv8, PyTorch (CRNN), TensorFlow      |
| **Processing** | OpenCV, Albumentations (augmentation)    |

## Conclusion

This project’s primary goal is to develop a system that can swiftly and precisely read Nepali
car license plates automatically and translate it into our native script. To ensure that it functions properly with the Devanagari script we will be using deep learning and OCR. To keep
things quick and easy, the system is made to process one image at a time and handle clean,
high-quality images. Because of this, it works well in practical applications like parking
systems and traffic monitoring.

## Author
*© 2025 Team 200_OK |6th Semester |Nepal College of Information Technology (NCIT)*  
