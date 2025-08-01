# Nepali License Plate Recognition System (NLPR)

## Introduction
Automatic Nepali License Plate Recognition (NLPR) is a system capable of extracting license plate characters written in the Nepali Devanagari script. This project addresses the gap in existing recognition systems that primarily support Latin characters, making it challenging for traffic monitoring and vehicle identification in Nepal. Our solution combines YOLOv8 for license plate detection with a custom OCR engine using CRNN (CNN + LSTM) for Nepali character recognition.

## Objectives
1. **Vehicle Detection**: Detect vehicles from input images/videos using YOLOv8.
2. **License Plate Extraction**: Accurately localize and crop license plate regions.
3. **OCR for Nepali Script**: Recognize Devanagari characters using a CRNN model with CTC loss.
4. **System Integration**: Deploy an end-to-end pipeline with a user-friendly interface.

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

### Dataset
- **YOLOv8 Training**: 6,000 annotated images (buses, cars, bikes) split into train/val/test sets.
- **OCR Training**: 580 high-quality Nepali license plate images with Devanagari labels.


## Tools & Technologies
| Category       | Tools/Libraries                          |
|----------------|------------------------------------------|
| **Frontend**   | React.js                                 |
| **Backend**    | FastAPI, Python                          |
| **ML Models**  | YOLOv8, PyTorch (CRNN), TensorFlow      |
| **Processing** | OpenCV, Albumentations (augmentation)    |
| **Deployment** | Docker                                   |

# Installation

1. Clone the repository:
   ```bash
    git clone https://github.com/your-repo/nlpr-system.git
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
    npm run dev 
