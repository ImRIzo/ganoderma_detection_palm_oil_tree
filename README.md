# Ganoderma Detection Project

This project is for detecting **Ganoderma** in oil palm plantations using YOLOv8.

---

## Setup Instructions

1. **Clone this repository:**

Download the dataset
Download the dataset from Roboflow: Ganoderma Detection Dataset
 as a ZIP file -> https://universe.roboflow.com/binus-ptpn4-ramadya/ganoderma-detection-oil-palm/dataset/3/download

2. **Extract the dataset:**
Extract the ZIP file into the project directory. After extraction, the project structure should look like this:

![Project Structure](project_structure.png?raw=true "Project Structure")


Install additional dependencies (if needed):

pip install -r requirements.txt


data/ contains the dataset split into train, valid, and test sets.

Each set has images/ and corresponding labels/ folders in YOLO format.

data.yaml contains paths, class names, and metadata.

**Visualization:**
![P](example.png?raw=true "P")
