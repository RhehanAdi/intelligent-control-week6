from ultralytics import YOLO
import cv2
import numpy as np
from canny_edge import canny_edge_detection  # Pastikan file canny_edge.py ada di folder yang sama

# Load YOLOv8 Instance Segmentation model
model = YOLO("yolov8n-seg.pt")

def combined_detection(image_path, output_path="combined_result.jpg"):
    """
    Menggabungkan Canny Edge Detection dengan YOLOv8 Lane Detection.
    
    Args:
        image_path (str): Path gambar input.
        output_path (str): Path untuk menyimpan hasil gabungan.
    
    Returns:
        str: Path file hasil gabungan.
    """
    # Jalankan Canny Edge Detection
    canny_result = canny_edge_detection(r"D:\Kuliah Politeknik Negeri Madiun\SEMESTER 7\8. PRAK. KONTROL CERDAS\6. Week 6\Hasil_Identifikasi\canny_sample_image.jpg")
    
    # Jalankan Lane Detection dengan YOLOv8-seg
    results = model(r"D:\Kuliah Politeknik Negeri Madiun\SEMESTER 7\8. PRAK. KONTROL CERDAS\6. Week 6\lane_detection_result.jpg")
    lane_img = results[0].plot()  # Plot hasil segmentasi YOLO
    
    # Baca hasil Canny Edge Detection
    edges = cv2.imread(canny_result, cv2.IMREAD_GRAYSCALE)
    
    # Overlay hasil Lane Detection dengan Canny Edge
    combined = cv2.addWeighted(lane_img, 0.7, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.3, 0)
    
    # Simpan dan tampilkan hasil
    cv2.imshow("Combined Detection", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite(output_path, combined)
    print(f"Hasil gabungan tersimpan di: {output_path}")
    
    return output_path

# Contoh penggunaan
combined_detection("Dataset/sample_image.jpg")
