from ultralytics import YOLO
import cv2

# Load model YOLOv8 Instance Segmentation
model = YOLO("yolov8n-seg.pt")

def detect_rail_lane(image_path, output_path="lane_detection_result.jpg"):
    """
    Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation.
    
    Args:
        image_path (str): Path gambar input.
        output_path (str): Path untuk menyimpan hasil deteksi.
    
    Returns:
        str: Path file hasil deteksi.
    """
    # Jalankan deteksi
    results = model(image_path, show=True)  # show=True akan menampilkan jendela hasil deteksi
    
    # Simpan hasil
    results[0].save(output_path)
    print(f"Hasil deteksi tersimpan di: {output_path}")
    
    return output_path

# Contoh penggunaan
detect_rail_lane("Dataset/sample_image.jpg")
