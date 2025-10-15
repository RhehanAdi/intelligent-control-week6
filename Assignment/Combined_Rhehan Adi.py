import cv2
import numpy as np
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

# =======================
# PATH MODEL YOLOv8
# =======================
MODEL_PATH = r"D:\Kuliah Politeknik Negeri Madiun\SEMESTER 7\8. PRAK. KONTROL CERDAS\6. Week 6\Assignment\rail-dataset.pt"

# =======================
# MEMUAT MODEL
# =======================
try:
    model = YOLO(MODEL_PATH)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# =======================
# KATEGORI THRESHOLD
# =======================
THRESHOLDS = {
    "1": {"name": "Low", "low": 10, "high": 50},
    "2": {"name": "Optimal", "low": 30, "high": 120},
    "3": {"name": "High", "low": 80, "high": 200}
}

def pilih_kategori_threshold():
    print("Pilih kategori threshold:")
    for key, val in THRESHOLDS.items():
        print(f"{key}. {val['name']} (Low={val['low']}, High={val['high']})")
    choice = input("Masukkan pilihan (1/2/3): ").strip()
    if choice in THRESHOLDS:
        selected = THRESHOLDS[choice]
        print(f"Threshold digunakan: Low={selected['low']}, High={selected['high']}")
        return selected['low'], selected['high'], selected['name'].lower()
    else:
        print("Pilihan tidak valid. Menggunakan threshold Optimal.")
        selected = THRESHOLDS["2"]
        return selected['low'], selected['high'], selected['name'].lower()

# =======================
# FUNGSI MASK YOLO
# =======================
def get_yolo_mask(img):
    results = model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    predictions = results[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_model = np.zeros_like(gray)

    if hasattr(predictions, 'masks') and predictions.masks is not None:
        for mask in predictions.masks.xy:
            mask = np.array(mask, np.int32)
            cv2.fillPoly(mask_model, [mask], 255)
    elif hasattr(predictions, 'boxes') and len(predictions.boxes.xyxy) > 0:
        for box in predictions.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(mask_model, (x1,y1), (x2,y2), 255, thickness=cv2.FILLED)
    return mask_model

# =======================
# FUNGSI CANNNY + YOLO
# =======================
def process_frame_or_image(img, low_threshold, high_threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
    gray_eq = cv2.equalizeHist(gray_blur)
    edges = cv2.Canny(gray_eq, low_threshold, high_threshold)

    mask_model = get_yolo_mask(img)
    overlay = img.copy()

    results = model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    predictions = results[0]

    # Overlay mask / bounding box
    if hasattr(predictions, 'masks') and predictions.masks is not None:
        for mask in predictions.masks.xy:
            mask = np.array(mask, np.int32)
            cv2.fillPoly(overlay, [mask], (255,0,0))
            cv2.polylines(img, [mask], isClosed=True, color=(0,255,0), thickness=2)
    elif hasattr(predictions, 'boxes') and len(predictions.boxes.xyxy) > 0:
        for i, box in enumerate(predictions.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(overlay, (x1,y1),(x2,y2),(255,0,0), thickness=cv2.FILLED)
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            if hasattr(predictions.boxes,"cls"):
                label = f"Objek {int(predictions.boxes.cls[i])}"
                cv2.putText(img,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_bgr[mask_model==255] = [0,0,255]
    edges_bgr[(mask_model==0) & (edges==255)] = [255,255,255]

    combined = np.hstack((img, edges_bgr))
    return combined

# =======================
# MODE WEBCAM (tidak disimpan)
# =======================
def mode_webcam():
    low, high, category = pilih_kategori_threshold()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera tidak dapat diakses!")
        return
    print("Tekan 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break
        combined = process_frame_or_image(frame, low, high)
        cv2.imshow("Webcam: Canny + YOLO", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# =======================
# MODE FOLDER (tidak menyimpan grafik)
# =======================
def mode_folder():
    input_folder = input("Masukkan path folder input: ").strip()
    output_folder = input("Masukkan path folder output: ").strip()
    low, high, category = pilih_kategori_threshold()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg','.png','.jpeg')):
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)
            if img is None:
                print(f"Gagal membaca {filename}")
                continue
            combined = process_frame_or_image(img, low, high)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{name}_result_{category}{ext}")
            cv2.imwrite(output_path, combined)
            print(f"Hasil disimpan: {output_path}")

# =======================
# MODE SINGLE IMAGE (simpan hasil + grafik)
# =======================
def mode_single_image():
    image_path = input("Masukkan path gambar: ").strip()
    img = cv2.imread(image_path)
    if img is None:
        print("Gagal membaca gambar.")
        return
    output_folder = input("Masukkan path folder output: ").strip()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    low, high, category = pilih_kategori_threshold()

    combined = process_frame_or_image(img, low, high)
    name, ext = os.path.splitext(os.path.basename(image_path))
    output_path = os.path.join(output_folder, f"{name}_result_{category}{ext}")
    graph_path = os.path.join(output_folder, f"{name}_threshold_{category}.png")

    cv2.imshow("Canny + YOLO", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(output_path, combined)
    print(f"Hasil deteksi disimpan: {output_path}")

    analyze_threshold(image_path, mask_model=get_yolo_mask(img), save_path=graph_path)

# =======================
# ANALISIS THRESHOLD + GRAFIK
# =======================
def analyze_threshold(image_path, mask_model=None, save_path=None):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
    gray_eq = cv2.equalizeHist(gray_blur)

    if mask_model is None:
        mask_model = get_yolo_mask(img)

    low_thresh_list = range(10, 101, 10)
    high_thresh_list = range(50, 201, 20)
    results = []

    for low in low_thresh_list:
        for high in high_thresh_list:
            if high <= low:
                continue
            edges = cv2.Canny(gray_eq, low, high)
            edges_in_mask = np.sum(edges[mask_model==255] > 0)
            results.append((low, high, edges_in_mask))

    results = np.array(results)
    lows = results[:,0]
    highs = results[:,1]
    counts = results[:,2]

    plt.figure(figsize=(10,6))
    plt.tricontourf(lows, highs, counts, levels=14, cmap='plasma')
    plt.colorbar(label='Pixel Tepi di Area Objek YOLO')
    plt.xlabel('Low Threshold')
    plt.ylabel('High Threshold')
    plt.title('Pengaruh Threshold terhadap Canny + Mask YOLO')

    if save_path:
        plt.savefig(save_path)
        print(f"Grafik threshold disimpan: {save_path}")

    plt.show()

# =======================
# MENU UTAMA
# =======================
def main():
    print("Pilih mode:")
    print("1. Webcam real-time")
    print("2. Folder gambar")
    print("3. Single image")
    choice = input("Masukkan pilihan (1/2/3): ").strip()
    if choice == '1':
        mode_webcam()
    elif choice == '2':
        mode_folder()
    elif choice == '3':
        mode_single_image()
    else:
        print("Pilihan tidak valid!")

if __name__ == "__main__":
    main()
