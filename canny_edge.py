import cv2
import os

def canny_edge_detection(image_path, output_folder="Hasil_Identifikasi", output_name=None, show_preview=False):
    """
    Mendeteksi tepi satu gambar menggunakan Canny Edge Detection
    dan menyimpannya ke folder output.
    
    Args:
        image_path (str): Path gambar input.
        output_folder (str): Folder untuk menyimpan hasil.
        output_name (str, optional): Nama file hasil. Jika None, gunakan 'canny_{nama_file}'.
        show_preview (bool, optional): Tampilkan preview hasil sebelum disimpan.
    
    Returns:
        str: Path file hasil deteksi tepi.
    """
    # Buat folder output jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Baca gambar
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan di path: {image_path}")
    
    # Blur dan deteksi tepi
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img_blur, 50, 150)
    
    # Tentukan nama output
    if output_name is None:
        filename = os.path.basename(image_path)
        output_name = f"canny_{filename}"
    output_path = os.path.join(output_folder, output_name)
    
    # Tampilkan preview jika diminta
    if show_preview:
        cv2.imshow("Canny Edge Detection", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Simpan hasil
    cv2.imwrite(output_path, edges)
    print(f"Berhasil memproses: {image_path} -> {output_path}")
    
    return output_path

# Contoh penggunaan
canny_edge_detection("Dataset/sample_image.jpg", show_preview=True)
