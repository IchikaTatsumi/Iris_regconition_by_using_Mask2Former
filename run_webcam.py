import cv2
import torch
import numpy as np
import json
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os

# Đảm bảo Python nhìn thấy thư mục src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import model class (Sửa lại đường dẫn import nếu cấu trúc thư mục của bạn khác)
try:
    from src.models.mask2former import EnhancedMask2Former
except ImportError:
    print("❌ LỖI: Không tìm thấy module 'src'. Hãy đảm bảo file này nằm ngang hàng với thư mục 'src'.")
    sys.exit(1)

class IrisSegmentor:
    def __init__(self, config_path, checkpoint_path):
        # Kiểm tra CUDA
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.device_name = torch.cuda.get_device_name(0)
            print(f"🚀 Hardware: {self.device_name} (Ready for RTX 3050 Optimization)")
        else:
            self.device = torch.device('cpu')
            print("⚠️ CẢNH BÁO: Không tìm thấy GPU. Tốc độ sẽ rất chậm!")

        # 1. Load Config
        print(f"📖 Loading config: {config_path}")
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # 2. Chuẩn bị config cho model (Loại bỏ các key thừa gây lỗi)
        model_cfg = self.config['model_config']
        
        # Xóa key 'use_checkpoint' nếu tồn tại (nguyên nhân gây lỗi trước đó)
        keys_to_remove = ['use_checkpoint']
        for key in keys_to_remove:
            if key in model_cfg:
                print(f"🔧 Removing incompatible key: {key}")
                del model_cfg[key]
        
        # 3. Khởi tạo Model
        print("🏗️ Initializing Model...")
        self.model = EnhancedMask2Former(**model_cfg)
        
        # 4. Load Weights (Trọng số)
        print(f"⚖️ Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval() # Bắt buộc: Chế độ Evaluation

        # 5. Cấu hình Transform (TỐI ƯU HÓA CHO RTX 3050)
        # 384x384 là điểm cân bằng tốt nhất giữa tốc độ và độ chính xác cho GPU 4GB
        self.img_size = 384 
        print(f"⚙️ Input Resolution set to: {self.img_size}x{self.img_size}")

        self.transform = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # Bảng màu: Class 0 (Trong suốt), Class 1 (Xanh lá - Iris)
        self.colors = np.array([
            [0, 0, 0],       # Background
            [0, 255, 0]      # Iris
        ], dtype=np.uint8)

    def predict(self, frame):
        """
        Dự đoán Mask từ frame ảnh (Webcam)
        """
        original_h, original_w = frame.shape[:2]

        # 1. Preprocess: Resize & Normalize
        # Chuyển BGR (OpenCV) -> RGB (Model)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image_rgb)
        x_tensor = augmented['image'].unsqueeze(0).to(self.device) # Shape: [1, 3, 384, 384]

        # 2. Inference (QUAN TRỌNG: Dùng FP16 để tăng tốc)
        with torch.no_grad():
            # Tự động dùng Mixed Precision (FP16) cho RTX 3050
            with torch.amp.autocast('cuda'): 
                outputs = self.model(x_tensor)
                
                # Xử lý output (tùy vào output của model là dict hay tensor)
                if isinstance(outputs, dict):
                     logits = outputs['pred_masks'] # Key phổ biến của Mask2Former
                else:
                     logits = outputs

                # Lấy class có xác suất cao nhất ngay trên GPU
                # [1, 2, H, W] -> [H, W]
                pred_mask = torch.argmax(logits, dim=1).squeeze(0)

        # 3. Post-process
        # Chuyển về CPU -> Numpy
        pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)
        
        # Resize mask về kích thước gốc của Webcam (dùng Nearest để giữ cạnh sắc nét)
        pred_mask_resized = cv2.resize(pred_mask_np, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        return pred_mask_resized

    def draw_overlay(self, frame, mask, alpha=0.4):
        """
        Vẽ mask chồng lên ảnh gốc
        """
        # Tạo ảnh màu từ mask index
        # mask có giá trị 0 hoặc 1. self.colors[mask] sẽ map ra màu tương ứng
        color_mask = self.colors[mask]
        
        # Chỉ blend màu tại vị trí mống mắt (mask == 1)
        iris_pixels = mask == 1
        
        overlay = frame.copy()
        # Công thức blend: img * (1-alpha) + mask * alpha
        overlay[iris_pixels] = cv2.addWeighted(
            frame[iris_pixels], 1-alpha, 
            color_mask[iris_pixels], alpha, 
            0
        )
        return overlay

# --- CHƯƠNG TRÌNH CHÍNH ---
def main():
    # --- CẤU HÌNH ĐƯỜNG DẪN (Sửa lại nếu tên file khác) ---
    CONFIG_PATH = 'configs/mask2former_config_kaggle.json'
    CHECKPOINT_PATH = 'checkpoints/best_checkpoint.pth' # Hoặc 'training_results/...'

    # Kiểm tra file tồn tại
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(CHECKPOINT_PATH):
        print("❌ LỖI: Không tìm thấy file Config hoặc Checkpoint!")
        print(f"   - Config: {CONFIG_PATH}")
        print(f"   - Checkpoint: {CHECKPOINT_PATH}")
        return

    # 1. Khởi tạo Model
    try:
        segmentor = IrisSegmentor(CONFIG_PATH, CHECKPOINT_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Mở Webcam
    print("🎥 Opening Webcam...")
    cap = cv2.VideoCapture(0)
    
    # Thiết lập độ phân giải Webcam (640x480 là chuẩn nhẹ nhất để hiển thị)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Không thể mở Webcam.")
        return

    print("\n" + "="*40)
    print("   NHẤN 'Q' ĐỂ THOÁT CHƯƠNG TRÌNH   ")
    print("="*40 + "\n")

    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame.")
            break

        # Flip gương để nhìn tự nhiên hơn
        frame = cv2.flip(frame, 1)

        # Đo FPS
        current_time = time.time()
        
        # --- CHẠY DỰ ĐOÁN ---
        mask = segmentor.predict(frame)
        
        # --- VẼ KẾT QUẢ ---
        result_frame = segmentor.draw_overlay(frame, mask)

        # Tính toán và hiển thị FPS
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        
        # Vẽ thông số lên màn hình
        cv2.putText(result_frame, f"FPS: {int(fps)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result_frame, f"Device: RTX 3050 (FP16)", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow('Real-time Iris Segmentation', result_frame)

        # Nhấn Q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Chương trình kết thúc.")

if __name__ == "__main__":
    main()