import os
import shutil

# Thư mục gốc chứa tất cả subfolder
root_dir = "/home/coung/Project/cafe_repo/data/pretrain"

# Các định dạng ảnh hợp lệ
image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
# 1️⃣ Duyệt toàn bộ cây thư mục và gom ảnh về thư mục gốc
for subdir, dirs, files in os.walk(root_dir):
    # print("🚀 Bắt đầu di chuyển ảnh...")
    for file in files:
        if file.lower().endswith(image_exts):
            src_path = os.path.join(subdir, file)
            dst_path = os.path.join(root_dir, file)

            # Nếu file trùng tên, đổi tên để không ghi đè
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(file)
                count = 1
                while os.path.exists(os.path.join(root_dir, f"{base}_{count}{ext}")):
                    count += 1
                dst_path = os.path.join(root_dir, f"{base}_{count}{ext}")

            shutil.move(src_path, dst_path)

# 2️⃣ Sau khi di chuyển, xóa tất cả thư mục con (bottom-up để tránh lỗi)
for subdir, dirs, files in os.walk(root_dir, topdown=False):
    for d in dirs:
        folder_path = os.path.join(subdir, d)
        try:
            os.rmdir(folder_path)  # chỉ xóa nếu rỗng
        except OSError:
            # Nếu còn file rác, bỏ qua
            pass

print("✅ Hoàn tất: đã di chuyển toàn bộ ảnh về thư mục ./pretrain và xoá mọi thư mục con.")
