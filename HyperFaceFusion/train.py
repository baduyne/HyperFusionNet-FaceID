# import torch
# from torch.utils.data import DataLoader, random_split
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.optim as optim
from torchvision.transforms.functional import rgb_to_grayscale
# import numpy as np
# from tqdm import tqdm
# import os

# # Import các lớp mô hình và dataset của bạn
# # CHỈ IMPORT HyperFacePipeline, pixel_loss, ssim_loss, gradient_loss
# # criterion_cls (CrossEntropyLoss) và ClassifierHead đã được loại bỏ khỏi mô hình
# from model import HyperFacePipeline, pixel_loss, ssim_loss, gradient_loss
# from dataset import HyperspectralFaceDataset

# # --- LỚP EARLY STOPPING ---
# class EarlyStopping:
#     """
#     Stops training if validation loss doesn't improve after a given patience.
#     """
#     def __init__(self, patience=7, verbose=False, delta=0, path='hyperface_best_model.pth'):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.inf
#         self.delta = delta
#         self.path = path

#     def __call__(self, val_loss, model):
#         score = -val_loss # Sử dụng âm của loss vì ta muốn loss càng nhỏ càng tốt

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, model):
#         '''Saves model when validation loss decrease.'''
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
#         torch.save(model.state_dict(), self.path)
#         self.val_loss_min = val_loss

# # --- LOAD DATASET ---
# transform = transforms.ToTensor()
# full_dataset = HyperspectralFaceDataset("./data/RGB_Thermal/rgb", "./data/RGB_Thermal/thermal", transform)

# train_size = int(0.8 * len(full_dataset))
# val_size = len(full_dataset) - train_size
# train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# # --- MODEL ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_classes_actual = 113 
# model = HyperFacePipeline(num_classes=num_classes_actual).to(device)


# if os.path.exists('hyperface_best_model.pth'):
#     print("Đang tải mô hình đã lưu từ 'hyperface_best_model.pth' để tiếp tục huấn luyện...")
#     model.load_state_dict(torch.load('hyperface_best_model.pth', map_location=device), strict=False) 

# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# early_stopping = EarlyStopping(patience=10, verbose=True, path='hyperface_best_model.pth')

# num_epochs = 400
# print(f"Bắt đầu huấn luyện trên {device}...")

# for epoch in range(num_epochs):
#     # --- Giai đoạn huấn luyện ---
#     model.train()
#     total_composite_loss = total_loss_pixel = total_loss_ssim = total_loss_gradient = 0

#     train_bar = tqdm(train_loader, desc=f"[Train {epoch+1}/{num_epochs}]")
#     for batch_idx, (ir, vis, labels) in enumerate(train_bar): # labels vẫn được tải nhưng không dùng để tính loss_cls
#         ir, vis, labels = ir.to(device), vis.to(device), labels.to(device)
#         optimizer.zero_grad()

#         # CHỈ LẤY embeddings VÀ fused_face TỪ MÔ HÌNH
#         embeddings, fused_face = model(ir, vis)

#         if ir.shape[1] == 3: ir = rgb_to_grayscale(ir)
#         if vis.shape[1] == 3: vis = rgb_to_grayscale(vis)
#         target_reconstruction = (ir + vis) / 2

#         loss_pixel = pixel_loss(fused_face, target_reconstruction)
#         loss_ssim = ssim_loss(fused_face, target_reconstruction)
#         loss_gradient = gradient_loss(fused_face)
#         composite_loss = loss_pixel + loss_ssim + loss_gradient

#         composite_loss.backward()
#         optimizer.step()

#         total_composite_loss += composite_loss.item()

#         total_loss_pixel += loss_pixel.item()
#         total_loss_ssim += loss_ssim.item()
#         total_loss_gradient += loss_gradient.item()

#         train_bar.set_postfix({
#             "loss": f"{total_composite_loss / (batch_idx + 1):.4f}",
#             "pxl": f"{total_loss_pixel / (batch_idx + 1):.4f}",
#             "ssim": f"{total_loss_ssim / (batch_idx + 1):.4f}",
#             "grad": f"{total_loss_gradient / (batch_idx + 1):.4f}"
#         })

#     avg_train_loss = total_composite_loss / len(train_loader)


#     # --- VALIDATION ---
#     model.eval()
#     val_total_loss = val_loss_pixel_total = val_loss_ssim_total = val_loss_gradient_total = 0

#     val_bar = tqdm(val_loader, desc=f"[Valid {epoch+1}/{num_epochs}]")
#     with torch.no_grad():
#         for batch_idx_val, (ir_val, vis_val, labels_val) in enumerate(val_bar): 
#             ir_val, vis_val, labels_val = ir_val.to(device), vis_val.to(device), labels_val.to(device)
#             embeddings_val, fused_face_val = model(ir_val, vis_val)
   
#             if ir_val.shape[1] == 3: ir_val = rgb_to_grayscale(ir_val)
#             if vis_val.shape[1] == 3: vis_val = rgb_to_grayscale(vis_val)
#             target_reconstruction_val = (ir_val + vis_val) / 2

#             val_loss_pixel = pixel_loss(fused_face_val, target_reconstruction_val)
#             val_loss_ssim = ssim_loss(fused_face_val, target_reconstruction_val)
#             val_loss_gradient = gradient_loss(fused_face_val)

#             val_composite_loss = val_loss_pixel + val_loss_ssim + val_loss_gradient
            
#             val_total_loss += val_composite_loss.item()
#             val_loss_pixel_total += val_loss_pixel.item()
#             val_loss_ssim_total += val_loss_ssim.item()
#             val_loss_gradient_total += val_loss_gradient.item()

#             val_bar.set_postfix({
#                 "loss": f"{val_total_loss / (batch_idx_val + 1):.4f}",
#                 "pxl": f"{val_loss_pixel_total / (batch_idx_val + 1):.4f}",
#                 "ssim": f"{val_loss_ssim_total / (batch_idx_val + 1):.4f}",
#                 "grad": f"{val_loss_gradient_total / (batch_idx_val + 1):.4f}"
#             })

#     avg_val_loss = val_total_loss / len(val_loader)

#     print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
#           f"Val Loss: {avg_val_loss:.4f}") 

#     early_stopping(avg_val_loss, model)
#     if early_stopping.early_stop:
#         print("Dừng sớm vì không cải thiện!")
#         break

# print("Huấn luyện hoàn tất!")
# print(f"Mô hình tốt nhất được lưu tại: {early_stopping.path}")