import os
from PIL import Image

folder = '/Users/sumankar/Desktop/HSI_SSFTT/cls_SSFTT_IP/P-R_curve_comapring_3models_with_3_bithashlen'
losses = ['dsh', 'hashnet', 'greedyhash', 'idhn', 'csq', 'dpn', 'orthohash', 'dspch', 'dhnn']
bits = ['16', '32', '64']

# Find one image to get dimensions
sample_img = Image.open(os.path.join(folder, f'grid_trento_{losses[0]}_bits{bits[0]}_pr.png'))
img_w, img_h = sample_img.size

# We want 3 rows (bits) x 9 columns (losses)
grid_w = img_w * len(losses)
grid_h = img_h * len(bits)

grid_img = Image.new('RGB', (grid_w, grid_h), color='white')

for row_idx, bit in enumerate(bits):
    for col_idx, loss in enumerate(losses):
        filename = f'grid_trento_{loss}_bits{bit}_pr.png'
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            img = Image.open(filepath)
            # Paste into the grid
            x_offset = col_idx * img_w
            y_offset = row_idx * img_h
            grid_img.paste(img, (x_offset, y_offset))
        else:
            print(f"Missing: {filename}")

save_path = '/Users/sumankar/Desktop/HSI_SSFTT/Massive_PR_Comparison_Grid.png'
grid_img.save(save_path)
print(f"Successfully saved massive grid to {save_path}")
