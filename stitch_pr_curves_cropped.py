import os
from PIL import Image, ImageChops

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im

folder = '/Users/sumankar/Desktop/HSI_SSFTT/cls_SSFTT_IP/P-R_curve_comapring_3models_with_3_bithashlen'
losses = ['csq', 'dpn', 'dsh', 'greedyhash', 'hashnet', 'idhn', 'orthohash', 'dspch', 'dhnn']
bits = ['16', '32', '64']

# Load and crop all images first
cropped_images = []
for row_idx, bit in enumerate(bits):
    row_imgs = []
    for col_idx, loss in enumerate(losses):
        filename = f'grid_trento_{loss}_bits{bit}_pr.png'
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            img = Image.open(filepath).convert('RGB')
            cropped = trim(img)
            row_imgs.append(cropped)
        else:
            print(f"Missing: {filename}")
    cropped_images.append(row_imgs)

# Find max width and height of cropped images to align them in a clean grid
max_w = max(img.size[0] for row in cropped_images for img in row)
max_h = max(img.size[1] for row in cropped_images for img in row)

# Add a small padding (e.g., 20 pixels) between plots
pad = 20

grid_w = (max_w + pad) * len(losses) - pad
grid_h = (max_h + pad) * len(bits) - pad

grid_img = Image.new('RGB', (grid_w, grid_h), color='white')

for r, row in enumerate(cropped_images):
    for c, img in enumerate(row):
        # Center the cropped image within its grid cell
        x_offset = c * (max_w + pad) + (max_w - img.size[0]) // 2
        y_offset = r * (max_h + pad) + (max_h - img.size[1]) // 2
        grid_img.paste(img, (x_offset, y_offset))

save_path = '/Users/sumankar/Desktop/HSI_SSFTT/Massive_PR_Comparison_Grid_Cropped.png'
grid_img.save(save_path)
print(f"Successfully saved cropped massive grid to {save_path}")
