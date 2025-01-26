import cv2
import numpy as np

def create_gradient_circle(radius, color):
    gradient_circle = np.zeros((radius*2, radius*2, 4), dtype=np.uint8)

    # Create gradient from center (fully opaque) to edges (fully transparent)
    for y in range(radius*2):
        for x in range(radius*2):
            distance = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)
            if distance < radius:
                alpha = 255 - int(255 * (distance / radius))
                gradient_circle[y, x] = [color[0], color[1], color[2], alpha]

    return gradient_circle

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if no overlap
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, 3] / 255.0
    alpha_inv = 1.0 - alpha

    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay_crop[:, :, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

# Example usage
background_img = cv2.imread(r"C:\Users\Bennett\Documents\WORKING_THESIS\RESOURCES\Examination Sample Images\TEMPLATE.png")
radius = 100
color = (0, 255, 0)  # Green color
gradient_circle = create_gradient_circle(radius, color)

# Convert the background to RGBA (if it's not already)
background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2BGRA)

# Overlay the gradient circle onto the background image
overlay_image_alpha(background_img, gradient_circle, (50, 50), gradient_circle)

# Save or display the result
cv2.imwrite('result.png', background_img)
cv2.imshow('Gradient Circle', background_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
