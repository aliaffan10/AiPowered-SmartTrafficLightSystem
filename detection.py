import torch
import cv2
import matplotlib.pyplot as plt

# === Load the Trained YOLOv5 Model ===
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.3  # Confidence threshold

# === Detection Function ===
def detect_image(image_path):
    print(f"\n📸 Detecting objects in: {image_path}")

    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(img_rgb)

    # Print results
    results.print()

    # Display image with bounding boxes
    results.render()  # draws boxes on image
    img_with_boxes = results.ims[0]  # get rendered image

    # Show image using matplotlib
    plt.imshow(img_with_boxes)
    plt.title("Detection Results")
    plt.axis("off")
    plt.show()

    # Optional: save output
    output_path = image_path.replace(".jpg", "_detected.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    print(f"✅ Output saved to {output_path}")


if __name__ == "__main__":
    detect_image("test_images/image2.jpeg")
