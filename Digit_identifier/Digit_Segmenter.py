import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

class DigitSegmenter:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def get_excel_label(self, index):
        """Convert index (0,1,2...) to Excel-like column labels (A, B, ..., Z, AA, AB, ..., AZ, BA, BB, ...)."""
        label = ""
        while index >= 0:
            label = chr((index % 26) + ord('A')) + label
            index = (index // 26) - 1  # Move to the next letter position
        return label

    def segment_digits_with_labels(self, image_path, padding=5):
        """Segment digits from an image and label them."""
        image = cv2.imread(image_path)  # Read in color (BGR) to allow colored labels
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digits = []
        label_index = 0  # Start at 0 for 'A'

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Expand bounding box with padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)

            if w > 10 and h > 10:
                digit_image = gray[y:y+h, x:x+w]
                digit_image_resized = cv2.resize(digit_image, (28, 28))
                digit_image_normalized = digit_image_resized / 255.0
                digit_image_normalized = digit_image_normalized.reshape(1, 28, 28, 1)

                prediction = self.model.predict(digit_image_normalized)
                predicted_digit = np.argmax(prediction)

                # Label the bounding box with Excel column labels
                label = self.get_excel_label(label_index)
                digits.append((predicted_digit, label, (x, y, w, h)))

                # Draw the bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw label above the box
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Draw predicted digit below the box
                cv2.putText(image, str(predicted_digit), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                label_index += 1  # Move to the next label

        return image, digits

    def display_image(self, image):
        """ Display the segmented image with bounding boxes and labels """
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()