import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import radon, rescale
from IPython.display import display
from ipywidgets import Button, HBox, VBox, Output, Layout


class ImageAlignment:
    def __init__(self, image_folder, output_folder, cropped):
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.cropped=cropped
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def compute_fft(self, image):
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        return fshift, magnitude_spectrum

    def find_fft_angle(self, power_spectrum):
        _, thresh = cv2.threshold(power_spectrum, 0.5 * power_spectrum.max(), 255, cv2.THRESH_BINARY)
        thresh = np.uint8(thresh)
        lines = cv2.HoughLines(thresh, 1, np.pi / 180, 100)
        if lines is not None:
            angles = [np.rad2deg(line[0][1]) - 90 for line in lines]
            return np.median(angles)
        return 0

    def compute_radon_transform(self, image, theta_range=np.arange(-90, 90, 1)):
        image_rescaled = rescale(image, scale=0.5, mode='reflect')
        sinogram = radon(image_rescaled, theta=theta_range, circle=False)
        return sinogram, theta_range

    def find_radon_angle(self, sinogram, theta_range):
        projection_sums = np.sum(sinogram, axis=0)
        return theta_range[np.argmax(projection_sums)]

    def calculate_contrast(self, image, translation):
        rows, cols = image.shape
        M_right = np.float32([[1, 0, translation], [0, 1, 0]])
        M_left = np.float32([[1, 0, -translation], [0, 1, 0]])
        translated_right = cv2.warpAffine(image, M_right, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        translated_left = cv2.warpAffine(image, M_left, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        return np.sum(np.abs(image - translated_right)) + np.sum(np.abs(image - translated_left))

    def find_translation_angle(self, image, angle_range=(-90, 90), step=1, translations=[5, 20, 50]):
        min_contrast, optimal_angle = float('inf'), 0
        for angle in range(angle_range[0], angle_range[1] + step, step):
            rows, cols = image.shape
            M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
            rotated = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
            total_contrast = sum(self.calculate_contrast(rotated, t) for t in translations)
            if total_contrast < min_contrast:
                min_contrast, optimal_angle = total_contrast, angle
        return optimal_angle

    def rotate_image(self, image, angle):
        rows, cols = image.shape
        M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
        return cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    def process_image(self, filename, method="fft"):
        image_path = os.path.join(self.image_folder, filename)
        image_in = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if not self.cropped:
            image = image_in[:int(image_in.shape[0] * 0.93), :]
        else:
            image = image_in

        if method == "fft":
            _, magnitude_spectrum = self.compute_fft(image)
            angle = self.find_fft_angle(magnitude_spectrum)
        elif method == "radon":
            sinogram, theta_range = self.compute_radon_transform(image)
            angle = self.find_radon_angle(sinogram, theta_range)
        elif method == "translation":
            angle = self.find_translation_angle(image)
        elif method == "crop_only":
            angle = 0
        elif method == "all":
            return self.process_all_methods(image, filename)  # New function f
        else:
            raise ValueError("Invalid method. Choose from 'fft', 'radon', or 'translation'.")
        rotated_image = self.rotate_image(image, angle)
        output_path = os.path.join(self.output_folder, filename)
        cv2.imwrite(output_path, rotated_image)
        return angle

    def process_all_images(self, method="fft"):
        angles = {}
        image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.tif'))]
        for filename in tqdm(image_files, desc=f"Processing Images ({method})"):
            angles[filename] = self.process_image(filename, method)
        print("Rotation Angles:", angles)
        return angles

    from ipywidgets import Button, HBox, VBox, Output, Layout



    def process_all_methods(self, image, filename):
        """Process an image using FFT, Radon, and Translation and allow the user to select the best output."""

        # Compute rotation angles using different methods
        _, magnitude_spectrum = self.compute_fft(image)
        angle_fft = self.find_fft_angle(magnitude_spectrum)

        sinogram, theta_range = self.compute_radon_transform(image)
        angle_radon = self.find_radon_angle(sinogram, theta_range)

        angle_translation = self.find_translation_angle(image)

        # Rotate images accordingly
        rotated_fft = self.rotate_image(image, angle_fft)
        rotated_radon = self.rotate_image(image, angle_radon)
        rotated_translation = self.rotate_image(image, angle_translation)

        # Create an output widget to display the images
        output = Output()

        with output:
            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
            ax[0].imshow(image, cmap='gray')
            ax[0].set_title("Original Image")
            ax[0].axis("off")

            ax[1].imshow(rotated_fft, cmap='gray')
            ax[1].set_title(f"FFT ({angle_fft:.2f}°)")
            ax[1].axis("off")

            ax[2].imshow(rotated_radon, cmap='gray')
            ax[2].set_title(f"Radon ({angle_radon:.2f}°)")
            ax[2].axis("off")

            ax[3].imshow(rotated_translation, cmap='gray')
            ax[3].set_title(f"Translation ({angle_translation:.2f}°)")
            ax[3].axis("off")

            plt.show()

        # Create buttons for selecting the angle
        btn_fft = Button(description="Select FFT")
        btn_radon = Button(description="Select Radon")
        btn_translation = Button(description="Select Trans.")

        # Create a container for the buttons
        buttons = HBox([btn_fft, btn_radon, btn_translation])

        # Create a container for the output and buttons
        container = VBox([output, buttons])

        # Display the container
        display(container)

        selected_angle = None

        # Function to set selected angle
        def select_angle(angle):
            nonlocal selected_angle
            selected_angle = angle
            output.clear_output(wait=True)  # Clear the output to hide the images

        # Set button click events
        btn_fft.on_click(lambda _: select_angle(angle_fft))
        btn_radon.on_click(lambda _: select_angle(angle_radon))
        btn_translation.on_click(lambda _: select_angle(angle_translation))

        # Wait for the user to select an angle
        while selected_angle is None:
            pass

        return selected_angle