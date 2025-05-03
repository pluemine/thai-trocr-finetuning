import os
import random
import re
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from augraphy import NoisyLines, LightingGradient, BleedThrough, LowInkRandomLines
import time
import multiprocessing
import aiofiles
import asyncio
from io import BytesIO

# Configuration Constants
NUM_PROCESSES = 32
TASK_NAME = "currency"
FONTS_FOLDER = "/project/lt200324-optmul/pluem/version_3/font/general_fonts"
NOSYMBOL_FONTS_FOLDER = "/project/lt200324-optmul/pluem/version_3/font/nosymbol_fonts"
BACKGROUND_LIGHT_FOLDER = "/project/lt200324-optmul/pluem/version_3/bg_images/light"
BACKGROUND_DARK_FOLDER = "/project/lt200324-optmul/pluem/version_3/bg_images/dark"
INPUT_TXT_FOLDER = f"/project/lt200324-optmul/pluem/version_3/dataset/{TASK_NAME}"
OUTPUT_IMAGE_DIR = f"/project/lt200324-optmul/pluem/version_3/images/{TASK_NAME}/"
OUTPUT_IMAGE_NAME = TASK_NAME
CHECKPOINT_DIR = f"/project/lt200324-optmul/pluem/version_3/checkpoint/{TASK_NAME}/"
FONT_SIZE_LIST = [64, 72]
PADDING = 32
CHECKPOINT_INTERVAL = 1000
BACKGROUND_TYPE = ["black", "light", "light"]
LIGHT_BACKGROUND_COLORS = [
    (255, 255, 255),
    (230, 230, 230),
    (205, 205, 205),
    (232, 220, 184),
    (201, 193, 181),
]
LIGHT_FONT_COLORS = [
    (0, 0, 0),
    (100, 0, 0),
    (0, 100, 0),
    (0, 0, 100),
    (50, 50, 50),
]
DARK_BACKGROUND_COLORS = [(0, 0, 0), (24, 0, 0), (0, 24, 0), (0, 0, 24)]
DARK_FONT_COLORS = [(255, 255, 255), (200, 200, 200)]

# Get the list of all font files in each folder
font_files = [
    os.path.join(FONTS_FOLDER, f)
    for f in os.listdir(FONTS_FOLDER)
    if f.endswith(".ttf")
]
nosymbol_font_files = [
    os.path.join(NOSYMBOL_FONTS_FOLDER, f)
    for f in os.listdir(NOSYMBOL_FONTS_FOLDER)
    if f.endswith(".ttf")
]

FONT_NAME_RESTRICT = font_files
FONT_NAME = 2 * nosymbol_font_files + font_files

BACKGROUND_LIGHT_IMAGES = [
    f
    for f in os.listdir(BACKGROUND_LIGHT_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
BACKGROUND_DARK_IMAGES = [
    f
    for f in os.listdir(BACKGROUND_DARK_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

SYMBOL_REGEX = re.compile(r"[^\w\s]", re.UNICODE)

# Augmentation configurations
NOISY_LINES = NoisyLines(
    noisy_lines_number_range=(1, 4), noisy_lines_thickness_range=(1, 1)
)
LIGHTING_GRADIENT = LightingGradient(max_brightness=180, min_brightness=80)
BLEED_THROUGH = BleedThrough(intensity_range=(0.2, 0.5), color_range=(120, 200))
LOW_INK_RANDOM_LINES = LowInkRandomLines(count_range=(1, 3))


def augment_img(img):
    """
    Apply augmentations to an image.

    Args:
        img (PIL.Image): Input image.

    Returns:
        PIL.Image: Augmented image.
    """
    if random.randint(1, 5) > 4:
        return img

    img = np.asarray(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if random.randint(1, 5) == 1:
        img = cv2.dilate(img, kernel, iterations=random.randint(1, 1))

    transform = A.Compose(
        [
            A.Affine(
                shear=random.randint(-2, 2),
                mode=cv2.BORDER_CONSTANT,
                cval=(255, 255, 255),
                p=0.75,
            ),
        ]
    )

    img = transform(image=img)["image"]

    if random.random() > 0.25:
        img = NOISY_LINES(img)
    if random.random() > 0.25:
        img = LIGHTING_GRADIENT(img)
    if random.random() > 0.25:
        img = BLEED_THROUGH(img)
    if random.random() > 0.25:
        img = LOW_INK_RANDOM_LINES(img)

    return Image.fromarray(img)


def create_image(text, font_path, font_size, background_color, font_color):
    """
    Create an image with specified text, font, and colors.

    Args:
        text (str): Text to be rendered on the image.
        font_path (str): Path to the font file.
        font_size (int): Size of the font.
        background_color (tuple): RGB values for the background color.
        font_color (tuple): RGB values for the font color.

    Returns:
        PIL.Image: Image with rendered text.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    dummy_img = Image.new("RGB", (1, 1), color=(255, 255, 255))
    d = ImageDraw.Draw(dummy_img)

    # Get the bounding box of the text
    left, top, right, bottom = d.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top

    image_width = text_width + 2 * PADDING
    image_height = text_height + 2 * PADDING

    img = Image.new("RGB", (image_width, image_height), color=background_color)
    d = ImageDraw.Draw(img)

    # Calculate text position
    x = PADDING - left  # Adjust for any negative left offset
    y = PADDING - top   # Adjust for any negative top offset

    # Draw the text
    d.text((x, y), text, fill=font_color, font=font)

    return img


def create_image_with_transparency(text, font_path, font_size, font_color):
    """
    Create an image with specified text, font, and transparent background.

    Args:
        text (str): Text to be rendered on the image.
        font_path (str): Path to the font file.
        font_size (int): Size of the font.
        font_color (tuple): RGB values for the font color.

    Returns:
        PIL.Image: Image with rendered text and a transparent background.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Create a dummy image to calculate text size
    dummy_img = Image.new("RGBA", (1, 1), color=(255, 255, 255, 0))
    d = ImageDraw.Draw(dummy_img)

    # Get the bounding box of the text
    left, top, right, bottom = d.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top

    # Calculate image dimensions
    image_width = text_width + 2 * PADDING
    image_height = text_height + 2 * PADDING

    # Create the actual image with transparent background
    img = Image.new("RGBA", (image_width, image_height), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)

    # Calculate text position
    x = PADDING - left  # Adjust for any negative left offset
    y = PADDING - top   # Adjust for any negative top offset

    # Draw the text
    d.text((x, y), text, fill=font_color, font=font)

    return img


def crop_text_from_image(img):
    """
    Crop the text region from an image.

    Args:
        img (PIL.Image): Input image.

    Returns:
        PIL.Image: Cropped image containing only the text region.
    """
    img_cv = np.array(img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img  # Return original image if no contours are found

    x_min, y_min, x_max, y_max = img_cv.shape[1], img_cv.shape[0], 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x + w), max(y_max, y + h)

    # Apply padding to ensure no text is cropped out
    x_min, y_min = max(x_min - 16, 0), max(y_min - 16, 0)
    x_max, y_max = min(x_max + 16, img_cv.shape[1]), min(y_max + 16, img_cv.shape[0])

    # Check for invalid cropping dimensions
    if x_max - x_min <= 0 or y_max - y_min <= 0:
        return img  # Return original image if the dimensions are invalid

    cropped_img_cv = img_cv[y_min:y_max, x_min:x_max]
    cropped_img = Image.fromarray(cropped_img_cv)

    return cropped_img


def apply_background_to_cropped_image(cropped_img, background_image_path):
    """
    Apply a background image to a cropped image with text.

    Args:
        cropped_img (PIL.Image): The cropped image containing the text.
        background_image_path (str): Path to the background image.

    Returns:
        PIL.Image: Image with the background applied.
    """
    try:
        background_img = Image.open(background_image_path).convert("RGB")

        # Check the size of the cropped image before resizing
        if cropped_img.size[0] > 0 and cropped_img.size[1] > 0:
            background_img = background_img.resize(cropped_img.size)
        else:
            raise ValueError("Invalid size for cropped_img: dimensions must be > 0")

        # Paste the cropped image onto the background
        background_img.paste(cropped_img, (0, 0), cropped_img)

        return background_img

    except Exception as e:
        print(f"Error in applying background: {e}")
        return cropped_img  # Return the cropped image in case of failure


async def save_image_async(img, image_path):
    """
    Save an image asynchronously to the specified path.

    Args:
        img (PIL.Image): Image to be saved.
        image_path (str): Path where the image will be saved.
    """
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    async with aiofiles.open(image_path, "wb") as out_file:
        await out_file.write(buffer.getvalue())


def get_last_checkpoint(process_id):
    """
    Retrieve the last checkpoint index for a specific process.

    Args:
        process_id (int): ID of the process.

    Returns:
        int: Last checkpoint index.
    """
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_{process_id}.txt")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as file:
            lines = file.readlines()
            # Only the latest index is needed for resuming
            latest_index = int(lines[0].strip())
            return latest_index
    return -1


def save_checkpoint(process_id, idx, completed_jobs, all_jobs):
    """
    Save the current checkpoint index and job status.

    Args:
        process_id (int): ID of the process.
        idx (int): Current index.
        completed_jobs (int): Number of completed jobs.
        all_jobs (int): Total number of jobs.
    """
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_{process_id}.txt")
    with open(checkpoint_file, "w") as file:
        file.write(f"{idx}\nCompleted: {completed_jobs}\nTotal: {all_jobs}")


async def generate_image_from_text_async(text, idx, output_dir):
    """
    Generate an image from the provided text asynchronously.

    Args:
        text (str): Text to be rendered on the image.
        idx (int): Index for the output image file name.
        output_dir (str): Directory where the image will be saved.
    """
    contains_symbol = SYMBOL_REGEX.search(text) is not None

    # Choose font based on presence of symbols
    selected_font = FONT_NAME_RESTRICT if contains_symbol else FONT_NAME
    font_path = selected_font[random.randint(0, len(selected_font) - 1)]
    font_size = FONT_SIZE_LIST[random.randint(0, len(FONT_SIZE_LIST) - 1)]

    # Choose background and font colors
    select_background_type = BACKGROUND_TYPE[
        int(random.random() * len(BACKGROUND_TYPE))
    ]
    background_color = (
        LIGHT_BACKGROUND_COLORS[int(random.random() * len(LIGHT_BACKGROUND_COLORS))]
        if select_background_type == "light"
        else DARK_BACKGROUND_COLORS[int(random.random() * len(DARK_BACKGROUND_COLORS))]
    )
    font_color = (
        LIGHT_FONT_COLORS[int(random.random() * len(LIGHT_FONT_COLORS))]
        if select_background_type == "light"
        else DARK_FONT_COLORS[int(random.random() * len(DARK_FONT_COLORS))]
    )

    image_background = random.random() > 0.9
    if image_background:
        background_folder = (
            BACKGROUND_LIGHT_FOLDER
            if select_background_type == "light"
            else BACKGROUND_DARK_FOLDER
        )
        background_images = (
            BACKGROUND_LIGHT_IMAGES
            if select_background_type == "light"
            else BACKGROUND_DARK_IMAGES
        )

        selected_background_image = background_images[
            int(random.random() * len(background_images))
        ]
        random_image_path = os.path.join(background_folder, selected_background_image)

        img = create_image_with_transparency(text, font_path, font_size, font_color)
        img = crop_text_from_image(img)
        img = apply_background_to_cropped_image(img, random_image_path)
        img_augmented = augment_img(img)
    else:
        img = create_image(text, font_path, font_size, background_color, font_color)
        img = crop_text_from_image(img)
        img_augmented = augment_img(img)

    # Save the augmented image asynchronously
    image_path = os.path.join(output_dir, f"{OUTPUT_IMAGE_NAME}_{idx}.jpg")
    await save_image_async(img_augmented, image_path)


async def process_chunk_async(lines, output_dir, process_id, start_idx):
    """
    Process a chunk of text lines asynchronously and generate images.

    Args:
        lines (list): List of text lines to process.
        output_dir (str): Directory where images will be saved.
        process_id (int): ID of the process.
        start_idx (int): Starting index for the process.
    """
    output_dir = os.path.join(output_dir, f"thread{process_id}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    last_checkpoint = get_last_checkpoint(process_id)
    tasks = []

    for idx, text in enumerate(lines):
        global_idx = start_idx + idx
        if global_idx <= last_checkpoint:
            continue

        if text.strip():
            task = generate_image_from_text_async(text.strip(), global_idx, output_dir)
            tasks.append(task)

        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            await asyncio.gather(*tasks)
            tasks = []
            save_checkpoint(process_id, global_idx, idx + 1, len(lines))

    if tasks:
        await asyncio.gather(*tasks)
        save_checkpoint(process_id, global_idx, idx + 1, len(lines))


def split_work(input_folder, num_processes):
    """
    Split the combined content of all text files in the input folder into chunks for parallel processing.

    Args:
        input_folder (str): Path to the input folder containing text files.
        num_processes (int): Number of processes to split the work among.

    Returns:
        list: List of chunks, each chunk is a list of text lines.
    """
    lines = []

    # Iterate over all files in the folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                lines.extend(file.readlines())
    # lines = lines[:10000]
    print(f"Total number of lines: {len(lines)}")

    # Determine the size of each chunk
    chunk_size = len(lines) // num_processes
    chunks = [
        lines[i * chunk_size : (i + 1) * chunk_size] for i in range(num_processes)
    ]

    # Handle any remaining lines that don't fit evenly into chunks
    if len(lines) % num_processes != 0:
        chunks[-1].extend(lines[num_processes * chunk_size :])

    return chunks


def main():
    """
    Main function to start the image generation process.
    """

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    num_processes = NUM_PROCESSES  # Number of parallel processes

    chunks = split_work(INPUT_TXT_FOLDER, num_processes)

    processes = []
    start_idx = 0
    for i, chunk in enumerate(chunks):
        process = multiprocessing.Process(
            target=asyncio.run,
            args=(process_chunk_async(chunk, OUTPUT_IMAGE_DIR, i, start_idx),),
        )
        processes.append(process)
        process.start()

        start_idx += len(chunk)

    for process in processes:
        process.join()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Time execution is: {end - start}")
