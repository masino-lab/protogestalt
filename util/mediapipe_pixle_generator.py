import cv2
import mediapipe as mp
import os


IMAGE_DIR = '/home/abamini/face2pheno/data/external/gestaltGAN'
ANNOTATED_IMAGE_DIR = '/home/abamini/face2pheno/data/external/gestaltGAN'  #  save annotated images
LANDMARK_SAVE_DIR = '/home/abamini/face2pheno/data/external/gestaltGAN'  # save landmarks

# Make sure save dir exists
os.makedirs(LANDMARK_SAVE_DIR, exist_ok=True)
os.makedirs(ANNOTATED_IMAGE_DIR, exist_ok=True) 

IMAGE_FILES = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        if image is None:
            print(f"Cannot read {file}")
            continue

        # Some UTKFace images might be grayscale, convert to BGR for processing
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        image_height, image_width, _ = image.shape

        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Print number of faces detected in this image
        num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
        print(f"{os.path.basename(file)}: {num_faces} face(s) detected")
        
        if not results.multi_face_landmarks:
            continue
            
        annotated_image = image.copy()
        for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
            # Save landmarks to a txt file
            landmark_filename = os.path.join(
                LANDMARK_SAVE_DIR,
                f'{os.path.splitext(os.path.basename(file))[0]}_landmarks.txt'
            )
            with open(landmark_filename, 'w') as lm_file:
                for i, lm in enumerate(face_landmarks.landmark):
                    pixel_x = int(lm.x * image_width)
                    pixel_y = int(lm.y * image_height)
                    
                    # Write the pixel coordinates to the file 
                    lm_file.write(f'{i},{pixel_x},{pixel_y}\n')
            
            print(f"Saved pixel landmarks to {landmark_filename}")

            # Drawing landmarks on the image (optional, for visualization)
            # OpenCV uses BGR: red = (0, 0, 255), yellow = (0, 255, 255)
            red_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=3)
            yellow_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=red_spec,
                connection_drawing_spec=yellow_spec)
                
        # Save annotated image
        output_path = os.path.join(
            ANNOTATED_IMAGE_DIR, f'annotated_{os.path.basename(file)}')
        cv2.imwrite(output_path, annotated_image)
        print(f'Saved annotated image to {output_path}')