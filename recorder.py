import cv2
import numpy as np
import os
import time
import random

# --- DNN face detector paths ---
proto_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

face_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# --- HOG full-body detector ---
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# --- Initialize webcam ---
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20.0  # target FPS

# --- Recording variables ---
recording = False
frames_buffer = []
recorded_files = []  # track individual segment filenames
last_person_time = 0  # last time a person was detected
grace_period = 5.0  # seconds

# --- Folder for individual recordings ---
recordings_folder = "recordings"
os.makedirs(recordings_folder, exist_ok=True)

def generate_unique_filename(prefix="person_recording", ext=".mp4"):
    timestamp = int(time.time())
    rand = random.randint(1000, 9999)
    return f"{prefix}_{timestamp}_{rand}{ext}"

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # --- Face detection ---
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            faces.append((x1, y1, x2 - x1, y2 - y1))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, "Face", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    # --- Full body detection ---
    rects, _ = hog.detectMultiScale(frame, winStride=(8, 8),
                                    padding=(8, 8), scale=1.05)
    for (x, y, w2, h2) in rects:
        cv2.rectangle(frame, (x, y), (x + w2, y + h2), (255, 0, 0), 2)
        cv2.putText(frame, "Body", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    total_persons = len(faces) + len(rects)
    current_time = time.time()
    cv2.putText(frame, f"Persons Detected: {total_persons}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # --- Recording logic with 5s grace period ---
    if total_persons > 0:
        frames_buffer.append(frame.copy())
        last_person_time = current_time
        if not recording:
            recording = True
            print("Person detected. Recording started...")
    else:
        if recording:
            if current_time - last_person_time > grace_period:
                # Stop recording, save current segment in recordings folder
                filename = generate_unique_filename()
                filepath = os.path.join(recordings_folder, filename)
                out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'),
                                      fps, (frame_width, frame_height))
                for f in frames_buffer:
                    out.write(f)
                out.release()
                print(f"Person gone. Saved {filepath}")
                recorded_files.append(filepath)
                frames_buffer = []
                recording = False
            else:
                # Still within grace period, keep recording
                frames_buffer.append(frame.copy())

    cv2.imshow("Smart Person Detection Recording", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Save any ongoing recording
        if frames_buffer:
            filename = generate_unique_filename()
            filepath = os.path.join(recordings_folder, filename)
            out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'),
                                  fps, (frame_width, frame_height))
            for f in frames_buffer:
                out.write(f)
            out.release()
            recorded_files.append(filepath)
            print(f"Final recording saved: {filepath}")
        break

cap.release()
cv2.destroyAllWindows()

# --- Merge all recorded MP4s manually using OpenCV ---
if recorded_files:
    combined_filename = generate_unique_filename(prefix="combined_recording")
    out = cv2.VideoWriter(combined_filename, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (frame_width, frame_height))

    for ffile in recorded_files:
        temp_cap = cv2.VideoCapture(ffile)
        while True:
            ret, frame = temp_cap.read()
            if not ret:
                break
            out.write(frame)
        temp_cap.release()

    out.release()
    print(f"All clips merged into: {combined_filename}")

    # Save log of all individual clips inside recordings folder
    log_path = os.path.join(recordings_folder, "recorded_files.log")
    with open(log_path, "w") as log_file:
        for f in recorded_files:
            log_file.write(f + "\n")
    print(f"Recorded segment filenames saved to {log_path}")
