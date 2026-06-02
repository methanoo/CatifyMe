import cv2
import mediapipe as mp
import math
import numpy as np
from cairosvg import svg2png
from io import BytesIO
import os

def svg_to_png(svg_path, width=600, height=600):
    try:
        png_buffer = BytesIO()
        svg2png(url=svg_path, write_to=png_buffer, output_width=width, output_height=height)
        png_buffer.seek(0)
        
        nparr = np.frombuffer(png_buffer.getvalue(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        print(f"Errore nel convertire {svg_path}: {e}")
        return None

def load_svg_components(img_size=600):
    components = {
        'body': svg_to_png('imgs/body.svg', img_size, img_size),
        'mouth': {
            'rest': svg_to_png('imgs/mouth_rest.svg', img_size, img_size),
            'happy': svg_to_png('imgs/mouth_happy.svg', img_size, img_size),
            'shock': svg_to_png('imgs/mouth_shock.svg', img_size, img_size),
            'smile': svg_to_png('imgs/mouth_smile.svg', img_size, img_size),
        },
        'eyes': {
            'rest': svg_to_png('imgs/eyes_rest.svg', img_size, img_size),
            'shock': svg_to_png('imgs/eyes_shock.svg', img_size, img_size),
            'blink': svg_to_png('imgs/eyes_blink.svg', img_size, img_size),
            'blink_sx': svg_to_png('imgs/eyes_sx_blink.svg', img_size, img_size),
            'blink_dx': svg_to_png('imgs/eyes_dx_blink.svg', img_size, img_size),
        },
        'hands': {
            'hi_dx': svg_to_png('imgs/hi_dx.svg', img_size, img_size),
            'hi2_dx': svg_to_png('imgs/hi2_dx.svg', img_size, img_size),
            'hi_sx': svg_to_png('imgs/hi_sx.svg', img_size, img_size),
            'hi2_sx': svg_to_png('imgs/hi2_sx.svg', img_size, img_size),
            'hand_dx_rest': svg_to_png('imgs/hand_dx_rest.svg', img_size, img_size),
            'hand_sx_rest': svg_to_png('imgs/hand_sx_rest.svg', img_size, img_size),
            'thumb_dx': svg_to_png('imgs/thumb_dx.svg', img_size, img_size),
            'thumb_sx': svg_to_png('imgs/thumb_sx.svg', img_size, img_size),
        }
    }
    return components

def applica_overlay(background, overlay, x=0, y=0):
    if overlay is None:
        return background
    
    h_overlay, w_overlay = overlay.shape[:2]
    h_bg, w_bg = background.shape[:2]
    
    if y + h_overlay > h_bg or x + w_overlay > w_bg:
        return background
    
    if overlay.shape[2] == 4:
        overlay_bgr = overlay[:, :, :3]
        alpha_mask = overlay[:, :, 3] / 255.0
        roi = background[y:y+h_overlay, x:x+w_overlay]
        
        for c in range(3):
            roi[:, :, c] = (alpha_mask * overlay_bgr[:, :, c] + 
                            (1.0 - alpha_mask) * roi[:, :, c])
        background[y:y+h_overlay, x:x+w_overlay] = roi
    else: 
        background[y:y+h_overlay, x:x+w_overlay] = overlay
    
    return background

def detect_hand_gestures(hand_landmarks, handedness):

    punta_pollice = hand_landmarks.landmark[4]
    nocca_pollice = hand_landmarks.landmark[2]
    punta_indice = hand_landmarks.landmark[8]
    nocca_indice = hand_landmarks.landmark[6]
    punta_medio = hand_landmarks.landmark[12]
    nocca_medio = hand_landmarks.landmark[10]
    punta_anulare = hand_landmarks.landmark[16]
    nocca_anulare = hand_landmarks.landmark[14]
    punta_mignolo = hand_landmarks.landmark[20]
    nocca_mignolo = hand_landmarks.landmark[18]
    
    y_pollice_tip = punta_pollice.y
    y_pollice_base = nocca_pollice.y
    y_indice_tip = punta_indice.y
    y_indice_base = nocca_indice.y
    y_medio_tip = punta_medio.y
    y_medio_base = nocca_medio.y
    y_anulare_tip = punta_anulare.y
    y_anulare_base = nocca_anulare.y
    y_mignolo_tip = punta_mignolo.y
    y_mignolo_base = nocca_mignolo.y
    
    pollice_up = y_pollice_tip < y_pollice_base
    indice_up = y_indice_tip < y_indice_base
    medio_up = y_medio_tip < y_medio_base
    anulare_up = y_anulare_tip < y_anulare_base
    mignolo_up = y_mignolo_tip < y_mignolo_base
    
    if pollice_up and indice_up and medio_up and anulare_up and mignolo_up:
        return 'hi'
    elif pollice_up and not indice_up: 
        return 'thumb'
    
    return None

def detect_face_expression(face_landmarks, frame_height, frame_width):

    eye_sx_sup = face_landmarks.landmark[386]
    eye_sx_inf = face_landmarks.landmark[374]
    eye_dx_sup = face_landmarks.landmark[159]
    eye_dx_inf = face_landmarks.landmark[145]
    
    eyebrow_sx_1 = face_landmarks.landmark[70]
    eyebrow_sx_2 = face_landmarks.landmark[63]
    eyebrow_sx_3 = face_landmarks.landmark[105]
    eyebrow_sx_4 = face_landmarks.landmark[66]
    eyebrow_sx_5 = face_landmarks.landmark[107]
    
    eyebrow_dx_1 = face_landmarks.landmark[336]
    eyebrow_dx_2 = face_landmarks.landmark[296]
    eyebrow_dx_3 = face_landmarks.landmark[334]
    eyebrow_dx_4 = face_landmarks.landmark[293]
    eyebrow_dx_5 = face_landmarks.landmark[300]
    
    labbro_sup = face_landmarks.landmark[13]
    labbro_inf = face_landmarks.landmark[14]
    labbro_sx = face_landmarks.landmark[61]
    labbro_dx = face_landmarks.landmark[291]
    
    x_sx_sup, y_sx_sup = int(eye_sx_sup.x * frame_width), int(eye_sx_sup.y * frame_height)
    x_sx_inf, y_sx_inf = int(eye_sx_inf.x * frame_width), int(eye_sx_inf.y * frame_height)
    x_dx_sup, y_dx_sup = int(eye_dx_sup.x * frame_width), int(eye_dx_sup.y * frame_height)
    x_dx_inf, y_dx_inf = int(eye_dx_inf.x * frame_width), int(eye_dx_inf.y * frame_height)
    x_labbro_sup, y_labbro_sup = int(labbro_sup.x * frame_width), int(labbro_sup.y * frame_height)
    x_labbro_inf, y_labbro_inf = int(labbro_inf.x * frame_width), int(labbro_inf.y * frame_height)
    x_labbro_sx, y_labbro_sx = int(labbro_sx.x * frame_width), int(labbro_sx.y * frame_height)
    x_labbro_dx, y_labbro_dx = int(labbro_dx.x * frame_width), int(labbro_dx.y * frame_height)
    
    eyebrow_sx_y_avg = (eyebrow_sx_1.y + eyebrow_sx_2.y + eyebrow_sx_3.y + eyebrow_sx_4.y + eyebrow_sx_5.y) / 5
    eyebrow_dx_y_avg = (eyebrow_dx_1.y + eyebrow_dx_2.y + eyebrow_dx_3.y + eyebrow_dx_4.y + eyebrow_dx_5.y) / 5
  
    distanza_bocca_sugiu = math.hypot(x_labbro_inf - x_labbro_sup, y_labbro_inf - y_labbro_sup)
    distanza_bocca_sxdx = math.hypot(x_labbro_dx - x_labbro_sx, y_labbro_dx - y_labbro_sx)
    distanza_eye_dx = math.hypot(x_dx_inf - x_dx_sup, y_dx_inf - y_dx_sup)
    distanza_eye_sx = math.hypot(x_sx_inf - x_sx_sup, y_sx_inf - y_sx_sup)
    
    mouth_state = 'rest'
    eyes_state = 'rest'
    is_blinking = False
    eye_sx_closed = False
    eye_dx_closed = False
    
    eye_sx_closed = distanza_eye_sx < 10
    eye_dx_closed = distanza_eye_dx < 10
    
    if eye_sx_closed or eye_dx_closed:
        is_blinking = True
        
        if eye_sx_closed and eye_dx_closed:
            eyes_state = 'blink'
        elif eye_sx_closed:
            eyes_state = 'blink_dx'
        else:
            eyes_state = 'blink_sx'
    elif distanza_eye_dx > 25 and distanza_eye_sx > 25 and eyebrow_sx_y_avg >0.25 and eyebrow_dx_y_avg > 0.25:
        eyes_state = 'shock'
    else:
        eyes_state = 'rest'
    
    if distanza_bocca_sxdx > 140 and distanza_bocca_sugiu < 40:
        mouth_state = 'smile' 
    elif distanza_bocca_sxdx > 140 and distanza_bocca_sugiu > 40:
        mouth_state = 'happy' 
    elif distanza_bocca_sugiu > 40:
        mouth_state = 'shock'
    else:
        mouth_state = 'rest'
    
    return mouth_state, eyes_state, is_blinking

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)

components = load_svg_components(600)

hand_positions = {'Left': None, 'Right': None} 
frame_counter = 0  
greeting_speed = 5  

eye_sx_closed_frames = 0 
eye_dx_closed_frames = 0  
WINK_THRESHOLD = 30

def scegli_webcam():
    webcam_disponibili = []

    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                webcam_disponibili.append(i)
        cap.release()

    if not webcam_disponibili:
        raise Exception("Nessuna webcam trovata")

    indice = 0

    cv2.namedWindow("Selezione Webcam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Selezione Webcam", 800, 600)

    while True:
        camera_id = webcam_disponibili[indice]

        cap_preview = cv2.VideoCapture(camera_id)

        while True:
            ret, frame = cap_preview.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            preview = cv2.resize(frame, (800, 500))

            cv2.putText(
                preview,
                f"Webcam {camera_id} ({indice+1}/{len(webcam_disponibili)})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                preview,
                "A/D = cambia webcam | INVIO = conferma",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow("Selezione Webcam", preview)

            key = cv2.waitKey(1)

            if key == ord('a'):
                indice = (indice - 1) % len(webcam_disponibili)
                break
            if key == ord('d'):
                indice = (indice + 1) % len(webcam_disponibili)
                break
            if key == 13:
                cap_preview.release()
                cv2.destroyWindow("Selezione Webcam")
                return camera_id
            if key == 27:
                cap_preview.release()
                cv2.destroyAllWindows()
                exit()

        cap_preview.release()


camera_id = scegli_webcam()
cap = cv2.VideoCapture(camera_id)

print("Premi 'q' per uscire")

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    
    frame_height, frame_width, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    frame_counter += 1
    
    hand_results = hands.process(img_rgb)
    face_results = face_mesh.process(img_rgb)
    
    face_layer = components['body'].copy() if components['body'] is not None else frame.copy()
    
    if components['eyes']['rest'] is not None:
        face_layer = applica_overlay(face_layer, components['eyes']['rest'], 0, 0)
    if components['mouth']['rest'] is not None:
        face_layer = applica_overlay(face_layer, components['mouth']['rest'], 0, 0)
    
    frame = applica_overlay(frame, face_layer, 0, 0)
    
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mouth_state, eyes_state, is_blinking = detect_face_expression(face_landmarks, frame_height, frame_width)
            
            if eyes_state == 'blink_sx':
                eye_sx_closed_frames += 1
            else:
                eye_sx_closed_frames = 0
            
            if eyes_state == 'blink_dx':
                eye_dx_closed_frames += 1
            else:
                eye_dx_closed_frames = 0
            
            if eyes_state == 'blink_sx' and eye_sx_closed_frames > WINK_THRESHOLD:
                eyes_state = 'blink_dx'  
            elif eyes_state == 'blink_dx' and eye_dx_closed_frames > WINK_THRESHOLD:
                eyes_state = 'blink_sx' 

            face_layer = components['body'].copy() if components['body'] is not None else frame.copy()
            
            if eyes_state in components['eyes'] and components['eyes'][eyes_state] is not None:
                face_layer = applica_overlay(face_layer, components['eyes'][eyes_state], 0, 0)
            
            if mouth_state in components['mouth'] and components['mouth'][mouth_state] is not None:
                face_layer = applica_overlay(face_layer, components['mouth'][mouth_state], 0, 0)
            
            frame = applica_overlay(frame, face_layer, 0, 0)
    else:
        eye_sx_closed_frames = 0
        eye_dx_closed_frames = 0
    
    hand_gestures = {} 
    hand_velocities = {} 
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks, handedness_info in zip(hand_results.multi_hand_landmarks, 
                                                     hand_results.multi_handedness):
            hand_side = handedness_info.classification[0].label 
            gesture = detect_hand_gestures(hand_landmarks, hand_side)
            hand_gestures[hand_side] = gesture
            
            current_x = hand_landmarks.landmark[9].x * frame_width 
            
            if hand_positions[hand_side] is not None:
                velocity = abs(current_x - hand_positions[hand_side])
                hand_velocities[hand_side] = velocity
            
            hand_positions[hand_side] = current_x

    if 'Left' not in hand_gestures or hand_gestures['Left'] is None:
        if components['hands']['hand_sx_rest'] is not None:
            frame = applica_overlay(frame, components['hands']['hand_sx_rest'], 0, 0)
    
    if 'Right' not in hand_gestures or hand_gestures['Right'] is None:
        if components['hands']['hand_dx_rest'] is not None:
            frame = applica_overlay(frame, components['hands']['hand_dx_rest'], 0, 0)
    
    for hand_side, gesture in hand_gestures.items():
        if gesture is None:
            continue  
            
        hand_key_suffix = "sx" if hand_side == "Left" else "dx"
        
        if gesture == 'hi':
            velocity = hand_velocities.get(hand_side, 0)
            
            if velocity > 20:  
                if (frame_counter // greeting_speed) % 2 == 0:
                    key = f'hi_{hand_key_suffix}'
                else:
                    key = f'hi2_{hand_key_suffix}'
                if key in components['hands'] and components['hands'][key] is not None:
                    frame = applica_overlay(frame, components['hands'][key], 0, 0)
            else:
                key = f'hi_{hand_key_suffix}'
                if key in components['hands'] and components['hands'][key] is not None:
                    frame = applica_overlay(frame, components['hands'][key], 0, 0)
        
        elif gesture == 'thumb':
            key = f'thumb_{hand_key_suffix}'
            if key in components['hands'] and components['hands'][key] is not None:
                frame = applica_overlay(frame, components['hands'][key], 0, 0)
    
    cv2.imshow("Avatar SVG", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
