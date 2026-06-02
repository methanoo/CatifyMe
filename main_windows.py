
import cv2
import mediapipe as mp
import math
import numpy as np
from cairosvg import svg2png
from io import BytesIO
import os
import sys 
if getattr(sys, 'frozen', False):
    os.environ['PATH'] = sys._MEIPASS + os.path.pathsep + os.environ.get('PATH', '')

def resource_path(relative_path):
    """ Ottiene il percorso assoluto delle risorse, funziona sia in sviluppo che dentro l'EXE """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def svg_to_png(svg_path, width=200, height=200):
    if not os.path.exists(svg_path):
        print(f"ATTENZIONE: Il file {svg_path} non esiste!")
        return None
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

def load_svg_components(img_size=200):
    components = {
        'body': svg_to_png(resource_path('imgs/body.svg'), img_size, img_size),
        'mouth': {
            'rest': svg_to_png(resource_path('imgs/mouth_rest.svg'), img_size, img_size),
            'happy': svg_to_png(resource_path('imgs/mouth_happy.svg'), img_size, img_size),
            'shock': svg_to_png(resource_path('imgs/mouth_shock.svg'), img_size, img_size),
            'smile': svg_to_png(resource_path('imgs/mouth_smile.svg'), img_size, img_size),
        },
        'eyes': {
            'rest': svg_to_png(resource_path('imgs/eyes_rest.svg'), img_size, img_size),
            'shock': svg_to_png(resource_path('imgs/eyes_shock.svg'), img_size, img_size),
            'blink': svg_to_png(resource_path('imgs/eyes_blink.svg'), img_size, img_size),
            'blink_sx': svg_to_png(resource_path('imgs/eyes_sx_blink.svg'), img_size, img_size),
            'blink_dx': svg_to_png(resource_path('imgs/eyes_dx_blink.svg'), img_size, img_size),
        },
        'hands': {
            'hi_dx': svg_to_png(resource_path('imgs/hi_dx.svg'), img_size, img_size),
            'hi2_dx': svg_to_png(resource_path('imgs/hi2_dx.svg'), img_size, img_size),
            'hi_sx': svg_to_png(resource_path('imgs/hi_sx.svg'), img_size, img_size),
            'hi2_sx': svg_to_png(resource_path('imgs/hi2_sx.svg'), img_size, img_size),
            'hand_dx_rest': svg_to_png(resource_path('imgs/hand_dx_rest.svg'), img_size, img_size),
            'hand_sx_rest': svg_to_png(resource_path('imgs/hand_sx_rest.svg'), img_size, img_size),
            'thumb_dx': svg_to_png(resource_path('imgs/thumb_dx.svg'), img_size, img_size),
            'thumb_sx': svg_to_png(resource_path('imgs/thumb_sx.svg'), img_size, img_size),
        }
    }
    return components

def applica_overlay(background, overlay, x=0, y=0):
    if overlay is None:
        return background
    
    h_overlay, w_overlay = overlay.shape[:2]
    h_bg, w_bg = background.shape[:2]
    
    if y + h_overlay > h_bg or x + w_overlay > w_bg:
        x_end = min(x + w_overlay, w_bg)
        y_end = min(y + h_overlay, h_bg)
        overlay = overlay[0:(y_end - y), 0:(x_end - x)]
        h_overlay, w_overlay = overlay.shape[:2]
        if h_overlay <= 0 or w_overlay <= 0:
            return background
            
    if overlay.shape[2] == 4:
        overlay_bgr = overlay[:, :, :3]
        alpha_mask = overlay[:, :, 3] / 255.0
        roi = background[y:y+h_overlay, x:x+w_overlay]
        
        for c in range(3):
            roi[:, :, c] = (alpha_mask * overlay_bgr[:, :, c] + 
                            (1.0 - alpha_mask) * roi[:, :, c])
            
        if background.shape[2] == 4:
            background[y:y+h_overlay, x:x+w_overlay, 3] = np.maximum(background[y:y+h_overlay, x:x+w_overlay, 3], overlay[:, :, 3])
            
        background[y:y+h_overlay, x:x+w_overlay, 0:3] = roi[:, :, 0:3]
    else: 
        background[y:y+h_overlay, x:x+w_overlay, 0:overlay.shape[2]] = overlay
    
    return background

def detect_hand_gestures(hand_landmarks, handedness):
    punta_pollice = hand_landmarks.landmark[4]
    nocca_pollice = hand_landmarks.landmark[2]
    punta_indice = hand_landmarks.landmark[8]
    nocca_indice = hand_landmarks.landmark[6]
    
    pollice_up = punta_pollice.y < nocca_pollice.y
    indice_up = punta_indice.y < nocca_indice.y
    
    if pollice_up and indice_up:
        return 'hi'
    elif pollice_up and not indice_up: 
        return 'thumb'
    return None

def detect_face_expression(face_landmarks, frame_height, frame_width):
    eye_sx_sup = face_landmarks.landmark[386]
    eye_sx_inf = face_landmarks.landmark[374]
    eye_dx_sup = face_landmarks.landmark[159]
    eye_dx_inf = face_landmarks.landmark[145]
    
    viso_sx = face_landmarks.landmark[234]
    viso_dx = face_landmarks.landmark[454]
    larghezza_viso = math.hypot((viso_dx.x - viso_sx.x) * frame_width, (viso_dx.y - viso_sx.y) * frame_height)
    
    labbro_sup = face_landmarks.landmark[13]
    labbro_inf = face_landmarks.landmark[14]
    labbro_sx = face_landmarks.landmark[61]
    labbro_dx = face_landmarks.landmark[291]
    
    distanza_bocca_sugiu = math.hypot((labbro_inf.x - labbro_sup.x) * frame_width, (labbro_inf.y - labbro_sup.y) * frame_height)
    distanza_bocca_sxdx = math.hypot((labbro_dx.x - labbro_sx.x) * frame_width, (labbro_dx.y - labbro_sx.y) * frame_height)
    
    distanza_eye_dx = math.hypot((eye_dx_inf.x - eye_dx_sup.x) * frame_width, (eye_dx_inf.y - eye_dx_sup.y) * frame_height)
    distanza_eye_sx = math.hypot((eye_sx_inf.x - eye_sx_sup.x) * frame_width, (eye_sx_inf.y - eye_sx_sup.y) * frame_height)
    
    soglia_occhi_chiusi = larghezza_viso * 0.048
    soglia_occhi_shock = larghezza_viso * 0.075  
    soglia_bocca_larga = larghezza_viso * 0.38   
    soglia_bocca_aperta = larghezza_viso * 0.10  

    mouth_state = 'rest'
    eyes_state = 'rest'

    if distanza_eye_sx < soglia_occhi_chiusi and distanza_eye_dx > (distanza_eye_sx * 1.5):
        eyes_state = 'blink_dx'
    elif distanza_eye_dx < soglia_occhi_chiusi and distanza_eye_sx > (distanza_eye_dx * 1.5):
        eyes_state = 'blink_sx'
    elif distanza_eye_sx < soglia_occhi_chiusi and distanza_eye_dx < soglia_occhi_chiusi:
        eyes_state = 'blink'
    elif distanza_eye_dx > soglia_occhi_shock or distanza_eye_sx > soglia_occhi_shock:
        eyes_state = 'shock'
    else:
        eyes_state = 'rest'
    
    if distanza_bocca_sxdx > soglia_bocca_larga and distanza_bocca_sugiu < soglia_bocca_aperta:
        mouth_state = 'smile' 
    elif distanza_bocca_sxdx > soglia_bocca_larga and distanza_bocca_sugiu > soglia_bocca_aperta:
        mouth_state = 'happy' 
    elif distanza_bocca_sugiu > soglia_bocca_aperta:
        mouth_state = 'shock'
    else:
        mouth_state = 'rest'
    
    return mouth_state, eyes_state

def scegli_webcam():
    webcam_disponibili = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret: webcam_disponibili.append(i)
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
            if not ret: break

            frame = cv2.flip(frame, 1)
            preview = cv2.resize(frame, (800, 500))

            cv2.putText(preview, f"Webcam {camera_id} ({indice+1}/{len(webcam_disponibili)})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(preview, "A/D = cambia webcam | INVIO = conferma", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)

camera_id = scegli_webcam()
cap = cv2.VideoCapture(camera_id)

img_size = 200 
components = load_svg_components(img_size)

hand_positions = {'Left': None, 'Right': None} 
frame_counter = 0  
greeting_speed = 5  

eye_sx_closed_frames = 0 
eye_dx_closed_frames = 0  
WINK_THRESHOLD = 20  

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
    
    avatar_layer = np.zeros((img_size, img_size, 4), dtype=np.uint8)
    
    if components['body'] is not None:
        avatar_layer = applica_overlay(avatar_layer, components['body'], 0, 0)
    
    current_eyes = 'rest'
    current_mouth = 'rest'
    
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mouth_state, eyes_state = detect_face_expression(face_landmarks, frame_height, frame_width)
            
            if eyes_state == 'blink_sx':
                eye_sx_closed_frames += 1
                eye_dx_closed_frames = 0
                eyes_state = 'blink_sx' if eye_sx_closed_frames > WINK_THRESHOLD else 'rest'
            elif eyes_state == 'blink_dx':
                eye_dx_closed_frames += 1
                eye_sx_closed_frames = 0
                eyes_state = 'blink_dx' if eye_dx_closed_frames > WINK_THRESHOLD else 'rest'
            elif eyes_state == 'blink':
                eye_sx_closed_frames = 0
                eye_dx_closed_frames = 0
            else:
                eye_sx_closed_frames = 0
                eye_dx_closed_frames = 0

            current_eyes = eyes_state
            current_mouth = mouth_state
    else:
        eye_sx_closed_frames = 0
        eye_dx_closed_frames = 0

    if current_eyes in components['eyes'] and components['eyes'][current_eyes] is not None:
        avatar_layer = applica_overlay(avatar_layer, components['eyes'][current_eyes], 0, 0)
    if current_mouth in components['mouth'] and components['mouth'][current_mouth] is not None:
        avatar_layer = applica_overlay(avatar_layer, components['mouth'][current_mouth], 0, 0)
        
    hand_gestures = {} 
    hand_velocities = {} 
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks, handedness_info in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
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
            avatar_layer = applica_overlay(avatar_layer, components['hands']['hand_sx_rest'], 0, 0)
    
    if 'Right' not in hand_gestures or hand_gestures['Right'] is None:
        if components['hands']['hand_dx_rest'] is not None:
            avatar_layer = applica_overlay(avatar_layer, components['hands']['hand_dx_rest'], 0, 0)
    
    for hand_side, gesture in hand_gestures.items():
        if gesture is None: continue  
        hand_key_suffix = "sx" if hand_side == "Left" else "dx"
        
        if gesture == 'hi':
            velocity = hand_velocities.get(hand_side, 0)
            key = f'hi_{hand_key_suffix}' if velocity > 20 and (frame_counter // greeting_speed) % 2 == 0 else f'hi_{hand_key_suffix}'
            if velocity > 20 and (frame_counter // greeting_speed) % 2 != 0:
                key = f'hi2_{hand_key_suffix}'
                
            if key in components['hands'] and components['hands'][key] is not None:
                avatar_layer = applica_overlay(avatar_layer, components['hands'][key], 0, 0)
        
        elif gesture == 'thumb':
            key = f'thumb_{hand_key_suffix}'
            if key in components['hands'] and components['hands'][key] is not None:
                avatar_layer = applica_overlay(avatar_layer, components['hands'][key], 0, 0)
    
    frame = applica_overlay(frame, avatar_layer, 20, 20)
    
    cv2.imshow("Avatar SVG", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()