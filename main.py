import cv2
import mediapipe as mp
import math  

def applica_overlay(background, overlay, x=0, y=0):
    h_overlay, w_overlay, canali = overlay.shape
    h_bg, w_bg, _ = background.shape

    if y + h_overlay > h_bg or x + w_overlay > w_bg:
        return background

    if canali == 4:
        overlay_bgr = overlay[:, :, :3]
        alpha_mask = overlay[:, :, 3] / 255.0

        roi = background[y:y+h_overlay, x:x+w_overlay]

        for c in range(0, 3):
            roi[:, :, c] = (alpha_mask * overlay_bgr[:, :, c] + 
                            (1.0 - alpha_mask) * roi[:, :, c])
            
        background[y:y+h_overlay, x:x+w_overlay] = roi
    else:
        background[y:y+h_overlay, x:x+w_overlay] = overlay

    return background

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)

thumb = cv2.imread("imgs/thumb.png", cv2.IMREAD_UNCHANGED)
thumb = cv2.resize(thumb, (600, 600))
omg = cv2.imread("imgs/omg.png", cv2.IMREAD_UNCHANGED)
omg = cv2.resize(omg, (600, 600))
smile = cv2.imread("imgs/smile.png", cv2.IMREAD_UNCHANGED)
smile = cv2.resize(smile, (600, 600))
HUH = cv2.imread("imgs/HUH.png", cv2.IMREAD_UNCHANGED)
HUH = cv2.resize(HUH, (600, 600))
pwease = cv2.imread("imgs/pwease.png", cv2.IMREAD_UNCHANGED)
pwease = cv2.resize(pwease, (600, 600))
nerd = cv2.imread("imgs/nerd.png", cv2.IMREAD_UNCHANGED)
nerd = cv2.resize(nerd, (600, 600))
hello = cv2.imread("imgs/hello.png", cv2.IMREAD_UNCHANGED)
hello = cv2.resize(hello, (600, 600))

cap = cv2.VideoCapture(0)

print("premi q per uscire")

while True:
    success, frame = cap.read()
    if not success:
        break

    altezza_frame, larghezza_frame, _ = frame.shape

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    risultati1 = hands.process(img_rgb)
    risultati2 = face_mesh.process(img_rgb)

    if risultati2.multi_face_landmarks:
        for face_landmarks in risultati2.multi_face_landmarks:
            
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

            eye_sx_sup = face_landmarks.landmark[386]
            eye_sx_inf = face_landmarks.landmark[374]
            eye_dx_sup = face_landmarks.landmark[159]
            eye_dx_inf = face_landmarks.landmark[145]

            labbro_sup = face_landmarks.landmark[13]
            labbro_inf = face_landmarks.landmark[14]

            labbro_sx = face_landmarks.landmark[61]
            labbro_dx = face_landmarks.landmark[291]

            x_sx_sup, y_sx_sup = int(eye_sx_sup.x * larghezza_frame), int(eye_sx_sup.y * altezza_frame)
            x_sx_inf, y_sx_inf = int(eye_sx_inf.x * larghezza_frame), int(eye_sx_inf.y * altezza_frame)
            x_dx_sup, y_dx_sup = int(eye_dx_sup.x * larghezza_frame), int(eye_dx_sup.y * altezza_frame)
            x_dx_inf, y_dx_inf = int(eye_dx_inf.x * larghezza_frame), int(eye_dx_inf.y * altezza_frame)

            x_labbro_sup, y_labbro_sup = int(labbro_sup.x * larghezza_frame), int(labbro_sup.y * altezza_frame)
            x_labbro_inf, y_labbro_inf = int(labbro_inf.x * larghezza_frame), int(labbro_inf.y * altezza_frame)

            x_labbro_sx, y_labbro_sx = int(labbro_sx.x * larghezza_frame), int(labbro_sx.y * altezza_frame)
            x_labbro_dx, y_labbro_dx = int(labbro_dx.x * larghezza_frame), int(labbro_dx.y * altezza_frame)

            distanza_bocca_sugiu = math.hypot(x_labbro_inf - x_labbro_sup, y_labbro_inf - y_labbro_sup)
            distanza_bocca_sxdx = math.hypot(x_labbro_dx - x_labbro_sx, y_labbro_dx - y_labbro_sx)
            distanza_eye_dx = math.hypot(x_dx_inf - x_dx_sup, y_dx_inf - y_dx_sup)
            distanza_eye_sx = math.hypot(x_sx_inf - x_sx_sup, y_sx_inf - y_sx_sup)

            if distanza_bocca_sugiu > 30 and distanza_eye_dx > 30 and distanza_eye_sx > 30:
                 if HUH is not None:
                     frame = applica_overlay(frame, HUH, 0, 0)
            else:
                if distanza_eye_dx > 30 and distanza_eye_sx > 30 and distanza_bocca_sugiu < 30:
                    if pwease is not None:
                        frame = applica_overlay(frame, pwease, 0, 0)
                else:
                    if distanza_bocca_sxdx > 140 and distanza_bocca_sugiu > 30:
                        if smile is not None:
                            frame = applica_overlay(frame, smile, 0, 0)

                    if distanza_bocca_sugiu > 40:
                        if omg is not None:
                            frame = applica_overlay(frame, omg, 0, 0)
                
                    

    if risultati1.multi_hand_landmarks:
        for hand_landmarks in risultati1.multi_hand_landmarks:
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            punta_indice = hand_landmarks.landmark[8]
            nocca_indice = hand_landmarks.landmark[6]

            punta_pollice = hand_landmarks.landmark[4]
            nocca_pollice = hand_landmarks.landmark[2]

            punta_medio = hand_landmarks.landmark[12]
            nocca_medio = hand_landmarks.landmark[10]

            punta_anulare = hand_landmarks.landmark[16]
            nocca_anulare = hand_landmarks.landmark[14]

            punta_mignolo = hand_landmarks.landmark[20]
            nocca_mignolo = hand_landmarks.landmark[18]

            y_puntaindice = int(punta_indice.y * altezza_frame)
            y_noccaindice = int(nocca_indice.y * altezza_frame)

            x_puntaanulare = int(punta_anulare.x * larghezza_frame)
            x_noccaanulare = int(nocca_anulare.x * larghezza_frame)

            y_puntaanulare = int(punta_anulare.y * altezza_frame)
            y_noccaanulare = int(nocca_anulare.y * altezza_frame)

            x_puntamignolo = int(punta_mignolo.x * larghezza_frame)
            x_noccamignolo = int(nocca_mignolo.x * larghezza_frame)

            y_puntamignolo = int(punta_mignolo.y * altezza_frame)
            y_noccamignolo = int(nocca_mignolo.y * altezza_frame)

            y_puntapollice = int(punta_pollice.y * altezza_frame)
            y_noccapollice = int(nocca_pollice.y * altezza_frame)

            y_puntamedio = int(punta_medio.y * altezza_frame)
            y_noccamedio = int(nocca_medio.y * altezza_frame)

            if y_puntapollice < y_noccapollice and y_puntaindice < y_noccaindice and y_puntamedio < y_noccamedio and y_puntaanulare < y_noccaanulare and y_puntamignolo < y_noccamignolo:
                if hello is not None:
                    frame = applica_overlay(frame, hello, 0, 0)
            else:
                if y_puntapollice < y_noccapollice and y_puntaindice > y_noccaindice and y_puntamedio > y_noccamedio and y_puntaanulare > y_noccaanulare and y_puntamignolo > y_noccamignolo:
                    if thumb is not None:
                        frame = applica_overlay(frame, thumb, 0, 0)
                    else:
                        cv2.putText(frame, "DOVE ESSERE IMMAGINE???", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if y_puntaindice < y_noccaindice and y_puntapollice < y_noccapollice and y_puntamedio > y_noccamedio and y_puntaanulare > y_noccaanulare and y_puntamignolo > y_noccamignolo:
                    if nerd is not None:
                        frame = applica_overlay(frame, nerd, 0, 0)
                    else:
                        cv2.putText(frame, "DOVE ESSERE IMMAGINE???", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("idek", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()