import cv2
import mediapipe as mp
import math  
 
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)
 
thumb = cv2.imread("imgs/thumb.png")
thumb = cv2.resize(thumb, (500, 500))
omg = cv2.imread("imgs/omg.png")
omg = cv2.resize(omg, (500, 500))
smile = cv2.imread("imgs/smile.png")
smile = cv2.resize(smile, (500, 500))
wow = cv2.imread("imgs/wow.jpg")
wow = cv2.resize(wow, (500, 500))
shock = cv2.imread("imgs/shock.png")
shock = cv2.resize(shock, (500, 500))
nerd = cv2.imread("imgs/nerd.jpg")
nerd = cv2.resize(nerd, (500, 500))
hello = cv2.imread("imgs/hello.jpg")
hello = cv2.resize(hello, (500, 500))
flipoff = cv2.imread("imgs/flipoff.png")
flipoff = cv2.resize(flipoff, (500, 500))
 
 
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
                 if wow is not None:
                    h_img, w_img, _ = wow.shape
                    if h_img < altezza_frame and w_img < larghezza_frame:
                        frame[0:h_img, 0:w_img] = wow
            else:
                if distanza_eye_dx > 30 and distanza_eye_sx > 30 and distanza_bocca_sugiu < 30:
                    if shock is not None:
                            h_img, w_img, _ = shock.shape
                            if h_img < altezza_frame and w_img < larghezza_frame:
                                frame[0:h_img, 0:w_img] = shock
                else:
                    if distanza_bocca_sugiu > 40:
                        if omg is not None:
                            h_img, w_img, _ = omg.shape
                            if h_img < altezza_frame and w_img < larghezza_frame:
                                frame[0:h_img, 0:w_img] = omg
                
                    if distanza_bocca_sxdx > 150 and distanza_bocca_sugiu > 30 and distanza_eye_dx < 30 and distanza_eye_sx < 30:
                        if smile is not None:
                            h_img, w_img, _ = smile.shape
                            if h_img < altezza_frame and w_img < larghezza_frame:
                                frame[0:h_img, 0:w_img] = smile
 
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

            x_puntamignolo = int(punta_mignolo.x * larghezza_frame)
            x_noccamignolo = int(nocca_mignolo.x * larghezza_frame)

            y_puntamignolo = int(punta_mignolo.y * altezza_frame)
            y_noccamignolo = int(nocca_mignolo.y * altezza_frame)

            y_puntapollice = int(punta_pollice.y * altezza_frame)
            y_noccapollice = int(nocca_pollice.y * altezza_frame)
 
            y_puntamedio = int(punta_medio.y * altezza_frame)
            y_noccamedio = int(nocca_medio.y * altezza_frame)
 
            if  y_puntapollice < y_noccapollice and y_puntaindice < y_noccaindice and y_puntamedio < y_noccamedio and y_puntaanulare < y_noccaanulare and y_puntamignolo < y_noccamignolo:
                if hello is not None:
                    h_img, w_img, _ = hello.shape
                    if h_img < altezza_frame and w_img < larghezza_frame:
                        frame[0:h_img, 0:w_img] = hello
            else:
                if y_puntapollice < y_noccapollice and y_puntaindice > y_noccaindice and y_puntamedio > y_noccamedio and y_puntaanulare > y_noccaanulare and y_puntamignolo > y_noccamignolo:
                    
                    if thumb is not None:
                        h_img, w_img, _ = thumb.shape
                        if h_img < altezza_frame and w_img < larghezza_frame:
                            frame[0:h_img, 0:w_img] = thumb
                    else:
                        cv2.putText(frame, "DOVE ESSERE IMMAGINE???", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
 
                if y_puntaindice < y_noccaindice and y_puntapollice < y_noccapollice and y_puntamedio > y_noccamedio and y_puntaanulare > y_noccaanulare and y_puntamignolo > y_noccamignolo:
                    
                    if nerd is not None:
                        h_img, w_img, _ = nerd.shape
                        if h_img < altezza_frame and w_img < larghezza_frame:
                            frame[0:h_img, 0:w_img] = nerd
                    else:
                        cv2.putText(frame, "DOVE ESSERE IMMAGINE???", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if y_puntamedio < y_noccamedio and y_puntaindice > y_noccaindice and y_puntapollice < y_noccapollice and y_puntaanulare > y_noccaanulare and y_puntamignolo > y_noccamignolo:
                    if flipoff is not None:
                        h_img, w_img, _ = flipoff.shape
                        
                        if h_img < altezza_frame and w_img < larghezza_frame:
                            frame[0:h_img, 0:w_img] = flipoff
                    else:
                        cv2.putText(frame, "DOVE ESSERE IMMAGINE???", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
 
    cv2.imshow("idek", frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()