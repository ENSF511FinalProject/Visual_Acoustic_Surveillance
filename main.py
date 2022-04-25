from email.mime import audio
import audio_classification
import email_send
import camera

import tkinter as tk

from fastai.vision.all import *

import threading
import time

import numpy as np
# audio_label = "general"
# video_label = "general"
audio_pred_prob =[]
video_pred_prob =[]
try:
    learn = load_learner(".//model_camera.pkl")
    print("[INFO] learner {} loaded".format("model_camera.pkl"))
except:
    print("[ERROR] Couldn't load {}".format("model_camera.pkl"))
    print("        Check that file exists")
    exit()

print("[INFO] starting...")
gui = tk.Tk()  
gui.title("Predict image")  
cam = camera.AudioClass(gui, learn)

audio_model = load_learner('mobilenet_v2.pkl')

def activate_cam():
    """activate camera"""
    # Load model from file
    # model_path = Path("C://Users//15047//Desktop//511training//model_camera.pkl")
    gui.mainloop()
    
def get_video_label():
    global video_pred_prob

    while True:
        if audio_classification.access_cam:  
            video_pred_prob =cam.pred_probs
            print("video label: {}".format(video_label))
            time.sleep(3)

def compare_result(audio_pred_prob,video_pred_prob):
    """compare the result for audio and video;
    if both are match and sensitive return True,
    otherwise return Fase"""
    #  reshape the video tensor
    audio_labels =['alarm','baby','crash','dog','fire','footsteps','general','knock','scream','speech']

    video_reshape = [4,0,5,2,3,1,6,1,1,1] # labels to the audio file
    video_pred_prob = np.array([video_pred_prob[i] for i in video_reshape])
    prob = video_pred_prob+np.array(audio_pred_prob)
    max_indx = np.argmax(prob)
    audio_label = audio_labels[max_indx]
    sensitive_labels =["footsteps",'scream',"baby","crash","dog","fire","alarm"]
    if(audio_label in sensitive_labels):
        return [True,audio_label]
    else:
        return [False, audio_label]
    
def activate_mic(audio_model =load_learner('mobilenet_v2.pkl')):
    """active microphone"""
    global audio_pred_prob
    while True:
        audio_pred_prob = audio_classification.audio_classify()

def notification():
    """send email notification if both audio and video label are sensitive and matches"""
    global audio_pred_prob
    global video_pred_prob
    while True:
        rslt = compare_result(audio_pred_prob,video_pred_prob)
        if rslt[0]:

            audio_msg = audio_classification.compile_message(rslt[1])
            audio_classification.send_TTS(audio_msg)

            msg = email_send.alert_message(rslt[1])
            email_send.sendMail(message =msg,Subject="alert",sender_show="xxx",recipient_show="xxx",to_addrs="'zt945625@gmail.com'")
    

mic_thread = threading.Thread(name='mic',target = activate_mic)
notification_thread = threading.Thread(name='notification',target = notification)
video_label_thread = threading.Thread(name='vl',target = get_video_label)


mic_thread.start()
notification_thread.start()
video_label_thread.start()
activate_cam()