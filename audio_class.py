from fastai.vision.all import *
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import sounddevice as sd
import noisereduce as nr
from PIL import Image
import time

# Loading the variables
model = load_learner('mobilenet_v2.pkl')
sample_rate = 44100
duration = 2.5
listen_time = 0.2
threshold = 10

print("Listening...")
state =  True

while True:
    listen_samples = sd.rec(int(listen_time*sample_rate), samplerate = sample_rate, channels = 2)
    sd.wait()
    s = librosa.feature.melspectrogram(y=listen_samples.flatten(),sr=(sample_rate/2))
    rms = librosa.feature.rms(S = s, frame_length=254)

    if rms.mean()*10 > threshold:
        samples = sd.rec(int(duration*sample_rate), samplerate = sample_rate, channels = 1, dtype='float32')
        print("Sound Detected. Predicting...")
        sd.wait()
        reduced_noise = nr.reduce_noise(y=samples.flatten(), sr=sample_rate)
        fig = plt.figure(figsize=[2,2])
        ax =fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        s = librosa.feature.melspectrogram(y=samples.flatten(),sr=(sample_rate/2))
        librosa.display.specshow(librosa.power_to_db(S=s,ref=np.max))

        canvas = FigureCanvasAgg(fig) 
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())

        # Pass it to PIL.
        im = Image.fromarray(rgba)
        im = im.convert('RGB')
        im = im.resize((128,128), resample = Image.NEAREST)

        pred,pred_idx,probs = model.predict(tensor(im))
        print("Prediction: {}, {}".format(pred, probs[pred_idx]))
        state = False #switch

    elif state == False:
        print("\nCooldown...")        
        time.sleep(2)
        state = True
        pred = None
        print("Ready!\nListening...")