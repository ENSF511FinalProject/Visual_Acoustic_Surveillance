# Set Up
import pychromecast
from azure.storage.blob import BlobServiceClient
import os

chromecast_device = input("Enter cast device name: ")

os.system('cls')
print("\nStarting and setting up program and cast device: {}".format(chromecast_device))

# For Azure Storage Container Resource
connection_str_blob = 'DefaultEndpointsProtocol=https;AccountName=xxx;AccountKey=xxx==;EndpointSuffix=xxx' # Details of the string connection removed
container_name = 'xxx';    blob_name = 'alert.mp3'
# For Pychromecast
url_blob_file = 'xxx' # link to the block file removed

# Pychromecast and Connecting to Device
services, browser = pychromecast.discovery.discover_chromecasts()
pychromecast.discovery.stop_discovery(browser)
chromecasts, browser = pychromecast.get_listed_chromecasts(friendly_names=[chromecast_device])
cast = chromecasts[0]
cast.wait()
media = cast.media_controller

# Connecting to the Blob Container and create Blob client
blob_service_client = BlobServiceClient.from_connection_string(connection_str_blob)
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

# Loading Libraries
from fastai.vision.all import *
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import sounddevice as sd
import noisereduce as nr
from PIL import Image
import time
from gtts import gTTS


#  Method -------------------------------------------------------
access_cam = False
def compile_message(result): #Testing (Unfinished...)
    '''Comparisons are made to determine what text will be notified if there are
       sensitive sounds heard.

       Input: temperture (float), humidity (float), comfort level/result (str)
       Output: message (str), update (boolean)
    '''   
    if result == 'footsteps':   return "Footsteps are heard nearby the house.".format(result), True

    if result == 'maleScream' or result == 'femaleScream':   return "Someone is screaming nearby the house.".format(result), True

    if result == 'baby':   return "A baby is heard nearby the house.".format(result), True

    if result == 'crash':  return "A crash is heard nearby the house.".format(result), True

    if result == 'dog':  return "A dog is heard nearby the house.".format(result), True

    if result == 'fire':  return "A fire is heard nearby the house.".format(result), True

    if result == 'alarm':  return "An alarm is heard nearby the house.".format(result), True

    return None, False  # Return False if result is non-sensitive

def send_chromecast_file():
    '''Get the request file for the blob created on storage container resource in Azure cloud.
       Connects to a chromcast device in the local Wi-Fi and gets the blob file to play
       notification

       Input: None; Output: None
    '''
    media.play_media(url_blob_file, content_type = 'audio/mp3')
    media.block_until_active()

def send_TTS(message):
    ''' The string message received will generate a .mp3 text to speech (TTS) file,
        which will then be sent to the container storage resource on the Azure cloud.

        Input: message (str); Output: None
    '''
    # Generating Text-To-Speech; Saved to current directory
    tts = gTTS(message)
    tts.save(blob_name)

    # Uploading the created file
    with open(blob_name, "rb") as data:
        blob_client.upload_blob(data, overwrite = True)

    # Method call to send TTS to Speaker
    send_chromecast_file()



def audio_classify(audio_model = load_learner('mobilenet_v2.pkl')):
    
    sample_rate = 44100
    duration = 2.5
    listen_time = 0.1
    threshold = 2
    pred = "general"

    print("\nListening...")
    state =  True
    
    listen_samples = sd.rec(int(listen_time*sample_rate), samplerate = sample_rate, channels = 2)
    sd.wait()
    s = librosa.feature.melspectrogram(y=listen_samples.flatten(),sr=(sample_rate/2))
    rms = librosa.feature.rms(S = s, frame_length=254)

    if rms.mean()*100 > threshold and state == True:
        access_cam = True
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
        plt.close()

        # Pass it to PIL.
        im = Image.fromarray(rgba)
        im = im.convert('RGB')
        im = im.resize((128,128), resample = Image.NEAREST)

        pred,pred_idx,probs = audio_model.predict(tensor(im))

    return probs

if __name__ == "__main__":
    while True:
        audio_classify(audio_model = load_learner('mobilenet_v2.pkl'))