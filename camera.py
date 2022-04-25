# Adapted from code in this thread:
# https://stackoverflow.com/questions/32342935/using-opencv-with-tkinter

from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk

import cv2
import os

from fastai.vision.all import *

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class FingerCountFrame(ttk.Frame):
    def __init__(self, parent, learn):
        """ Initialize frame which uses OpenCV + Tkinter. 
            The frame:
            - Uses OpenCV video capture and periodically captures an image.
            - Uses a fastai learner to predict the finger count
            - Overlays the label and probability on the image
            - and shows it in Tkinter
            
            attributes:
                vs (cv2 VideoSource): webcam to capture images from
                learn (fastai Learner): CNN to generate prediction.
                current_image (PIL Image): current image displayed
                pil_font (PIL ImageFont): font for text overlay
                panel (ttk Label): to display image in frame
        """
        super().__init__(parent)
        self.pack()
        
        # 0 is your default video camera
        self.vs = cv2.VideoCapture(0) 
        
        self.learn = learn
        
        self.current_image = None 
        self.pil_font = ImageFont.truetype("fonts/DejaVuSans.ttf", 40)
        
        # self.destructor function gets fired when the window is closed
        parent.protocol('WM_DELETE_WINDOW', self.destructor)

        
        # Label will display image
        self.panel = ttk.Label(self)  
        self.panel.pack(padx=10, pady=10)

        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame
        self.video_loop()

        
    def video_loop(self):
        """ Get frame from the video stream and show it in Tkinter 
            
            The image is processed using PIL: 
            - crop left and right to make image smaller
            - mirror 
            - convert to Tkinter image
            
            Uses fastai learner to predict label and probability,
            overlayed as text onto image displayed.
            
            Uses after() to call itself again after 30 msec.
        
        """
        # read frame from video stream
        ok, frame = self.vs.read()  
        # frame captured without any errors
        if ok:  
            # convert colors from BGR (opencv) to RGB (PIL)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            # convert image for PIL
            self.current_image = Image.fromarray(cv2image)  
            # camera is wide: crop 200 from left and right
            self.current_image = ImageOps.crop(self.current_image, (200,0,200,0)) 
            # mirror, easier to locate objects
            self.current_image = ImageOps.mirror(self.current_image) 
            
            #predict
            pred,pred_idx,probs = self.learn.predict(tensor(self.current_image))
            pred_str = f"{pred} ({probs[pred_idx]:.2f})"
            self.pred = pred
            self.prob = probs[pred_idx]
            #add text
            draw = ImageDraw.Draw(self.current_image)
            draw.text((10, 10), pred_str, font=self.pil_font, fill='aqua')
            
            # convert image to tkinter for display
            imgtk = ImageTk.PhotoImage(image=self.current_image) 
            # anchor imgtk so it does not get deleted by garbage-collector
            self.panel.imgtk = imgtk  
             # show the image
            self.panel.config(image=imgtk)
            # save results
            self.save_image(pred)
        # do this again after 300 milliseconds
        self.after(300, self.video_loop) 

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.master.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application

    def get_current_count(self, folder):
        """Checks for existing images and returns current file number.
            (code source from lab 2)
            floder(str): directory to search for existing images (a folder within .//images//)
            
            return: (int) the current highest number in jpg filename
                          or -1 if no jpg files present.
        """
        folder_path = './/images//'+folder
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        file_list = os.listdir(folder_path)
        legal_file = [i for i in file_list if re.match('^[0-9]+.jpg',i)]

        if len(legal_file)!= 0:
            filename_number = [int(i.replace('.jpg','')) for i in legal_file]
            return max(filename_number)+1

        else:
            return 0

    def save_image(self,pred):
        """ Save current image to the file (code source from lab2)
        
        Current image is saved to output_path(folder named by the predicted number) using
        consecutive numbering in 
        zero-filled, eight-number format, e.g. 00000000.jpg.
        
        pred(str or index): predicted number of the image
        """
        pic_index = str(self.get_current_count(folder=str(pred)))
        file_name = '0'*(8-len(pic_index))+pic_index+'.jpg'
        file_path = './/images//'+ str(pred)+'//'+file_name
        self.current_image.save(file_path)
        
if __name__ == '__main__':
    

    # Load model from file
    # model_path = Path("C://Users//15047//Desktop//511training//model_camera.pkl")
    try:
        learn = load_learner("model_camera.pkl")
        print("[INFO] learner {} loaded".format("model_camera.pkl"))
    except:
        print("[ERROR] Couldn't load {}".format("model_camera.pkl"))
        print("        Check that file exists")
        exit()

    # start the app
    print("[INFO] starting...")
    gui = tk.Tk() 
    gui.title("Predict Class")  
    FingerCountFrame(gui, learn)
    gui.mainloop()


