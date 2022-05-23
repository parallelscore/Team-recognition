import tkinter as tk
from tkinter import *
import threading


import put_players_into_folders

root =tk.Tk()

root.geometry('800x200')
root.title('Player classification')
root.resizable(False,False)




def start_button():
   classification_starter = threading.Thread(target=put_players_into_folders.start_analysis)
   classification_starter.start()

def stop_button():
   put_players_into_folders.stop_video()

startvideo = Button(root, text = "Start Player Classification", width=30,command=start_button)
startvideo.place(x =100, y=70)

stopvideo = Button(root, text = "Stop Player Classification", width=30,command=stop_button)
stopvideo.place(x =400, y=70)


root.mainloop()