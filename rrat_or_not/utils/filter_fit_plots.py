import numpy as np
import os
import subprocess
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import PySimpleGUI as sg
fit_plots = 'fit_plots/'
plot_files = os.listdir(fit_plots)
file_paths = [fit_plots + f for f in plot_files]
image_viewer_column = [
    [sg.Text("r for refit, d for delete, g for good fit",key="-TEXT-")],
    [sg.Image(key="-IMAGE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(image_viewer_column),
    ]
]
def resize(image_file, new_size, encode_format='PNG'):
    im = Image.open(image_file)
    new_im = im.resize(new_size)
    with BytesIO() as buffer:
        new_im.save(buffer, format=encode_format)
        data = buffer.getvalue()
    return data

window = sg.Window("Image Viewer", layout, return_keyboard_events=True,finalize=True, font="Any 18")
image_ind = 0
#check if the "refit" folder exists
if not os.path.exists("refit"):
    os.makedirs("refit")
#check if the delete folder exists
if not os.path.exists("delete_fit"):
    os.makedirs("delete_fit")
refit_img_list = os.listdir("refit")
delete_img_list = os.listdir("delete_fit")
#find the last image that was refit
plot_files = np.array(plot_files)
file_paths = np.array(file_paths)
if len(refit_img_list) > 0:
    for refit_img in refit_img_list:
        refit_ind = np.argwhere(refit_img == plot_files).flatten()
        if refit_ind[0] > image_ind:
            image_ind = refit_ind
if len(delete_img_list) > 0:
    for delete_img in delete_img_list:
        delete_ind = np.argwhere(delete_img == plot_files).flatten()
        if delete_ind[0] > image_ind:
            image_ind = delete_ind
window["-IMAGE-"].update(data=resize(file_paths[image_ind][0],(1500,900)))
window["-TEXT-"].update(f"r for refit, d for delete, g for good fit {image_ind[0]} , total images {len(file_paths)}")
while True:
    event, values = window.read()
    window["-TEXT-"].update(f"r for refit, d for delete, g for good fit {image_ind} , total images {len(file_paths)}")
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event.startswith("r"):
        #copy the file to the refit folder
        subprocess.run(["cp", file_paths[image_ind][0], "refit/"])
        image_ind += 1
        window["-IMAGE-"].update(data=resize(file_paths[image_ind][0],(1500,900)))
    if event.startswith("d"):
        #copy the file to the delete folder
        subprocess.run(["cp", file_paths[image_ind][0], "delete_fit/"])
        image_ind += 1
        window["-IMAGE-"].update(data=resize(file_paths[image_ind][0],(1500,900)))
    if event.startswith("g"):
        image_ind += 1
        window["-IMAGE-"].update(data=resize(file_paths[image_ind][0],(1500,900)))
    print(image_ind)
    if image_ind == len(file_paths):
        break
window.close()
