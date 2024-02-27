from threading import Thread
from multiprocessing import Process

import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter.filedialog import *

import customtkinter as ctk
# from ttkwidgets.frames import Tooltip
from CTkToolTip import *

import distutils.util

import os
import sys
import ctypes

from PIL import Image, ImageOps

# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import multiprocessing
multiprocessing.freeze_support()

from topovis_main import *
from pc_mesh import *
from pc_image import *


def dist_path(relative_path):
    base_path = sys._MEIPASS
    return os.path.join(base_path, relative_path)

####Help texts####
overview_txt="""Topography Visualisation Toolbox was originally developed using landscape archaeology methods to highlight the detail of carved rock art. This basic version of the toolbox includes a tool that improves the visualisation of carved surfaces, particularly in cultural heritage applications. 
                                    
Click on the Topo Vis tab to visualise mesh data as images using a range of parameters. 
                                    
The tool runs many functions on CPU currently, and processes leverage multiprocessing where possible. As such, the interface may freeze while tools are running in the background. The program is provided "AS IS" under MIT copyright license.

For further guidance on how to use the tool, hover over the information icons within with Topo Vis tab.  A pop-up will appear with information about the required input(s) and suggestions for how to use the tools with your data."""

mesh_info_txt='Path of folder containing mesh files (.obj, .stl, .ply)'
out_info_txt='Path of folder the processed images will be saved in. If this folder does not exist, manually type the path in the text box and it will be created during processing.'
downsample_txt='Simplifies the mesh with vertex clustering.  Set the voxel multiplier to determine the voxel (region) size - final voxel size is equivalent to voxel multiplier * mesh edge resolution.'
update_res_txt='Scales the values in the point cloud derived from the mesh.  Set the scale multiplier to the desired conversion.  Setting this parameter >1 can be useful for meshes derived from SfM data.'
fill_txt='Fills holes in the mesh using Poisson surface reconstruction (Khazdan,2006; Open3D).  Set the octree depth parameter to influence the resolution of the resulting mesh - a higher value will generate a mesh with more detail.'
rotate_txt='Rotate the mesh 180° around centre.'
out_array_txt='Saves .pcd and .npz files of the original and transformed points.'
visualise_txt='Will display the results of the mesh and point cloud processing stages.'
transparent_txt='Export each plot with a transparent background. If unchecked, plots will have a black background.  Please also choose a suitable export format when using a transparent background.'
rgb_txt='Export only plots in RGB colourmap.'
grey_txt='Export only plot in greyscale.'


class TvtApp(ctk.CTk, tk.Tk):

    def __init__(self):
        super().__init__()


#Define app
        self.title('Topography Visualisation Toolbox')
        self.wm_iconbitmap(default=dist_path('001.ico'))

        self.resizable(False, False)
        # self.wm_minsize(int(start_width/1.5),int(start_height/1.5))
        ctk.set_appearance_mode('Dark')
        ctk.set_default_color_theme(dist_path('custom_theme.json'))

        self.geometry('1000x800')
    
        self.lift()
        self.attributes('-transparentcolor', 'pink')
        
        ctk.FontManager.load_font(dist_path('BarlowSemiCondensed-Thin.ttf'))
        ctk.FontManager.load_font(dist_path('BarlowSemiCondensed-Light.ttf'))
        self.overview_font = ctk.CTkFont(family="Barlow Semi Condensed Thin", size=20)
        self.menu_font = ctk.CTkFont(family="Barlow Semi Condensed Thin", size=18)
        self.tooltip_font = ctk.CTkFont(family="Barlow Semi Condensed Light", size=14, weight='bold')
        self.input_font= ctk.CTkFont(family="Barlow Semi Condensed Light", size=14, slant='italic')

        self.columnconfigure(0,weight=1)
        self.rowconfigure(0,weight=0)
        self.rowconfigure(1, weight=1)

        # Define colours to override those in json and tkinter style
        self.background = "gray25"
        self.frame = "gray22"
        self.entry = "gray17"
        self.button = "gray22"
        self.hover = "gray10"
        self.text = 'white'
        self.tooltip = 'gray10'


        # Load icons
        self.info_icon = ctk.CTkImage(Image.open(dist_path('info.png')), size=(13, 13)) 
        self.folder = ctk.CTkImage(Image.open(dist_path('folder.png')), size=(21, 21))
        self.play_icon = dist_path('play.png')
        self.play = ctk.CTkImage(Image.open(self.play_icon), size=(35, 35))
        self.wait_icon = dist_path('restart.png')
        self.wait = ctk.CTkImage(Image.open(self.wait_icon), size=(35, 35))


        self.topview = ctk.CTkTabview(self, corner_radius=20)
        self.topview.grid(row=1, column=0, padx=(20, 20), pady=(5, 5), sticky='nsew')
    # Overview Tab
        self.topview.add('Overview')
        self.topview.tab('Overview').columnconfigure(0, weight=1)
        self.topview.tab('Overview').rowconfigure((0,1,2), weight=1)

        self.header_img_dark = Image.open(dist_path('header_landscape.png'))
        # self.header_img_light = Image.open(dist_path('header_landscape_light.png'))
        self.img_width = 1000
        self.img_height =500
        self.header = ctk.CTkImage(light_image=self.header_img_dark,
                                   dark_image=self.header_img_dark,
                                   size=(self.img_width,self.img_height))

        self.sidebar_img = ctk.CTkLabel(self.topview.tab('Overview'), text=None, image=self.header, corner_radius=0, fg_color=self.frame)
        self.sidebar_img.grid(row=0, column=0,sticky='nsew', pady=(0,5))
        
        self.summary = ctk.CTkScrollableFrame(self.topview.tab('Overview'))
        
        
        self.summary.grid(row=1, column=0, pady=(5, 0), ipadx=150, ipady=10, sticky='nsew')
        self.overview_text=ctk.CTkLabel(self.summary, justify='left', wraplength=850,
                                    text=overview_txt,font=self.overview_font)
        self.overview_text.grid(row=0,column=0, sticky='nsew')



    # Topo Vis Tab
        self.topview.add('Topo Vis')
        self.topview.tab('Topo Vis').rowconfigure(0, weight=1)
        self.topview.tab('Topo Vis').columnconfigure((0), weight=1)

        def run_tvt():
            self.tv_run_button.configure(image=self.wait, text=None)
            self.tv_run_button.update()

            data_path = self.tv_direct_entry.get() + '/'
            save_path = self.tv_out_entry.get() + '/'
            meta_data_path = save_path + 'output_summary.csv'
            visualize_steps = bool(self.visualise_spinbox.get())
            downsample_mesh = bool(self.downsampling_spinbox.get())
            voxel_multiplier = float(self.voxel_multiplier.get()) if self.downsampling_spinbox.get() > 0 else 1.0
            fill_holes = bool(self.fill_spinbox.get())
            depth_size = int(self.depth.get()) if self.fill_spinbox.get() > 0 else 9
            flip_mesh = bool(self.invert_spinbox.get())
            update_resolution = bool(self.updateres_spinbox.get())
            scale_multiplier = float(self.res_multiplier.get()) if self.updateres_spinbox.get() > 0 else 1.0
            progress_text=self.tv_run_text
            array = bool(self.outarray_spinbox.get())
            export_rgb = bool(self.rgb_spinbox.get())
            export_grey = bool(self.grey_spinbox.get())
            transparency=bool(self.transparency_spinbox.get())
            img_format = self.img_ext_entry.get()

            files = []
            for dp, dn, filenames in os.walk(data_path):
                for file in filenames:
                    if os.path.splitext(file)[1] in ['.stl', '.ply', '.obj']:
                        files.append((' '.join(file.split('.')[0].split(' ')), os.path.join(dp, file),
                                      ''.join(dp.replace(data_path, save_path))))
                               
            # self.tvresultsview.grid(column=1,row=0,sticky='nsew', padx=(10,0))
            # self.tvresultsview.rowconfigure((1,2), weight=1)
            # self.tvresultsview.columnconfigure((1,2), weight=1)

            try:
                topovis.start_tv(
                data_path, save_path, meta_data_path, visualize_steps, downsample_mesh, voxel_multiplier, fill_holes,
                depth_size, flip_mesh, update_resolution, scale_multiplier, progress_text, array, 
                files=files, export_rgb=export_rgb,export_grey=export_grey,transparency=transparency,img_type=img_format)
            except MemoryError as error:
                    topovis.log_exception(error, False)
            except Exception as exception:
                    topovis.log_exception(exception, True)


            self.tv_run_button.configure(image=self.play, text=None, command=run_tvt)
            self.tv_run_button.update()

            self.tv_run_text.configure(text='')
            self.tv_run_text.update()

    #Set parameters frame
        self.tv_inputarea = ctk.CTkScrollableFrame(self.topview.tab('Topo Vis'), corner_radius=10)
        self.tv_inputarea.grid(row=0,column=0,sticky='nsew')
        self.tv_inputarea.rowconfigure((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22), weight=5)
        self.tv_inputarea.columnconfigure((1,2,3,4), weight=1)

    # Select input
        self.mesh_info=ctk.CTkButton(self.tv_inputarea,image=self.info_icon, text='', fg_color='transparent', command=None, width=30)
        self.mesh_info.grid(row=0, column=0, sticky='e', pady=5)
        self.mesh_info_balloon = CTkToolTip(self.mesh_info, message=mesh_info_txt, wraplength=250, text_color='whitesmoke', alpha=1, font=self.tooltip_font)
        self.mesh_direct = ctk.CTkLabel(self.tv_inputarea, text='Input Directory', justify='left', anchor='w', font=self.menu_font)
        self.mesh_direct.grid(row=0, column=1, columnspan=3, padx=0, sticky='nsew')
        self.tv_direct_entry = ctk.CTkEntry(self.tv_inputarea, placeholder_text=' ',font=self.input_font)
        self.tv_direct_entry.grid(row=1, column=1, columnspan=3, sticky='nsew')
        self.tv_in_button = ctk.CTkButton(self.tv_direct_entry, image=self.folder,
                                       command=lambda: self.browse_in_dir(self.tv_direct_entry, self.tv_inputarea), text='', width=20, height=20, bg_color=self.frame)
        self.tv_in_button.grid(row=0, column=1, padx=0, sticky='nsw')

    # Select output
        self.out_info = ctk.CTkButton(self.tv_inputarea,image=self.info_icon, text='', fg_color='transparent', command=None, width=30)
        self.out_info.grid(row=2, column=0, sticky='ns', pady=5)
        self.out_info_balloon = CTkToolTip(self.out_info, message=out_info_txt, wraplength=250, text_color='whitesmoke', alpha=1, font=self.tooltip_font)
        self.tv_out_direct = ctk.CTkLabel(self.tv_inputarea, text='Output Directory', justify='left', anchor='w', font=self.menu_font)
        self.tv_out_direct.grid(row=2, column=1, columnspan=3, padx=0, sticky='nsew')
        self.tv_out_entry = ctk.CTkEntry(self.tv_inputarea, placeholder_text=' ',font=self.input_font)
        self.tv_out_entry.grid(row=3, column=1, columnspan=3, padx=0, sticky='nsew')
        self.tv_out_button = ctk.CTkButton(self.tv_out_entry, image=self.folder, bg_color=self.frame,
                                        command=lambda: self.browse_out_dir(self.tv_out_entry, self.tv_inputarea), text='', width=20, height=20)
        self.tv_out_button.grid(row=0, column=1, padx=0, sticky='nsw')

        # self.tv_settings_info = ctk.CTkButton(self.tv_inputarea,image=self.info_icon, text='', fg_color='transparent', command=None, width=30)
        # self.tv_settings_info.grid(row=6, column=0, sticky='ns', pady=(40,0))

        self.params_header = ctk.CTkLabel(self.tv_inputarea, text='Settings', justify='left', anchor='w', font=self.menu_font)
        self.params_header.grid(row=6, column=0, columnspan=3, padx=3, pady=(40,0), sticky='ew')


        self.downsampling_info = ctk.CTkButton(self.tv_inputarea,image=self.info_icon, text='', fg_color='transparent', command=None, width=30)
        self.downsampling_info.grid(row=7, column=0, sticky='ns', pady=5)
        self.downsampling_info_balloon = CTkToolTip(self.downsampling_info, message=downsample_txt, wraplength=250, text_color='whitesmoke', alpha=1,font=self.tooltip_font)
        self.downsampling_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Downsample Mesh', onvalue=1,
                                                    offvalue=0, font=self.menu_font,
                                                    command=lambda: self.change_entry_state(self.downsampling_spinbox,self.voxel_multiplier, 1.0))
        self.downsampling_spinbox.grid(row=7, column=1, columnspan=3, padx=5, pady=5, sticky='ew')
        self.voxel_multiplier = ctk.CTkEntry(self.tv_inputarea, placeholder_text='Voxel Multiplier: e.g. 1.0',font=self.input_font, border_width=1.5,
                                             state='disabled')
        self.voxel_multiplier.grid(row=8, column=1, columnspan=3, padx=20, pady=5, sticky='ew')


        self.res_info = ctk.CTkButton(self.tv_inputarea,image=self.info_icon, text='', fg_color='transparent', command=None, width=30)
        self.res_info.grid(row=9, column=0, sticky='ns', pady=5)
        self.res_info_balloon = CTkToolTip(self.res_info, message=update_res_txt, wraplength=250, text_color='whitesmoke', alpha=1,font=self.tooltip_font)
        self.updateres_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Update Resolution', onvalue=1,
                                                 offvalue=0, font=self.menu_font,
                                                 command=lambda: self.change_entry_state(self.updateres_spinbox,self.res_multiplier, 1.0))
        self.updateres_spinbox.grid(row=9, column=1, columnspan=3, padx=5, pady=5, sticky='ew')
        self.res_multiplier = ctk.CTkEntry(self.tv_inputarea, placeholder_text='Scale Multiplier: e.g. 1.0',font=self.input_font, border_width=1.5,
                                           state='disabled')
        self.res_multiplier.grid(row=10, column=1, columnspan=3, padx=20, pady=5, sticky='ew')



        self.fill_info = ctk.CTkButton(self.tv_inputarea,image=self.info_icon, text='', fg_color='transparent', command=None, width=30)
        self.fill_info.grid(row=11, column=0, sticky='ns', pady=5)
        self.fill_info_balloon = CTkToolTip(self.fill_info, message=fill_txt, wraplength=250, text_color='whitesmoke', alpha=1,font=self.tooltip_font)
        self.fill_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Fill Mesh', onvalue=1,
                                            offvalue=0, font=self.menu_font,
                                            command=lambda: self.change_entry_state(self.fill_spinbox,self.depth, 10))
        self.fill_spinbox.grid(row=11, column=1, columnspan=3, padx=5, pady=5, sticky='ew')
        self.depth = ctk.CTkEntry(self.tv_inputarea, placeholder_text='Octree Depth: e.g. 9',font=self.input_font, border_width=1.5,
                                     state='disabled')
        self.depth.grid(row=12, column=1, columnspan=3, padx=20, pady=5, sticky='ew')


        self.invert_info = ctk.CTkButton(self.tv_inputarea,image=self.info_icon, text='', fg_color='transparent', command=None, width=30)
        self.invert_info.grid(row=13, column=0, sticky='ns', pady=5)
        self.invert_info_balloon = CTkToolTip(self.invert_info, message=rotate_txt, wraplength=250, text_color='whitesmoke', alpha=1,font=self.tooltip_font)
        self.invert_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Rotate Mesh', onvalue=1,
                                              offvalue=0, font=self.menu_font)
        self.invert_spinbox.grid(row=13, column=1, columnspan=3, padx=5, pady=5, sticky='ew')



        self.array_info = ctk.CTkButton(self.tv_inputarea,image=self.info_icon, text='', fg_color='transparent', command=None, width=30)
        self.array_info.grid(row=14, column=0, sticky='ns', pady=5)
        self.array_info_balloon = CTkToolTip(self.array_info, message=out_array_txt, wraplength=250, text_color='whitesmoke', alpha=1,font=self.tooltip_font)
        self.outarray_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Output Array Files',
                                                onvalue=1, offvalue=0, font=self.menu_font)
        self.outarray_spinbox.grid(row=14, column=1, columnspan=3, padx=5, pady=5, sticky='ew')


        self.visualise_info = ctk.CTkButton(self.tv_inputarea,image=self.info_icon, text='', fg_color='transparent', command=None, width=30)
        self.visualise_info.grid(row=15, column=0, sticky='ns', pady=5)
        self.visualise_info_balloon = CTkToolTip(self.visualise_info, message=visualise_txt, wraplength=250, text_color='whitesmoke', alpha=1,font=self.tooltip_font)
        self.visualise_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Visualise Processing Stages',
                                                 onvalue=1, offvalue=0, font=self.menu_font)
        self.visualise_spinbox.grid(row=15, column=1, columnspan=3, padx=5, pady=5, sticky='ew')


        self.transparency_info = ctk.CTkButton(self.tv_inputarea,image=self.info_icon, text='', fg_color='transparent', command=None, width=30)
        self.transparency_info.grid(row=16, column=0, sticky='ns', pady=5)
        self.transparency_info_balloon = CTkToolTip(self.transparency_info, message=transparent_txt, wraplength=250, text_color='whitesmoke', alpha=1,font=self.tooltip_font)
        self.transparency_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Transparent Background',
                                                 onvalue=1, offvalue=0, font=self.menu_font)
        self.transparency_spinbox.grid(row=16, column=1, columnspan=3, padx=5, pady=5, sticky='ew')



        self.rgb_info = ctk.CTkButton(self.tv_inputarea,image=self.info_icon, text='', fg_color='transparent', command=None, width=30)
        self.rgb_info.grid(row=17, column=0, sticky='ns', pady=5)
        self.rgb_info_balloon = CTkToolTip(self.rgb_info, message=rgb_txt, wraplength=250, text_color='whitesmoke', alpha=1,font=self.tooltip_font)
        self.rgb_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Export RGB Images',
                                                 onvalue=1, offvalue=0, font=self.menu_font)
        self.rgb_spinbox.grid(row=17, column=1, columnspan=3, padx=5, pady=5, sticky='ew')


        self.grey_info = ctk.CTkButton(self.tv_inputarea,image=self.info_icon, text='', fg_color='transparent', command=None, width=30)
        self.grey_info.grid(row=18, column=0, sticky='ns', pady=5)
        self.grey_info_balloon = CTkToolTip(self.grey_info, message=rgb_txt, wraplength=250, text_color='whitesmoke', alpha=1,font=self.tooltip_font)
        self.grey_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Export Greyscale Images',
                                                 onvalue=1, offvalue=0, font=self.menu_font)
        self.grey_spinbox.grid(row=18, column=1, columnspan=3, padx=5, pady=5, sticky='ew')

        self.img_ext = ctk.CTkLabel(self.tv_inputarea, text='Image Type', justify='left', anchor='w', font=self.menu_font)
        self.img_ext.grid(row=19, column=1, columnspan=1, padx=0, sticky='nsew')
        self.img_ext_entry=ctk.CTkComboBox(self.tv_inputarea, values=['jpg','png','tif'], font=self.menu_font, border_width=1.5)
        self.img_ext_entry.grid(row=19, column=2, columnspan=2, padx=0, sticky='nsew')

        self.tv_run_button = ctk.CTkButton(self.tv_inputarea, image=self.play,  command=run_tvt,
                                        fg_color='transparent', text='', width=35, height=35)
        self.tv_run_button.grid(row=21, column=4, rowspan=2, padx=10, pady=40)

        self.tv_run_text = ctk.CTkLabel(self.tv_inputarea, text='', font=self.menu_font, justify='center', wraplength=400)
        self.tv_run_text.grid(row=21, column=1, columnspan=3, rowspan=2, padx=10, pady=40)


    def browse_in_dir(self, field, tab):
        in_directory = askdirectory(parent=tab)
        field.delete(0, END)
        field.insert(tk.END, in_directory)


    def browse_out_dir(self, field, tab):
        out_directory = askdirectory(parent=tab)
        field.delete(0, END)
        field.insert(tk.END, out_directory)


    def change_entry_state(self, checkbox, entry, default_val):
        if checkbox.get() !=0:
            entry.configure(state='normal')
            entry.delete(0, END)
            entry.insert(END, default_val)
            entry.update()
        else:
            entry.delete(0, END)
            entry.configure(state='disabled')
            entry.update()


    def run_progress(self, stage_text):
        self.run_text.configure(text=stage_text)
        self.run_text.update()


    def clear_results(self, tab_name):
        tab_name.destroy()


    def plot_topovis_img(self, count, file):
        tab_name = '{}:{}...'.format(count, file.replace('.stl', '').replace('.obj', '').replace('.ply', ''))
        self.topovis_tab_list.append(tab_name)




if __name__ == '__main__':

    try:
       import pyi_splash

       pyi_splash.close()
    except:
       pass

    app = TvtApp()
    app.mainloop()

    app.protocol('WM_DELETE_WINDOW', sys.exit())
