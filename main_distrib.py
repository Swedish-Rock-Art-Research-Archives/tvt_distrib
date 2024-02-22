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

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import multiprocessing
multiprocessing.freeze_support()

from topovis_main import *
from pc_mesh import *
from pc_image import *


def dist_path(relative_path):
    base_path = sys._MEIPASS
    return os.path.join(base_path, relative_path)

####Help texts####
overview_txt="""Topography Visualisation Toolbox was originally developed using landscape archaeology methods to highlight the detail of carved rock art. The toolbox includes two visualisation tools that improve the visualisation of surfaces, particularly in cultural heritage applications, and a deep learning tool to highlight motifs in carved rock art. 
                                    
Click on the Topo Vis tab to visualise mesh data as images, then use the output from this tool in the Segmentation tab to interpret motifs in the image.  The Digital Frottage and Segmentation tools are still undergoing development and improvements; the Segmentation tools currently only includes the first iteration of models for prediction. 
                                    
The tools run many functions on CPU currently, and processes leverage threading and multiprocessing where possible. As such, the interface may freeze while tools are running. The program is provided "AS IS".

For further guidance on how to use the tools, hover over the information icons within each tool tab.  A pop-up will appear with information about the required input(s) and suggestions for how to use the tools with your data."""

mesh_info_txt='Path of folder containing mesh files (.obj, .stl, .ply)'
out_info_txt='Path of folder the processed images will be saved in. If this folder does not exist, manually type the path in the text box and it will be created during processing.'
downsample_txt='Simplifies the mesh with vertex clustering.  Set the voxel multiplier to determine the voxel (region) size - final voxel size is equivalent to voxel multiplier * mesh edge resolution.'
update_res_txt='Scales the values in the point cloud derived from the mesh.  Set the scale multiplier to the desired conversion.  Setting this parameter >1 can be useful for meshes derived from SfM data.'
fill_txt='Fills holes in the mesh using Poisson surface reconstruction (Khazdan,2006; Open3D).  Set the octree depth parameter to influence the resolution of the resulting mesh - a higher value will generate a mesh with more detail.'
rotate_txt='Rotate the mesh 180° around centre.'
out_array_txt='Saves .pcd and .npz files of the original and transformed points.'
visualise_txt='Will display the results of the mesh and point cloud processing stages.'
tiff_txt='Export each plot in tiff format.'
rgb_txt='Export only plots in RGB colourmap.'
grey_txt='Export only plot in greyscale.'


class TvtApp(ctk.CTk, tk.Tk):

    def __init__(self):
        super().__init__()


#Define app
        self.title('Topography Visualisation Toolbox')
        self.wm_iconbitmap(default='001.ico')

        self.resizable(False, False)
        ctk.set_appearance_mode('Dark')
        ctk.set_default_color_theme('custom_theme.json')

        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        scaling=self._get_window_scaling()
        self.geometry(f'{screenwidth/2}x{screenheight/2}+{((screenwidth/2)-(screenwidth/1.5))*scaling}+{((screenheight/2)-(screenheight/2))*scaling}')

        self.lift()
        self.attributes('-transparentcolor', 'pink')
        ctk.FontManager.load_font('fonts/BarlowSemiCondensed-Medium.ttf')


#Resize the images as window size changes
        def resize(event):
            self.new_height = int(event.height/scaling)
            self.new_width = int(event.width/scaling)
            self.correction=(self.new_width,self.new_height)
            corrected_header=ImageOps.contain(self.header_img_light,(self.correction))

            self.new_header = ctk.CTkImage(light_image=self.header_img_light, dark_image=self.header_img_dark, size=(corrected_header.width,corrected_header.height))
            self.sidebar_img.configure(image=self.new_header)
            self.update_idletasks()
            self.summary.configure(wraplength=(self.summary.winfo_width()/scaling))


#Toggle between light and dark mode
        def switch_mode():
            self.colour_mode.configure(image=self.lighticon if self.colour_mode._image is self.darkicon else self.darkicon,
                                       bg_color=self._fg_color, fg_color=self._fg_color)
            ctk.set_default_color_theme('custom_theme.json')
            ctk.set_appearance_mode('Light' if self.colour_mode._image is self.darkicon else 'Dark')
            self.colour_mode.update()

            self.info_img = PhotoImage(file='info_light.png' if self.colour_mode._image is self.darkicon else 'info.png')  # Loaded in old tk style to use with hover tooltip
            self.info_icon = self.info_img.subsample(50, 50)  # Loaded in old tk style to use with hover tooltip
            self.play_icon = 'play_light.png' if self.colour_mode._image is self.darkicon else 'play.png'
            self.play = ctk.CTkImage(Image.open(self.play_icon), size=(35, 35))
            self.wait_icon = 'wait_light.png' if self.colour_mode._image is self.darkicon else 'wait.png'
            self.wait = ctk.CTkImage(Image.open(self.wait_icon), size=(35, 35))
            self.palette_icon = 'palette_light.png' if self.colour_mode._image is self.darkicon else 'palette.png'
            self.palette = ctk.CTkImage(Image.open(self.palette_icon), size=(21, 21))
            self.restart_icon = 'restart_light.png' if self.colour_mode._image is self.darkicon else 'restart.png'
            self.restart = ctk.CTkImage(Image.open(self.restart_icon), size=(25, 25))

            self.tv_run_button.configure(image=self.play, text=None)
            self.tv_run_button.update()

            # Define colours to override those in json and tkinter style
            self.background = '#f2f2f2' if self.colour_mode._image is self.darkicon else "gray25"
            self.frame = '#f2f2f2' if self.colour_mode._image is self.darkicon else "gray22"
            self.entry = '#f2f2f2' if self.colour_mode._image is self.darkicon else "gray17"
            self.button = '#f2f2f2' if self.colour_mode._image is self.darkicon else "gray22"
            self.hover = '#61506B' if self.colour_mode._image is self.darkicon else "gray10"
            self.text = 'black' if self.colour_mode._image is self.darkicon else 'white'
            self.clickable_button = '#f2f2f2' if self.colour_mode._image is self.darkicon else "gray22"
            self.tooltip = '#f2f2f2' if self.colour_mode._image is self.darkicon else 'gray10'

            

            for widget in [self.mesh_info,self.out_info,self.tv_settings_info,self.surf_info, self.df_settings_info, self.seg_img_info, self.seg_out_info, self.seg_settings_info]:
                widget.configure(image=self.info_icon)

            for widget in [self.mesh_info_balloon,self.out_info_balloon,self.surf_info_balloon,self.df_settings_info_balloon,self.seg_img_info_balloon,self.seg_out_info_balloon,self.seg_settings_info_balloon]:
                widget.config(background=self.tooltip,borderwidth=0)

            for widget in [self.tv_in_button,self.tv_out_button,self.df_in_button,self.relief_cmap_btn,self.erelief_cmap_btn,self.seg_in_button,self.seg_out_button]:
                widget.configure(bg_color=self.button)

            self.update()


        self.lighticon = ctk.CTkImage(Image.open(('lightmode.png')), size=(24, 24))
        self.darkicon = ctk.CTkImage(Image.open(('darkmode.png')), size=(24,24))

        self.colour_mode = ctk.CTkButton(self,image=self.lighticon,text='',command=switch_mode, bg_color=self._fg_color, fg_color=self._fg_color, width=25)
        self.colour_mode.grid(row=0,column=0, sticky='nse', ipadx=2, ipady=2, padx=(0,20), pady=5)

        self.columnconfigure(0,weight=1)
        self.rowconfigure(0,weight=0)
        self.rowconfigure(1, weight=1)

        # Define colours to override those in json and tkinter style
        self.background = '#f2f2f2' if self.colour_mode._image is self.darkicon else "gray25"
        self.frame = '#f2f2f2' if self.colour_mode._image is self.darkicon else "gray22"
        self.entry = '#f2f2f2' if self.colour_mode._image is self.darkicon else "gray17"
        self.button = '#f2f2f2' if self.colour_mode._image is self.darkicon else "gray22"
        self.hover = '#83579E' if self.colour_mode._image is self.darkicon else "gray10"
        self.text = 'black' if self.colour_mode._image is self.darkicon else 'white'
        self.tooltip = '#f2f2f2' if self.colour_mode._image is self.darkicon else 'gray10'


        # Load icons
        self.info_img = PhotoImage(file=('info.png'))  # Loaded in old tk style to use with hover tooltip
        self.info_icon = self.info_img.subsample(50, 50)  # Loaded in old tk style to use with hover tooltip
        self.folder = ctk.CTkImage(Image.open(('folder.png')), size=(21, 21))
        self.play_icon = 'play_light.png'if self.colour_mode._image is self.darkicon else 'play.png'
        self.play = ctk.CTkImage(Image.open(self.play_icon), size=(35, 35))
        self.wait_icon = 'wait_light.png'if self.colour_mode._image is self.darkicon else 'wait.png'
        self.wait = ctk.CTkImage(Image.open(self.wait_icon), size=(35, 35))
        self.palette_icon = 'palette_light.png'if self.colour_mode._image is self.darkicon else 'palette.png'
        self.palette = ctk.CTkImage(Image.open(self.palette_icon), size=(21, 21))
        self.restart_icon = 'restart_light.png'if self.colour_mode._image is self.darkicon else 'restart.png'
        self.restart = ctk.CTkImage(Image.open(self.restart_icon), size=(25, 25))


        self.topview = ctk.CTkTabview(self, corner_radius=20)
        self.topview.grid(row=1, column=0, padx=(20, 20), pady=(5, 5), sticky='nsew')
    # Overview Tab
        self.topview.add('Overview')
        self.topview.tab('Overview').columnconfigure(0, weight=1)
        self.topview.tab('Overview').rowconfigure((0,1,2), weight=1)

        self.summary = ctk.CTkLabel(self.topview.tab('Overview'), justify='left', wraplength=(screenwidth-300),
                                    text=overview_txt,font=('BarlowSemiCondensed-Medium',28))
        self.summary.grid(row=1, column=0, pady=(5, 0), ipadx=150, ipady=10, sticky='ns')

        self.update()
        self.header_img_dark = Image.open(('header_landscape.png'))
        self.header_img_light = Image.open(('header_landscape_light.png'))
        self.img_width = int(self.summary.winfo_width())
        self.img_height =int((screenheight)*0.75)
        self.header = ctk.CTkImage(light_image=self.header_img_light,
                                   dark_image=self.header_img_dark,
                                   size=(self.img_width,self.img_height))

        self.sidebar_img = ctk.CTkLabel(self.topview.tab('Overview'), text=None, image=self.header, corner_radius=0, fg_color=self.frame)
        self.sidebar_img.grid(row=0, column=0,sticky='nsew', pady=(0,5))
        self.sidebar_img.bind('<Configure>', resize)


    # Topo Vis Tab
        self.topview.add('Topo Vis')
        self.topview.tab('Topo Vis').rowconfigure(0, weight=1)
        self.topview.tab('Topo Vis').columnconfigure((0,1,2,3,4), weight=1)

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
            progress_text = self.tv_run_text
            array = bool(self.outarray_spinbox.get())
            export_rgb = bool(self.rgb_spinbox.get())
            export_grey = bool(self.grey_spinbox.get())
            img_format = self.img_ext_entry.get()
            transparency=bool(self.transparency_spinbox.get())
            

            image_tab = self.tvresultsview

            files = []
            for dp, dn, filenames in os.walk(data_path):
                for file in filenames:
                    if os.path.splitext(file)[1] in ['.stl', '.ply', '.obj']:
                        files.append((' '.join(file.split('.')[0].split(' ')), os.path.join(dp, file),
                                      ''.join(dp.replace(data_path, save_path))))
                               
            self.tvresultsview.grid(column=1,row=0,sticky='nsew', padx=(10,0))
            # self.tvresultsview.rowconfigure((1,2), weight=1)
            # self.tvresultsview.columnconfigure((1,2), weight=1)

            topovis.start_tv(
                data_path, save_path, meta_data_path, visualize_steps, downsample_mesh, voxel_multiplier, fill_holes,
                depth_size, flip_mesh, update_resolution, scale_multiplier, progress_text, image_tab, array, files=files, export_rgb=export_rgb,export_grey=export_grey,img_format=img_format,transparency=transparency)

            self.tv_clear_tabs.grid(row=4, column=0, sticky='ew', padx=2, pady=2)

            self.tv_run_button.configure(image=self.play, text=None, command=lambda:Thread.stop())
            self.tv_run_button.update()

            self.tv_run_text.configure(text='')
            self.tv_run_text.update()

    #Set parameters frame
        self.tv_inputarea = ctk.CTkScrollableFrame(self.topview.tab('Topo Vis'), corner_radius=10)
        self.tv_inputarea.grid(row=0,column=0,sticky='nsew')
        self.tv_inputarea.rowconfigure((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18), weight=5)
        self.tv_inputarea.columnconfigure((1,2,3,4), weight=1)

    # Select input
        self.mesh_info=ctk.CTkButton(self.tv_inputarea,image=self.info_icon, text='', fg_color='transparent', command=None, width=30)
        self.mesh_info.grid(row=0, column=0, sticky='e', pady=5)
        self.out_info_balloon = CTkToolTip(self.mesh_info, message=mesh_info_txt, wraplength=350,alpha=1,text_color='lightblue')
        self.mesh_direct = ctk.CTkLabel(self.tv_inputarea, text='Input Directory', justify='left', anchor='w')
        self.mesh_direct.grid(row=0, column=1, columnspan=3, padx=0, sticky='nsew')
        self.tv_direct_entry = ctk.CTkEntry(self.tv_inputarea, placeholder_text=' ')
        self.tv_direct_entry.grid(row=1, column=1, columnspan=3, sticky='nsew')
        self.tv_in_button = ctk.CTkButton(self.tv_direct_entry, image=self.folder,
                                       command=lambda: self.browse_in_dir(self.tv_direct_entry, self.tv_inputarea), text='', width=20, height=20, bg_color=self.frame)
        self.tv_in_button.grid(row=0, column=1, padx=0, sticky='nsw')

    # Select output
        self.out_info = ctk.CTkButton(self.tv_inputarea,image=self.info_icon, text='', fg_color='transparent', command=None, width=30)
        self.out_info.grid(row=2, column=0, sticky='ns', pady=5)
        self.out_info_balloon = CTkToolTip(self.out_info, message=out_info_txt, wraplength=350,alpha=1,text_color='lightblue')
        self.tv_out_direct = ctk.CTkLabel(self.tv_inputarea, text='Output Directory', justify='left', anchor='w')
        self.tv_out_direct.grid(row=2, column=1, columnspan=3, padx=0, sticky='nsew')
        self.tv_out_entry = ctk.CTkEntry(self.tv_inputarea, placeholder_text=' ')
        self.tv_out_entry.grid(row=3, column=1, columnspan=3, padx=0, sticky='nsew')
        self.tv_out_button = ctk.CTkButton(self.tv_out_entry, image=self.folder, bg_color=self.frame,
                                        command=lambda: self.browse_out_dir(self.tv_out_entry, self.tv_inputarea), text='', width=20, height=20)
        self.tv_out_button.grid(row=0, column=1, padx=0, sticky='nsw')

        self.tv_settings_info = ctk.CTkButton(self.tv_inputarea,image=self.info_icon, text='', fg_color='transparent', command=None, width=30)
        self.tv_settings_info.grid(row=6, column=0, sticky='ns', pady=(40,0))

        self.params_header = ctk.CTkLabel(self.tv_inputarea, text='Settings', justify='left', anchor='w')
        self.params_header.grid(row=6, column=1, columnspan=3, padx=3, pady=(40,0), sticky='ew')

        self.downsampling_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Downsample Mesh', onvalue=1,
                                                    offvalue=0, font=ctk.CTkFont(size=14, weight="normal"),
                                                    command=lambda: self.change_entry_state(self.downsampling_spinbox,self.voxel_multiplier, 1.0))
        self.downsampling_spinbox.grid(row=7, column=1, columnspan=3, padx=5, pady=5, sticky='ew')
        self.voxel_multiplier = ctk.CTkEntry(self.tv_inputarea, placeholder_text='Voxel Multiplier: e.g. 1.0',
                                             font=ctk.CTkFont(size=12, weight="normal", slant='italic'), border_width=1.5,
                                             state='disabled')
        self.voxel_multiplier.grid(row=8, column=1, columnspan=3, padx=20, pady=5, sticky='ew')

        self.updateres_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Update Resolution', onvalue=1,
                                                 offvalue=0, font=ctk.CTkFont(size=14, weight="normal"),
                                                 command=lambda: self.change_entry_state(self.updateres_spinbox,self.res_multiplier, 1.0))
        self.updateres_spinbox.grid(row=9, column=1, columnspan=3, padx=5, pady=5, sticky='ew')
        self.res_multiplier = ctk.CTkEntry(self.tv_inputarea, placeholder_text='Scale Multiplier: e.g. 1.0',
                                           font=ctk.CTkFont(size=12, weight="normal", slant='italic'), border_width=1.5,
                                           state='disabled')
        self.res_multiplier.grid(row=10, column=1, columnspan=3, padx=20, pady=5, sticky='ew')

        self.fill_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Fill Mesh', onvalue=1,
                                            offvalue=0, font=ctk.CTkFont(size=14, weight="normal"),
                                            command=lambda: self.change_entry_state(self.fill_spinbox,self.depth, 10))
        self.fill_spinbox.grid(row=11, column=1, columnspan=3, padx=5, pady=5, sticky='ew')
        self.depth = ctk.CTkEntry(self.tv_inputarea, placeholder_text='Octree Depth: e.g. 9',
                                     font=ctk.CTkFont(size=12, weight="normal", slant='italic'), border_width=1.5,
                                     state='disabled')
        self.depth.grid(row=12, column=1, columnspan=3, padx=20, pady=5, sticky='ew')

        self.invert_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Rotate Mesh', onvalue=1,
                                              offvalue=0, font=ctk.CTkFont(size=14, weight="normal"))
        self.invert_spinbox.grid(row=13, column=1, columnspan=3, padx=5, pady=5, sticky='ew')

        self.outarray_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Output Array Files',
                                                onvalue=1, offvalue=0,
                                                font=ctk.CTkFont(size=14, weight='normal'))
        self.outarray_spinbox.grid(row=14, column=1, columnspan=3, padx=5, pady=5, sticky='ew')

        self.visualise_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Visualise Processing Stages',
                                                 onvalue=1, offvalue=0,
                                                 font=ctk.CTkFont(size=14, weight="normal"))
        self.visualise_spinbox.grid(row=15, column=1, columnspan=3, padx=5, pady=5, sticky='ew')

        
        self.transparency_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Transparent Background',
                                                 onvalue=1, offvalue=0,
                                                 font=ctk.CTkFont(size=14, weight="normal"))
        self.transparency_spinbox.grid(row=16, column=1, columnspan=3, padx=5, pady=5, sticky='ew')

        self.rgb_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Export RGB Images',
                                                 onvalue=1, offvalue=0,
                                                 font=ctk.CTkFont(size=14, weight="normal"))
        self.rgb_spinbox.grid(row=17, column=1, columnspan=3, padx=5, pady=5, sticky='ew')
        self.grey_spinbox = ctk.CTkCheckBox(self.tv_inputarea, text='Export Greyscale Images',
                                                 onvalue=1, offvalue=0,
                                                 font=ctk.CTkFont(size=14, weight="normal"))
        self.grey_spinbox.grid(row=18, column=1, columnspan=3, padx=5, pady=5, sticky='ew')

        self.img_ext = ctk.CTkLabel(self.tv_inputarea, text='Image Type', justify='left', anchor='w', font=ctk.CTkFont(size=14, weight="normal"))
        self.img_ext.grid(row=19, column=1, columnspan=1, padx=0, sticky='nsew')
        self.img_ext_entry=ctk.CTkComboBox(self.tv_inputarea, values=['jpg','png','tif'],
                                           font=ctk.CTkFont(size=12, weight="normal"), border_width=1.5)
        self.img_ext_entry.grid(row=19, column=2, columnspan=2, padx=0, sticky='nsew')

        self.tv_run_button = ctk.CTkButton(self.tv_inputarea, image=self.play,  command=lambda: Thread(target=run_tvt, daemon=True).start(),
                                        fg_color='transparent', text='', width=35, height=35)
        self.tv_run_button.grid(row=21, column=4, rowspan=2, padx=10, pady=40)

        self.tv_run_text = ctk.CTkLabel(self.tv_inputarea, text='',
                                     font=ctk.CTkFont(size=14, weight="bold"), justify='center', wraplength=400)
        self.tv_run_text.grid(row=21, column=1, columnspan=3, rowspan=2, padx=10, pady=40)


        self.tvresultsview = ctk.CTkTabview(self.topview.tab('Topo Vis'), corner_radius=10)
        #self.tvresultsview.grid(column=1,row=0,sticky='nsew')

        self.tv_clear_tabs = ctk.CTkButton(self.tvresultsview, image=self.restart, command=lambda: self.clear_results(self.tvresultsview),
                                        fg_color='gray40', text='Clear Tabs', width=75, height=25, compound='right', anchor='center')
        # self.clear_tabs.grid(row=1, column=0, columnspan=2, sticky='ew', padx=2, pady=2)


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
