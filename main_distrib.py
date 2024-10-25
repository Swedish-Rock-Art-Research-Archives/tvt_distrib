# from pc_image import *
# from pc_mesh import *
from topovis_main import *
import ftfy
from multiprocessing import Process

import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter.filedialog import *

import customtkinter as ctk
from CTkToolTip import *

import distutils.util

import os
import sys
import ctypes

from PIL import Image

import datetime

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import multiprocessing
multiprocessing.freeze_support()


def dist_path(relative_path):
    base_path = sys._MEIPASS
    return os.path.join(base_path, relative_path)

#### Help texts####
overview_txt = """Topography Visualisation Toolbox was originally developed using landscape archaeology methods to highlight the detail of carved rock art. This basic version of the toolbox includes a tool that improves the visualisation of carved surfaces, particularly in cultural heritage applications. 
                                    
Click on the Topo Vis tab to visualise mesh data as images using a range of parameters. For further guidance on how to use the tool, hover over the information icons within with Topo Vis tab.  A pop-up will appear with information about the required input(s) and suggestions for how to use the tools with your data.
                                    
The program is provided "AS IS" under MIT copyright license.

"""

mesh_info_txt = 'Path of folder containing mesh files (.obj, .stl, .ply)'
out_info_txt = 'Path of folder the processed images will be saved in. If this folder does not exist, manually type the path in the text box and it will be created during processing.  If this field is empty when you start a process, a default directory will be created in the input directory.'
downsample_txt = 'Simplifies the mesh with vertex clustering.  Set the voxel multiplier to determine the voxel (region) size - final voxel size is equivalent to voxel multiplier * mesh edge resolution.'
update_res_txt = 'Scales the values in the point cloud derived from the mesh.  Set the scale multiplier to the desired conversion.  Setting this parameter >1 can be useful for meshes derived from SfM data.'
fill_txt = 'Fills holes in the mesh using Poisson surface reconstruction (Khazdan,2006; Open3D).  Set the octree depth parameter to influence the resolution of the resulting mesh - a higher value will generate a mesh with more detail.'
rotate_txt = 'STRICT will rotate the resulting point cloud using a [[1, 0, 0, 0], [0, -1, 0, 0],[0, 0, -1, 0], [0, 0, 0, 1]] matrix.  AUTO will rotate the point cloud using incremental principal component analysis to find the best angle of rotation.'
out_array_txt = 'Saves .pcd and .npz files of the original and transformed points.'
visualise_txt = 'Will display the results of the mesh and point cloud processing stages.'
transparent_txt = 'Export each plot with a transparent background. If unchecked, plots will have a black background.  Please also choose a suitable export format when using a transparent background.'
rgb_txt = 'Export only plots in RGB colourmap.'
grey_txt = 'Export only plot in greyscale.'


info_icon = ctk.CTkImage(Image.open(dist_path('info.png')), size=(13, 13))
folder = ctk.CTkImage(Image.open(dist_path('folder.png')), size=(21, 21))
play = ctk.CTkImage(Image.open(dist_path('play.png')), size=(35, 35))
wait = ctk.CTkImage(Image.open(dist_path('restart.png')), size=(35, 35))
generate = ctk.CTkImage(Image.open(dist_path('generate.png')), size=(21, 21))
expand = ctk.CTkImage(Image.open(dist_path('expand.png')), size=(12, 12))
expand_rot = ctk.CTkImage(Image.open(dist_path('expand_rot.png')), size=(12, 12))

header_img_dark = Image.open(dist_path('header_landscape.png'))
img_width = 1000
img_height = 500
header = ctk.CTkImage(light_image=header_img_dark,
                      dark_image=header_img_dark,
                      size=(img_width,img_height))


ctk.FontManager.load_font(dist_path('BarlowSemiCondensed-Thin.ttf'))
ctk.FontManager.load_font(dist_path('BarlowSemiCondensed-Light.ttf'))



# Define colours to override those in json and tkinter style
background = "#202020"
frame = "#202020"
entry = "gray17"
button = "gray22"
hover = "gray10"
text = 'white'
tooltip = 'gray10'

class AppOverview(ctk.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.menu_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Thin", size=16)

        
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        
        self.sidebar_img = ctk.CTkLabel(self, text=None, image=header, corner_radius=0)
        self.sidebar_img.grid(row=0, column=0, sticky='nsew', pady=(0, 5))
        self.overview_text = ctk.CTkLabel(self, justify='left', wraplength=850,text=overview_txt, font=self.menu_font)
        self.overview_text.grid(row=1, column=0, sticky='nsew')

class InputParameters(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.columnconfigure((1,2), weight=1)
        self.rowconfigure(0, weight=1)
        
        self.menu_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Thin", size=16)
        self.tooltip_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Light", size=14, weight='bold')
        self.input_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Light", size=14, slant='italic')

        self.mesh_info = ctk.CTkButton(
            self, image=info_icon, text='', fg_color='transparent', command=None, width=30)
        self.mesh_info.grid(row=0, column=0, sticky='e', pady=5)
        self.mesh_info_balloon = CTkToolTip(self.mesh_info, wraplength=250, text_color='whitesmoke', alpha=1, font=self.tooltip_font)
        self.mesh_direct = ctk.CTkLabel(
            self, justify='left', anchor='w', font=self.menu_font)
        self.mesh_direct.grid(
            row=0, column=1, columnspan=3, padx=0, sticky='nsew')
        self.tv_direct_entry = ctk.CTkEntry(
            self, placeholder_text=' ', font=self.input_font)
        self.tv_direct_entry.grid(row=1, column=1, columnspan=3, sticky='nsew')
        self.tv_in_button = ctk.CTkButton(self.tv_direct_entry, image=folder,
                                          command=lambda: self.browse_dir(self.tv_direct_entry, self), text='', width=20, height=20, bg_color=frame)
        self.tv_in_button.grid(row=0, column=2, padx=0, sticky='nsw')

        
    def browse_dir(self, field, tab):
        in_directory = askdirectory(parent=tab)
        field.delete(0, END)
        field.insert(tk.END, in_directory)
        
        
    @classmethod   
    def generate_dir(self,field,processing_section,input_dir):
        if bool(processing_section.downsampling_spinbox.get()):
            ds_val = processing_section.voxel_multiplier.get()
        else:
            ds_val = 'No'
        if bool(processing_section.updateres_spinbox.get()):
            res_val = processing_section.res_multiplier.get()
        else:
            res_val = 'No'
        if bool(processing_section.clean_spinbox.get()):
            cleaning_val = 'Auto'
        else:
            cleaning_val = 'No'
        
        curr_time = datetime.datetime.now()
        date = f"{curr_time.day}{curr_time.month}{curr_time.year}"
        time = f"{curr_time.hour}{curr_time.minute}"
        
        new_out_dir = f"{input_dir.get()}/TVT/DS{ds_val}_RES{res_val}_{cleaning_val}CLEAN_{date}{time}"
        field.delete(0, END)
        field.insert(tk.END, new_out_dir)
        return new_out_dir        
         
class ProcessingSettings(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.menu_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Thin", size=16)
        self.tooltip_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Light", size=14, weight='bold')
        self.input_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Light", size=14, slant='italic')
        
        self.downsampling_info = ctk.CTkButton(
            self, image=info_icon, text='', fg_color='transparent', command=None, width=30)
        self.downsampling_info.grid(row=0, column=0, sticky='ns', pady=5)
        self.downsampling_info_balloon = CTkToolTip(
            self.downsampling_info, message=downsample_txt, wraplength=250, text_color='whitesmoke', alpha=1, font=self.tooltip_font)
        self.downsampling_spinbox = ctk.CTkCheckBox(self, text='Downsample Mesh', onvalue=1, offvalue=0, font=self.menu_font,
                                                    command=lambda: self.change_entry_state(self.downsampling_spinbox, self.voxel_multiplier, 1.0))
        self.downsampling_spinbox.grid(
            row=0, column=1, columnspan=3, padx=5, pady=5, sticky='ew')
        self.voxel_multiplier = ctk.CTkEntry(self, placeholder_text='Voxel Multiplier: e.g. 1.0', font=self.input_font, border_width=1.5,
                                             state='disabled')
        self.voxel_multiplier.grid(
            row=1, column=1, columnspan=3, padx=20, pady=5, sticky='ew')

        self.res_info = ctk.CTkButton(
            self, image=info_icon, text='', fg_color='transparent', command=None, width=30)
        self.res_info.grid(row=2, column=0, sticky='ns', pady=5)
        self.res_info_balloon = CTkToolTip(self.res_info, message=update_res_txt,
                                           wraplength=250, text_color='whitesmoke', alpha=1, font=self.tooltip_font)
        self.updateres_spinbox = ctk.CTkCheckBox(self, text='Update Resolution', onvalue=1,
                                                 offvalue=0, font=self.menu_font,
                                                 command=lambda: self.change_entry_state(self.updateres_spinbox, self.res_multiplier, 1.0))
        self.updateres_spinbox.grid(
            row=2, column=1, columnspan=3, padx=5, pady=5, sticky='ew')
        self.res_multiplier = ctk.CTkEntry(self, placeholder_text='Scale Multiplier: e.g. 1.0', font=self.input_font, border_width=1.5,
                                           state='disabled')
        self.res_multiplier.grid(
            row=3, column=1, columnspan=3, padx=20, pady=5, sticky='ew')
        
        self.rotation_info = ctk.CTkButton(
            self, image=info_icon, text='', fg_color='transparent', command=None, width=30)
        self.rotation_info.grid(row=4, column=0, sticky='ns', pady=5)
        self.rotation_info_balloon = CTkToolTip(self.rotation_info, message=rotate_txt, wraplength=250,text_color='whitesmoke', alpha=1, font=self.tooltip_font)
        
        self.rotation_label = ctk.CTkCheckBox(self, text='Rotate Mesh',
                                                    onvalue=1, offvalue=0, font=self.menu_font, command = lambda: self.change_combo_state(self.rotation_label, self.rotation_entry))
        self.rotation_label.grid(row=4, column=1, columnspan=3, padx=5, pady=5, sticky='ew')
        self.rotation_entry = ctk.CTkComboBox(self, font=self.menu_font, border_width=1.5, state='disabled',dropdown_font=self.menu_font, values=['AUTO','STRICT'])
        self.rotation_entry.grid(
            row=5, column=1, columnspan=3, padx=20, pady=5, sticky='ew')
        
        
        self.clean_info = ctk.CTkButton(
            self, image=info_icon, text='', fg_color='transparent', command=None, width=30)
        self.clean_info.grid(row=6, column=0, sticky='ns', pady=5)
        self.clean_info_balloon = CTkToolTip(
            self.clean_info, message='Remove noise from converted mesh based on the number of neighbours', wraplength=250, text_color='whitesmoke', alpha=1, font=self.tooltip_font)
        self.clean_spinbox = ctk.CTkCheckBox(self, text='Outlier Removal',
                                                    onvalue=1, offvalue=0, font=self.menu_font)
        self.clean_spinbox.grid(
            row=6, column=1, columnspan=3, padx=5, pady=5, sticky='ew')
    @classmethod  
    def change_entry_state(self, checkbox, entry, default_val):
        if checkbox.get() != 0:
            entry.configure(state='normal')
            if entry.get() == '':
                entry.delete(0, END)
                entry.insert(END, default_val)
            entry.update()
        else:
            entry.configure(state='disabled')
            entry.update()
            
    def change_combo_state(self,checkbox,combobox):
        if checkbox.get() != 0:
            combobox.configure(state='normal')
            if combobox.get() != 'STRICT':
                combobox.set('AUTO')
            combobox.update()
        else:
            combobox.configure(state='disabled')
            combobox.update()
                    
class OutputSettings(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.menu_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Thin", size=16)
        self.tooltip_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Light", size=14, weight='bold')
        self.input_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Light", size=14, slant='italic')
        
        self.transparency_info = ctk.CTkButton(
            self, image=info_icon, text='', fg_color='transparent', command=None, width=30)
        self.transparency_info.grid(row=0, column=0, sticky='ns', pady=5)
        self.transparency_info_balloon = CTkToolTip(
            self.transparency_info, message=transparent_txt, wraplength=250, text_color='whitesmoke', alpha=1, font=self.tooltip_font)
        self.transparency_spinbox = ctk.CTkCheckBox(self, text='Transparent Background',
                                                    onvalue=1, offvalue=0, font=self.menu_font)
        self.transparency_spinbox.grid(
            row=0, column=1, columnspan=3, padx=5, pady=5, sticky='ew')
        
        self.colour_info = ctk.CTkButton(self, image=info_icon, text='', fg_color='transparent', command=None, width=30)
        self.colour_info.grid(row=1, column=0, sticky='ns', pady=5)
        self.colour_info_balloon = CTkToolTip(self.colour_info, message='RGB will only export the RGB colour plots.  GREY will only export the plots in greyscale.  BOTH will export both RGB and greyscale plots.', wraplength=250, text_color='whitesmoke', alpha=1, font=self.tooltip_font)
        self.colour_picker = ctk.CTkLabel(self, text='Output Type', justify='left', anchor='w', font=self.menu_font)
        self.colour_picker.grid(row=1, column=1, columnspan=1, padx=0, sticky='nsew')
        self.colour_entry = ctk.CTkComboBox(self, values=['BOTH', 'GREY', 'RGB'], font=self.menu_font, dropdown_font=self.menu_font, border_width=1.5)
        self.colour_entry.grid(row=2, column=2, columnspan=2, padx=0, sticky='ew')
        
        self.img_ext = ctk.CTkLabel(
            self, text='Image Type', justify='left', anchor='w', font=self.menu_font)
        self.img_ext.grid(row=3, column=1, columnspan=1,
                          padx=0, sticky='nsew')
        self.img_ext_entry = ctk.CTkComboBox(self, values=[
                                             'png', 'jpg','tif'], font=self.menu_font, dropdown_font=self.menu_font, border_width=1.5)
        self.img_ext_entry.grid(
            row=4, column=2, columnspan=2, padx=0, sticky='ew')
        
class OptionalSettings(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.menu_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Thin", size=16)
        self.tooltip_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Light", size=14, weight='bold')
        self.input_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Light", size=14, slant='italic')
        
        self.fill_info = ctk.CTkButton(
            self, image=info_icon, text='', fg_color='transparent', command=None, width=30)
        self.fill_info.grid(row=0, column=0, sticky='ns', pady=5)
        self.fill_info_balloon = CTkToolTip(
            self.fill_info, message=fill_txt, wraplength=250, text_color='whitesmoke', alpha=1, font=self.tooltip_font)
        self.fill_spinbox = ctk.CTkCheckBox(self, text='Fill Mesh', onvalue=1,
                                            offvalue=0, font=self.menu_font,
                                            command=lambda: ProcessingSettings.change_entry_state(self.fill_spinbox, self.depth, 10))
        self.fill_spinbox.grid(
            row=0, column=1, columnspan=3, padx=5, pady=5, sticky='ew')
        self.depth = ctk.CTkEntry(self, placeholder_text='Octree Depth: e.g. 9', font=self.input_font, border_width=1.5,
                                  state='disabled')
        self.depth.grid(row=1, column=1, columnspan=3,
                        padx=20, pady=5, sticky='ew')

        self.array_info = ctk.CTkButton(
            self, image=info_icon, text='', fg_color='transparent', command=None, width=30)
        self.array_info.grid(row=2, column=0, sticky='ns', pady=5)
        self.array_info_balloon = CTkToolTip(
            self.array_info, message=out_array_txt, wraplength=250, text_color='whitesmoke', alpha=1, font=self.tooltip_font)
        self.outarray_spinbox = ctk.CTkCheckBox(self, text='Output Array Files',
                                                onvalue=1, offvalue=0, font=self.menu_font)
        self.outarray_spinbox.grid(
            row=2, column=1, columnspan=3, padx=5, pady=5, sticky='ew')

        self.visualise_info = ctk.CTkButton(
            self, image=info_icon, text='', fg_color='transparent', command=None, width=30)
        self.visualise_info.grid(row=3, column=0, sticky='ns', pady=5)
        self.visualise_info_balloon = CTkToolTip(
            self.visualise_info, message=visualise_txt, wraplength=250, text_color='whitesmoke', alpha=1, font=self.tooltip_font)
        self.visualise_spinbox = ctk.CTkCheckBox(self, text='Visualise Processing Stages',
                                                 onvalue=1, offvalue=0, font=self.menu_font)
        self.visualise_spinbox.grid(
            row=3, column=1, columnspan=3, padx=5, pady=5, sticky='ew')
        
class TvtApp(ctk.CTk, tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.heading_font = ctk.CTkFont(
            family="Barlow Semi Condensed Thin", size=20, weight='bold')   
        self.overview_font = ctk.CTkFont(family="Barlow Semi Condensed Thin", size=20)
        self.menu_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Thin", size=16)
        self.tooltip_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Light", size=14, weight='bold')
        self.input_font = ctk.CTkFont(
                    family="Barlow Semi Condensed Light", size=14, slant='italic')

# Define app
        self.title('Topography Visualisation Toolbox')
        self.wm_iconbitmap(default=dist_path('001.ico'))

        self.resizable(False, True)
        # self.wm_minsize(int(start_width/1.5),int(start_height/1.5))
        ctk.set_appearance_mode('Dark')
        ctk.set_default_color_theme(dist_path('custom_theme.json'))

        self.geometry('1000x850')

        self.lift()
        self.attributes('-transparentcolor', 'pink')

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        self.topview = ctk.CTkTabview(self, corner_radius=20)
        self.topview.grid(row=1, column=0, padx=(
            20, 20), pady=(5, 5), sticky='nsew')
    # Overview Tab
        self.topview.add('Overview')
        self.topview.tab('Overview').columnconfigure(0, weight=1)
        self.topview.tab('Overview').rowconfigure((0), weight=1)
        
        self.overview = AppOverview(self.topview.tab('Overview'))
        self.overview.grid(row=0, column=0, sticky='nsew')

        

    # Topo Vis Tab
        self.topview.add('Topo Vis')
        self.topview.tab('Topo Vis').rowconfigure((0), weight=1)
        self.topview.tab('Topo Vis').columnconfigure((0), weight=1)
            
        def run_tvt():
            self.tv_run_button.configure(image=wait, text=None)
            self.tv_run_button.update()
            
            if bool(self.output_directory.tv_direct_entry.get()) == False:
                default_output_dir = InputParameters.generate_dir(self.output_directory.tv_direct_entry,self.processing_section,self.input_directory.tv_direct_entry)
            else:
                default_output_dir = f"{ftfy.fix_text(str(self.output_directory.tv_direct_entry.get()))}/"

            data_path = f"{ftfy.fix_text(str(self.input_directory.tv_direct_entry.get()))}/"
            
            save_path = ftfy.fix_text(default_output_dir)
            
            meta_data_path = f"{default_output_dir}/output_summary.csv"
            visualize_steps = bool(self.optional_section.visualise_spinbox.get())
            
            downsample_mesh = bool(self.processing_section.downsampling_spinbox.get())
            voxel_multiplier = float(self.processing_section.voxel_multiplier.get(
            )) if self.processing_section.downsampling_spinbox.get() > 0 else 1.0
            fill_holes = bool(self.optional_section.fill_spinbox.get())
            depth_size = int(
                self.optional_section.depth.get()) if self.optional_section.fill_spinbox.get() > 0 else 9
            rotation_choice = self.processing_section.rotation_entry.get()
            if not bool(self.processing_section.rotation_label.get()):
                flip_mesh = False
                auto_rotate = False
            else:
                if rotation_choice == 'AUTO':
                    flip_mesh = False
                    auto_rotate = True
                else:
                    flip_mesh = True
                    auto_rotate = False
                
            update_resolution = bool(self.processing_section.updateres_spinbox.get())
            scale_multiplier = float(self.processing_section.res_multiplier.get(
            )) if self.processing_section.updateres_spinbox.get() > 0 else 1.0
            progress_text = self.tv_run_text
            array = bool(self.optional_section.outarray_spinbox.get())
            
            if self.output_section.colour_entry.get() == 'BOTH':
                export_rgb = True
                export_grey = True
            elif self.output_section.colour_entry.get() == 'RGB':
                export_rgb = True
                export_grey = False
            else:
                export_rgb = False
                export_grey = True
                
            transparency = bool(self.output_section.transparency_spinbox.get())
            img_format = self.output_section.img_ext_entry.get()
            
            clean_noise = bool(self.processing_section.clean_spinbox.get())

            files = []
            for dp, dn, filenames in os.walk(data_path):
                for file in filenames:
                    if os.path.splitext(file)[1] in ['.stl', '.ply', '.obj']:
                        files.append((' '.join(file.split('.')[0].split(' ')), os.path.join(
                            dp, file), ''.join(dp.replace(data_path, save_path))))
            
            try:
                topovis.start_tv(
                    data_path, save_path, meta_data_path, visualize_steps, downsample_mesh, voxel_multiplier, fill_holes,
                    depth_size, flip_mesh, update_resolution, scale_multiplier, progress_text, array,
                    files=files, export_rgb=export_rgb, export_grey=export_grey, img_type=img_format, transparency=transparency, clean_noise=clean_noise,auto_rotate=auto_rotate, dpi=900)
                
            except MemoryError as error:
                topovis.log_exception(error, False)
            except Exception as exception:
                topovis.log_exception(exception, True)
            except:
                topovis.log_exception('An error occurred', True)
                
            self.output_directory.tv_direct_entry.delete(0, END)
            self.output_directory.tv_direct_entry.update()

            self.tv_run_button.configure(
                image=play, text=None, command=run_tvt)
            self.tv_run_button.update()

            self.tv_run_text.configure(text='')
            self.tv_run_text.update()
            

        self.frame_container = ctk.CTkScrollableFrame(self.topview.tab('Topo Vis'))
        self.frame_container.grid(row=0, column=0, columnspan=2, padx=15, pady=10, sticky='nsew')
        self.frame_container.columnconfigure((0), weight=1)
        self.frame_container.rowconfigure((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18), weight=1)
        

        self.input_directory = InputParameters(self.frame_container,border_width=0)
        self.input_directory.mesh_direct.configure(text='Input Directory')
        self.input_directory.mesh_info_balloon.configure(message=mesh_info_txt)
        self.input_directory.grid(row=0, column=0, padx=20, pady=0, sticky='nsew')

        
        processing_header_offset = self.input_directory.grid_info()['row']+self.input_directory.grid_info()['rowspan']
        
        self.processing_header= ctk.CTkLabel(
            self.frame_container, text='Mesh Settings', justify='left', anchor='w', font=self.heading_font)
        self.processing_header.grid(
            row=processing_header_offset, column=0, padx=3, pady=(40, 0), sticky='ew')
        
        processing_sect_offset = self.processing_header.grid_info()['row']+self.processing_header.grid_info()['rowspan']
        self.processing_section = ProcessingSettings(self.frame_container, border_width=0)
        self.processing_section.grid(row=processing_sect_offset, column=0, padx=20, pady=10, sticky='nsew')
        
        
        output_header_offset = self.processing_section.grid_info()['row']+self.processing_section.grid_info()['rowspan']
        
        self.output_header = ctk.CTkLabel(self.frame_container, text='Output Settings', justify='left', anchor='w', font=self.heading_font)
        self.output_header.grid(row=output_header_offset, column=0, padx=3, pady=(40, 0), sticky='ew')
        
        output_directory_offset = self.output_header.grid_info()['row']+self.output_header.grid_info()['rowspan']
        
        
        self.output_directory = InputParameters(self.frame_container,border_width=0)
        self.output_directory.mesh_direct.configure(text='Output Directory')
        self.output_directory.mesh_info_balloon.configure(message=out_info_txt)
        self.output_directory.grid(row=output_directory_offset, column=0, padx=20, pady=0, sticky='nsew')
        
        self.generate_button = ctk.CTkButton(self.output_directory.tv_direct_entry, image=generate, bg_color=frame, text='', width=20, height=20)
        
        self.generate_button.grid(row=0, column=1, padx=0, sticky='nsw')
        
        self.generate_button.configure(command=lambda: InputParameters.generate_dir(self.output_directory.tv_direct_entry, self.processing_section, self.input_directory.tv_direct_entry))
        
        self.generate_info_balloon = CTkToolTip(
            self.generate_button, message='Automatically generate the output folder name based on you processing settings.', wraplength=250, text_color='whitesmoke', alpha=1, font=self.tooltip_font)
        
        output_sect_offset = self.output_directory.grid_info()['row']+self.output_directory.grid_info()['rowspan']
        self.output_section = OutputSettings(self.frame_container, border_width=0)
        self.output_section.grid(row=output_sect_offset, column=0, padx=20, pady=10, sticky='ew')
        
        
        optional_header_offset = self.output_section.grid_info()['row']+self.output_section.grid_info()['rowspan']
        self.optional_header = ctk.CTkLabel(self.frame_container, text='Optional Settings', justify='left', anchor='w', font=self.heading_font)
        self.optional_header.grid(row=optional_header_offset, column=0, padx=3, pady=(40,0), sticky='ew')
        
        optional_sect_offset = self.optional_header.grid_info()['row']+self.optional_header.grid_info()['rowspan']
        self.optional_section = OptionalSettings(self.frame_container, border_width=0)
        
        self.expand_section = ctk.CTkButton(self.frame_container, image=expand_rot, text='', fg_color='transparent', command=lambda: self.expand_menu(self.optional_section,optional_sect_offset),width=20)
        self.expand_section.grid(row=optional_header_offset, column=1, padx=0, pady=(40,0), sticky='ew')
        
        run_area_offset = self.frame_container.grid_info()['row']+self.frame_container.grid_info()['rowspan']
        self.tv_run_button = ctk.CTkButton(self.topview.tab('Topo Vis'), image=play,  command=run_tvt,fg_color='transparent', text='', width=35, height=35)
        self.tv_run_button.grid(row=run_area_offset, column=1, rowspan=2, padx=10, pady=40)

        self.tv_run_text = ctk.CTkLabel(
            self.topview.tab('Topo Vis'), text='', font=self.menu_font, justify='center', wraplength=200)
        self.tv_run_text.grid(row=run_area_offset, column=0, columnspan=1,
                              rowspan=4, padx=10, pady=40)

        
        
    def expand_menu(self,section,offset):
        if section.winfo_manager() != 'grid':
            section.grid(row=offset, column=0, padx=10, pady=10, sticky='ew')
            self.expand_section.configure(image=expand)
            self.expand_section.update()
            self.configure(scrollregion=self.bbox('all'))
            self.frame_container.update()
            self.frame_container._parent_canvas.yview_moveto(1.0)
            
        else:
            self.configure(scrollregion=self.bbox('all'))
            section.grid_remove()
            self.expand_section.configure(image=expand_rot)
            self.expand_section.update()
            self.frame_container._parent_canvas.yview_moveto(1.0)


if __name__ == '__main__':

    try:
        import pyi_splash

        pyi_splash.close()
    except:
        pass

    app = TvtApp()
    app.mainloop()

    app.protocol('WM_DELETE_WINDOW', sys.exit())
