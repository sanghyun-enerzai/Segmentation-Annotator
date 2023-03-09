import cv2
import tkinter as tk
import numpy as np

from pathlib import Path
from tkinter import filedialog
from PIL import Image, ImageTk


class SegmentationAnnotator(tk.Tk):
    def __init__(self):
        super(SegmentationAnnotator, self).__init__()
        self.title('Segmentation Annotator')
        self.geometry('800x600+100+100')

        # loader frame
        loader = tk.Frame(self)
        loader.pack(side='top', fill='x', expand=False)

        # image loader
        image_loader = tk.Frame(loader)
        image_loader.pack(side='top', fill='both', expand=True)

        image_loader_label = tk.Label(image_loader, text='Load Image File: ')
        image_loader_label.pack(side='left')

        self.image_loader_entry = tk.Entry(image_loader, takefocus=0, state='readonly')
        self.image_loader_entry.pack(side='left', fill='x', expand=True)

        image_loader_button = tk.Button(image_loader, text='Find', command=self.image_filedialog)
        image_loader_button.pack(side='right')

        # annotation loader
        annot_loader = tk.Frame(loader)
        annot_loader.pack(side='top', fill='both', expand=True)

        annot_loader_label = tk.Label(annot_loader, text='Load Annotation File: ')
        annot_loader_label.pack(side='left')

        self.annot_loader_entry = tk.Entry(annot_loader, takefocus=0, state='readonly')
        self.annot_loader_entry.pack(side='left', fill='x', expand=True)

        annot_loader_button = tk.Button(annot_loader, text='Find', command=self.annot_filedialog)
        annot_loader_button.pack(side='right')

        # start and save frame
        buttons = tk.Frame(loader)
        buttons.pack(side='top', fill='x', expand=True)

        # start annotation button
        start_button = tk.Button(buttons, text='Start Annotation', command=self.start_annotation)
        start_button.pack(side='left', fill='x', expand=True)

        # save annotation button
        self.annot_save_button = tk.Button(buttons, text='Save Annotation (Crtl+S)', command=self.save_annotation, state='disabled')
        self.annot_save_button.pack(side='right', fill='x', expand=True)

        self.image_path = None
        self.annot_path = None
        self.cv_image = None
        self.cv_annot = None
        self.original_canvas_image = None
        self.loaded = False

        # draw and erase mode
        mode_frame = tk.Frame(loader)
        mode_frame.pack(side='top', fill='x', expand=True)

        self.is_draw = tk.IntVar(self, value=1)
        self.mode_draw = tk.Radiobutton(mode_frame, text='Draw Label (1)', value=1, variable=self.is_draw)
        self.mode_draw.pack(side='left', fill='x', expand=True)
        self.mode_erase = tk.Radiobutton(mode_frame, text='Erase Label (2)', value=2, variable=self.is_draw)
        self.mode_erase.pack(side='right', fill='x', expand=True)

        self.bind('1', lambda e: self.mode_draw.select())
        self.bind('2', lambda e: self.mode_erase.select())

        # annotator
        annotation = tk.Frame(self)
        annotation.pack(side='bottom', fill='both', expand=True)
        self.annot_canvas = tk.Canvas(annotation, bg='white')
        self.annot_canvas.pack(side='bottom', fill='both', expand=True)

        # variables
        self.color = (132, 12, 234)
        self.color_map = None
        self.opacity = 0.4
        self.h, self.w = 0, 0
        self.move_canvas_flag = False
        self.move_point_flag = False
        self.canvas_xpos, self.canvas_ypos = 0, 0
        self.mouse_xpos, self.mouse_ypos = 0, 0
        self.image_obj = None
        self.points = []
        self.canvas_points = []
        self.canvas_lines = []
        self.created_annots = []
        self.removed_annots = []
        
    def image_filedialog(self):
        if self.image_path is None:
            path = filedialog.askopenfilename(filetypes=[('Image File', '.jpg'), ('Image File', '.png')])
        else:
            path = filedialog.askopenfilename(initialdir=self.image_path.parent, filetypes=[('Image File', '.jpg'), ('Image File', '.png')])
        if path == '':
            return
        self.image_loader_entry.config(state='normal')
        self.image_loader_entry.delete(0, tk.END)
        self.image_loader_entry.insert(0, path)
        self.image_loader_entry.config(state='readonly')

    def annot_filedialog(self):
        if self.annot_path is None:
            path = filedialog.askopenfilename(filetypes=[('Image File', '.png')])
        else:
            path = filedialog.askopenfilename(initialdir=self.annot_path.parent, filetypes=[('Image File', '.png')])
        if path == '':
            return
        self.annot_loader_entry.config(state='normal')
        self.annot_loader_entry.delete(0, tk.END)
        self.annot_loader_entry.insert(0, path)
        self.annot_loader_entry.config(state='readonly')

    def start_annotation(self):
        self.annot_canvas.delete('all')
        self.image_path = Path(self.image_loader_entry.get())
        self.annot_path = Path(self.annot_loader_entry.get())

        self.cv_image = np.array(Image.open(str(self.image_path)).convert('RGB'))[:, :, ::-1]
        self.cv_image = np.ascontiguousarray(self.cv_image, dtype=np.uint8)
        self.cv_annot = np.array(Image.open(str(self.annot_path)).convert('RGB'))[:, :, ::-1]
        self.cv_annot = np.ascontiguousarray(self.cv_annot, dtype=np.uint8)
        cv_annot_gray = cv2.cvtColor(self.cv_annot, cv2.COLOR_BGR2GRAY)
        self.h, self.w, c = self.cv_image.shape
        self.color_map = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for i in range(3):
             self.color_map[:, :, i] = self.color[i]
        self.cv_overlay = self.cv_image.copy()
        self.cv_overlay[cv_annot_gray > 0, :] = (1 - self.opacity) * self.cv_image[cv_annot_gray > 0, :] + self.opacity * self.color_map[cv_annot_gray > 0, :]

        self.cv_overlay = cv2.cvtColor(self.cv_overlay, cv2.COLOR_BGR2RGB)
        self.cv_overlay = Image.fromarray(self.cv_overlay)

        cv_image_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        cv_image_rgb = Image.fromarray(cv_image_rgb)
        self.original_canvas_image = ImageTk.PhotoImage(image=cv_image_rgb)

        self.canvas_image = ImageTk.PhotoImage(image=self.cv_overlay)
        self.canvas_xpos, self.canvas_ypos = self.w // 2, self.h // 2
        self.canvas_image_obj = self.annot_canvas.create_image(self.canvas_xpos, self.canvas_ypos, image=self.canvas_image)

        self.annot_canvas.bind('<Button3-Motion>', self.move_canvas)
        self.annot_canvas.bind('<ButtonRelease-3>', self.release_canvas)
        self.annot_canvas.bind('<Button-1>', self.add_point)
        self.bind('<Return>', self.make_polygon)
        self.bind('<space>', self.make_polygon)
        self.bind('<Control-z>', self.remove_last_point)
        self.bind('<Control-s>', lambda e: self.save_annotation())
        self.bind('<KeyPress-q>', self.change_to_original)
        self.bind('<KeyRelease-q>', self.change_to_overlay)
        self.bind('<KeyPress-Left>', self.go_back_annot)
        self.bind('<KeyPress-Right>', self.go_front_annot)
        
        self.annot_save_button.config(state='normal')
        self.loaded = True

        self.points = []
        self.canvas_points = []
        self.canvas_lines = []
        self.created_annots = [self.cv_annot.copy()]
        self.removed_annots = []

        self.title(f'Segmentation Annotator - {self.image_path.name} | {self.annot_path.name}')

    def move_canvas(self, event):
        if self.move_canvas_flag:
            new_xpos, new_ypos = event.x, event.y
            x_diff = new_xpos - self.mouse_xpos
            y_diff = new_ypos - self.mouse_ypos
            self.canvas_xpos += x_diff
            self.canvas_ypos += y_diff
            self.annot_canvas.move(self.canvas_image_obj, x_diff, y_diff)
            for i in range(len(self.canvas_points)):
                self.annot_canvas.move(self.canvas_points[i], x_diff, y_diff)
            for i in range(len(self.canvas_lines)):
                self.annot_canvas.move(self.canvas_lines[i], x_diff, y_diff)
            self.mouse_xpos = new_xpos
            self.mouse_ypos = new_ypos
        else:
            self.move_canvas_flag = True
            self.mouse_xpos = event.x
            self.mouse_ypos = event.y

    def release_canvas(self, event):
        self.move_canvas_flag = False

    def add_point(self, event):
        x = self.annot_canvas.canvasx(event.x)
        y = self.annot_canvas.canvasy(event.y)

        img_x = self.w // 2 - self.canvas_xpos + x
        img_y = self.h // 2 - self.canvas_ypos + y

        # check invalid point
        if img_x < 0 or img_x >= self.w or img_y < 0 or img_y >= self.h:
            return

        pt = self.annot_canvas.create_polygon(x-1, y-1, x+1, y-1, x-1, y+1, x+1, y+1, fill='red', outline='red', width=1)
        self.canvas_points.append(pt)

        if len(self.points) > 0:
            last_img_x, last_img_y = self.points[-1]
            last_x = self.canvas_xpos - self.w // 2 + last_img_x
            last_y = self.canvas_ypos - self.h // 2 + last_img_y
            line = self.annot_canvas.create_line(last_x, last_y, x, y, fill='red', width=1)
            self.canvas_lines.append(line)
        
        self.points.append([img_x, img_y])

    def make_polygon(self, event):
        if len(self.points) < 3:
            return

        polygon = np.array(self.points).reshape(-1, 1, 2).astype(np.int32)
        if self.is_draw.get() == 1:
            self.cv_annot = cv2.drawContours(self.cv_annot, [polygon], -1, color=(255, 255, 255), thickness=-1)
        else:
            self.cv_annot = cv2.drawContours(self.cv_annot, [polygon], -1, color=(0, 0, 0), thickness=-1)
        cv_annot_gray = cv2.cvtColor(self.cv_annot, cv2.COLOR_BGR2GRAY)
        self.cv_overlay = self.cv_image.copy()
        self.cv_overlay[cv_annot_gray > 0, :] = (1 - self.opacity) * self.cv_image[cv_annot_gray > 0, :] + self.opacity * self.color_map[cv_annot_gray > 0, :]

        for i in range(len(self.canvas_points)):
            self.annot_canvas.delete(self.canvas_points[i])
        for i in range(len(self.canvas_lines)):
            self.annot_canvas.delete(self.canvas_lines[i])

        self.cv_overlay = cv2.cvtColor(self.cv_overlay, cv2.COLOR_BGR2RGB)
        self.cv_overlay = Image.fromarray(self.cv_overlay)
        self.canvas_image = ImageTk.PhotoImage(image=self.cv_overlay)
        self.annot_canvas.itemconfig(self.canvas_image_obj, image=self.canvas_image)

        self.points = []
        self.canvas_points = []
        self.canvas_lines = []

        self.created_annots.append(self.cv_annot.copy())
        self.removed_annots = []

    def remove_last_point(self, event):
        if not self.loaded:
            return
        if len(self.canvas_points) > 0:
            self.annot_canvas.delete(self.canvas_points.pop())
            self.points.pop()
        if len(self.canvas_lines) > 0:
            self.annot_canvas.delete(self.canvas_lines.pop())

    def save_annotation(self):
        path = filedialog.asksaveasfilename(initialfile=self.annot_path.name, filetypes=[('Image File', '.png')])
        if path[-4:] != '.png':
            path += '.png'
        cv_annot_img = Image.fromarray(self.cv_annot)
        cv_annot_img.save(path)

    def change_to_original(self, event):
        if not self.loaded:
            return
        self.annot_canvas.itemconfig(self.canvas_image_obj, image=self.original_canvas_image)

    def change_to_overlay(self, event):
        if not self.loaded:
            return
        self.annot_canvas.itemconfig(self.canvas_image_obj, image=self.canvas_image)

    def go_back_annot(self, event):
        if len(self.created_annots) > 0:
            self.cv_annot = self.created_annots.pop()
            self.removed_annots.append(self.cv_annot.copy())
            
            cv_annot_gray = cv2.cvtColor(self.cv_annot, cv2.COLOR_BGR2GRAY)
            self.cv_overlay = self.cv_image.copy()
            self.cv_overlay[cv_annot_gray > 0, :] = (1 - self.opacity) * self.cv_image[cv_annot_gray > 0, :] + self.opacity * self.color_map[cv_annot_gray > 0, :]
            self.cv_overlay = cv2.cvtColor(self.cv_overlay, cv2.COLOR_BGR2RGB)
            self.cv_overlay = Image.fromarray(self.cv_overlay)
            self.canvas_image = ImageTk.PhotoImage(image=self.cv_overlay)
            self.annot_canvas.itemconfig(self.canvas_image_obj, image=self.canvas_image)

    def go_front_annot(self, event):
        if len(self.removed_annots) > 0:
            self.cv_annot = self.removed_annots.pop()
            self.created_annots.append(self.cv_annot.copy())

            cv_annot_gray = cv2.cvtColor(self.cv_annot, cv2.COLOR_BGR2GRAY)
            self.cv_overlay = self.cv_image.copy()
            self.cv_overlay[cv_annot_gray > 0, :] = (1 - self.opacity) * self.cv_image[cv_annot_gray > 0, :] + self.opacity * self.color_map[cv_annot_gray > 0, :]
            self.cv_overlay = cv2.cvtColor(self.cv_overlay, cv2.COLOR_BGR2RGB)
            self.cv_overlay = Image.fromarray(self.cv_overlay)
            self.canvas_image = ImageTk.PhotoImage(image=self.cv_overlay)
            self.annot_canvas.itemconfig(self.canvas_image_obj, image=self.canvas_image)


if __name__ == '__main__':
        annotator = SegmentationAnnotator()
        annotator.mainloop()
