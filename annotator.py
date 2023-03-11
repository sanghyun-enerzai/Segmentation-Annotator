import cv2
import tkinter as tk
import tkinter.messagebox as messagebox
import numpy as np

from pathlib import Path
from tkinter import filedialog
from PIL import Image, ImageTk


class SegmentationAnnotator(tk.Tk):
    def __init__(self):
        super(SegmentationAnnotator, self).__init__()
        self.title('Segmentation Annotator')
        self.geometry('800x600+100+100')

        tool = tk.Frame(self)
        tool.pack(side='top', anchor='n', fill='x')

        # loader frame
        loader = tk.Frame(tool)
        loader.pack(side='top', fill='x', expand=True)

        # image loader
        image_loader = tk.Frame(loader)
        image_loader.pack(side='left', fill='x', expand=True)

        image_finder = tk.Frame(image_loader)
        image_finder.pack(side='top', fill='x', expand=True)

        image_loader_label = tk.Label(image_finder, text='Image Files')
        image_loader_label.pack(side='left', fill='x', expand=True)

        image_loader_button = tk.Button(image_finder, text='Find', command=self.image_filedialog)
        image_loader_button.pack(side='right')

        image_list = tk.Frame(image_loader)
        image_list.pack(side='top', fill='x', expand=True)

        self.image_loader_listbox = tk.Listbox(image_list, selectmode='extended', height=6)
        self.image_loader_listbox.pack(side='left', fill='x', expand=True)
        self.image_loader_listbox.bind('<Delete>', self.delete_image_from_loader)

        image_loader_scroll = tk.Scrollbar(image_list)
        image_loader_scroll.pack(side='right', fill='y')

        self.image_loader_listbox.config(yscrollcommand=image_loader_scroll.set)
        image_loader_scroll.config(command=self.image_loader_listbox.yview)

        # annotation loader
        annot_loader = tk.Frame(loader)
        annot_loader.pack(side='left', fill='x', expand=True)

        annot_finder = tk.Frame(annot_loader)
        annot_finder.pack(side='top', fill='x', expand=True)

        annot_loader_label = tk.Label(annot_finder, text='Annotation Files')
        annot_loader_label.pack(side='left', fill='x', expand=True)

        annot_loader_scratch = tk.Frame(annot_finder)
        annot_loader_scratch.pack(side='left')

        annot_loader_scratch_label = tk.Label(annot_loader_scratch, text='From scratch')
        annot_loader_scratch_label.pack(side='left')

        self.from_scratch = tk.IntVar(self, value=0)
        annot_loader_scratch_button = tk.Checkbutton(annot_loader_scratch, variable=self.from_scratch)
        annot_loader_scratch_button.pack(side='right')

        annot_loader_button = tk.Button(annot_finder, text='Find', command=self.annot_filedialog)
        annot_loader_button.pack(side='right')

        annot_list = tk.Frame(annot_loader)
        annot_list.pack(side='top', fill='x', expand=True)

        self.annot_loader_listbox = tk.Listbox(annot_list, selectmode='extended', height=6)
        self.annot_loader_listbox.pack(side='left', fill='x', expand=True)
        self.annot_loader_listbox.bind('<Delete>', self.delete_annot_from_loader)

        annot_loader_scroll = tk.Scrollbar(annot_list)
        annot_loader_scroll.pack(side='right', fill='y')

        self.annot_loader_listbox.config(yscrollcommand=annot_loader_scroll.set)
        annot_loader_scroll.config(command=self.annot_loader_listbox.yview)

        # start annotation button
        start_button = tk.Button(loader, text='Start Annotation', command=self.start_annotation)
        start_button.pack(side='right', fill='y')

        # annotation selector
        annot_selector = tk.Frame(tool)
        annot_selector.pack(side='top', fill='x')

        annot_selector_left = tk.Button(annot_selector, text='(Page Up) <--', command=self.go_prev_image)
        annot_selector_left.pack(side='left')
        self.bind('<Prior>', lambda e: self.go_prev_image())

        self.annot_selector_filename_label = tk.Label(annot_selector, text='')
        self.annot_selector_filename_label.pack(side='left', fill='x', expand=True)

        self.annot_selector_number_label = tk.Label(annot_selector, text='(0/0)')
        self.annot_selector_number_label.pack(side='left', fill='x', expand=True)

        annot_selector_right = tk.Button(annot_selector, text='--> (Page Down)', command=self.go_next_image)
        annot_selector_right.pack(side='right')
        self.bind('<Next>', lambda e: self.go_next_image())

        self.image_paths = []
        self.annot_paths = []
        self.cv_image = None
        self.cv_annots = []
        self.original_canvas_image = None
        self.loaded = False

        # draw and erase mode
        mode_frame = tk.Frame(tool, background='blue')
        mode_frame.pack(side='top', fill='x')

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
        self.scratch = False
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
        self.total = 0
        self.current = 0
        
    def image_filedialog(self):
        if len(self.image_paths) == 0:
            paths = filedialog.askopenfilename(multiple=True, filetypes=[('Image File', '.jpg'), ('Image File', '.png')])
        else:
            paths = filedialog.askopenfilename(multiple=True, initialdir=self.image_paths[0].parent, filetypes=[('Image File', '.jpg'), ('Image File', '.png')])
        if paths == '':
            return
        self.image_paths = sorted([Path(path) for path in paths])
        self.image_loader_listbox.delete(0, tk.END)
        for i in range(len(self.image_paths)):
            self.image_loader_listbox.insert(i, self.image_paths[i].parent.name + '/' + self.image_paths[i].name)

    def annot_filedialog(self):
        if len(self.annot_paths) == 0:
            paths = filedialog.askopenfilename(multiple=True, filetypes=[('Image File', '.png')])
        else:
            paths = filedialog.askopenfilename(multiple=True, initialdir=self.annot_paths[0].parent, filetypes=[('Image File', '.png')])
        if paths == '':
            return
        self.annot_paths = sorted([Path(path) for path in paths])
        self.annot_loader_listbox.delete(0, tk.END)
        for i in range(len(self.annot_paths)):
            self.annot_loader_listbox.insert(i, self.annot_paths[i].parent.name + '/' + self.annot_paths[i].name)

    def start_annotation(self):
        self.scratch = self.from_scratch.get() == 1
        if (not self.scratch) and len(self.image_paths) != len(self.annot_paths):
            messagebox.showerror('Error', 'Number of Image Files != Number of Annotation Files')
            return
        
        valid_flag = True
        if not self.scratch:
            for i in range(len(self.image_paths)):
                image_name = self.image_paths[i].name
                annot_name = self.annot_paths[i].name
                if image_name[:-4] != annot_name[:-4]:
                    valid_flag = False
                break
        if not valid_flag:
            messagebox.showerror('Error', 'Name of Image File != Name of Annotation File')
            return

        self.total = len(self.image_paths)
        if self.total == 0:
            self.current = 0
            self.annot_selector_filename_label.config(text='')
            self.annot_selector_number_label.config(text=f'0/0')
            return
        
        self.loaded = True
        self.current = 0
        if not self.scratch:
            self.cv_annots = [np.ascontiguousarray(np.array(Image.open(str(annot_path)).convert('RGB'))[:, :, ::-1], dtype=np.uint8) for annot_path in self.annot_paths]
        else:
            self.cv_annots = []
            for i in range(self.total):
                w, h = Image.open(str(self.image_paths[i])).size
                self.cv_annots.append(np.zeros((h, w, 3), dtype=np.uint8))
        self.load_new_image()

        self.annot_canvas.bind('<Button3-Motion>', self.move_canvas)
        self.annot_canvas.bind('<ButtonRelease-3>', self.release_canvas)
        self.annot_canvas.bind('<Button-1>', self.add_point)
        self.bind('<Return>', self.make_polygon)
        self.bind('<space>', self.make_polygon)
        self.bind('<Control-z>', self.remove_last_point)
        self.bind('<Control-Z>', self.remove_last_point)
        self.bind('<Control-s>', lambda e: self.save_annotation())
        self.bind('<Control-S>', lambda e: self.save_annotation())
        self.bind('<KeyPress-q>', self.change_to_original)
        self.bind('<KeyPress-Q>', self.change_to_original)
        self.bind('<KeyRelease-q>', self.change_to_overlay)
        self.bind('<KeyRelease-Q>', self.change_to_overlay)
        self.bind('<KeyPress-Left>', self.go_back_annot)
        self.bind('<KeyPress-Right>', self.go_front_annot)
        self.loaded = True

    def load_new_image(self):
        self.annot_canvas.delete('all')
        self.annot_selector_filename_label.config(text=f'{self.image_paths[self.current].name}')
        self.annot_selector_number_label.config(text=f'{self.current + 1}/{self.total}')
        self.title(f'Segmentation Annotator - {self.image_paths[self.current].name}')
        
        self.cv_image = np.array(Image.open(str(self.image_paths[self.current])).convert('RGB'))[:, :, ::-1]
        cv_annot_gray = cv2.cvtColor(self.cv_annots[self.current], cv2.COLOR_BGR2GRAY)
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

        self.points = []
        self.canvas_points = []
        self.canvas_lines = []
        self.created_annots = [self.cv_annots[self.current].copy()]
        self.removed_annots = []

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
        margin = 50
        if img_x < -margin or img_x >= self.w + margin or img_y < -margin or img_y >= self.h + margin:
            return
        
        img_x = min(max(0, img_x), self.w - 1)
        img_y = min(max(0, img_y), self.h - 1)
        x = img_x + self.canvas_xpos - self.w // 2
        y = img_y + self.canvas_ypos - self.h // 2

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
            self.cv_annots[self.current] = cv2.drawContours(self.cv_annots[self.current], [polygon], -1, color=(255, 255, 255), thickness=-1)
        else:
            self.cv_annots[self.current] = cv2.drawContours(self.cv_annots[self.current], [polygon], -1, color=(0, 0, 0), thickness=-1)
        cv_annot_gray = cv2.cvtColor(self.cv_annots[self.current], cv2.COLOR_BGR2GRAY)
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

        self.created_annots.append(self.cv_annots[self.current].copy())
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
        if not self.scratch:
            path = Path(filedialog.askdirectory(initialdir=self.annot_paths[0].parent, mustexist=True))
        else:
            path = Path(filedialog.askdirectory(initialdir=self.image_paths[0].parent, mustexist=True))
        for i in range(self.total):
            cv_annot_img = Image.fromarray(self.cv_annots[i])
            cv_annot_img.save(path / (self.image_paths[i].name[:-3] + 'png'))

    def change_to_original(self, event):
        if not self.loaded:
            return
        self.annot_canvas.itemconfig(self.canvas_image_obj, image=self.original_canvas_image)

    def change_to_overlay(self, event):
        if not self.loaded:
            return
        self.annot_canvas.itemconfig(self.canvas_image_obj, image=self.canvas_image)

    def go_back_annot(self, event):
        if len(self.created_annots) > 1:
            self.removed_annots.append(self.created_annots.pop())
            self.cv_annots[self.current] = self.created_annots[-1].copy()
            
            cv_annot_gray = cv2.cvtColor(self.cv_annots[self.current], cv2.COLOR_BGR2GRAY)
            self.cv_overlay = self.cv_image.copy()
            self.cv_overlay[cv_annot_gray > 0, :] = (1 - self.opacity) * self.cv_image[cv_annot_gray > 0, :] + self.opacity * self.color_map[cv_annot_gray > 0, :]
            self.cv_overlay = cv2.cvtColor(self.cv_overlay, cv2.COLOR_BGR2RGB)
            self.cv_overlay = Image.fromarray(self.cv_overlay)
            self.canvas_image = ImageTk.PhotoImage(image=self.cv_overlay)
            self.annot_canvas.itemconfig(self.canvas_image_obj, image=self.canvas_image)

    def go_front_annot(self, event):
        if len(self.removed_annots) > 0:
            self.cv_annots[self.current] = self.removed_annots.pop()
            self.created_annots.append(self.cv_annots[self.current].copy())

            cv_annot_gray = cv2.cvtColor(self.cv_annots[self.current], cv2.COLOR_BGR2GRAY)
            self.cv_overlay = self.cv_image.copy()
            self.cv_overlay[cv_annot_gray > 0, :] = (1 - self.opacity) * self.cv_image[cv_annot_gray > 0, :] + self.opacity * self.color_map[cv_annot_gray > 0, :]
            self.cv_overlay = cv2.cvtColor(self.cv_overlay, cv2.COLOR_BGR2RGB)
            self.cv_overlay = Image.fromarray(self.cv_overlay)
            self.canvas_image = ImageTk.PhotoImage(image=self.cv_overlay)
            self.annot_canvas.itemconfig(self.canvas_image_obj, image=self.canvas_image)

    def delete_image_from_loader(self, event):
        selected = sorted(list(self.image_loader_listbox.curselection()), reverse=True)
        for idx in selected:
            self.image_paths.pop(idx)
            self.image_loader_listbox.delete(idx)

    def delete_annot_from_loader(self, event):
        selected = sorted(list(self.annot_loader_listbox.curselection()), reverse=True)
        for idx in selected:
            self.annot_paths.pop(idx)
            self.annot_loader_listbox.delete(idx)

    def go_prev_image(self):
        if self.current > 0:
            self.current -= 1
            self.load_new_image()
    
    def go_next_image(self):
        if self.current + 1 < self.total:
            self.current += 1
            self.load_new_image()


if __name__ == '__main__':
        annotator = SegmentationAnnotator()
        annotator.mainloop()
