import numpy as np
import segyio
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

def get_desktop_path():
    """Получение пути к рабочему столу"""
    return os.path.join(os.path.expanduser("~"), "Desktop")

def effective_to_interval_velocities(t0_list, v_eff_list):
    """
    Конвертация эффективных скоростей в интервальные
    """
    t0_list = np.array(t0_list) / 1000.0  # мс в с
    v_eff_list = np.array(v_eff_list)
    
    v_int = np.zeros_like(v_eff_list)
    v_int[0] = v_eff_list[0]  # Первый слой
    
    for k in range(1, len(t0_list)):
        # Формула Диксера для интервальной скорости
        numerator = v_eff_list[k]**2 * t0_list[k] - v_eff_list[k-1]**2 * t0_list[k-1]
        denominator = t0_list[k] - t0_list[k-1]
        v_int[k] = np.sqrt(numerator / denominator) if denominator > 0 else v_eff_list[k]
    
    return v_int

def create_synthetic_seismogram(record_length_ms, dt_ms, boundaries_data, 
                               sou_x, rec_x, velocity_list):
    """
    Создание синтетической сейсмограммы с заданными координатами
    
    Parameters:
    - record_length_ms: длина записи в мс
    - dt_ms: шаг дискретизации в мс
    - boundaries_data: данные границ
    - sou_x: координата источника (м)
    - rec_x: массив координат приемников (м)
    - velocity_list: список скоростей для каждой границы
    """
    # Извлекаем данные о границах
    boundaries = [b[0] for b in boundaries_data]
    t0_list = [b[1] for b in boundaries_data]  # в мс
    
    # Параметры сейсмограммы
    dt_sec = dt_ms / 1000.0
    samples_per_trace = int(record_length_ms / dt_ms)
    num_traces = len(rec_x)
    
    # Расчет разносов
    offsets = np.abs(np.array(rec_x) - sou_x)
    
    # Создание пустой сейсмограммы
    seismogram = np.zeros((num_traces, samples_per_trace), dtype=np.float32)
    
    # Заполнение годографов
    for boundary_idx in range(len(boundaries)):
        t0_ms = t0_list[boundary_idx]
        v_current = velocity_list[boundary_idx]
        
        for trace_idx in range(num_traces):
            x = offsets[trace_idx]
            # Время прихода для отраженной волны (в мс)
            t_arrival_ms = np.sqrt((t0_ms/1000.0)**2 + (x/v_current)**2) * 1000.0
            
            # Находим ближайший сэмпл
            sample_idx = int(round(t_arrival_ms / dt_ms))
            
            # Если время в пределах записи, ставим 1
            if 0 <= sample_idx < samples_per_trace:
                seismogram[trace_idx, sample_idx] = 1.0
    
    return seismogram, offsets, dt_ms

def create_velocity_model(record_length_ms, dt_ms, boundaries_data, v_int_list):
    """
    Создание модели интервальных скоростей
    """
    samples_per_trace = int(record_length_ms / dt_ms)
    num_traces = 1  # Одна трасса для модели
    
    # Создание модели скоростей
    velocity_model = np.zeros((num_traces, samples_per_trace), dtype=np.float32)
    
    # Заполнение модели
    for i in range(len(boundaries_data)):
        t0_ms = boundaries_data[i][1]  # время границы в мс
        sample_idx = int(round(t0_ms / dt_ms))
        
        if sample_idx < samples_per_trace:
            # Заполняем от текущей границы до следующей (или до конца)
            end_sample = samples_per_trace
            if i < len(boundaries_data) - 1:
                next_t0_ms = boundaries_data[i+1][1]
                end_sample = min(int(round(next_t0_ms / dt_ms)), samples_per_trace)
            
            velocity_model[0, sample_idx:end_sample] = v_int_list[i]
    
    return velocity_model

def save_segy_file(seismogram, offsets, rec_x, dt_ms, filename, sou_x, ffid=1, description=""):
    """
    Сохранение сейсмограммы в формате SEG-Y с правильными заголовками
    """
    num_traces = seismogram.shape[0]
    samples_per_trace = seismogram.shape[1]
    
    # Спецификация для SEG-Y файла
    spec = segyio.spec()
    spec.format = 1  # IBM float
    spec.sorting = 2  # CDP sorting
    spec.samples = range(samples_per_trace)
    spec.tracecount = num_traces
    
    # Полный путь к файлу
    full_path = filename
    
    # Создание SEG-Y файла
    with segyio.create(full_path, spec) as f:
        # Заголовок двоичного файла
        f.bin = {
            segyio.BinField.Traces: num_traces,
            segyio.BinField.Samples: samples_per_trace,
            segyio.BinField.Interval: int(dt_ms * 1000),  # микросекунды
        }
        
        # Заголовки трасс с правильными полями
        for i in range(num_traces):
            f.header[i] = {
                # Основные поля SEG-Y
                segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1,
                segyio.TraceField.FieldRecord: ffid,
                segyio.TraceField.TraceNumber: i + 1,
                
                # Координаты и геометрия
                segyio.TraceField.CDP: i + 1,
                segyio.TraceField.CDP_TRACE: 1,
                segyio.TraceField.offset: int(offsets[i]),
                
                # Координаты источника
                segyio.TraceField.SourceX: int(sou_x),
                segyio.TraceField.SourceY: 0,
                
                # Координаты приемника
                segyio.TraceField.GroupX: int(rec_x[i]),
                segyio.TraceField.GroupY: 0,
                
                # Номер канала
                segyio.TraceField.TraceIdentificationCode: i + 1,
                
                # Дополнительные технические поля
                segyio.TraceField.SourceSurfaceElevation: 0,
                segyio.TraceField.ReceiverGroupElevation: 0,
                segyio.TraceField.SourceDepth: 0,
                segyio.TraceField.ReceiverDatumElevation: 0,
                segyio.TraceField.SourceDatumElevation: 0,
                
                # Временные параметры
                segyio.TraceField.DelayRecordingTime: 0,
                
                # Масштабы
                segyio.TraceField.ElevationScalar: 1,
                segyio.TraceField.SourceGroupScalar: 1,
                segyio.TraceField.CoordinateUnits: 1,
            }
        
        # Запись данных
        for i in range(num_traces):
            f.trace[i] = seismogram[i, :]
    
    return full_path

class SeismogramGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Генератор синтетических сейсмограмм")
        self.root.geometry("1200x800")
        
        # Данные границ по умолчанию
        self.boundaries_data = [
            [1, 23, 1420],
            [2, 53, 1590],
            [3, 69, 1670],
            [4, 88, 1770],
            [5, 132, 1930],
            [6, 141, 2020],
            [7, 155, 2140],
            [8, 177, 2440],
            [9, 202, 2700],
            [10, 247, 3040]
        ]
        
        self.setup_ui()
        self.update_boundaries_table()
        self.update_preview()
        
    def setup_ui(self):
        # Создаем вкладки
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Вкладка 1: Основные параметры
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Основные параметры")
        
        # Вкладка 2: Границы и скорости
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text="Границы и скорости")
        
        # Вкладка 3: Геометрия
        tab3 = ttk.Frame(notebook)
        notebook.add(tab3, text="Геометрия")
        
        # Вкладка 4: Предпросмотр и генерация
        tab4 = ttk.Frame(notebook)
        notebook.add(tab4, text="Предпросмотр и генерация")
        
        self.setup_tab1(tab1)
        self.setup_tab2(tab2)
        self.setup_tab3(tab3)
        self.setup_tab4(tab4)
        
        # Кнопка запуска внизу окна
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill='x', padx=10, pady=10)
        
        self.start_button = tk.Button(
            btn_frame, 
            text="Запустить генерацию", 
            command=self.start_generation,
            font=("Arial", 10, "bold"),
            fg="black",
            bg="#4CAF50",
            padx=20,
            pady=5
        )
        self.start_button.pack(side='right', padx=5)
        
        ttk.Button(btn_frame, text="Выход", command=self.root.quit).pack(side='right', padx=5)
        
    def setup_tab1(self, parent):
        # Параметры сейсмограммы
        frame = ttk.LabelFrame(parent, text="Параметры сейсмограммы", padding=10)
        frame.pack(fill='x', padx=10, pady=10)
        
        # Длина записи
        ttk.Label(frame, text="Длина записи (мс):").grid(row=0, column=0, sticky='w', pady=5)
        self.record_length_var = tk.StringVar(value="500")
        ttk.Entry(frame, textvariable=self.record_length_var, width=15).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        # Шаг дискретизации
        ttk.Label(frame, text="Шаг дискретизации (мс):").grid(row=1, column=0, sticky='w', pady=5)
        self.dt_var = tk.StringVar(value="1")
        ttk.Entry(frame, textvariable=self.dt_var, width=15).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        # Номер FFID
        ttk.Label(frame, text="Номер удара (FFID):").grid(row=2, column=0, sticky='w', pady=5)
        self.ffid_var = tk.StringVar(value="1")
        ttk.Entry(frame, textvariable=self.ffid_var, width=15).grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        # Тип скоростей
        ttk.Label(frame, text="Тип скоростей:").grid(row=3, column=0, sticky='w', pady=5)
        self.velocity_type_var = tk.StringVar(value="both")
        ttk.Radiobutton(frame, text="Только эффективные", variable=self.velocity_type_var, 
                       value="effective").grid(row=3, column=1, sticky='w', padx=5, pady=2)
        ttk.Radiobutton(frame, text="Только интервальные", variable=self.velocity_type_var, 
                       value="interval").grid(row=4, column=1, sticky='w', padx=5, pady=2)
        ttk.Radiobutton(frame, text="Оба типа", variable=self.velocity_type_var, 
                       value="both").grid(row=5, column=1, sticky='w', padx=5, pady=2)
        
        # Создать модель скоростей
        self.create_model_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Создать файл с моделью интервальных скоростей", 
                       variable=self.create_model_var).grid(row=6, column=0, columnspan=2, sticky='w', pady=10)
        
    def setup_tab2(self, parent):
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Левая часть: таблица границ
        left_frame = ttk.LabelFrame(main_frame, text="Границы раздела", padding=10)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Таблица
        columns = ('№', 't0 (мс)', 'Vэфф (м/с)', 'Vинт (м/с)')
        self.tree = ttk.Treeview(left_frame, columns=columns, show='headings', height=12)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor='center')
        
        self.tree.pack(side='left', fill='both', expand=True)
        
        # Полоса прокрутки
        scrollbar = ttk.Scrollbar(left_frame, orient='vertical', command=self.tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Правая часть: редактирование
        right_frame = ttk.LabelFrame(main_frame, text="Редактирование границ", padding=10)
        right_frame.pack(side='right', fill='y', padx=(5, 0))
        
        # Поля для редактирования
        ttk.Label(right_frame, text="Номер границы:").pack(anchor='w', pady=(0, 5))
        self.edit_num_var = tk.StringVar()
        self.edit_num_entry = ttk.Entry(right_frame, textvariable=self.edit_num_var, width=15, state='readonly')
        self.edit_num_entry.pack(fill='x', pady=(0, 10))
        
        ttk.Label(right_frame, text="t0 (мс):").pack(anchor='w', pady=(0, 5))
        self.edit_t0_var = tk.StringVar()
        self.edit_t0_entry = ttk.Entry(right_frame, textvariable=self.edit_t0_var, width=15)
        self.edit_t0_entry.pack(fill='x', pady=(0, 10))
        
        ttk.Label(right_frame, text="Vэфф (м/с):").pack(anchor='w', pady=(0, 5))
        self.edit_veff_var = tk.StringVar()
        self.edit_veff_entry = ttk.Entry(right_frame, textvariable=self.edit_veff_var, width=15)
        self.edit_veff_entry.pack(fill='x', pady=(0, 10))
        
        # Кнопки редактирования
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill='x', pady=10)
        
        ttk.Button(btn_frame, text="Добавить новую", command=self.add_new_boundary).pack(side='left', fill='x', expand=True, padx=(0, 2))
        ttk.Button(btn_frame, text="Обновить", command=self.update_selected_boundary).pack(side='left', fill='x', expand=True, padx=2)
        
        btn_frame2 = ttk.Frame(right_frame)
        btn_frame2.pack(fill='x', pady=5)
        
        ttk.Button(btn_frame2, text="Удалить", command=self.delete_selected_boundary).pack(side='left', fill='x', expand=True, padx=(0, 2))
        ttk.Button(btn_frame2, text="Очистить все", command=self.clear_all_boundaries).pack(side='left', fill='x', expand=True, padx=2)
        
        ttk.Button(right_frame, text="Загрузить по умолчанию", command=self.load_default_boundaries).pack(fill='x', pady=10)
        
        # Кнопка для ручного изменения номера
        self.renumber_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(right_frame, text="Автонумерация при удалении", 
                       variable=self.renumber_var).pack(anchor='w', pady=5)
        
        # Связывание выбора в таблице с полями редактирования
        self.tree.bind('<<TreeviewSelect>>', self.on_boundary_select)
        
        # Связывание изменений в полях с обновлением
        self.edit_t0_var.trace_add('write', self.on_parameter_change)
        self.edit_veff_var.trace_add('write', self.on_parameter_change)
        
    def setup_tab3(self, parent):
        # Геометрия съемки
        frame = ttk.LabelFrame(parent, text="Геометрия съемки", padding=10)
        frame.pack(fill='x', padx=10, pady=10)
        
        # Координата источника
        ttk.Label(frame, text="Координата источника X (м):").grid(row=0, column=0, sticky='w', pady=5)
        self.sou_x_var = tk.StringVar(value="0")
        ttk.Entry(frame, textvariable=self.sou_x_var, width=15).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        # Массив приемников
        ttk.Label(frame, text="Приемники:").grid(row=1, column=0, sticky='w', pady=5)
        
        rec_frame = ttk.Frame(frame)
        rec_frame.grid(row=1, column=1, columnspan=3, sticky='w', pady=5)
        
        ttk.Label(rec_frame, text="От:").pack(side='left')
        self.rec_start_var = tk.StringVar(value="0")
        ttk.Entry(rec_frame, textvariable=self.rec_start_var, width=10).pack(side='left', padx=5)
        
        ttk.Label(rec_frame, text="До:").pack(side='left')
        self.rec_end_var = tk.StringVar(value="1000")
        ttk.Entry(rec_frame, textvariable=self.rec_end_var, width=10).pack(side='left', padx=5)
        
        ttk.Label(rec_frame, text="Шаг:").pack(side='left')
        self.rec_spacing_var = tk.StringVar(value="10")
        ttk.Entry(rec_frame, textvariable=self.rec_spacing_var, width=10).pack(side='left', padx=5)
        
        ttk.Button(rec_frame, text="Обновить", command=self.update_preview).pack(side='left', padx=10)
        
    def setup_tab4(self, parent):
        # Предпросмотр и выходные файлы
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Левая панель: предпросмотр
        left_frame = ttk.LabelFrame(main_frame, text="Предпросмотр геометрии", padding=10)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, left_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Правая панель: параметры и лог
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', padx=(5, 0))
        
        # Имя файла
        file_frame = ttk.LabelFrame(right_frame, text="Выходные файлы", padding=10)
        file_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(file_frame, text="Базовое имя файлов:").pack(anchor='w')
        self.filename_var = tk.StringVar(value="synthetic")
        ttk.Entry(file_frame, textvariable=self.filename_var, width=30).pack(fill='x', pady=5)
        
        ttk.Label(file_frame, text="Папка для сохранения:").pack(anchor='w', pady=(10, 0))
        
        folder_frame = ttk.Frame(file_frame)
        folder_frame.pack(fill='x', pady=5)
        
        self.folder_var = tk.StringVar(value=get_desktop_path())
        ttk.Entry(folder_frame, textvariable=self.folder_var).pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(folder_frame, text="...", width=3, command=self.browse_folder).pack(side='right')
        
        # Лог
        log_frame = ttk.LabelFrame(right_frame, text="Лог выполнения", padding=10)
        log_frame.pack(fill='both', expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=40)
        self.log_text.pack(fill='both', expand=True)
        
    def update_boundaries_table(self):
        """Обновление таблицы границ"""
        # Очистка таблицы
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Сортировка границ по номеру
        self.boundaries_data.sort(key=lambda x: x[0])
        
        # Автонумерация, если включена
        if self.renumber_var.get():
            self.renumber_boundaries()
        
        # Расчет интервальных скоростей
        if self.boundaries_data:
            t0_list = [b[1] for b in self.boundaries_data]
            v_eff_list = [b[2] for b in self.boundaries_data]
            v_int_list = effective_to_interval_velocities(t0_list, v_eff_list)
            
            # Заполнение таблицы
            for i, (b, t0, v_eff) in enumerate(self.boundaries_data):
                v_int = v_int_list[i]
                self.tree.insert('', 'end', values=(b, t0, v_eff, f"{v_int:.0f}"))
        else:
            self.log("Таблица границ пуста")
            
        # Обновляем список в логе
        self.log_boundaries_info()
    
    def renumber_boundaries(self):
        """Автоматическая перенумерация границ"""
        if not self.boundaries_data:
            return
            
        # Сортируем по t0 для правильной последовательности
        self.boundaries_data.sort(key=lambda x: x[1])
        
        # Перенумеровываем
        for i in range(len(self.boundaries_data)):
            self.boundaries_data[i][0] = i + 1
    
    def on_parameter_change(self, *args):
        """Обработка изменений в полях параметров"""
        # Автоматическое обновление при изменении t0 или Vэфф
        pass  # Можно добавить автоматическое обновление, если нужно
    
    def on_boundary_select(self, event):
        """Заполнение полей редактирования при выборе границы"""
        selected = self.tree.selection()
        if not selected:
            self.clear_edit_fields()
            return
        
        item = self.tree.item(selected[0])
        values = item['values']
        
        # Временно отключаем трассировку для избежания рекурсии
        self.edit_t0_var.trace_vdelete('write', self.edit_t0_var.trace_info()[0][1])
        self.edit_veff_var.trace_vdelete('write', self.edit_veff_var.trace_info()[0][1])
        
        self.edit_num_var.set(str(values[0]))
        self.edit_t0_var.set(str(values[1]))
        self.edit_veff_var.set(str(values[2]))
        
        # Восстанавливаем трассировку
        self.edit_t0_var.trace_add('write', self.on_parameter_change)
        self.edit_veff_var.trace_add('write', self.on_parameter_change)
    
    def add_new_boundary(self):
        """Добавление новой границы"""
        try:
            # Получаем последний номер
            last_num = 0
            if self.boundaries_data:
                last_num = max([b[0] for b in self.boundaries_data])
            
            # Создаем диалог для ввода параметров
            dialog = tk.Toplevel(self.root)
            dialog.title("Добавить новую границу")
            dialog.geometry("400x250")
            dialog.transient(self.root)
            dialog.grab_set()
            
            ttk.Label(dialog, text="Параметры новой границы:", font=("Arial", 10, "bold")).pack(pady=10)
            
            # Поля ввода
            input_frame = ttk.Frame(dialog)
            input_frame.pack(pady=10, padx=20, fill='x')
            
            ttk.Label(input_frame, text="t0 (мс):").grid(row=0, column=0, sticky='w', pady=5)
            new_t0_var = tk.StringVar(value="100")
            ttk.Entry(input_frame, textvariable=new_t0_var, width=15).grid(row=0, column=1, sticky='w', padx=10, pady=5)
            
            ttk.Label(input_frame, text="Vэфф (м/с):").grid(row=1, column=0, sticky='w', pady=5)
            new_veff_var = tk.StringVar(value="2000")
            ttk.Entry(input_frame, textvariable=new_veff_var, width=15).grid(row=1, column=1, sticky='w', padx=10, pady=5)
            
            # Функция сохранения
            def save_new_boundary():
                try:
                    t0 = float(new_t0_var.get())
                    v_eff = float(new_veff_var.get())
                    
                    if t0 <= 0:
                        messagebox.showerror("Ошибка", "t0 должно быть положительным")
                        return
                    
                    if v_eff <= 0:
                        messagebox.showerror("Ошибка", "Скорость должна быть положительной")
                        return
                    
                    # Проверяем, чтобы t0 было больше предыдущих
                    if self.boundaries_data:
                        max_t0 = max([b[1] for b in self.boundaries_data])
                        if t0 <= max_t0:
                            messagebox.showerror("Ошибка", f"Новое t0 должно быть больше максимального ({max_t0} мс)")
                            return
                    
                    # Добавляем новую границу
                    new_num = last_num + 1
                    self.boundaries_data.append([new_num, t0, v_eff])
                    self.update_boundaries_table()
                    
                    self.log(f"Добавлена новая граница №{new_num}: t0={t0} мс, Vэфф={v_eff} м/с")
                    dialog.destroy()
                    
                except ValueError:
                    messagebox.showerror("Ошибка", "Введите корректные числовые значения")
            
            # Кнопки
            btn_frame = ttk.Frame(dialog)
            btn_frame.pack(pady=20)
            
            ttk.Button(btn_frame, text="Добавить", command=save_new_boundary, width=15).pack(side='left', padx=5)
            ttk.Button(btn_frame, text="Отмена", command=dialog.destroy, width=15).pack(side='left', padx=5)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при добавлении границы: {str(e)}")
    
    def update_selected_boundary(self):
        """Обновление выбранной границы"""
        try:
            selected = self.tree.selection()
            if not selected:
                messagebox.showwarning("Внимание", "Выберите границу для обновления")
                return
            
            # Получение текущих значений
            item = self.tree.item(selected[0])
            old_values = item['values']
            old_num = int(old_values[0])
            
            # Получение новых значений
            t0 = float(self.edit_t0_var.get())
            v_eff = float(self.edit_veff_var.get())
            
            if t0 <= 0:
                messagebox.showerror("Ошибка", "t0 должно быть положительным")
                return
            
            if v_eff <= 0:
                messagebox.showerror("Ошибка", "Скорость должна быть положительной")
                return
            
            # Проверяем, чтобы t0 было больше предыдущих и меньше следующих
            if self.boundaries_data:
                # Находим индекс текущей границы
                for i, b in enumerate(self.boundaries_data):
                    if b[0] == old_num:
                        current_idx = i
                        break
                
                # Проверка с предыдущей границей
                if current_idx > 0:
                    prev_t0 = self.boundaries_data[current_idx-1][1]
                    if t0 <= prev_t0:
                        messagebox.showerror("Ошибка", f"t0 должно быть больше предыдущего ({prev_t0} мс)")
                        return
                
                # Проверка со следующей границей
                if current_idx < len(self.boundaries_data) - 1:
                    next_t0 = self.boundaries_data[current_idx+1][1]
                    if t0 >= next_t0:
                        messagebox.showerror("Ошибка", f"t0 должно быть меньше следующего ({next_t0} мс)")
                        return
            
            # Обновление данных
            for i, b in enumerate(self.boundaries_data):
                if b[0] == old_num:
                    self.boundaries_data[i] = [old_num, t0, v_eff]
                    break
            
            self.update_boundaries_table()
            self.log(f"Обновлена граница №{old_num}: t0={t0} мс, Vэфф={v_eff} м/с")
            
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числовые значения")
    
    def delete_selected_boundary(self):
        """Удаление выбранной границы"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Внимание", "Выберите границу для удаления")
            return
        
        item = self.tree.item(selected[0])
        values = item['values']
        
        if messagebox.askyesno("Подтверждение", f"Удалить границу №{values[0]}?"):
            # Удаление из данных
            self.boundaries_data = [b for b in self.boundaries_data if b[0] != values[0]]
            
            # Если включена автонумерация, пересчитываем номера
            if self.renumber_var.get():
                self.update_boundaries_table()  # Автонумерация сработает внутри
            else:
                self.update_boundaries_table()
            
            self.clear_edit_fields()
            
            self.log(f"Удалена граница №{values[0]}")
    
    def clear_all_boundaries(self):
        """Очистка всех границ"""
        if not self.boundaries_data:
            return
            
        if messagebox.askyesno("Подтверждение", "Вы действительно хотите очистить все границы?"):
            self.boundaries_data = []
            self.update_boundaries_table()
            self.clear_edit_fields()
            self.log("Все границы очищены")
    
    def clear_edit_fields(self):
        """Очистка полей редактирования"""
        # Временно отключаем трассировку
        self.edit_t0_var.trace_vdelete('write', self.edit_t0_var.trace_info()[0][1])
        self.edit_veff_var.trace_vdelete('write', self.edit_veff_var.trace_info()[0][1])
        
        self.edit_num_var.set("")
        self.edit_t0_var.set("")
        self.edit_veff_var.set("")
        
        # Восстанавливаем трассировку
        self.edit_t0_var.trace_add('write', self.on_parameter_change)
        self.edit_veff_var.trace_add('write', self.on_parameter_change)
    
    def load_default_boundaries(self):
        """Загрузка границ по умолчанию"""
        self.boundaries_data = [
            [1, 23, 1420],
            [2, 53, 1590],
            [3, 69, 1670],
            [4, 88, 1770],
            [5, 132, 1930],
            [6, 141, 2020],
            [7, 155, 2140],
            [8, 177, 2440],
            [9, 202, 2700],
            [10, 247, 3040]
        ]
        self.update_boundaries_table()
        self.clear_edit_fields()
        self.log("Загружены границы по умолчанию")
    
    def log_boundaries_info(self):
        """Логирование информации о границах"""
        if self.boundaries_data:
            self.log(f"Всего границ: {len(self.boundaries_data)}")
            for b in self.boundaries_data:
                self.log(f"  Граница {b[0]}: t0={b[1]} мс, Vэфф={b[2]} м/с")
    
    def update_preview(self):
        try:
            self.ax.clear()
            
            # Параметры геометрии
            sou_x = float(self.sou_x_var.get())
            rec_start = float(self.rec_start_var.get())
            rec_end = float(self.rec_end_var.get())
            rec_spacing = float(self.rec_spacing_var.get())
            
            # Создание массива приемников
            num_receivers = int((rec_end - rec_start) / rec_spacing) + 1
            rec_x = np.linspace(rec_start, rec_end, num_receivers)
            
            # Построение геометрии - сначала приемники
            self.ax.scatter(rec_x, np.zeros_like(rec_x), s=50, c='blue', marker='^', label='Приемники', zorder=1)
            
            # Затем источник поверх приемников
            self.ax.scatter([sou_x], [0], s=200, c='red', marker='*', label='Источник', zorder=2, edgecolors='black', linewidth=1)
            
            # Настройки графика
            self.ax.set_xlabel('Расстояние (м)')
            self.ax.set_ylabel('Z (м)')
            self.ax.set_title('Геометрия съемки')
            self.ax.legend(loc='upper right')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_ylim(-5, 5)  # Фиксируем масштаб по вертикали
            
            # Добавление информации
            offsets = np.abs(rec_x - sou_x)
            info_text = f"Приемников: {len(rec_x)}\nМин. разнос: {min(offsets):.1f} м\nМакс. разнос: {max(offsets):.1f} м"
            self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        zorder=3)
            
            self.canvas.draw()
            
        except Exception as e:
            self.log(f"Ошибка при построении предпросмотра: {str(e)}")
    
    def browse_folder(self):
        folder = filedialog.askdirectory(initialdir=self.folder_var.get())
        if folder:
            self.folder_var.set(folder)
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert('end', f"[{timestamp}] {message}\n")
        self.log_text.see('end')
        self.root.update_idletasks()
    
    def create_receiver_array(self, start_x, end_x, spacing):
        """Создание массива координат приемников"""
        num_receivers = int((end_x - start_x) / spacing) + 1
        return np.linspace(start_x, end_x, num_receivers)
    
    def start_generation(self):
        # Проверка ввода
        try:
            # Параметры сейсмограммы
            record_length_ms = float(self.record_length_var.get())
            dt_ms = float(self.dt_var.get())
            ffid = int(self.ffid_var.get())
            
            # Геометрия
            sou_x = float(self.sou_x_var.get())
            rec_start = float(self.rec_start_var.get())
            rec_end = float(self.rec_end_var.get())
            rec_spacing = float(self.rec_spacing_var.get())
            
            # Проверка границ
            if not self.boundaries_data:
                messagebox.showerror("Ошибка", "Добавьте хотя бы одну границу")
                return
            
            # Проверка геометрии
            if rec_start >= rec_end:
                messagebox.showerror("Ошибка", "Начальная координата должна быть меньше конечной")
                return
            
            if rec_spacing <= 0:
                messagebox.showerror("Ошибка", "Шаг между приемниками должен быть положительным")
                return
            
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректные значения параметров: {str(e)}")
            return
        
        # Запуск генерации в отдельном потоке
        thread = threading.Thread(target=self.generate_seismograms, daemon=True)
        thread.start()
    
    def generate_seismograms(self):
        try:
            self.log("=" * 50)
            self.log("Начало генерации сейсмограмм")
            self.log("=" * 50)
            
            # Получение параметров
            record_length_ms = float(self.record_length_var.get())
            dt_ms = float(self.dt_var.get())
            ffid = int(self.ffid_var.get())
            sou_x = float(self.sou_x_var.get())
            rec_start = float(self.rec_start_var.get())
            rec_end = float(self.rec_end_var.get())
            rec_spacing = float(self.rec_spacing_var.get())
            
            # Создание массива приемников
            rec_x = self.create_receiver_array(rec_start, rec_end, rec_spacing)
            num_traces = len(rec_x)
            
            self.log(f"Создано {num_traces} приемников")
            self.log(f"Геометрия: источник на {sou_x} м, приемники от {rec_start} м до {rec_end} м")
            
            # Эффективные скорости
            v_eff_list = [b[2] for b in self.boundaries_data]
            
            # Базовое имя файлов
            base_filename = self.filename_var.get()
            folder = self.folder_var.get()
            
            velocity_type = self.velocity_type_var.get()
            
            # Создание сейсмограммы с эффективными скоростями (для статистики)
            seismogram_eff, offsets, actual_dt_ms = create_synthetic_seismogram(
                record_length_ms, dt_ms, self.boundaries_data, sou_x, rec_x, v_eff_list
            )
            
            if velocity_type in ["both", "effective"]:
                # Файл с эффективными скоростями
                self.log("Создание сейсмограммы с эффективными скоростями...")
                filename = os.path.join(folder, f"{base_filename}_effective.sgy")
                save_segy_file(seismogram_eff, offsets, rec_x, dt_ms, filename, sou_x, ffid)
                self.log(f"✓ Создан файл: {filename}")
            
            if velocity_type in ["both", "interval"]:
                # Файл с интервальными скоростями
                self.log("Создание сейсмограммы с интервальными скоростями...")
                v_int_list = effective_to_interval_velocities(
                    [b[1] for b in self.boundaries_data], 
                    v_eff_list
                )
                
                seismogram_int, offsets, actual_dt_ms = create_synthetic_seismogram(
                    record_length_ms, dt_ms, self.boundaries_data, sou_x, rec_x, v_int_list
                )
                
                filename = os.path.join(folder, f"{base_filename}_interval.sgy")
                save_segy_file(seismogram_int, offsets, rec_x, dt_ms, filename, sou_x, ffid)
                self.log(f"✓ Создан файл: {filename}")
                
                # Вывод интервальных скоростей
                self.log("\nИнтервальные скорости:")
                for i, v_int in enumerate(v_int_list):
                    self.log(f"  Слой {i+1}: {v_int:.0f} м/с")
            
            # Файл с моделью скоростей
            if self.create_model_var.get():
                self.log("Создание модели интервальных скоростей...")
                v_int_list = effective_to_interval_velocities(
                    [b[1] for b in self.boundaries_data], 
                    v_eff_list
                )
                
                velocity_model = create_velocity_model(record_length_ms, dt_ms, self.boundaries_data, v_int_list)
                model_filename = os.path.join(folder, f"{base_filename}_velocity_model.sgy")
                dummy_offsets = [0]
                dummy_rec_x = [0]
                save_segy_file(velocity_model, dummy_offsets, dummy_rec_x, dt_ms, model_filename, sou_x, ffid)
                self.log(f"✓ Создан файл: {model_filename}")
            
            # Статистика
            self.log("\n" + "=" * 50)
            self.log("ГЕНЕРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
            self.log("=" * 50)
            self.log(f"Количество границ: {len(self.boundaries_data)}")
            self.log(f"Размер сейсмограммы: {seismogram_eff.shape[1]} сэмплов × {seismogram_eff.shape[0]} трасс")
            self.log(f"Интервал дискретизации: {dt_ms} мс")
            self.log(f"Длина записи: {record_length_ms} мс")
            self.log(f"Максимальный разнос: {max(offsets):.1f} м")
            self.log(f"Минимальный разнос: {min(offsets):.1f} м")
            self.log(f"Номер удара (FFID): {ffid}")
            self.log(f"Файлы сохранены в: {folder}")
            
            messagebox.showinfo("Успех", "Сейсмограммы успешно созданы!")
            
        except Exception as e:
            self.log(f"ОШИБКА: {str(e)}")
            messagebox.showerror("Ошибка", f"Произошла ошибка:\n{str(e)}")

def main():
    """Точка входа в программу"""
    root = tk.Tk()
    app = SeismogramGeneratorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()