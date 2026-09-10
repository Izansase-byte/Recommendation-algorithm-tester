from __future__ import annotations

import os
from pathlib import Path
import re
import queue
import tempfile
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Matplotlib necesita una caché escribible incluso en carpetas restringidas.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "movielab-matplotlib"))

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from recommender.data import (
    load_demo_data,
    load_movielens_files,
    load_sqlite_database,
    save_sqlite_database,
)
from recommender.evaluation import (
    MODEL_NAMES,
    evaluate_models,
    optimize_hybrid_weights,
    recommend_for_user,
    train_test_split_by_user,
)
from recommender.models import HybridRecommender, build_model
from desktop_panels import AnalysisPanelMixin, NewUserPanelMixin


BASE_DIR = Path(__file__).resolve().parent
COLORS = {
    "navy": "#14213D",
    "blue": "#2563EB",
    "green": "#16A34A",
    "background": "#F3F4F6",
    "surface": "#FFFFFF",
    "text": "#111827",
    "muted": "#6B7280",
}


def _year_from_title(title: str) -> int | None:
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(match.group(1)) if match else None


def _clear_tree(tree: ttk.Treeview) -> None:
    children = tree.get_children()
    if children:
        tree.delete(*children)


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, background=COLORS["background"])
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, padding=18)
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.inner.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.window, width=e.width))
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class MovieLabDesktop(AnalysisPanelMixin, NewUserPanelMixin):
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MovieLab · Comparador de recomendadores")
        self.root.geometry("1380x860")
        self.root.minsize(1080, 700)
        self.root.configure(background=COLORS["background"])

        self.movies, self.ratings = load_demo_data(BASE_DIR / "data" / "movies_demo.db")
        self.source_name = "Demostración integrada"
        self.results: pd.DataFrame | None = None
        self.recommendations: pd.DataFrame | None = None
        self.fitted_models: dict[str, object] = {}
        self.full_model_cache: dict[tuple, object] = {}
        self.active_recommendation_model = None
        self.user_names: dict[int, str] = {}
        self.busy = False

        self._create_variables()
        self._configure_styles()
        self._create_menu()
        self._create_layout()
        self._bind_shortcuts()
        self._refresh_dataset_views()
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    def _create_variables(self) -> None:
        self.model_vars = {name: tk.BooleanVar(value=True) for name in MODEL_NAMES}
        self.test_ratio = tk.DoubleVar(value=0.20)
        self.top_k = tk.IntVar(value=10)
        self.relevance = tk.DoubleVar(value=4.0)
        self.seed = tk.IntVar(value=42)

        self.base_regularization = tk.DoubleVar(value=10.0)
        self.base_iterations = tk.IntVar(value=10)
        self.user_k = tk.IntVar(value=25)
        self.user_similarity = tk.DoubleVar(value=0.0)
        self.user_shrinkage = tk.DoubleVar(value=10.0)
        self.item_k = tk.IntVar(value=25)
        self.item_similarity = tk.DoubleVar(value=0.0)
        self.item_shrinkage = tk.DoubleVar(value=10.0)
        self.content_title = tk.BooleanVar(value=False)
        self.content_genre_weight = tk.IntVar(value=3)

        self.hybrid_weights = {
            "base": tk.DoubleVar(value=0.10),
            "user": tk.DoubleVar(value=0.30),
            "item": tk.DoubleVar(value=0.30),
            "content": tk.DoubleVar(value=0.30),
        }
        self.normalized_weights = tk.StringVar()
        for variable in self.hybrid_weights.values():
            variable.trace_add("write", self._update_normalized_weights)

        self.optimization_objective = tk.StringVar(value="RMSE")
        self.optimization_trials = tk.IntVar(value=80)
        self.metric_var = tk.StringVar(value="RMSE")

        self.rec_user = tk.StringVar()
        self.rec_model = tk.StringVar(value="Modelo híbrido")
        self.rec_count = tk.IntVar(value=10)
        self.rec_genre = tk.StringVar(value="Todos")
        self.rec_year_min = tk.StringVar(value="")
        self.rec_year_max = tk.StringVar(value="")
        self.rec_min_votes = tk.IntVar(value=0)
        self.rec_exclude_seen = tk.BooleanVar(value=True)
        self.rec_diversity = tk.DoubleVar(value=0.0)

        self.status_text = tk.StringVar(value="Listo")
        self.data_source_text = tk.StringVar()
        self.data_stats_text = tk.StringVar()
        self.interpretation_text = tk.StringVar(value="Ejecuta una comparación para ver el análisis.")
        self.optimization_text = tk.StringVar(value="Aún no se ha optimizado el híbrido.")

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        # Clam respects explicit colours on macOS; Aqua inherits dark-mode colours
        # and can otherwise make custom navy headings unreadable.
        style.theme_use("clam")
        style.configure("TFrame", background=COLORS["background"])
        style.configure("Card.TFrame", background=COLORS["surface"], relief="solid", borderwidth=1)
        style.configure("Header.TFrame", background=COLORS["navy"])
        style.configure("TLabel", background=COLORS["background"], foreground=COLORS["text"], font=("Helvetica", 11))
        style.configure("Card.TLabel", background=COLORS["surface"], foreground=COLORS["text"], font=("Helvetica", 11))
        style.configure("Title.TLabel", background=COLORS["navy"], foreground="white", font=("Helvetica", 24, "bold"))
        style.configure("Subtitle.TLabel", background=COLORS["navy"], foreground="#CBD5E1", font=("Helvetica", 11))
        style.configure("Section.TLabel", font=("Helvetica", 15, "bold"), foreground=COLORS["navy"])
        style.configure("Stat.TLabel", background=COLORS["surface"], font=("Helvetica", 18, "bold"), foreground=COLORS["blue"])
        style.configure("Muted.TLabel", background=COLORS["surface"], foreground=COLORS["muted"], font=("Helvetica", 10))
        style.configure("Accent.TButton", font=("Helvetica", 11, "bold"))
        style.configure("TButton", background="#E5E7EB", foreground=COLORS["text"], padding=(9, 5))
        style.map("TButton", background=[("active", "#D1D5DB")])
        style.configure("TCheckbutton", background=COLORS["surface"], foreground=COLORS["text"])
        style.configure("TEntry", fieldbackground="white", foreground=COLORS["text"])
        style.configure("TCombobox", fieldbackground="white", foreground=COLORS["text"])
        style.configure("TSpinbox", fieldbackground="white", foreground=COLORS["text"])
        style.configure(
            "Treeview",
            rowheight=29,
            font=("Helvetica", 10),
            background="white",
            fieldbackground="white",
            foreground=COLORS["text"],
        )
        style.map(
            "Treeview",
            background=[("selected", COLORS["blue"])],
            foreground=[("selected", "white")],
        )
        style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"), background="#E5E7EB", foreground=COLORS["text"])
        style.configure("TNotebook", background=COLORS["background"], borderwidth=0)
        style.configure("TNotebook.Tab", font=("Helvetica", 11, "bold"), padding=(18, 10))
        style.configure("TLabelframe", background=COLORS["surface"], padding=10)
        style.configure("TLabelframe.Label", background=COLORS["surface"], foreground=COLORS["navy"], font=("Helvetica", 12, "bold"))
        self.root.option_add("*TCombobox*Listbox.background", "white")
        self.root.option_add("*TCombobox*Listbox.foreground", COLORS["text"])

    def _create_menu(self) -> None:
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Cargar carpeta MovieLens…", command=self.load_movielens_folder, accelerator="⌘O")
        file_menu.add_command(label="Cargar dos archivos CSV…", command=self.load_csv_pair)
        file_menu.add_command(label="Abrir base SQLite…", command=self.load_sqlite)
        file_menu.add_separator()
        file_menu.add_command(label="Guardar base como SQLite…", command=self.export_sqlite)
        file_menu.add_command(label="Exportar comparación…", command=self.export_results)
        file_menu.add_command(label="Exportar recomendaciones…", command=self.export_recommendations)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.root.destroy)
        menu.add_cascade(label="Archivo", menu=file_menu)

        data_menu = tk.Menu(menu, tearoff=False)
        data_menu.add_command(label="Usar demostración integrada", command=self.reset_demo)
        data_menu.add_command(label="Actualizar vistas", command=self._refresh_dataset_views)
        menu.add_cascade(label="Datos", menu=data_menu)

        run_menu = tk.Menu(menu, tearoff=False)
        run_menu.add_command(label="Comparar algoritmos", command=self.run_comparison, accelerator="⌘R")
        run_menu.add_command(label="Optimizar híbrido", command=self.optimize_hybrid)
        run_menu.add_command(label="Generar recomendaciones", command=self.generate_recommendations, accelerator="⌘G")
        run_menu.add_command(label="Recomendar a un usuario nuevo", command=lambda: self.notebook.select(self.new_user_tab))
        menu.add_cascade(label="Ejecutar", menu=run_menu)
        self.root.configure(menu=menu)

    def _create_layout(self) -> None:
        header = ttk.Frame(self.root, padding=(24, 16), style="Header.TFrame")
        header.pack(fill="x")
        title_group = ttk.Frame(header, style="Header.TFrame")
        title_group.pack(side="left")
        ttk.Label(title_group, text="MOVIELAB", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_group, text="Comparación y experimentación con sistemas de recomendación", style="Subtitle.TLabel").pack(anchor="w")
        ttk.Label(header, textvariable=self.data_stats_text, style="Subtitle.TLabel", justify="right").pack(side="right")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=(12, 8))
        self.data_tab = ttk.Frame(self.notebook, padding=12)
        self.analysis_tab = ttk.Frame(self.notebook, padding=12)
        self.config_tab = ttk.Frame(self.notebook)
        self.comparison_tab = ttk.Frame(self.notebook, padding=12)
        self.recommendation_tab = ttk.Frame(self.notebook, padding=12)
        self.new_user_tab = ttk.Frame(self.notebook, padding=12)
        self.guide_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.data_tab, text="  Datos  ")
        self.notebook.add(self.analysis_tab, text="  Análisis  ")
        self.notebook.add(self.config_tab, text="  Configuración  ")
        self.notebook.add(self.comparison_tab, text="  Comparación  ")
        self.notebook.add(self.recommendation_tab, text="  Recomendaciones  ")
        self.notebook.add(self.new_user_tab, text="  Nuevo usuario  ")
        self.notebook.add(self.guide_tab, text="  Guía  ")

        self._build_data_tab()
        self._build_analysis_tab()
        self._build_config_tab()
        self._build_comparison_tab()
        self._build_recommendation_tab()
        self._build_new_user_tab()
        self._build_guide_tab()

        status = ttk.Frame(self.root, padding=(14, 5))
        status.pack(fill="x", side="bottom")
        ttk.Label(status, textvariable=self.status_text).pack(side="left")
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=180)
        self.progress.pack(side="right")

    def _build_data_tab(self) -> None:
        toolbar = ttk.Frame(self.data_tab)
        toolbar.pack(fill="x", pady=(0, 10))
        ttk.Label(toolbar, text="Base de películas", style="Section.TLabel").pack(side="left")
        ttk.Button(toolbar, text="MovieLens…", command=self.load_movielens_folder).pack(side="right", padx=4)
        ttk.Button(toolbar, text="Abrir SQLite…", command=self.load_sqlite).pack(side="right", padx=4)
        ttk.Button(toolbar, text="Base de ejemplo", command=self.reset_demo).pack(side="right", padx=4)

        cards = ttk.Frame(self.data_tab)
        cards.pack(fill="x", pady=(0, 10))
        self.stat_labels: dict[str, ttk.Label] = {}
        for key, label in [("movies", "Películas"), ("users", "Usuarios"), ("ratings", "Valoraciones"), ("density", "Densidad")]:
            card = ttk.Frame(cards, padding=14, style="Card.TFrame")
            card.pack(side="left", fill="x", expand=True, padx=4)
            ttk.Label(card, text=label, style="Muted.TLabel").pack(anchor="w")
            value = ttk.Label(card, text="—", style="Stat.TLabel")
            value.pack(anchor="w")
            self.stat_labels[key] = value

        content = ttk.Panedwindow(self.data_tab, orient="horizontal")
        content.pack(fill="both", expand=True)
        table_card = ttk.Frame(content, padding=10, style="Card.TFrame")
        chart_card = ttk.Frame(content, padding=10, style="Card.TFrame")
        content.add(table_card, weight=3)
        content.add(chart_card, weight=2)
        ttk.Label(table_card, textvariable=self.data_source_text, style="Card.TLabel", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 8))
        tree_frame = ttk.Frame(table_card, style="Card.TFrame")
        tree_frame.pack(fill="both", expand=True)
        self.movies_tree = ttk.Treeview(tree_frame, columns=("id", "title", "genres"), show="headings")
        for column, text, width in [("id", "ID", 65), ("title", "Título", 300), ("genres", "Géneros", 300)]:
            self.movies_tree.heading(column, text=text)
            self.movies_tree.column(column, width=width, anchor="w")
        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.movies_tree.yview)
        self.movies_tree.configure(yscrollcommand=scroll.set)
        self.movies_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        ttk.Label(chart_card, text="Distribución de valoraciones", style="Card.TLabel", font=("Helvetica", 12, "bold")).pack(anchor="w")
        self.data_figure = Figure(figsize=(5, 4), dpi=100, facecolor="white")
        self.data_axis = self.data_figure.add_subplot(111)
        self.data_canvas = FigureCanvasTkAgg(self.data_figure, master=chart_card)
        self.data_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _field(self, parent, row: int, label: str, variable, values=None, width=12, help_text: str = ""):
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=5)
        if values is not None:
            widget = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=width)
        else:
            widget = ttk.Entry(parent, textvariable=variable, width=width)
        widget.grid(row=row, column=1, sticky="w", pady=5)
        if help_text:
            ttk.Label(parent, text=help_text, style="Muted.TLabel").grid(row=row, column=2, sticky="w", padx=8)
        return widget

    def _build_config_tab(self) -> None:
        scrollable = ScrollableFrame(self.config_tab)
        scrollable.pack(fill="both", expand=True)
        body = scrollable.inner
        ttk.Label(body, text="Configuración del experimento", style="Section.TLabel").pack(anchor="w", pady=(0, 10))

        models = ttk.LabelFrame(body, text="Algoritmos incluidos")
        models.pack(fill="x", pady=7)
        for index, name in enumerate(MODEL_NAMES):
            ttk.Checkbutton(models, text=name, variable=self.model_vars[name]).grid(row=index // 3, column=index % 3, sticky="w", padx=12, pady=6)

        row = ttk.Frame(body)
        row.pack(fill="x")
        validation = ttk.LabelFrame(row, text="Validación")
        baseline = ttk.LabelFrame(row, text="Modelo base")
        validation.pack(side="left", fill="both", expand=True, padx=(0, 5), pady=7)
        baseline.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=7)
        self._field(validation, 0, "Proporción de prueba", self.test_ratio, help_text="0,10 – 0,40")
        self._field(validation, 1, "Tamaño del top-K", self.top_k, help_text="3 – 50")
        self._field(validation, 2, "Nota relevante", self.relevance, help_text="2,5 – 5")
        self._field(validation, 3, "Semilla aleatoria", self.seed)
        self._field(baseline, 0, "Regularización", self.base_regularization, help_text="Reduce sobreajuste")
        self._field(baseline, 1, "Iteraciones", self.base_iterations, help_text="1 – 100")

        knn_row = ttk.Frame(body)
        knn_row.pack(fill="x")
        user_box = ttk.LabelFrame(knn_row, text="K-NN usuario-usuario")
        item_box = ttk.LabelFrame(knn_row, text="K-NN ítem-ítem")
        user_box.pack(side="left", fill="both", expand=True, padx=(0, 5), pady=7)
        item_box.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=7)
        self._field(user_box, 0, "Número de vecinos", self.user_k, help_text="2 – 200")
        self._field(user_box, 1, "Similitud mínima", self.user_similarity, help_text="0 – 1")
        self._field(user_box, 2, "Regularización", self.user_shrinkage, help_text="Solapamientos pequeños")
        self._field(item_box, 0, "Número de vecinos", self.item_k, help_text="2 – 200")
        self._field(item_box, 1, "Similitud mínima", self.item_similarity, help_text="0 – 1")
        self._field(item_box, 2, "Regularización", self.item_shrinkage, help_text="Solapamientos pequeños")

        content_box = ttk.LabelFrame(body, text="Modelo basado en contenido")
        content_box.pack(fill="x", pady=7)
        ttk.Checkbutton(content_box, text="Incluir palabras del título", variable=self.content_title).grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Label(content_box, text="Peso relativo de los géneros", style="Card.TLabel").grid(row=0, column=1, padx=(30, 8))
        ttk.Spinbox(content_box, from_=1, to=10, textvariable=self.content_genre_weight, width=7).grid(row=0, column=2)

        hybrid = ttk.LabelFrame(body, text="Modelo híbrido")
        hybrid.pack(fill="x", pady=7)
        labels = [("base", "Base"), ("user", "Usuario"), ("item", "Ítem"), ("content", "Contenido")]
        for column, (key, label) in enumerate(labels):
            holder = ttk.Frame(hybrid, style="Card.TFrame")
            holder.grid(row=0, column=column, sticky="ew", padx=10, pady=6)
            hybrid.columnconfigure(column, weight=1)
            ttk.Label(holder, text=f"Peso {label}", style="Card.TLabel").pack(anchor="w")
            ttk.Spinbox(holder, from_=0.0, to=1.0, increment=0.05, textvariable=self.hybrid_weights[key], width=8).pack(anchor="w", pady=3)
        ttk.Label(hybrid, textvariable=self.normalized_weights, style="Muted.TLabel").grid(row=1, column=0, columnspan=4, sticky="w", padx=10, pady=4)
        optimizer = ttk.Frame(hybrid, style="Card.TFrame")
        optimizer.grid(row=2, column=0, columnspan=4, sticky="ew", padx=8, pady=8)
        ttk.Label(optimizer, text="Objetivo", style="Card.TLabel").pack(side="left")
        ttk.Combobox(optimizer, textvariable=self.optimization_objective, values=["RMSE", "MAE"], state="readonly", width=8).pack(side="left", padx=6)
        ttk.Label(optimizer, text="Intentos", style="Card.TLabel").pack(side="left", padx=(12, 0))
        ttk.Spinbox(optimizer, from_=10, to=1000, increment=10, textvariable=self.optimization_trials, width=8).pack(side="left", padx=6)
        ttk.Button(optimizer, text="Optimizar pesos automáticamente", command=self.optimize_hybrid).pack(side="left", padx=12)
        ttk.Label(hybrid, textvariable=self.optimization_text, style="Muted.TLabel", wraplength=950).grid(row=3, column=0, columnspan=4, sticky="w", padx=10, pady=4)

        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=14)
        ttk.Button(actions, text="Restablecer parámetros", command=self.reset_parameters).pack(side="left")
        ttk.Button(actions, text="Ejecutar comparación", command=self.run_comparison, style="Accent.TButton").pack(side="right")
        self._update_normalized_weights()

    def _build_comparison_tab(self) -> None:
        toolbar = ttk.Frame(self.comparison_tab)
        toolbar.pack(fill="x", pady=(0, 9))
        ttk.Label(toolbar, text="Resultados comparables", style="Section.TLabel").pack(side="left")
        ttk.Button(toolbar, text="Ejecutar", command=self.run_comparison, style="Accent.TButton").pack(side="right", padx=4)
        ttk.Button(toolbar, text="Exportar CSV…", command=self.export_results).pack(side="right", padx=4)
        ttk.Label(toolbar, text="Gráfico:").pack(side="right", padx=(12, 4))
        self.metric_combo = ttk.Combobox(toolbar, textvariable=self.metric_var, values=["RMSE", "MAE"], state="readonly", width=23)
        self.metric_combo.pack(side="right")
        self.metric_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_comparison_chart())

        table_card = ttk.Frame(self.comparison_tab, padding=8, style="Card.TFrame")
        table_card.pack(fill="x")
        tree_holder = ttk.Frame(table_card, style="Card.TFrame")
        tree_holder.pack(fill="x")
        self.results_tree = ttk.Treeview(tree_holder, show="headings", height=6)
        vscroll = ttk.Scrollbar(tree_holder, orient="vertical", command=self.results_tree.yview)
        hscroll = ttk.Scrollbar(table_card, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
        self.results_tree.pack(side="left", fill="x", expand=True)
        vscroll.pack(side="right", fill="y")
        hscroll.pack(fill="x")

        lower = ttk.Panedwindow(self.comparison_tab, orient="horizontal")
        lower.pack(fill="both", expand=True, pady=(10, 0))
        chart_card = ttk.Frame(lower, padding=10, style="Card.TFrame")
        analysis_card = ttk.Frame(lower, padding=14, style="Card.TFrame")
        lower.add(chart_card, weight=3)
        lower.add(analysis_card, weight=2)
        self.comparison_figure = Figure(figsize=(7, 4), dpi=100, facecolor="white")
        self.comparison_axis = self.comparison_figure.add_subplot(111)
        self.comparison_canvas = FigureCanvasTkAgg(self.comparison_figure, master=chart_card)
        self.comparison_canvas.get_tk_widget().pack(fill="both", expand=True)
        ttk.Label(analysis_card, text="Lectura rápida", style="Card.TLabel", font=("Helvetica", 14, "bold")).pack(anchor="w")
        ttk.Label(analysis_card, textvariable=self.interpretation_text, style="Card.TLabel", justify="left", wraplength=360).pack(anchor="w", pady=12)
        ttk.Label(analysis_card, text="RMSE, MAE y tiempos: menor es mejor.\nPrecisión, recall, NDCG, cobertura, diversidad y novedad: mayor es mejor.", style="Muted.TLabel", justify="left").pack(anchor="w", pady=8)

    def _build_recommendation_tab(self) -> None:
        controls = ttk.LabelFrame(self.recommendation_tab, text="Opciones de recomendación")
        controls.pack(fill="x", pady=(0, 10))
        ttk.Label(controls, text="Usuario", style="Card.TLabel").grid(row=0, column=0, padx=7, pady=6, sticky="w")
        self.user_combo = ttk.Combobox(controls, textvariable=self.rec_user, state="readonly", width=11)
        self.user_combo.grid(row=1, column=0, padx=7, pady=(0, 8))
        self.user_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_history())
        ttk.Label(controls, text="Algoritmo", style="Card.TLabel").grid(row=0, column=1, padx=7, pady=6, sticky="w")
        ttk.Combobox(controls, textvariable=self.rec_model, values=MODEL_NAMES, state="readonly", width=25).grid(row=1, column=1, padx=7, pady=(0, 8))
        ttk.Label(controls, text="Cantidad", style="Card.TLabel").grid(row=0, column=2, padx=7, pady=6, sticky="w")
        ttk.Spinbox(controls, from_=1, to=100, textvariable=self.rec_count, width=8).grid(row=1, column=2, padx=7, pady=(0, 8))
        ttk.Label(controls, text="Género", style="Card.TLabel").grid(row=0, column=3, padx=7, pady=6, sticky="w")
        self.genre_combo = ttk.Combobox(controls, textvariable=self.rec_genre, state="readonly", width=17)
        self.genre_combo.grid(row=1, column=3, padx=7, pady=(0, 8))
        ttk.Label(controls, text="Año desde", style="Card.TLabel").grid(row=0, column=4, padx=7, pady=6, sticky="w")
        ttk.Entry(controls, textvariable=self.rec_year_min, width=9).grid(row=1, column=4, padx=7, pady=(0, 8))
        ttk.Label(controls, text="Año hasta", style="Card.TLabel").grid(row=0, column=5, padx=7, pady=6, sticky="w")
        ttk.Entry(controls, textvariable=self.rec_year_max, width=9).grid(row=1, column=5, padx=7, pady=(0, 8))
        ttk.Label(controls, text="Mín. votos", style="Card.TLabel").grid(row=0, column=6, padx=7, pady=6, sticky="w")
        ttk.Spinbox(controls, from_=0, to=10000, textvariable=self.rec_min_votes, width=9).grid(row=1, column=6, padx=7, pady=(0, 8))

        second = ttk.Frame(controls, style="Card.TFrame")
        second.grid(row=2, column=0, columnspan=7, sticky="ew", padx=7, pady=(2, 8))
        ttk.Checkbutton(second, text="Excluir películas ya valoradas", variable=self.rec_exclude_seen).pack(side="left")
        ttk.Label(second, text="Diversidad", style="Card.TLabel").pack(side="left", padx=(22, 5))
        ttk.Scale(second, from_=0.0, to=0.8, variable=self.rec_diversity, orient="horizontal", length=150).pack(side="left")
        ttk.Label(second, text="0 = máxima afinidad · 0,8 = más variedad", style="Muted.TLabel").pack(side="left", padx=5)
        ttk.Button(second, text="Generar recomendaciones", command=self.generate_recommendations, style="Accent.TButton").pack(side="right")

        panes = ttk.Panedwindow(self.recommendation_tab, orient="horizontal")
        panes.pack(fill="both", expand=True)
        rec_card = ttk.Frame(panes, padding=9, style="Card.TFrame")
        detail_card = ttk.Frame(panes, padding=12, style="Card.TFrame")
        panes.add(rec_card, weight=3)
        panes.add(detail_card, weight=2)
        ttk.Label(rec_card, text="Películas recomendadas", style="Card.TLabel", font=("Helvetica", 13, "bold")).pack(anchor="w", pady=(0, 6))
        rec_holder = ttk.Frame(rec_card, style="Card.TFrame")
        rec_holder.pack(fill="both", expand=True)
        self.rec_tree = ttk.Treeview(rec_holder, columns=("rank", "id", "title", "genres", "score"), show="headings")
        for col, label, width, anchor in [
            ("rank", "#", 40, "center"), ("id", "ID", 55, "center"), ("title", "Título", 290, "w"),
            ("genres", "Géneros", 260, "w"), ("score", "Nota", 70, "center")
        ]:
            self.rec_tree.heading(col, text=label)
            self.rec_tree.column(col, width=width, anchor=anchor)
        rec_scroll = ttk.Scrollbar(rec_holder, orient="vertical", command=self.rec_tree.yview)
        self.rec_tree.configure(yscrollcommand=rec_scroll.set)
        self.rec_tree.pack(side="left", fill="both", expand=True)
        rec_scroll.pack(side="right", fill="y")
        self.rec_tree.bind("<<TreeviewSelect>>", self._show_recommendation_explanation)

        ttk.Label(detail_card, text="Explicación", style="Card.TLabel", font=("Helvetica", 13, "bold")).pack(anchor="w")
        self.explanation = tk.Text(detail_card, height=12, wrap="word", borderwidth=0, background="white", foreground=COLORS["text"], insertbackground=COLORS["text"], font=("Helvetica", 10), padx=4, pady=8)
        self.explanation.pack(fill="x")
        self.explanation.insert("1.0", "Selecciona una recomendación para entender por qué aparece.")
        self.explanation.configure(state="disabled")
        ttk.Separator(detail_card).pack(fill="x", pady=8)
        ttk.Label(detail_card, text="Historial del usuario", style="Card.TLabel", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 5))
        self.history_tree = ttk.Treeview(detail_card, columns=("title", "rating"), show="headings", height=8)
        self.history_tree.heading("title", text="Película")
        self.history_tree.heading("rating", text="Nota")
        self.history_tree.column("title", width=260)
        self.history_tree.column("rating", width=55, anchor="center")
        self.history_tree.pack(fill="both", expand=True)

    def _build_guide_tab(self) -> None:
        guide = tk.Text(self.guide_tab, wrap="word", borderwidth=0, background="white", foreground=COLORS["text"], font=("Helvetica", 12), padx=28, pady=24, spacing1=4, spacing3=10)
        guide.pack(fill="both", expand=True)
        guide.tag_configure("title", font=("Helvetica", 20, "bold"), foreground=COLORS["navy"])
        guide.tag_configure("heading", font=("Helvetica", 14, "bold"), foreground=COLORS["blue"])
        sections = [
            ("title", "Cómo usar MovieLab\n"),
            ("heading", "1. Elige los datos\n"),
            (None, "Usa la demostración integrada, una carpeta MovieLens con movies.csv y ratings.csv, dos CSV separados o una base SQLite compatible.\n"),
            ("heading", "2. Configura el experimento\n"),
            (None, "Selecciona algoritmos, partición de prueba, vecinos, regularización, contenido y pesos del híbrido. Todos los modelos se evalúan con la misma partición para que la comparación sea justa.\n"),
            ("heading", "3. Analiza la base\n"),
            (None, "La pestaña Análisis permite buscar películas, estudiar la actividad de cada usuario, comparar géneros, detectar datos ausentes y observar la concentración de popularidad mediante la curva de cola larga.\n"),
            ("heading", "4. Compara\n"),
            (None, "RMSE y MAE miden el error de nota. Precisión, recall y NDCG miden el ranking. Cobertura, diversidad y novedad muestran propiedades que la exactitud no captura. Los tiempos permiten comparar eficiencia.\n"),
            ("heading", "5. Optimiza el híbrido\n"),
            (None, "El optimizador prueba combinaciones de pesos sobre la partición reservada y conserva la mejor según RMSE o MAE. Para un trabajo académico, utiliza después otra partición final que no haya participado en la optimización.\n"),
            ("heading", "6. Recomienda y explica\n"),
            (None, "Filtra por género, año y popularidad. El control de diversidad reordena el resultado para evitar una lista demasiado repetitiva. Selecciona una fila para ver puntuaciones, popularidad, gustos del usuario y aportación de cada componente híbrido.\n"),
            ("heading", "7. Introduce un usuario nuevo\n"),
            (None, "En Nuevo usuario puedes elegir el formulario sencillo, con opiniones cotidianas y configuración automática, o el avanzado, con control del algoritmo y sus ajustes. En ambos casos necesitas al menos tres películas. Puedes recomendar inmediatamente o guardar sus valoraciones en la base activa.\n"),
        ]
        for tag, text in sections:
            guide.insert("end", text, tag or ())
        guide.configure(state="disabled")

    def _bind_shortcuts(self) -> None:
        modifier = "Command" if self.root.tk.call("tk", "windowingsystem") == "aqua" else "Control"
        self.root.bind_all(f"<{modifier}-o>", lambda _e: self.load_movielens_folder())
        self.root.bind_all(f"<{modifier}-r>", lambda _e: self.run_comparison())
        self.root.bind_all(f"<{modifier}-g>", lambda _e: self.generate_recommendations())

    def _update_normalized_weights(self, *_args) -> None:
        try:
            raw = {key: max(0.0, variable.get()) for key, variable in self.hybrid_weights.items()}
            total = sum(raw.values()) or 1.0
            self.normalized_weights.set("Pesos normalizados: " + " · ".join(f"{key} {value / total:.0%}" for key, value in raw.items()))
        except (tk.TclError, ValueError):
            self.normalized_weights.set("Introduce pesos numéricos válidos.")

    def _validate_configuration(self) -> None:
        if not 0.05 <= self.test_ratio.get() <= 0.5:
            raise ValueError("La proporción de prueba debe estar entre 0,05 y 0,50.")
        if not 1 <= self.top_k.get() <= 100:
            raise ValueError("El top-K debe estar entre 1 y 100.")
        if not 0.5 <= self.relevance.get() <= 5.0:
            raise ValueError("La nota relevante debe estar entre 0,5 y 5.")
        if self.user_k.get() < 1 or self.item_k.get() < 1:
            raise ValueError("El número de vecinos debe ser positivo.")
        if not 0 <= self.user_similarity.get() <= 1 or not 0 <= self.item_similarity.get() <= 1:
            raise ValueError("La similitud mínima debe estar entre 0 y 1.")
        if self.base_regularization.get() < 0 or self.base_iterations.get() < 1:
            raise ValueError("Revisa la regularización y las iteraciones del modelo base.")

    def _model_parameters(self) -> dict[str, dict]:
        base = {"regularization": self.base_regularization.get(), "iterations": self.base_iterations.get()}
        common_baseline = {"baseline_regularization": self.base_regularization.get(), "baseline_iterations": self.base_iterations.get()}
        return {
            "Modelo base": base,
            "K-NN usuario-usuario": {"k": self.user_k.get(), "min_similarity": self.user_similarity.get(), "shrinkage": self.user_shrinkage.get(), **common_baseline},
            "K-NN ítem-ítem": {"k": self.item_k.get(), "min_similarity": self.item_similarity.get(), "shrinkage": self.item_shrinkage.get(), **common_baseline},
            "Basado en contenido": {"use_title": self.content_title.get(), "genre_weight": self.content_genre_weight.get(), **common_baseline},
            "Modelo híbrido": {
                "weights": {key: variable.get() for key, variable in self.hybrid_weights.items()},
                "user_k": self.user_k.get(), "item_k": self.item_k.get(),
                "user_min_similarity": self.user_similarity.get(), "item_min_similarity": self.item_similarity.get(),
                "user_shrinkage": self.user_shrinkage.get(), "item_shrinkage": self.item_shrinkage.get(),
                "content_use_title": self.content_title.get(), "content_genre_weight": self.content_genre_weight.get(),
                "baseline_regularization": self.base_regularization.get(), "baseline_iterations": self.base_iterations.get(),
            },
        }

    def _run_async(self, status: str, worker, success) -> None:
        if self.busy:
            messagebox.showinfo("MovieLab", "Ya hay una operación en curso.")
            return
        self.busy = True
        self.status_text.set(status)
        self.progress.start(12)

        messages: queue.Queue = queue.Queue(maxsize=1)

        def target():
            try:
                result = worker()
            except Exception as error:
                details = traceback.format_exc()
                messages.put(("error", error, details))
            else:
                messages.put(("success", result, None))

        def poll_worker():
            try:
                kind, payload, details = messages.get_nowait()
            except queue.Empty:
                self.root.after(50, poll_worker)
                return
            if kind == "error":
                self._finish_error(payload, details)
            else:
                self._finish_success(payload, success)

        threading.Thread(target=target, daemon=True).start()
        self.root.after(50, poll_worker)

    def _finish_error(self, error: Exception, details: str) -> None:
        self.progress.stop()
        self.busy = False
        self.status_text.set("La operación no se pudo completar")
        messagebox.showerror("Error", f"{error}\n\nDetalle técnico:\n{details[-1200:]}")

    def _finish_success(self, result, callback) -> None:
        self.progress.stop()
        self.busy = False
        try:
            callback(result)
        except Exception as error:
            self.status_text.set("Error al mostrar el resultado")
            messagebox.showerror("Error", str(error))

    def _apply_dataset(self, movies: pd.DataFrame, ratings: pd.DataFrame, source: str) -> None:
        if movies.empty or ratings.empty:
            raise ValueError("La base de datos no contiene películas y valoraciones utilizables.")
        if ratings["userId"].nunique() < 2:
            raise ValueError("Se necesitan valoraciones de al menos dos usuarios.")
        if movies["movieId"].nunique() > 2500:
            reduce = messagebox.askyesno("Base grande", "La base contiene más de 2.500 películas. Las matrices K-NN pueden usar mucha memoria.\n\n¿Limitarla a las 1.500 películas y 2.000 usuarios con más valoraciones?")
            if reduce:
                users = ratings["userId"].value_counts().head(2000).index
                ratings = ratings[ratings["userId"].isin(users)]
                items = ratings["movieId"].value_counts().head(1500).index
                ratings = ratings[ratings["movieId"].isin(items)]
                movies = movies[movies["movieId"].isin(items)]
                source += " (muestra limitada)"
        counts = ratings["userId"].value_counts()
        valid_users = counts[counts >= 4].index
        ratings = ratings[ratings["userId"].isin(valid_users)].reset_index(drop=True)
        movies = movies.reset_index(drop=True)
        if ratings.empty:
            raise ValueError("No quedan usuarios con al menos cuatro valoraciones.")
        self.movies, self.ratings, self.source_name = movies, ratings, source
        self.results = None
        self.recommendations = None
        self.fitted_models.clear()
        self.full_model_cache.clear()
        self._refresh_dataset_views()
        self.status_text.set(f"Datos cargados: {source}")

    def load_movielens_folder(self) -> None:
        folder = filedialog.askdirectory(title="Selecciona la carpeta de MovieLens")
        if not folder:
            return
        movies_path, ratings_path = Path(folder) / "movies.csv", Path(folder) / "ratings.csv"
        if not movies_path.exists() or not ratings_path.exists():
            messagebox.showerror("Archivos no encontrados", "La carpeta debe contener movies.csv y ratings.csv.")
            return
        try:
            self._apply_dataset(*load_movielens_files(movies_path, ratings_path), f"MovieLens · {Path(folder).name}")
        except Exception as error:
            messagebox.showerror("No se pudo abrir MovieLens", str(error))

    def load_csv_pair(self) -> None:
        movies_path = filedialog.askopenfilename(title="Selecciona movies.csv", filetypes=[("CSV", "*.csv")])
        if not movies_path:
            return
        ratings_path = filedialog.askopenfilename(title="Selecciona ratings.csv", filetypes=[("CSV", "*.csv")])
        if not ratings_path:
            return
        try:
            self._apply_dataset(*load_movielens_files(movies_path, ratings_path), "Archivos CSV")
        except Exception as error:
            messagebox.showerror("No se pudieron abrir los CSV", str(error))

    def load_sqlite(self) -> None:
        path = filedialog.askopenfilename(title="Abrir base SQLite", filetypes=[("SQLite", "*.db *.sqlite *.sqlite3"), ("Todos", "*.*")])
        if not path:
            return
        try:
            self._apply_dataset(*load_sqlite_database(path), f"SQLite · {Path(path).name}")
        except Exception as error:
            messagebox.showerror("No se pudo abrir la base", str(error))

    def reset_demo(self) -> None:
        try:
            self._apply_dataset(*load_demo_data(BASE_DIR / "data" / "movies_demo.db"), "Demostración integrada")
        except Exception as error:
            messagebox.showerror("Error", str(error))

    def export_sqlite(self) -> None:
        path = filedialog.asksaveasfilename(title="Guardar base SQLite", defaultextension=".db", filetypes=[("SQLite", "*.db")])
        if path:
            try:
                save_sqlite_database(self.movies, self.ratings, path)
                self.status_text.set(f"Base guardada en {Path(path).name}")
            except Exception as error:
                messagebox.showerror("No se pudo guardar", str(error))

    def _refresh_dataset_views(self) -> None:
        movie_count = self.movies["movieId"].nunique()
        user_count = self.ratings["userId"].nunique()
        rating_count = len(self.ratings)
        density = 100 * rating_count / max(1, movie_count * user_count)
        values = {"movies": f"{movie_count:,}", "users": f"{user_count:,}", "ratings": f"{rating_count:,}", "density": f"{density:.2f}%"}
        for key, value in values.items():
            self.stat_labels[key].configure(text=value)
        self.data_source_text.set(f"Catálogo · {self.source_name}")
        self.data_stats_text.set(f"{movie_count:,} películas · {user_count:,} usuarios · {rating_count:,} valoraciones")

        _clear_tree(self.movies_tree)
        for row in self.movies.head(500).itertuples(index=False):
            self.movies_tree.insert("", "end", values=(row.movieId, row.title, row.genres))

        distribution = self.ratings["rating"].value_counts().sort_index()
        self.data_axis.clear()
        self.data_axis.bar([str(value) for value in distribution.index], distribution.values, color=COLORS["blue"])
        self.data_axis.set_xlabel("Nota")
        self.data_axis.set_ylabel("Número de valoraciones")
        self.data_axis.grid(axis="y", alpha=0.2)
        self.data_figure.tight_layout()
        self.data_canvas.draw_idle()

        users = [str(value) for value in sorted(self.ratings["userId"].unique())]
        self.user_combo.configure(values=users)
        if users:
            self.rec_user.set(users[0])
        genres = sorted({genre for value in self.movies["genres"].dropna() for genre in str(value).split("|")})
        self.genre_combo.configure(values=["Todos", *genres])
        self.rec_genre.set("Todos")
        self._refresh_history()
        self._refresh_analysis_views()
        self._refresh_new_user_sources()

    def reset_parameters(self) -> None:
        self.test_ratio.set(0.20)
        self.top_k.set(10)
        self.relevance.set(4.0)
        self.seed.set(42)
        self.base_regularization.set(10.0)
        self.base_iterations.set(10)
        self.user_k.set(25)
        self.user_similarity.set(0.0)
        self.user_shrinkage.set(10.0)
        self.item_k.set(25)
        self.item_similarity.set(0.0)
        self.item_shrinkage.set(10.0)
        self.content_title.set(False)
        self.content_genre_weight.set(3)
        for key, value in {"base": 0.10, "user": 0.30, "item": 0.30, "content": 0.30}.items():
            self.hybrid_weights[key].set(value)
        self.optimization_text.set("Parámetros restablecidos.")

    def run_comparison(self) -> None:
        try:
            self._validate_configuration()
            selected = [name for name, variable in self.model_vars.items() if variable.get()]
            if not selected:
                raise ValueError("Selecciona al menos un algoritmo.")
            params = self._model_parameters()
            train, test = train_test_split_by_user(self.ratings, self.test_ratio.get(), self.seed.get())
            if test.empty:
                raise ValueError("No hay suficientes valoraciones para crear una prueba.")
            top_k, relevance = self.top_k.get(), self.relevance.get()
        except (ValueError, tk.TclError) as error:
            messagebox.showwarning("Revisa la configuración", str(error))
            return

        def worker():
            return evaluate_models(selected, train, test, self.movies, top_k=top_k, relevance_threshold=relevance, model_params=params)

        def success(payload):
            self.results, self.fitted_models = payload
            self._display_results()
            self.notebook.select(self.comparison_tab)
            self.status_text.set(f"Comparación completada · {len(train):,} entrenamiento / {len(test):,} prueba")

        self._run_async("Entrenando y evaluando algoritmos…", worker, success)

    def _display_results(self) -> None:
        if self.results is None:
            return
        columns = list(self.results.columns)
        self.results_tree.configure(columns=columns)
        for column in columns:
            self.results_tree.heading(column, text=column)
            width = 205 if column == "Algoritmo" else max(95, min(170, len(column) * 8))
            self.results_tree.column(column, width=width, anchor="w" if column == "Algoritmo" else "center")
        _clear_tree(self.results_tree)
        for _, row in self.results.iterrows():
            values = [row[col] if col == "Algoritmo" else f"{float(row[col]):.4f}" for col in columns]
            self.results_tree.insert("", "end", values=values)

        numeric = [column for column in columns if column != "Algoritmo"]
        self.metric_combo.configure(values=numeric)
        if self.metric_var.get() not in numeric:
            self.metric_var.set(numeric[0])
        self._update_comparison_chart()
        rmse_best = self.results.loc[self.results["RMSE"].idxmin()]
        ndcg_col = next((c for c in columns if c.startswith("NDCG@")), None)
        fastest = self.results.loc[self.results["Tiempo entrenamiento (s)"].idxmin()]
        text = f"Mejor error de valoración\n{rmse_best['Algoritmo']} · RMSE {rmse_best['RMSE']:.3f}\n\n"
        if ndcg_col:
            ranking_best = self.results.loc[self.results[ndcg_col].idxmax()]
            text += f"Mejor ranking\n{ranking_best['Algoritmo']} · {ndcg_col} {ranking_best[ndcg_col]:.3f}\n\n"
        text += f"Entrenamiento más rápido\n{fastest['Algoritmo']} · {fastest['Tiempo entrenamiento (s)']:.3f} s"
        self.interpretation_text.set(text)

    def _update_comparison_chart(self) -> None:
        if self.results is None or self.metric_var.get() not in self.results:
            return
        metric = self.metric_var.get()
        values = self.results[metric].to_numpy(dtype=float)
        labels = self.results["Algoritmo"].str.replace("K-NN ", "KNN ", regex=False).to_list()
        lower_is_better = metric in {"RMSE", "MAE", "Tiempo entrenamiento (s)", "Tiempo predicción (s)", "Tiempo recomendación (s)"}
        best_index = int(np.argmin(values) if lower_is_better else np.argmax(values))
        colors = [COLORS["blue"]] * len(values)
        colors[best_index] = COLORS["green"]
        self.comparison_axis.clear()
        bars = self.comparison_axis.barh(labels, values, color=colors)
        self.comparison_axis.invert_yaxis()
        self.comparison_axis.set_title(metric)
        self.comparison_axis.grid(axis="x", alpha=0.2)
        for bar, value in zip(bars, values):
            self.comparison_axis.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.3f}", va="center", fontsize=9)
        self.comparison_figure.tight_layout()
        self.comparison_canvas.draw_idle()

    def optimize_hybrid(self) -> None:
        try:
            self._validate_configuration()
            trials = self.optimization_trials.get()
            if not 10 <= trials <= 2000:
                raise ValueError("Los intentos deben estar entre 10 y 2.000.")
            objective = self.optimization_objective.get()
            seed = self.seed.get()
            params = self._model_parameters()
            train, validation = train_test_split_by_user(self.ratings, self.test_ratio.get(), seed)
        except (ValueError, tk.TclError) as error:
            messagebox.showwarning("Revisa la configuración", str(error))
            return

        def worker():
            return optimize_hybrid_weights(train, validation, self.movies, params, trials, objective, seed)

        def success(payload):
            weights, attempts = payload
            for key, value in weights.items():
                self.hybrid_weights[key].set(round(value, 4))
            best = attempts.iloc[0]
            self.optimization_text.set(f"Mejor combinación de {len(attempts)}: {objective} {best[objective]:.4f} · " + " · ".join(f"{key} {weights[key]:.1%}" for key in ["base", "user", "item", "content"]))
            self.status_text.set("Pesos híbridos optimizados y aplicados")
            messagebox.showinfo("Optimización terminada", "Se han aplicado los mejores pesos encontrados al modelo híbrido.")

        self._run_async("Optimizando pesos del modelo híbrido…", worker, success)

    def _recommendation_candidates(self) -> list[int]:
        table = self.movies.copy()
        genre = self.rec_genre.get()
        if genre and genre != "Todos":
            table = table[table["genres"].fillna("").str.split("|").apply(lambda values: genre in values)]
        years = table["title"].map(_year_from_title)
        if self.rec_year_min.get().strip():
            minimum = int(self.rec_year_min.get())
            table = table[years >= minimum]
            years = table["title"].map(_year_from_title)
        if self.rec_year_max.get().strip():
            maximum = int(self.rec_year_max.get())
            table = table[years <= maximum]
        min_votes = self.rec_min_votes.get()
        if min_votes > 0:
            counts = self.ratings["movieId"].value_counts()
            table = table[table["movieId"].map(counts).fillna(0) >= min_votes]
        return table["movieId"].astype(int).tolist()

    def generate_recommendations(self) -> None:
        try:
            self._validate_configuration()
            user = int(self.rec_user.get())
            name = self.rec_model.get()
            count = self.rec_count.get()
            if not 1 <= count <= 100:
                raise ValueError("La cantidad debe estar entre 1 y 100.")
            candidates = self._recommendation_candidates()
            if not candidates:
                raise ValueError("Ninguna película cumple los filtros actuales.")
            params = self._model_parameters()[name]
            diversity = self.rec_diversity.get()
            exclude_seen = self.rec_exclude_seen.get()
            signature = (self.source_name, len(self.ratings), name, repr(params))
        except (ValueError, tk.TclError) as error:
            messagebox.showwarning("Revisa las opciones", str(error))
            return

        def worker():
            model = self.full_model_cache.get(signature)
            if model is None:
                model = build_model(name, params).fit(self.ratings, self.movies)
                self.full_model_cache[signature] = model
            recs = recommend_for_user(model, user, self.movies, self.ratings, count, candidates, exclude_seen=exclude_seen, diversity_strength=diversity)
            return model, recs

        def success(payload):
            self.active_recommendation_model, self.recommendations = payload
            self._display_recommendations()
            self.notebook.select(self.recommendation_tab)
            self.status_text.set(f"{len(self.recommendations)} recomendaciones generadas para el usuario {user}")

        self._run_async("Calculando recomendaciones…", worker, success)

    def _display_recommendations(self) -> None:
        _clear_tree(self.rec_tree)
        if self.recommendations is None:
            return
        for row in self.recommendations.itertuples(index=False):
            self.rec_tree.insert("", "end", values=(row.posición, row.movieId, row.title, row.genres, f"{row.predicción:.3f}"))
        children = self.rec_tree.get_children()
        if children:
            self.rec_tree.selection_set(children[0])
            self.rec_tree.focus(children[0])
            self._show_recommendation_explanation()

    def _show_recommendation_explanation(self, _event=None) -> None:
        selection = self.rec_tree.selection()
        if not selection or self.active_recommendation_model is None:
            return
        values = self.rec_tree.item(selection[0], "values")
        movie_id, title, genres, score = int(values[1]), values[2], values[3], float(values[4])
        user = int(self.rec_user.get())
        movie_ratings = self.ratings[self.ratings["movieId"] == movie_id]["rating"]
        history = self.ratings[self.ratings["userId"] == user].merge(self.movies, on="movieId")
        liked = history[history["rating"] >= max(4.0, history["rating"].mean())]
        liked_genres = liked["genres"].str.split("|").explode().value_counts().head(4).index.tolist()
        shared = [genre for genre in str(genres).split("|") if genre in liked_genres]
        lines = [
            title, f"Nota estimada: {score:.3f} / 5", f"Géneros: {genres}", "",
            f"Popularidad en la base: {len(movie_ratings)} valoraciones",
            f"Nota media real: {movie_ratings.mean():.2f}" if len(movie_ratings) else "Sin valoraciones previas",
            f"Géneros preferidos del usuario: {', '.join(liked_genres) or 'sin datos suficientes'}",
        ]
        if shared:
            lines.append(f"Coincidencia con sus gustos: {', '.join(shared)}")
        if isinstance(self.active_recommendation_model, HybridRecommender):
            components = self.active_recommendation_model.component_predictions(user, [movie_id])
            raw_weights = self.active_recommendation_model.weights
            total = sum(max(0.0, float(value)) for value in raw_weights.values()) or 1.0
            lines.extend(["", "Aportación del modelo híbrido:"])
            names = {"base": "Base", "user": "KNN usuario", "item": "KNN ítem", "content": "Contenido"}
            for key in ["base", "user", "item", "content"]:
                weight = max(0.0, float(raw_weights.get(key, 0))) / total
                lines.append(f"• {names[key]}: {components[key][0]:.3f} × {weight:.1%}")
        if self.rec_diversity.get() > 0:
            lines.extend(["", f"La lista fue reordenada con diversidad {self.rec_diversity.get():.2f}."])
        self.explanation.configure(state="normal")
        self.explanation.delete("1.0", "end")
        self.explanation.insert("1.0", "\n".join(lines))
        self.explanation.configure(state="disabled")

    def _refresh_history(self) -> None:
        if not hasattr(self, "history_tree") or not self.rec_user.get():
            return
        _clear_tree(self.history_tree)
        try:
            user = int(self.rec_user.get())
        except ValueError:
            return
        history = self.ratings[self.ratings["userId"] == user].merge(self.movies, on="movieId").sort_values("rating", ascending=False)
        for row in history.itertuples(index=False):
            self.history_tree.insert("", "end", values=(row.title, f"{row.rating:.1f}"))

    def export_results(self) -> None:
        if self.results is None:
            messagebox.showinfo("Sin resultados", "Primero ejecuta una comparación.")
            return
        path = filedialog.asksaveasfilename(title="Exportar comparación", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.results.to_csv(path, index=False)
            self.status_text.set(f"Comparación exportada a {Path(path).name}")

    def export_recommendations(self) -> None:
        if self.recommendations is None:
            messagebox.showinfo("Sin recomendaciones", "Primero genera recomendaciones.")
            return
        path = filedialog.asksaveasfilename(title="Exportar recomendaciones", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.recommendations.to_csv(path, index=False)
            self.status_text.set(f"Recomendaciones exportadas a {Path(path).name}")


def main() -> None:
    root = tk.Tk()
    MovieLabDesktop(root)
    root.mainloop()


if __name__ == "__main__":
    main()
