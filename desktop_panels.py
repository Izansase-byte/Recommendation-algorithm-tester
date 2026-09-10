from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from recommender.analysis import (
    data_quality_report,
    genre_statistics,
    movie_statistics,
    popularity_curve,
    user_statistics,
)
from recommender.evaluation import MODEL_NAMES, recommend_for_user
from recommender.models import HybridRecommender, build_model


BLUE = "#2563EB"
GREEN = "#16A34A"
NAVY = "#14213D"


def _clear(tree: ttk.Treeview) -> None:
    children = tree.get_children()
    if children:
        tree.delete(*children)


class AnalysisPanelMixin:
    """Interactive catalogue, user, genre and data-quality analysis panels."""

    def _build_analysis_tab(self) -> None:
        self.analysis_cache: dict[str, object] = {}
        self.analysis_movie_query = tk.StringVar()
        self.analysis_movie_genre = tk.StringVar(value="Todos")
        self.analysis_movie_min_votes = tk.IntVar(value=0)
        self.analysis_movie_min_score = tk.DoubleVar(value=0.0)
        self.analysis_movie_order = tk.StringVar(value="Más valoradas")
        self.analysis_user = tk.StringVar()
        self.analysis_genre_metric = tk.StringVar(value="Valoraciones")

        toolbar = ttk.Frame(self.analysis_tab)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Label(toolbar, text="Herramientas de análisis", style="Section.TLabel").pack(side="left")
        ttk.Button(toolbar, text="Actualizar estadísticas", command=lambda: self._refresh_analysis_views(force=True)).pack(side="right", padx=4)
        ttk.Button(toolbar, text="Exportar tabla actual…", command=self._export_analysis_table).pack(side="right", padx=4)

        self.analysis_notebook = ttk.Notebook(self.analysis_tab)
        self.analysis_notebook.pack(fill="both", expand=True)
        movies_page = ttk.Frame(self.analysis_notebook, padding=10)
        users_page = ttk.Frame(self.analysis_notebook, padding=10)
        genres_page = ttk.Frame(self.analysis_notebook, padding=10)
        quality_page = ttk.Frame(self.analysis_notebook, padding=10)
        self.analysis_notebook.add(movies_page, text="  Películas  ")
        self.analysis_notebook.add(users_page, text="  Usuarios  ")
        self.analysis_notebook.add(genres_page, text="  Géneros  ")
        self.analysis_notebook.add(quality_page, text="  Calidad y cola larga  ")

        filters = ttk.LabelFrame(movies_page, text="Buscar y filtrar")
        filters.pack(fill="x", pady=(0, 8))
        fields = [
            ("Título", ttk.Entry(filters, textvariable=self.analysis_movie_query, width=25)),
            ("Género", ttk.Combobox(filters, textvariable=self.analysis_movie_genre, state="readonly", width=16)),
            ("Mín. votos", ttk.Spinbox(filters, from_=0, to=100000, textvariable=self.analysis_movie_min_votes, width=8)),
            ("Nota mínima", ttk.Spinbox(filters, from_=0, to=5, increment=.1, textvariable=self.analysis_movie_min_score, width=8)),
            ("Orden", ttk.Combobox(filters, textvariable=self.analysis_movie_order, values=["Más valoradas", "Mejor nota", "Más polémicas", "Título", "Más recientes"], state="readonly", width=17)),
        ]
        self.analysis_genre_combo = fields[1][1]
        for column, (label, widget) in enumerate(fields):
            ttk.Label(filters, text=label, style="Card.TLabel").grid(row=0, column=column, padx=7, pady=(5, 2), sticky="w")
            widget.grid(row=1, column=column, padx=7, pady=(0, 7), sticky="w")
        ttk.Button(filters, text="Aplicar filtros", command=self._filter_movie_analysis).grid(row=1, column=len(fields), padx=12)
        fields[0][1].bind("<Return>", lambda _e: self._filter_movie_analysis())

        movie_pane = ttk.Panedwindow(movies_page, orient="horizontal")
        movie_pane.pack(fill="both", expand=True)
        movie_table_frame = ttk.Frame(movie_pane, style="Card.TFrame", padding=7)
        movie_chart_frame = ttk.Frame(movie_pane, style="Card.TFrame", padding=7)
        movie_pane.add(movie_table_frame, weight=3)
        movie_pane.add(movie_chart_frame, weight=2)
        self.analysis_movies_tree = ttk.Treeview(movie_table_frame, columns=("id", "title", "genres", "year", "votes", "mean", "median", "std"), show="headings")
        headings = [("id", "ID", 55), ("title", "Título", 260), ("genres", "Géneros", 240), ("year", "Año", 60), ("votes", "Votos", 65), ("mean", "Media", 65), ("median", "Mediana", 70), ("std", "Desv.", 65)]
        for col, label, width in headings:
            self.analysis_movies_tree.heading(col, text=label)
            self.analysis_movies_tree.column(col, width=width, anchor="w" if col in {"title", "genres"} else "center")
        scroll = ttk.Scrollbar(movie_table_frame, orient="vertical", command=self.analysis_movies_tree.yview)
        self.analysis_movies_tree.configure(yscrollcommand=scroll.set)
        self.analysis_movies_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.analysis_movie_figure = Figure(figsize=(5, 4), dpi=100, facecolor="white")
        self.analysis_movie_axis = self.analysis_movie_figure.add_subplot(111)
        self.analysis_movie_canvas = FigureCanvasTkAgg(self.analysis_movie_figure, master=movie_chart_frame)
        self.analysis_movie_canvas.get_tk_widget().pack(fill="both", expand=True)

        user_controls = ttk.Frame(users_page)
        user_controls.pack(fill="x", pady=(0, 8))
        ttk.Label(user_controls, text="Usuario:").pack(side="left")
        self.analysis_user_combo = ttk.Combobox(user_controls, textvariable=self.analysis_user, state="readonly", width=12)
        self.analysis_user_combo.pack(side="left", padx=6)
        self.analysis_user_combo.bind("<<ComboboxSelected>>", lambda _e: self._show_user_analysis())
        self.analysis_user_summary = tk.StringVar(value="Selecciona un usuario.")
        ttk.Label(user_controls, textvariable=self.analysis_user_summary).pack(side="left", padx=18)
        user_pane = ttk.Panedwindow(users_page, orient="horizontal")
        user_pane.pack(fill="both", expand=True)
        history_frame = ttk.Frame(user_pane, style="Card.TFrame", padding=7)
        user_chart_frame = ttk.Frame(user_pane, style="Card.TFrame", padding=7)
        user_pane.add(history_frame, weight=3)
        user_pane.add(user_chart_frame, weight=2)
        self.analysis_user_tree = ttk.Treeview(history_frame, columns=("title", "genres", "rating"), show="headings")
        for col, label, width in [("title", "Película", 300), ("genres", "Géneros", 260), ("rating", "Nota", 60)]:
            self.analysis_user_tree.heading(col, text=label)
            self.analysis_user_tree.column(col, width=width, anchor="center" if col == "rating" else "w")
        self.analysis_user_tree.pack(fill="both", expand=True)
        self.analysis_user_figure = Figure(figsize=(5, 4), dpi=100, facecolor="white")
        self.analysis_user_axis = self.analysis_user_figure.add_subplot(111)
        self.analysis_user_canvas = FigureCanvasTkAgg(self.analysis_user_figure, master=user_chart_frame)
        self.analysis_user_canvas.get_tk_widget().pack(fill="both", expand=True)

        genre_toolbar = ttk.Frame(genres_page)
        genre_toolbar.pack(fill="x", pady=(0, 8))
        ttk.Label(genre_toolbar, text="Gráfico:").pack(side="left")
        combo = ttk.Combobox(genre_toolbar, textvariable=self.analysis_genre_metric, values=["Valoraciones", "Nota media", "Películas", "Usuarios"], state="readonly", width=18)
        combo.pack(side="left", padx=6)
        combo.bind("<<ComboboxSelected>>", lambda _e: self._draw_genre_chart())
        genre_pane = ttk.Panedwindow(genres_page, orient="horizontal")
        genre_pane.pack(fill="both", expand=True)
        genre_table_frame = ttk.Frame(genre_pane, style="Card.TFrame", padding=7)
        genre_chart_frame = ttk.Frame(genre_pane, style="Card.TFrame", padding=7)
        genre_pane.add(genre_table_frame, weight=2)
        genre_pane.add(genre_chart_frame, weight=3)
        self.analysis_genre_tree = ttk.Treeview(genre_table_frame, columns=("genre", "movies", "ratings", "users", "mean", "std"), show="headings")
        for col, label, width in [("genre", "Género", 130), ("movies", "Películas", 70), ("ratings", "Votos", 70), ("users", "Usuarios", 70), ("mean", "Media", 65), ("std", "Desv.", 65)]:
            self.analysis_genre_tree.heading(col, text=label)
            self.analysis_genre_tree.column(col, width=width, anchor="w" if col == "genre" else "center")
        self.analysis_genre_tree.pack(fill="both", expand=True)
        self.analysis_genre_figure = Figure(figsize=(7, 4), dpi=100, facecolor="white")
        self.analysis_genre_axis = self.analysis_genre_figure.add_subplot(111)
        self.analysis_genre_canvas = FigureCanvasTkAgg(self.analysis_genre_figure, master=genre_chart_frame)
        self.analysis_genre_canvas.get_tk_widget().pack(fill="both", expand=True)

        quality_pane = ttk.Panedwindow(quality_page, orient="horizontal")
        quality_pane.pack(fill="both", expand=True)
        report_frame = ttk.Frame(quality_pane, style="Card.TFrame", padding=7)
        tail_frame = ttk.Frame(quality_pane, style="Card.TFrame", padding=7)
        quality_pane.add(report_frame, weight=2)
        quality_pane.add(tail_frame, weight=3)
        self.analysis_quality_tree = ttk.Treeview(report_frame, columns=("metric", "value"), show="headings")
        self.analysis_quality_tree.heading("metric", text="Indicador")
        self.analysis_quality_tree.heading("value", text="Valor")
        self.analysis_quality_tree.column("metric", width=240)
        self.analysis_quality_tree.column("value", width=120, anchor="center")
        self.analysis_quality_tree.pack(fill="both", expand=True)
        self.analysis_tail_figure = Figure(figsize=(7, 4), dpi=100, facecolor="white")
        self.analysis_tail_axis = self.analysis_tail_figure.add_subplot(111)
        self.analysis_tail_canvas = FigureCanvasTkAgg(self.analysis_tail_figure, master=tail_frame)
        self.analysis_tail_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _refresh_analysis_views(self, force: bool = False) -> None:
        signature = (self.source_name, len(self.movies), len(self.ratings), int(self.ratings["userId"].nunique()))
        if force or self.analysis_cache.get("signature") != signature:
            self.analysis_cache = {
                "signature": signature,
                "movies": movie_statistics(self.movies, self.ratings),
                "users": user_statistics(self.movies, self.ratings),
                "genres": genre_statistics(self.movies, self.ratings),
                "quality": data_quality_report(self.movies, self.ratings),
                "tail": popularity_curve(self.movies, self.ratings),
            }
        genres = ["Todos", *sorted(self.analysis_cache["genres"]["género"].astype(str))]
        self.analysis_genre_combo.configure(values=genres)
        if self.analysis_movie_genre.get() not in genres:
            self.analysis_movie_genre.set("Todos")
        users = [str(value) for value in self.analysis_cache["users"]["userId"]]
        self.analysis_user_combo.configure(values=users)
        if users and self.analysis_user.get() not in users:
            self.analysis_user.set(users[0])
        self._filter_movie_analysis()
        self._show_user_analysis()
        self._display_genre_analysis()
        self._display_quality_analysis()

    def _filter_movie_analysis(self) -> None:
        if not self.analysis_cache:
            return
        table = self.analysis_cache["movies"].copy()
        query = self.analysis_movie_query.get().strip()
        if query:
            table = table[table["title"].str.contains(query, case=False, na=False, regex=False)]
        genre = self.analysis_movie_genre.get()
        if genre and genre != "Todos":
            table = table[table["genres"].fillna("").str.split("|").apply(lambda values: genre in values)]
        try:
            table = table[(table["votos"] >= self.analysis_movie_min_votes.get()) & (table["media"].fillna(0) >= self.analysis_movie_min_score.get())]
        except tk.TclError:
            return
        order = self.analysis_movie_order.get()
        sort_map = {
            "Más valoradas": (["votos", "media"], [False, False]),
            "Mejor nota": (["media", "votos"], [False, False]),
            "Más polémicas": (["desviación", "votos"], [False, False]),
            "Título": (["title"], [True]),
            "Más recientes": (["año", "votos"], [False, False]),
        }
        columns, ascending = sort_map.get(order, sort_map["Más valoradas"])
        table = table.sort_values(columns, ascending=ascending, na_position="last")
        self.analysis_cache["filtered_movies"] = table
        _clear(self.analysis_movies_tree)
        for row in table.head(1000).itertuples(index=False):
            year = "" if pd.isna(row.año) else int(row.año)
            values = (row.movieId, row.title, row.genres, year, row.votos, self._fmt(row.media), self._fmt(row.mediana), self._fmt(row.desviación))
            self.analysis_movies_tree.insert("", "end", values=values)
        chart = table.head(15).sort_values("media")
        self.analysis_movie_axis.clear()
        self.analysis_movie_axis.barh(chart["title"].str.slice(0, 28), chart["media"].fillna(0), color=BLUE)
        self.analysis_movie_axis.set_xlim(0, 5)
        self.analysis_movie_axis.set_title(f"Nota media · {len(table)} resultados")
        self.analysis_movie_axis.grid(axis="x", alpha=.2)
        self.analysis_movie_figure.tight_layout()
        self.analysis_movie_canvas.draw_idle()

    @staticmethod
    def _fmt(value) -> str:
        return "—" if pd.isna(value) else f"{float(value):.2f}"

    def _show_user_analysis(self) -> None:
        if not self.analysis_cache or not self.analysis_user.get():
            return
        try:
            user_id = int(self.analysis_user.get())
        except ValueError:
            return
        stats = self.analysis_cache["users"]
        row = stats[stats["userId"] == user_id]
        if row.empty:
            return
        item = row.iloc[0]
        self.analysis_user_summary.set(
            f"{int(item['valoraciones'])} valoraciones · media {item['media']:.2f} · género favorito: {item['género_favorito']}"
        )
        history = self.ratings[self.ratings["userId"] == user_id].merge(self.movies, on="movieId").sort_values("rating", ascending=False)
        _clear(self.analysis_user_tree)
        for value in history.itertuples(index=False):
            self.analysis_user_tree.insert("", "end", values=(value.title, value.genres, f"{value.rating:.1f}"))
        distribution = history["rating"].value_counts().sort_index()
        self.analysis_user_axis.clear()
        self.analysis_user_axis.bar([str(value) for value in distribution.index], distribution.values, color=GREEN)
        self.analysis_user_axis.set_title(f"Distribución de notas · usuario {user_id}")
        self.analysis_user_axis.set_xlabel("Nota")
        self.analysis_user_axis.set_ylabel("Películas")
        self.analysis_user_axis.grid(axis="y", alpha=.2)
        self.analysis_user_figure.tight_layout()
        self.analysis_user_canvas.draw_idle()

    def _display_genre_analysis(self) -> None:
        table = self.analysis_cache["genres"]
        _clear(self.analysis_genre_tree)
        for row in table.itertuples(index=False):
            self.analysis_genre_tree.insert("", "end", values=(row.género, row.películas, row.valoraciones, row.usuarios, self._fmt(row.media), self._fmt(row.desviación)))
        self._draw_genre_chart()

    def _draw_genre_chart(self) -> None:
        if not self.analysis_cache:
            return
        table = self.analysis_cache["genres"].copy()
        mapping = {"Valoraciones": "valoraciones", "Nota media": "media", "Películas": "películas", "Usuarios": "usuarios"}
        column = mapping[self.analysis_genre_metric.get()]
        table = table.sort_values(column, ascending=False).head(18).sort_values(column)
        self.analysis_genre_axis.clear()
        self.analysis_genre_axis.barh(table["género"], table[column], color=BLUE)
        self.analysis_genre_axis.set_title(self.analysis_genre_metric.get())
        if column == "media":
            self.analysis_genre_axis.set_xlim(0, 5)
        self.analysis_genre_axis.grid(axis="x", alpha=.2)
        self.analysis_genre_figure.tight_layout()
        self.analysis_genre_canvas.draw_idle()

    def _display_quality_analysis(self) -> None:
        report = self.analysis_cache["quality"]
        _clear(self.analysis_quality_tree)
        for metric, value in report.items():
            shown = f"{value:.3f}" if isinstance(value, float) else value
            self.analysis_quality_tree.insert("", "end", values=(metric, shown))
        curve = self.analysis_cache["tail"]
        self.analysis_tail_axis.clear()
        self.analysis_tail_axis.plot(100 * curve["posición"] / max(1, len(curve)), curve["porcentaje_acumulado"], color=BLUE, linewidth=2.5)
        self.analysis_tail_axis.axhline(80, color="#DC2626", linestyle="--", linewidth=1)
        self.analysis_tail_axis.set_title("Curva de concentración de popularidad")
        self.analysis_tail_axis.set_xlabel("% del catálogo, ordenado por popularidad")
        self.analysis_tail_axis.set_ylabel("% acumulado de valoraciones")
        self.analysis_tail_axis.set_xlim(0, 100)
        self.analysis_tail_axis.set_ylim(0, 100)
        self.analysis_tail_axis.grid(alpha=.2)
        self.analysis_tail_figure.tight_layout()
        self.analysis_tail_canvas.draw_idle()

    def _export_analysis_table(self) -> None:
        if not self.analysis_cache:
            return
        index = self.analysis_notebook.index(self.analysis_notebook.select())
        tables = {
            0: (self.analysis_cache.get("filtered_movies", self.analysis_cache["movies"]), "peliculas"),
            1: (self.analysis_cache["users"], "usuarios"),
            2: (self.analysis_cache["genres"], "generos"),
            3: (pd.DataFrame(list(self.analysis_cache["quality"].items()), columns=["indicador", "valor"]), "calidad"),
        }
        table, name = tables[index]
        path = filedialog.asksaveasfilename(title="Exportar análisis", initialfile=f"analisis_{name}.csv", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            table.to_csv(path, index=False)
            self.status_text.set(f"Análisis exportado a {Path(path).name}")


class NewUserPanelMixin:
    """Form for cold-start profiles based on explicit ratings and preferred genres."""

    def _build_new_user_tab(self) -> None:
        self.new_profile: list[dict[str, float | int]] = []
        self.new_user_mode = tk.StringVar(value="Sencillo")
        self.new_user_name = tk.StringVar()
        self.new_user_model = tk.StringVar(value="Modelo híbrido")
        self.new_user_count = tk.IntVar(value=10)
        self.new_user_search = tk.StringVar()
        self.new_user_rating = tk.DoubleVar(value=4.0)
        self.new_user_genre_boost = tk.DoubleVar(value=.25)
        self.new_user_diversity = tk.DoubleVar(value=.10)
        self.new_user_summary = tk.StringVar(value="Añade al menos tres valoraciones.")
        self.active_new_user_model = None
        self.active_new_user_id: int | None = None
        self.new_user_recommendations: pd.DataFrame | None = None

        mode_bar = ttk.Frame(self.new_user_tab)
        mode_bar.pack(fill="x", pady=(0, 8))
        ttk.Label(mode_bar, text="Tipo de formulario:").pack(side="left")
        ttk.Button(mode_bar, text="Sencillo", command=lambda: self._show_new_user_mode("Sencillo"), style="Accent.TButton").pack(side="left", padx=(8, 4))
        ttk.Button(mode_bar, text="Avanzado", command=lambda: self._show_new_user_mode("Avanzado")).pack(side="left", padx=4)
        ttk.Label(mode_bar, text="Elige el nivel de detalle que prefieras.", style="Muted.TLabel").pack(side="left", padx=14)

        self.simple_user_frame = ttk.Frame(self.new_user_tab)
        self.advanced_user_frame = ttk.Frame(self.new_user_tab)

        header = ttk.Frame(self.advanced_user_frame)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="Formulario avanzado", style="Section.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.new_user_summary).pack(side="right")

        profile = ttk.LabelFrame(self.advanced_user_frame, text="1. Datos y preferencias")
        profile.pack(fill="x", pady=(0, 8))
        for col, label in enumerate(["Nombre (opcional)", "Algoritmo", "Recomendaciones", "Impulso por género", "Diversidad"]):
            ttk.Label(profile, text=label, style="Card.TLabel").grid(row=0, column=col, padx=7, pady=(5, 2), sticky="w")
        ttk.Entry(profile, textvariable=self.new_user_name, width=22).grid(row=1, column=0, padx=7, pady=(0, 7))
        ttk.Combobox(profile, textvariable=self.new_user_model, values=MODEL_NAMES, state="readonly", width=25).grid(row=1, column=1, padx=7, pady=(0, 7))
        ttk.Spinbox(profile, from_=1, to=100, textvariable=self.new_user_count, width=9).grid(row=1, column=2, padx=7, pady=(0, 7))
        ttk.Spinbox(profile, from_=0, to=1.5, increment=.05, textvariable=self.new_user_genre_boost, width=9).grid(row=1, column=3, padx=7, pady=(0, 7))
        ttk.Spinbox(profile, from_=0, to=.8, increment=.05, textvariable=self.new_user_diversity, width=9).grid(row=1, column=4, padx=7, pady=(0, 7))
        ttk.Label(profile, text="Géneros preferidos (puedes elegir varios):", style="Card.TLabel").grid(row=0, column=5, padx=(20, 7), sticky="w")
        self.new_genres_list = tk.Listbox(
            profile,
            selectmode="multiple",
            exportselection=False,
            height=3,
            width=28,
            background="white",
            foreground="#111827",
            selectbackground=BLUE,
            selectforeground="white",
        )
        self.new_genres_list.grid(row=1, column=5, padx=(20, 7), pady=(0, 7), sticky="ew")
        genre_scroll = ttk.Scrollbar(profile, orient="vertical", command=self.new_genres_list.yview)
        genre_scroll.grid(row=1, column=6, sticky="ns", pady=(0, 7))
        self.new_genres_list.configure(yscrollcommand=genre_scroll.set)
        profile.columnconfigure(5, weight=1)

        panes = ttk.Panedwindow(self.advanced_user_frame, orient="horizontal")
        panes.pack(fill="both", expand=True)
        input_card = ttk.Frame(panes, style="Card.TFrame", padding=9)
        output_card = ttk.Frame(panes, style="Card.TFrame", padding=9)
        panes.add(input_card, weight=3)
        panes.add(output_card, weight=3)

        ttk.Label(input_card, text="2. Valora películas conocidas", style="Card.TLabel", font=("Helvetica", 13, "bold")).pack(anchor="w")
        search = ttk.Frame(input_card, style="Card.TFrame")
        search.pack(fill="x", pady=6)
        ttk.Entry(search, textvariable=self.new_user_search).pack(side="left", fill="x", expand=True)
        ttk.Button(search, text="Buscar", command=self._search_new_movies).pack(side="left", padx=5)
        self.new_search_tree = ttk.Treeview(input_card, columns=("id", "title", "genres"), show="headings", height=8)
        for col, label, width in [("id", "ID", 55), ("title", "Título", 280), ("genres", "Géneros", 230)]:
            self.new_search_tree.heading(col, text=label)
            self.new_search_tree.column(col, width=width, anchor="w")
        self.new_search_tree.pack(fill="both", expand=True)
        add = ttk.Frame(input_card, style="Card.TFrame")
        add.pack(fill="x", pady=7)
        ttk.Label(add, text="Nota:", style="Card.TLabel").pack(side="left")
        ttk.Spinbox(add, from_=0.5, to=5, increment=.5, textvariable=self.new_user_rating, width=7).pack(side="left", padx=5)
        ttk.Button(add, text="Añadir valoración", command=self._add_new_rating).pack(side="left")
        ttk.Button(add, text="Quitar seleccionada", command=self._remove_new_rating).pack(side="right")
        ttk.Button(add, text="Vaciar perfil", command=self._clear_new_profile).pack(side="right", padx=5)
        self.new_profile_tree = ttk.Treeview(input_card, columns=("id", "title", "rating"), show="headings", height=6)
        self.new_profile_tree.heading("id", text="ID")
        self.new_profile_tree.heading("title", text="Películas del perfil")
        self.new_profile_tree.heading("rating", text="Nota")
        self.new_profile_tree.column("id", width=55, anchor="center")
        self.new_profile_tree.column("title", width=430)
        self.new_profile_tree.column("rating", width=60, anchor="center")
        self.new_profile_tree.pack(fill="both", expand=True)

        output_toolbar = ttk.Frame(output_card, style="Card.TFrame")
        output_toolbar.pack(fill="x")
        ttk.Label(output_toolbar, text="3. Recomendaciones", style="Card.TLabel", font=("Helvetica", 13, "bold")).pack(side="left")
        ttk.Button(output_toolbar, text="Recomendar", command=self.generate_new_user_recommendations, style="Accent.TButton").pack(side="right")
        ttk.Button(output_toolbar, text="Guardar usuario", command=self.save_new_user).pack(side="right", padx=5)
        self.new_recs_tree = ttk.Treeview(output_card, columns=("rank", "id", "title", "genres", "score"), show="headings", height=12)
        for col, label, width in [("rank", "#", 35), ("id", "ID", 50), ("title", "Título", 260), ("genres", "Géneros", 210), ("score", "Nota", 60)]:
            self.new_recs_tree.heading(col, text=label)
            self.new_recs_tree.column(col, width=width, anchor="center" if col in {"rank", "id", "score"} else "w")
        self.new_recs_tree.pack(fill="both", expand=True, pady=7)
        self.new_recs_tree.bind("<<TreeviewSelect>>", self._explain_new_recommendation)
        self.new_explanation = tk.Text(output_card, height=7, wrap="word", borderwidth=0, background="white", foreground="#111827", insertbackground="#111827", padx=5, pady=5)
        self.new_explanation.pack(fill="x")
        self.new_explanation.insert("1.0", "Aquí aparecerá la explicación de la recomendación seleccionada.")
        self.new_explanation.configure(state="disabled")
        self._build_simple_user_form()
        self._show_new_user_mode("Sencillo")

    def _build_simple_user_form(self) -> None:
        self.simple_genre_vars: dict[str, tk.BooleanVar] = {}
        self.simple_movie_vars = [tk.StringVar() for _ in range(5)]
        self.simple_rating_vars = [tk.StringVar(value="Me gustó") for _ in range(5)]

        intro = ttk.Frame(self.simple_user_frame, style="Card.TFrame", padding=14)
        intro.pack(fill="x", pady=(0, 8))
        ttk.Label(intro, text="Recomendación rápida", style="Card.TLabel", font=("Helvetica", 16, "bold")).pack(anchor="w")
        ttk.Label(
            intro,
            text="Cuéntanos qué te gusta. MovieLab elegirá automáticamente el modelo híbrido y sus ajustes recomendados.",
            style="Card.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        content = ttk.Panedwindow(self.simple_user_frame, orient="horizontal")
        content.pack(fill="both", expand=True)
        form = ttk.Frame(content, style="Card.TFrame", padding=14)
        result = ttk.Frame(content, style="Card.TFrame", padding=14)
        content.add(form, weight=2)
        content.add(result, weight=3)

        name_row = ttk.Frame(form, style="Card.TFrame")
        name_row.pack(fill="x", pady=(0, 10))
        ttk.Label(name_row, text="Tu nombre", style="Card.TLabel").pack(side="left")
        ttk.Entry(name_row, textvariable=self.new_user_name, width=24).pack(side="right", fill="x", expand=True, padx=(12, 0))

        ttk.Label(form, text="1. Elige tus géneros favoritos", style="Card.TLabel", font=("Helvetica", 12, "bold")).pack(anchor="w")
        self.simple_genres_frame = ttk.Frame(form, style="Card.TFrame")
        self.simple_genres_frame.pack(fill="x", pady=(5, 12))

        ttk.Label(form, text="2. Valora al menos tres películas", style="Card.TLabel", font=("Helvetica", 12, "bold")).pack(anchor="w")
        ttk.Label(form, text="No hace falta ser exacto: elige la opción que mejor describa tu opinión.", style="Muted.TLabel").pack(anchor="w", pady=(2, 5))
        self.simple_movie_combos: list[ttk.Combobox] = []
        opinions = ["No me gustó", "Regular", "Me gustó", "Me encantó"]
        for index in range(5):
            row = ttk.Frame(form, style="Card.TFrame")
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=f"Película {index + 1}", style="Card.TLabel", width=10).pack(side="left")
            combo = ttk.Combobox(row, textvariable=self.simple_movie_vars[index], state="readonly", width=34)
            combo.pack(side="left", fill="x", expand=True, padx=5)
            ttk.Combobox(row, textvariable=self.simple_rating_vars[index], values=opinions, state="readonly", width=14).pack(side="right")
            self.simple_movie_combos.append(combo)

        ttk.Button(form, text="Recomendarme películas", command=self._generate_simple_recommendations, style="Accent.TButton").pack(fill="x", pady=(14, 5))
        ttk.Button(form, text="Guardar mi perfil en la base", command=self._save_simple_user).pack(fill="x", pady=4)
        ttk.Button(form, text="Quiero ajustar más opciones", command=lambda: self._show_new_user_mode("Avanzado")).pack(fill="x", pady=4)

        ttk.Label(result, text="Películas para ti", style="Card.TLabel", font=("Helvetica", 14, "bold")).pack(anchor="w")
        self.simple_recs_tree = ttk.Treeview(result, columns=("rank", "id", "title", "genres", "score"), show="headings", height=13)
        for col, label, width in [("rank", "#", 35), ("id", "ID", 45), ("title", "Título", 250), ("genres", "Géneros", 210), ("score", "Nota", 55)]:
            self.simple_recs_tree.heading(col, text=label)
            self.simple_recs_tree.column(col, width=width, anchor="center" if col in {"rank", "id", "score"} else "w")
        self.simple_recs_tree.pack(fill="both", expand=True, pady=(7, 5))
        self.simple_recs_tree.bind("<<TreeviewSelect>>", self._explain_new_recommendation)
        self.simple_explanation = tk.Text(result, height=7, wrap="word", borderwidth=0, background="white", foreground="#111827", insertbackground="#111827", padx=5, pady=5)
        self.simple_explanation.pack(fill="x")
        self.simple_explanation.insert("1.0", "Selecciona una recomendación para saber por qué puede gustarte.")
        self.simple_explanation.configure(state="disabled")

    def _show_new_user_mode(self, mode: str) -> None:
        self.new_user_mode.set(mode)
        self.simple_user_frame.pack_forget()
        self.advanced_user_frame.pack_forget()
        target = self.simple_user_frame if mode == "Sencillo" else self.advanced_user_frame
        target.pack(fill="both", expand=True)

    def _generate_simple_recommendations(self) -> None:
        if self._sync_simple_profile():
            self.generate_new_user_recommendations()

    def _sync_simple_profile(self) -> bool:
        opinion_scores = {"No me gustó": 1.0, "Regular": 2.5, "Me gustó": 4.0, "Me encantó": 5.0}
        title_to_id = self.movies.set_index("title")["movieId"].to_dict()
        profile: dict[int, float] = {}
        for movie_var, opinion_var in zip(self.simple_movie_vars, self.simple_rating_vars):
            title = movie_var.get()
            if title in title_to_id:
                profile[int(title_to_id[title])] = opinion_scores[opinion_var.get()]
        if len(profile) < 3:
            messagebox.showwarning("Faltan películas", "Elige al menos tres películas diferentes para poder recomendarte.")
            return False
        self.new_profile = [{"movieId": movie_id, "rating": rating} for movie_id, rating in profile.items()]
        self.new_user_model.set("Modelo híbrido")
        self.new_user_count.set(10)
        self.new_user_genre_boost.set(.30)
        self.new_user_diversity.set(.10)
        selected_genres = {genre for genre, variable in self.simple_genre_vars.items() if variable.get()}
        self.new_genres_list.selection_clear(0, "end")
        for index in range(self.new_genres_list.size()):
            if self.new_genres_list.get(index) in selected_genres:
                self.new_genres_list.selection_set(index)
        self._render_new_profile()
        return True

    def _save_simple_user(self) -> None:
        if self._sync_simple_profile():
            self.save_new_user()

    def _refresh_new_user_sources(self) -> None:
        genres = sorted({genre for value in self.movies["genres"].fillna("") for genre in str(value).split("|") if genre})
        selected_names = {self.new_genres_list.get(i) for i in self.new_genres_list.curselection()}
        self.new_genres_list.delete(0, "end")
        for index, genre in enumerate(genres):
            self.new_genres_list.insert("end", genre)
            if genre in selected_names:
                self.new_genres_list.selection_set(index)
        previous_simple = {genre for genre, variable in self.simple_genre_vars.items() if variable.get()}
        for child in self.simple_genres_frame.winfo_children():
            child.destroy()
        self.simple_genre_vars = {}
        for index, genre in enumerate(genres):
            variable = tk.BooleanVar(value=genre in previous_simple)
            self.simple_genre_vars[genre] = variable
            ttk.Checkbutton(self.simple_genres_frame, text=genre, variable=variable).grid(
                row=index // 4, column=index % 4, sticky="w", padx=4, pady=2
            )
        popularity = self.ratings["movieId"].value_counts().rename("votos")
        popular_titles = (
            self.movies[["movieId", "title"]]
            .merge(popularity, left_on="movieId", right_index=True, how="left")
            .sort_values(["votos", "title"], ascending=[False, True])["title"]
            .head(250)
            .tolist()
        )
        for combo, variable in zip(self.simple_movie_combos, self.simple_movie_vars):
            combo.configure(values=popular_titles)
            if variable.get() not in popular_titles:
                variable.set("")
        valid_ids = set(self.movies["movieId"].astype(int))
        self.new_profile = [row for row in self.new_profile if int(row["movieId"]) in valid_ids]
        self._search_new_movies()
        self._render_new_profile()

    def _search_new_movies(self) -> None:
        query = self.new_user_search.get().strip()
        table = self.movies
        if query:
            table = table[table["title"].str.contains(query, case=False, na=False, regex=False)]
        rated = {int(row["movieId"]) for row in self.new_profile}
        _clear(self.new_search_tree)
        for row in table[~table["movieId"].isin(rated)].head(150).itertuples(index=False):
            self.new_search_tree.insert("", "end", values=(row.movieId, row.title, row.genres))

    def _add_new_rating(self) -> None:
        selection = self.new_search_tree.selection()
        if not selection:
            messagebox.showinfo("Selecciona una película", "Selecciona una película de la lista de búsqueda.")
            return
        try:
            rating = self.new_user_rating.get()
            if not .5 <= rating <= 5:
                raise ValueError
        except (tk.TclError, ValueError):
            messagebox.showwarning("Nota no válida", "La nota debe estar entre 0,5 y 5.")
            return
        values = self.new_search_tree.item(selection[0], "values")
        movie_id = int(values[0])
        self.new_profile = [row for row in self.new_profile if int(row["movieId"]) != movie_id]
        self.new_profile.append({"movieId": movie_id, "rating": float(rating)})
        self._render_new_profile()
        self._search_new_movies()

    def _remove_new_rating(self) -> None:
        selection = self.new_profile_tree.selection()
        if not selection:
            return
        movie_id = int(self.new_profile_tree.item(selection[0], "values")[0])
        self.new_profile = [row for row in self.new_profile if int(row["movieId"]) != movie_id]
        self._render_new_profile()
        self._search_new_movies()

    def _clear_new_profile(self) -> None:
        self.new_profile.clear()
        self.new_user_recommendations = None
        for variable in self.simple_movie_vars:
            variable.set("")
        self._render_new_profile()
        self._search_new_movies()
        _clear(self.new_recs_tree)
        _clear(self.simple_recs_tree)

    def _render_new_profile(self) -> None:
        _clear(self.new_profile_tree)
        titles = self.movies.set_index("movieId")["title"].to_dict()
        for row in sorted(self.new_profile, key=lambda value: value["rating"], reverse=True):
            self.new_profile_tree.insert("", "end", values=(row["movieId"], titles.get(int(row["movieId"]), ""), f"{row['rating']:.1f}"))
        count = len(self.new_profile)
        self.new_user_summary.set(f"{count} valoraciones añadidas" if count else "Añade al menos tres valoraciones.")

    def _selected_new_genres(self) -> list[str]:
        return [self.new_genres_list.get(index) for index in self.new_genres_list.curselection()]

    def _new_profile_dataframe(self, user_id: int) -> pd.DataFrame:
        rows = [{"userId": user_id, "movieId": int(row["movieId"]), "rating": float(row["rating"])} for row in self.new_profile]
        profile = pd.DataFrame(rows)
        if "timestamp" in self.ratings.columns:
            start = int(self.ratings["timestamp"].max()) + 1 if self.ratings["timestamp"].notna().any() else 1
            profile["timestamp"] = np.arange(start, start + len(profile))
        return profile

    def generate_new_user_recommendations(self) -> None:
        try:
            if len(self.new_profile) < 3:
                raise ValueError("Añade al menos tres películas valoradas para crear un perfil útil.")
            name = self.new_user_model.get()
            count = self.new_user_count.get()
            if not 1 <= count <= 100:
                raise ValueError("La cantidad debe estar entre 1 y 100.")
            boost = self.new_user_genre_boost.get()
            diversity = self.new_user_diversity.get()
            if not 0 <= boost <= 2 or not 0 <= diversity <= .9:
                raise ValueError("Revisa los valores de impulso y diversidad.")
            preferred = self._selected_new_genres()
            params = self._model_parameters()[name]
            new_id = int(self.ratings["userId"].max()) + 1
            profile = self._new_profile_dataframe(new_id)
            combined = pd.concat([self.ratings, profile], ignore_index=True)
            adjustments: dict[int, float] = {}
            if preferred and boost > 0:
                preferred_set = set(preferred)
                for row in self.movies.itertuples(index=False):
                    matches = len(preferred_set.intersection(str(row.genres).split("|")))
                    adjustments[int(row.movieId)] = boost * matches / len(preferred_set)
        except (ValueError, tk.TclError) as error:
            messagebox.showwarning("Perfil incompleto", str(error))
            return

        def worker():
            model = build_model(name, params).fit(combined, self.movies)
            recs = recommend_for_user(
                model, new_id, self.movies, combined, count,
                diversity_strength=diversity, score_adjustments=adjustments,
            )
            return model, recs, combined, new_id, preferred

        def success(payload):
            self.active_new_user_model, self.new_user_recommendations, self.new_user_combined_ratings, self.active_new_user_id, self.active_new_preferred = payload
            self._display_new_user_recommendations()
            label = self.new_user_name.get().strip() or f"Usuario {new_id}"
            self.new_user_summary.set(f"{label}: {len(self.new_user_recommendations)} recomendaciones generadas")
            self.status_text.set(f"Recomendaciones preparadas para {label}")

        self._run_async("Creando el perfil y entrenando el recomendador…", worker, success)

    def _display_new_user_recommendations(self) -> None:
        _clear(self.new_recs_tree)
        _clear(self.simple_recs_tree)
        if self.new_user_recommendations is None:
            return
        for row in self.new_user_recommendations.itertuples(index=False):
            values = (row.posición, row.movieId, row.title, row.genres, f"{row.predicción:.3f}")
            self.new_recs_tree.insert("", "end", values=values)
            self.simple_recs_tree.insert("", "end", values=values)
        for tree in (self.new_recs_tree, self.simple_recs_tree):
            children = tree.get_children()
            if children:
                tree.selection_set(children[0])
        self._explain_new_recommendation()

    def _explain_new_recommendation(self, _event=None) -> None:
        tree = getattr(_event, "widget", None)
        if tree not in (self.new_recs_tree, self.simple_recs_tree):
            tree = self.simple_recs_tree if self.new_user_mode.get() == "Sencillo" else self.new_recs_tree
        selection = tree.selection()
        if not selection or self.active_new_user_model is None:
            return
        values = tree.item(selection[0], "values")
        movie_id, title, genres, score = int(values[1]), values[2], values[3], float(values[4])
        matches = sorted(set(str(genres).split("|")).intersection(getattr(self, "active_new_preferred", [])))
        lines = [title, f"Nota estimada: {score:.3f} / 5", f"Géneros: {genres}", ""]
        if matches:
            lines.append(f"Coincide con los géneros preferidos: {', '.join(matches)}")
        else:
            lines.append("La posición procede del patrón de valoraciones introducido.")
        if isinstance(self.active_new_user_model, HybridRecommender):
            components = self.active_new_user_model.component_predictions(self.active_new_user_id, [movie_id])
            weights = self.active_new_user_model.weights
            total = sum(max(0.0, float(value)) for value in weights.values()) or 1.0
            lines.extend(["", "Componentes del híbrido:"])
            for key, label in [("base", "Base"), ("user", "KNN usuario"), ("item", "KNN ítem"), ("content", "Contenido")]:
                lines.append(f"• {label}: {components[key][0]:.3f} × {max(0, weights.get(key, 0)) / total:.1%}")
        target = self.simple_explanation if tree is self.simple_recs_tree else self.new_explanation
        target.configure(state="normal")
        target.delete("1.0", "end")
        target.insert("1.0", "\n".join(lines))
        target.configure(state="disabled")

    def save_new_user(self) -> None:
        if len(self.new_profile) < 3:
            messagebox.showwarning("Perfil incompleto", "Añade al menos tres valoraciones antes de guardar el usuario.")
            return
        new_id = int(self.ratings["userId"].max()) + 1
        profile = self._new_profile_dataframe(new_id)
        self.ratings = pd.concat([self.ratings, profile], ignore_index=True)
        if not hasattr(self, "user_names"):
            self.user_names = {}
        self.user_names[new_id] = self.new_user_name.get().strip() or f"Usuario {new_id}"
        self.results = None
        self.fitted_models.clear()
        self.full_model_cache.clear()
        self.analysis_cache.clear()
        self._refresh_dataset_views()
        self.rec_user.set(str(new_id))
        self._refresh_history()
        self.status_text.set(f"Usuario {new_id} guardado con {len(profile)} valoraciones")
        messagebox.showinfo("Usuario guardado", f"El perfil se ha añadido a la base activa con el ID {new_id}.\nGuarda la base como SQLite si quieres conservarlo al cerrar.")
