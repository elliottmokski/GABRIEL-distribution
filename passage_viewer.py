import tkinter as tk
from tkinter import ttk, scrolledtext
import pandas as pd
import random
import re
from typing import List, Dict, Any, Optional, Union
import colorsys
import matplotlib.pyplot as plt

class PassageViewer:
    def __init__(self, df: pd.DataFrame, text_column: str, categories: Optional[Union[List[str], str]] = None):
        self.df = df.copy()
        self.text_column = text_column
        self.current_index = 0
        self.last_tooltip_cats = None
        self.selected_snippet_tag = None
        self.dark_mode = True  # Default to dark mode
        # Detect mode: static categories or dynamic coded_passages
        if categories is None and 'coded_passages' in df.columns:
            self.dynamic_mode = True
            all_categories = set()
            for coded_passages in df['coded_passages']:
                if coded_passages and isinstance(coded_passages, dict):
                    all_categories.update(coded_passages.keys())
            self.categories = sorted(list(all_categories))
        elif isinstance(categories, str) and categories == 'coded_passages':
            self.dynamic_mode = True
            all_categories = set()
            for coded_passages in df['coded_passages']:
                if coded_passages and isinstance(coded_passages, dict):
                    all_categories.update(coded_passages.keys())
            self.categories = sorted(list(all_categories))
        else:
            self.dynamic_mode = False
            self.categories = categories if categories else []
        self.colors = self._generate_distinct_colors(len(self.categories))
        self.category_colors = dict(zip(self.categories, self.colors))
        self.tooltip = None
        self._setup_gui()
        self._display_current_text()

    def _generate_distinct_colors(self, n: int) -> List[str]:
        # Use matplotlib tab20 palette for up to 20, then HSV for overflow
        base_colors = []
        if n <= 20:
            cmap = plt.get_cmap('tab20')
            for i in range(n):
                rgb = cmap(i)[:3]
                base_colors.append('#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
            return base_colors
        else:
            cmap = plt.get_cmap('tab20')
            for i in range(20):
                rgb = cmap(i)[:3]
                base_colors.append('#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
            for i in range(20, n):
                hue = (i * 1.0 / n) % 1.0
                r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 1.0)
                base_colors.append('#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)))
            return base_colors[:n]

    def _setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Passage Viewer - Modern Text Analysis")
        self.root.geometry("1600x1000")
        self._apply_theme()

    def _apply_theme(self):
        # Set up theme colors and fonts
        if self.dark_mode:
            bg_main = '#181a1b'
            bg_secondary = '#23272a'
            text_primary = '#f7f7f7'
            text_accent = '#00bcd4'
            text_info = '#b0b0b0'
            legend_border = '#444'
            highlight_sel = '#fff176'
            font_main = ('Quicksand', 20, 'bold')
            font_header = ('Quicksand', 16, 'bold')
            font_legend = ('Quicksand', 15)
            font_info = ('Quicksand', 14)
            font_text = ('Quicksand', 20)
            font_popup = ('Quicksand', 22, 'bold')
        else:
            bg_main = '#f7f7f7'
            bg_secondary = '#eaeaea'
            text_primary = '#23272a'
            text_accent = '#00bcd4'
            text_info = '#444'
            legend_border = '#bbb'
            highlight_sel = '#fff176'
            font_main = ('Quicksand', 20, 'bold')
            font_header = ('Quicksand', 16, 'bold')
            font_legend = ('Quicksand', 15)
            font_info = ('Quicksand', 14)
            font_text = ('Quicksand', 20)
            font_popup = ('Quicksand', 22, 'bold')
        self.bg_main = bg_main
        self.bg_secondary = bg_secondary
        self.text_primary = text_primary
        self.text_accent = text_accent
        self.text_info = text_info
        self.legend_border = legend_border
        self.highlight_sel = highlight_sel
        self.font_main = font_main
        self.font_header = font_header
        self.font_legend = font_legend
        self.font_info = font_info
        self.font_text = font_text
        self.font_popup = font_popup
        self._build_gui()

    def _build_gui(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=self.font_main, background=self.bg_main, foreground=self.text_primary)
        style.configure('Header.TLabel', font=self.font_header, background=self.bg_main, foreground=self.text_accent)
        style.configure('Info.TLabel', font=self.font_info, background=self.bg_main, foreground=self.text_info)
        style.configure('Legend.TLabel', font=self.font_legend, background=self.bg_main, foreground=self.text_primary)
        style.configure('TFrame', background=self.bg_main)
        style.configure('TLabelFrame', background=self.bg_main, foreground=self.text_primary, borderwidth=0, relief='flat')
        style.configure('TLabelFrame.Label', background=self.bg_main, foreground=self.text_primary, font=self.font_main)
        style.configure('Modern.TButton', font=self.font_header, padding=(20, 10), background=self.bg_secondary, foreground=self.text_primary, borderwidth=0, relief='flat')
        style.map('Modern.TButton', background=[('active', self.text_accent), ('pressed', self.text_accent)])
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 15))
        self.info_label = ttk.Label(top_frame, text="", style='Title.TLabel')
        self.info_label.pack(side=tk.LEFT)
        button_frame = ttk.Frame(top_frame)
        button_frame.pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="◀ Previous", command=self._previous_text, style='Modern.TButton').pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_frame, text="Next ▶", command=self._next_text, style='Modern.TButton').pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_frame, text="🎲 Random", command=self._random_text, style='Modern.TButton').pack(side=tk.LEFT, padx=(0, 8))
        self.mode_toggle = ttk.Button(button_frame, text="🌙" if self.dark_mode else "☀️", command=self._toggle_mode, style='Modern.TButton')
        self.mode_toggle.pack(side=tk.LEFT, padx=(0, 8))
        legend_frame = ttk.LabelFrame(main_frame, text="Categories", padding=15)
        legend_frame.pack(fill=tk.X, pady=(0, 0))
        legend_canvas = tk.Canvas(legend_frame, bg=self.bg_main, highlightthickness=0, bd=0, height=120)
        legend_canvas.pack(fill=tk.X, expand=False)
        legend_inner = ttk.Frame(legend_canvas)
        legend_window = legend_canvas.create_window((0, 0), window=legend_inner, anchor='nw')
        n_cats = len(self.categories)
        n_cols = min(5, max(3, (n_cats + 7) // 8))
        self.legend_labels = {}
        self.category_snippet_positions = {cat: [] for cat in self.categories}
        self.category_snippet_indices = {cat: 0 for cat in self.categories}
        for i, (category, color) in enumerate(self.category_colors.items()):
            row = i // n_cols
            col = i % n_cols
            category_frame = ttk.Frame(legend_inner)
            category_frame.grid(row=row, column=col, padx=18, pady=6, sticky='w')
            color_canvas = tk.Canvas(category_frame, width=38, height=24, bg=self.bg_main, highlightthickness=0, bd=0)
            color_canvas.pack(side=tk.LEFT, padx=(0, 12))
            color_canvas.create_rectangle(4, 4, 34, 20, fill=color, outline=self.legend_border, width=2)
            legend_label = ttk.Label(category_frame, text=f"{category.replace('_', ' ').title()} (0)", style='Legend.TLabel', cursor="hand2")
            legend_label.pack(side=tk.LEFT)
            legend_label.bind('<Button-1>', lambda e, cat=category: self._find_next_snippet(cat))
            legend_label.bind('<Enter>', lambda e, lbl=legend_label: lbl.config(foreground=self.text_accent))
            legend_label.bind('<Leave>', lambda e, lbl=legend_label: lbl.config(foreground=self.text_primary))
            self.legend_labels[category] = legend_label
        legend_inner.update_idletasks()
        legend_canvas.config(scrollregion=legend_canvas.bbox("all"))
        if legend_inner.winfo_reqwidth() > legend_canvas.winfo_width():
            legend_canvas.config(width=legend_inner.winfo_reqwidth())
        # Add separator line
        sep = tk.Frame(main_frame, height=2, bg=self.legend_border)
        sep.pack(fill=tk.X, pady=(0, 0))
        text_frame = ttk.LabelFrame(main_frame, text="Text Content", padding=15)
        text_frame.pack(fill=tk.BOTH, expand=True)
        self.text_widget = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=self.font_text,
            bg='#23272a' if self.dark_mode else '#ffffff',
            fg='#f7f7f7' if self.dark_mode else '#000000',
            relief=tk.FLAT,
            borderwidth=2,
            padx=15,
            pady=15,
            selectbackground=self.text_accent,
            selectforeground='#23272a' if self.dark_mode else '#ffffff',
            insertbackground='#f7f7f7' if self.dark_mode else '#000000',
            spacing1=4,
            spacing3=4
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        for category, color in self.category_colors.items():
            self.text_widget.tag_configure(
                category,
                background=color,
                foreground='#23272a' if self.dark_mode else '#000000',
                relief=tk.RAISED,
                borderwidth=1,
                font=self.font_text
            )
            # Emphasis tag for selected snippet
            self.text_widget.tag_configure(f"{category}_emph", background=color, borderwidth=4, relief=tk.SOLID)
        for i, category in enumerate(self.categories):
            self.text_widget.tag_raise(category)
        self.snippet_info = ttk.Label(main_frame, text="", style='Info.TLabel')
        self.snippet_info.pack(pady=(10, 0))
        self.text_widget.bind('<Motion>', self._on_mouse_motion)
        self.text_widget.bind('<Leave>', self._on_mouse_leave)

    def _toggle_mode(self):
        self.dark_mode = not self.dark_mode
        self._apply_theme()
        self._display_current_text()

    def _letters_only(self, text: str) -> str:
        """Keep only lowercase letters a-z, remove everything else."""
        if not text:
            return ""
        return re.sub(r'[^a-z]', '', text.lower())

    def _find_text_position(self, text: str, snippet: str) -> tuple:
        """Robust text position finding using the same logic as codify."""
        clean_snippet = snippet.strip()
        if not clean_snippet:
            return None, None
        
        # Strategy 1: Direct exact match
        start = text.find(clean_snippet)
        if start != -1:
            return start, start + len(clean_snippet)
        
        # Strategy 2: Case-insensitive match  
        start = text.lower().find(clean_snippet.lower())
        if start != -1:
            return start, start + len(clean_snippet)
        
        # Strategy 3: Letters-only matching (most robust)
        text_letters = self._letters_only(text)
        snippet_letters = self._letters_only(clean_snippet)
        
        if snippet_letters and snippet_letters in text_letters:
            # Find approximate position using letters-only
            letters_idx = text_letters.find(snippet_letters)
            ratio = letters_idx / len(text_letters) if text_letters else 0
            approx_start = int(ratio * len(text))
            
            # Search in a window around the approximate position
            window_size = len(clean_snippet) * 3
            search_start = max(0, approx_start - window_size)
            search_end = min(len(text), approx_start + window_size)
            search_text = text[search_start:search_end]
            
            # Try to find exact match in this window
            local_pos = self._find_in_window(search_text, clean_snippet)
            if local_pos is not None:
                return search_start + local_pos[0], search_start + local_pos[1]
        
        # Strategy 4: Try with first/last parts for partial matches
        if len(snippet_letters) >= 20:
            # Try first 20 letters
            first_20 = snippet_letters[:20]
            if first_20 in text_letters:
                letters_idx = text_letters.find(first_20)
                ratio = letters_idx / len(text_letters) if text_letters else 0
                approx_start = int(ratio * len(text))
                
                # Search window
                window_size = len(clean_snippet) * 2
                search_start = max(0, approx_start - window_size//2)
                search_end = min(len(text), approx_start + window_size)
                search_text = text[search_start:search_end]
                
                local_pos = self._find_in_window(search_text, clean_snippet)
                if local_pos is not None:
                    return search_start + local_pos[0], search_start + local_pos[1]
        
        # Strategy 5: Fallback to regex (last resort)
        try:
            pattern = re.escape(clean_snippet[:50])
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.start(), match.end()
        except:
            pass
        
        return None, None

    def _find_in_window(self, window_text: str, target: str) -> tuple:
        """Find target in window using multiple strategies."""
        # Direct match
        idx = window_text.find(target)
        if idx != -1:
            return idx, idx + len(target)
        
        # Case insensitive
        idx = window_text.lower().find(target.lower())
        if idx != -1:
            return idx, idx + len(target)
        
        # Try with some normalization
        normalized_window = re.sub(r'\s+', ' ', window_text.lower())
        normalized_target = re.sub(r'\s+', ' ', target.lower())
        
        idx = normalized_window.find(normalized_target)
        if idx != -1:
            return idx, idx + len(normalized_target)
        
        return None

    def _display_current_text(self):
        if self.current_index >= len(self.df):
            self.current_index = 0
        row = self.df.iloc[self.current_index]
        text = str(row[self.text_column])
        additional_info = ""
        if 'conversation_id' in self.df.columns:
            additional_info = f" | ID: {row['conversation_id']}"
        elif 'id' in self.df.columns:
            additional_info = f" | ID: {row['id']}"
        self.info_label.config(
            text=f"Text {self.current_index + 1} of {len(self.df)}{additional_info}"
        )
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(1.0, text)
        highlights = []
        snippet_count = 0
        self.category_snippet_positions = {cat: [] for cat in self.categories}
        if self.dynamic_mode:
            coded_passages = row['coded_passages'] if 'coded_passages' in row else {}
            for category in self.categories:
                if category in coded_passages and coded_passages[category]:
                    snippets = coded_passages[category]
                    if isinstance(snippets, list):
                        for snippet in snippets:
                            if snippet and isinstance(snippet, str):
                                start_pos, end_pos = self._find_text_position(text, snippet)
                                if start_pos is not None:
                                    highlights.append({
                                        'start': start_pos,
                                        'end': end_pos,
                                        'category': category,
                                        'snippet': snippet
                                    })
                                    self.category_snippet_positions[category].append((start_pos, end_pos))
                                    snippet_count += 1
        else:
            for category in self.categories:
                if category in row and row[category]:
                    snippets = row[category]
                    if isinstance(snippets, list):
                        for snippet in snippets:
                            if snippet and isinstance(snippet, str):
                                start_pos, end_pos = self._find_text_position(text, snippet)
                                if start_pos is not None:
                                    highlights.append({
                                        'start': start_pos,
                                        'end': end_pos,
                                        'category': category,
                                        'snippet': snippet
                                    })
                                    self.category_snippet_positions[category].append((start_pos, end_pos))
                                    snippet_count += 1
        highlights.sort(key=lambda x: x['start'])
        self.text_widget.tag_remove('highlight', '1.0', tk.END)
        for tag in self.text_widget.tag_names():
            if tag.startswith('hover_') or tag.endswith('_emph'):
                self.text_widget.tag_delete(tag)
        for highlight in highlights:
            start_idx = f"1.0+{highlight['start']}c"
            end_idx = f"1.0+{highlight['end']}c"
            tag_name = f"{highlight['category']}_{highlight['start']}_{highlight['end']}"
            self.text_widget.tag_add(tag_name, start_idx, end_idx)
            color = self.category_colors[highlight['category']]
            self.text_widget.tag_configure(
                tag_name,
                background=color,
                foreground='#23272a' if self.dark_mode else '#000000',
                relief=tk.RAISED,
                borderwidth=1,
                font=self.font_text
            )
        self.text_widget.config(state=tk.DISABLED)
        self.current_highlights = highlights
        self._update_legend_counts()
        self._update_snippet_info()
        self.selected_snippet_tag = None

    def _update_legend_counts(self):
        for cat, label in self.legend_labels.items():
            count = len(self.category_snippet_positions.get(cat, []))
            label.config(text=f"{cat.replace('_', ' ').title()} ({count})")

    def _find_next_snippet(self, category):
        positions = self.category_snippet_positions.get(category, [])
        if not positions:
            return  # No snippets for this category
        idx = self.category_snippet_indices.get(category, 0)
        start, end = positions[idx]
        start_idx = f"1.0+{start}c"
        end_idx = f"1.0+{end}c"
        self.text_widget.tag_remove('sel', '1.0', tk.END)
        # Remove previous emphasis
        if self.selected_snippet_tag:
            self.text_widget.tag_remove(self.selected_snippet_tag, '1.0', tk.END)
        # Add emphasis to the selected snippet
        emph_tag = f"{category}_emph"
        self.text_widget.tag_add(emph_tag, start_idx, end_idx)
        self.text_widget.tag_raise(emph_tag)
        self.selected_snippet_tag = emph_tag
        self.text_widget.see(start_idx)
        idx = (idx + 1) % len(positions)
        self.category_snippet_indices[category] = idx

    def _update_snippet_info(self):
        if not hasattr(self, 'current_highlights'):
            self.snippet_info.config(text="")
            return
        category_counts = {}
        for highlight in self.current_highlights:
            cat = highlight['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        if category_counts:
            count_text = ", ".join([f"{cat.replace('_', ' ').title()}: {count}" for cat, count in category_counts.items()])
            self.snippet_info.config(text=f"Highlighted snippets - {count_text}")
        else:
            self.snippet_info.config(text="No snippets found for this text")

    def _next_text(self):
        self.current_index = (self.current_index + 1) % len(self.df)
        self._display_current_text()
    
    def _previous_text(self):
        self.current_index = (self.current_index - 1) % len(self.df)
        self._display_current_text()
    
    def _random_text(self):
        self.current_index = random.randint(0, len(self.df) - 1)
        self._display_current_text()
    
    def show(self):
        self.root.mainloop()
    
    def destroy(self):
        self.root.destroy()

    def _on_mouse_motion(self, event):
        index = self.text_widget.index(f"@{event.x},{event.y}")
        pos = self.text_widget.count('1.0', index, 'chars')[0]
        hovered = []
        for highlight in self.current_highlights:
            if highlight['start'] <= pos < highlight['end']:
                hovered.append(highlight)
        if hovered:
            cats = sorted(set(h['category'] for h in hovered))
            if cats != self.last_tooltip_cats:
                # Show each category in its highlight color
                label_text = " | ".join([
                    f"\u25A0 {cat.replace('_', ' ').title()}" for cat in cats
                ])
                self._show_tooltip(event.x_root, event.y_root, cats)
                self.last_tooltip_cats = cats
        else:
            self._hide_tooltip()
            self.last_tooltip_cats = None

    def _on_mouse_leave(self, event):
        self._hide_tooltip()
        self.last_tooltip_cats = None

    def _show_tooltip(self, x, y, cats):
        if self.tooltip:
            self.tooltip.destroy()
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x+20}+{y+20}")
        frame = tk.Frame(self.tooltip, bg='#23272a' if self.dark_mode else '#f7f7f7', bd=0, highlightthickness=0)
        frame.pack()
        for cat in cats:
            color = self.category_colors.get(cat, '#00bcd4')
            label = tk.Label(frame, text=cat.replace('_', ' ').title(), font=self.font_popup,
                             background='#23272a' if self.dark_mode else '#f7f7f7',
                             foreground=color, padx=18, pady=8, borderwidth=0)
            label.pack(anchor='w')
        # Drop shadow effect
        self.tooltip.lift()
        self.tooltip.attributes('-topmost', True)
        try:
            self.tooltip.attributes('-alpha', 0.98)
        except Exception:
            pass

    def _hide_tooltip(self):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


def view_coded_passages(df: pd.DataFrame, text_column: str, categories: Optional[Union[List[str], str]] = None):
    viewer = PassageViewer(df, text_column, categories)
    viewer.show()
    return viewer


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Sample data
    sample_data = {
        'id': [1, 2, 3],
        'text': [
            "This is a great example of positive text. I really appreciate your help with this matter.",
            "I can't believe how terrible this service is. This is absolutely unacceptable behavior.",
            "Could you please explain how this works? I'm genuinely curious about the process."
        ],
        'positive_sentiment': [
            ["This is a great example of positive text", "I really appreciate your help"],
            [],
            ["I'm genuinely curious about the process"]
        ],
        'negative_sentiment': [
            [],
            ["I can't believe how terrible this service is", "This is absolutely unacceptable behavior"],
            []
        ],
        'questions': [
            [],
            [],
            ["Could you please explain how this works?"]
        ]
    }
    
    df = pd.DataFrame(sample_data)
    categories = ['positive_sentiment', 'negative_sentiment', 'questions']
    
    view_coded_passages(df, 'text', categories) 