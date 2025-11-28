import datetime as dt
import math
import os
import json

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, callback_context
from dash.dependencies import Input, Output, State
import dash_auth

# ---------- CONFIG ----------
SHEET_KEY = "15BbXIsTKpRxfHlmVGXudwIfG_a2lOsvOoju45XNqV9o"
DATA_SHEET_NAME = "NW_data"
PLANNED_SHEET_NAME = "Planned_NW"

GRAPH1_COLS = ["D"]                # Graph 1: only D
GRAPH2_COLS = ["H"]                # Graph 2: only H
GRAPH3_COLS = ["O"]                # Graph 3: column O
GRAPH4_COLS = ["P"]                # Graph 4: column P
GRAPH5_COLS = ["Q"]                # Graph 5: column Q
GRAPH6_COLS = ["R"]                # Graph 6: column R
GRAPH7_COLS = ["S"]                # Graph 7: column S
GRAPH7_LABEL_CELL = "S1"           # Label for Graph 7

# Get refresh interval from environment variable
REFRESH_MINUTES = float(os.environ.get("REFRESH_MINUTES", "5"))
REFRESH_MS = int(REFRESH_MINUTES * 60 * 1000)

# ---------- HELPERS ----------

def human_short(n):
    """Convert number into K, Lakh, Crore format (for hover text)."""
    try:
        n = float(n)
    except:
        return ""
    abs_n = abs(n)
    if abs_n >= 1e7:
        return f"{n/1e7:.2f} Cr"
    elif abs_n >= 1e5:
        return f"{n/1e5:.2f} L"
    elif abs_n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return str(int(round(n)))

def connect_sheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not found")
    info = json.loads(sa_json)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scope)
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_KEY)

def load_df(ws):
    records = ws.get_all_records()
    if not records:
        return pd.DataFrame(columns=["timestamp"])
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df

def col_header(ws, col_letter):
    return ws.acell(f"{col_letter}1").value

def load_config_data(ws_planned):
    """Load configuration data from Planned_NW sheet (H2:K9, L3:M5, and L7)."""
    try:
        # Get data from H2:K9, L3:M5, and L7
        data = ws_planned.batch_get(['H2:K9', 'L3:M5', 'L7'])
        if not data:
            return {}
        
        config = {}
        
        # Parse H2:K9
        if data[0]:
            rows = data[0]
            for i, row in enumerate(rows):
                row_num = i + 2  # Starting from row 2
                for j, val in enumerate(row):
                    col_letter = chr(72 + j)  # H=72, I=73, J=74, K=75
                    cell_ref = f"{col_letter}{row_num}"
                    config[cell_ref] = val if val else ""
        
        # Parse L3:M5
        if len(data) > 1 and data[1]:
            rows = data[1]
            for i, row in enumerate(rows):
                row_num = i + 3  # Starting from row 3
                for j, val in enumerate(row):
                    col_letter = chr(76 + j)  # L=76, M=77
                    cell_ref = f"{col_letter}{row_num}"
                    config[cell_ref] = val if val else ""
        
        # Parse L7
        if len(data) > 2 and data[2] and data[2][0]:
            config['L7'] = data[2][0][0] if data[2][0][0] else ""
        
        return config
    except Exception as e:
        print(f"Error loading config data: {e}")
        return {}

def save_config_data(ws_planned, cell_ref, value):
    """Save a single cell value back to the Google Sheet."""
    try:
        # Convert to appropriate type
        if value is not None and value != '':
            try:
                # Try to convert to number if possible
                numeric_value = float(value)
                # If it's a whole number, convert to int
                if numeric_value.is_integer():
                    value = int(numeric_value)
                else:
                    value = numeric_value
            except (ValueError, TypeError):
                # Keep as string if not numeric
                pass
        
        # Use USER_ENTERED mode so Google Sheets interprets values correctly
        ws_planned.update(values=[[value]], range_name=cell_ref, value_input_option='USER_ENTERED')
        return True
    except Exception as e:
        print(f"Error saving to {cell_ref}: {e}")
        return False

def load_trajectory_data(ws_planned):
    """Load trajectory data from Planned_NW sheet, filtering by column E (TRUE)."""
    try:
        
        # Get data from columns C, D, E, Z for rows 14 to 1000 (enough to reach age 100+)
        # Column C is age, D is date, E is TRUE/FALSE filter, Z is value
        all_data = ws_planned.batch_get(['C14:C1000', 'D14:D1000', 'E14:E1000', 'Z14:Z1000'])
        
        if len(all_data) < 4:
            print("Not enough data returned from batch_get")
            return [], [], []
            
        c_values = all_data[0]
        d_values = all_data[1]
        e_values = all_data[2]
        z_values = all_data[3]
        
        # Find the first row where E is TRUE
        x_data = []
        age_data = []
        y_data = []
        
        max_len = max(len(c_values), len(d_values), len(e_values), len(z_values))
        
        true_count = 0
        last_true_age = None
        
        for i in range(max_len):
            # Get values safely
            c_row = c_values[i] if i < len(c_values) else []
            e_row = e_values[i] if i < len(e_values) else []
            d_row = d_values[i] if i < len(d_values) else []
            z_row = z_values[i] if i < len(z_values) else []
            
            e_val = e_row[0] if e_row and len(e_row) > 0 else ''
            
            # Check if E column is TRUE (handle multiple formats)
            is_true = False
            if isinstance(e_val, bool):
                is_true = e_val
            elif isinstance(e_val, str):
                is_true = e_val.strip().upper() in ['TRUE', 'T', '1', 'YES']
            elif isinstance(e_val, (int, float)):
                is_true = bool(e_val)
            
            if is_true:
                true_count += 1
                try:
                    # Column C is age
                    c_raw = c_row[0] if c_row and len(c_row) > 0 else None
                    # Column D can be a date string (like 'Nov-2025') or numeric
                    d_raw = d_row[0] if d_row and len(d_row) > 0 else None
                    z_raw = z_row[0] if z_row and len(z_row) > 0 else None
                    
                    if d_raw and z_raw:
                        # Get age
                        try:
                            age_val = int(float(str(c_raw).replace(',', ''))) if c_raw else None
                            if age_val is not None:
                                last_true_age = age_val
                        except (ValueError, TypeError):
                            age_val = None
                        
                        
                        # Try to convert D to float, if fails keep as string/date
                        try:
                            d_val = float(d_raw)
                        except (ValueError, TypeError):
                            # D is a date string, keep it as is
                            d_val = str(d_raw)
                        
                        # Z should be numeric (remove rupee symbol and commas)
                        try:
                            z_str = str(z_raw).replace('₹', '').replace(',', '').strip()
                            z_val = float(z_str)
                            
                            x_data.append(d_val)
                            age_data.append(age_val)
                            y_data.append(z_val)
                        except (ValueError, TypeError):
                            continue
                except Exception:
                    continue
        
        print(f"Trajectory data: Loaded {len(x_data)} points from {true_count} rows with E=TRUE")
        if age_data:
            valid_ages = [a for a in age_data if a is not None]
            if valid_ages:
                print(f"  Age range with E=TRUE: {min(valid_ages)} to {max(valid_ages)}")
        if last_true_age:
            print(f"  Last row with E=TRUE has age: {last_true_age}")
        return x_data, age_data, y_data
    except Exception as e:
        print(f"Error loading trajectory data: {e}")
        return [], [], []

def add_wealth_zones(fig, y_min_padded, y_max_padded, row=None, col=None):
    """Add progressive color zones for wealth levels."""
    # Define zones with progressive green shading
    zones = [
        (float('-inf'), 0, "rgba(200, 0, 0, 0.4)", "Below 0"),           # Dark Red
        (0, 3e7, "rgba(255, 255, 0, 0.3)", "0-3 Cr"),                   # Yellow
        (3e7, 5e7, "rgba(200, 255, 200, 0.3)", "3-5 Cr"),               # Light Green
        (5e7, 8e7, "rgba(150, 255, 150, 0.35)", "5-8 Cr"),              # Slightly Darker Green
        (8e7, 1e8, "rgba(100, 230, 100, 0.35)", "8-10 Cr"),             # Darker Green
        (1e8, 1.2e8, "rgba(80, 210, 80, 0.35)", "10-12 Cr"),            # Progressive Green
        (1.2e8, 1.4e8, "rgba(60, 190, 60, 0.35)", "12-14 Cr"),          # Progressive Green
        (1.4e8, 1.6e8, "rgba(50, 170, 50, 0.4)", "14-16 Cr"),           # Progressive Green
        (1.6e8, 1.8e8, "rgba(40, 150, 40, 0.4)", "16-18 Cr"),           # Progressive Green
        (1.8e8, 2e8, "rgba(30, 130, 30, 0.4)", "18-20 Cr"),             # Progressive Green
        (2e8, float('inf'), "rgba(20, 110, 20, 0.45)", "Above 20 Cr"),  # Darkest Green (capped)
    ]
    
    for zone_min, zone_max, color, label in zones:
        # Check if this zone is within the visible range
        if y_min_padded < zone_max and y_max_padded > zone_min:
            y0 = max(zone_min, y_min_padded)
            y1 = min(zone_max, y_max_padded)
            
            if row is not None and col is not None:
                fig.add_hrect(
                    y0=y0, y1=y1,
                    fillcolor=color, layer="below",
                    line_width=0, row=row, col=col
                )
            else:
                fig.add_hrect(
                    y0=y0, y1=y1,
                    fillcolor=color, layer="below",
                    line_width=0
                )

def add_wealth_threshold_lines(fig, y_min_padded, y_max_padded, row=None, col=None):
    """Add threshold lines at key wealth levels."""
    thresholds = [
        (0, "0", "gray", 1.5),
        (3e7, "3 Cr", "orange", 1.5),
        (5e7, "5 Cr", "yellowgreen", 1.5),
        (8e7, "8 Cr", "green", 2),
        (1e8, "10 Cr", "darkgreen", 2),
        (1.5e8, "15 Cr", "darkgreen", 1.5),
        (2e8, "20 Cr", "darkgreen", 2),
    ]
    
    for value, label, color, width in thresholds:
        if y_min_padded <= value <= y_max_padded:
            if row is not None and col is not None:
                fig.add_hline(
                    y=value, line_dash="dash", line_color=color, line_width=width,
                    annotation_text=label, annotation_position="right",
                    annotation_font_size=10, annotation_font_color=color,
                    row=row, col=col
                )
            else:
                fig.add_hline(
                    y=value, line_dash="dash", line_color=color, line_width=width,
                    annotation_text=label, annotation_position="right",
                    annotation_font_size=10, annotation_font_color=color
                )

def apply_custom_ticks(fig, df_range, headers, row, col):
    """Dynamic, rounded Y ticks (Cr / L / K) depending on range."""
    series_list = []
    for h in headers:
        if h in df_range.columns:
            series_list.append(df_range[h].dropna())

    if not series_list:
        return

    y_all = pd.concat(series_list)
    if y_all.empty:
        return

    y_min = float(y_all.min())
    y_max = float(y_all.max())

    if y_min == y_max:
        fig.update_yaxes(
            row=row, col=col,
            tickmode="array",
            tickvals=[y_min],
            ticktext=[human_short(y_min)]
        )
        return

    max_abs = max(abs(y_min), abs(y_max))
    span = y_max - y_min

    # Decide unit
    if max_abs >= 1e7:          # Crores
        unit = 1e7
        span_units = span / unit
        suffix = " Cr"
    elif max_abs >= 1e5:        # Lakhs
        unit = 1e5
        span_units = span / unit
        suffix = " L"
    elif max_abs >= 1e3:        # Thousands
        unit = 1e3
        span_units = span / unit
        suffix = "K"
    else:
        unit = 1.0
        span_units = span / unit
        suffix = ""

    target_ticks = 6
    raw_step_units = span_units / target_ticks if span_units > 0 else 1

    nice_candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    mag = 10 ** math.floor(math.log10(raw_step_units)) if raw_step_units > 0 else 1
    stepped = [c * mag for c in nice_candidates]
    step_units = min(stepped, key=lambda x: abs(x - raw_step_units))

    step = step_units * unit

    tick_min = math.floor(y_min / step) * step
    tick_max = math.ceil(y_max / step) * step

    tickvals = []
    v = tick_min
    for _ in range(100):
        if v > tick_max + 1e-9:
            break
        tickvals.append(v)
        v += step

    ticktext = []
    for tv in tickvals:
        val_units = tv / unit
        if abs(step_units) < 1:
            label = f"{val_units:.2f}{suffix}"
        else:
            label = f"{val_units:.2f}{suffix}"
        ticktext.append(label)

    fig.update_yaxes(
        row=row, col=col,
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext
    )

def add_series_labels(fig, df_range, headers, xref, yref):
    """Put the series name at the last data point of each line."""
    for h in headers:
        if h not in df_range.columns:
            continue
        series = df_range[["timestamp", h]].dropna()
        if series.empty:
            continue
        last_row = series.iloc[-1]
        x_last = last_row["timestamp"]
        y_last = last_row[h]
        fig.add_annotation(
            x=x_last,
            y=y_last,
            xref=xref,
            yref=yref,
            text=h,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            xshift=5,
            font=dict(size=10)
        )

# ---------- INITIAL SETUP (one-time) ----------

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
if not sa_json:
    raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON env var not set")

info = json.loads(sa_json)
creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scope)
client = gspread.authorize(creds)
ws = client.open_by_key(SHEET_KEY).worksheet(DATA_SHEET_NAME)
ws_planned = client.open_by_key(SHEET_KEY).worksheet(PLANNED_SHEET_NAME)

# ---------- Start date from ENV (for cloud) ----------
start_str = os.environ.get("START_DATE", "").strip()

df_all = load_df(ws)
if df_all.empty:
    print("NW_data is empty — nothing to plot.")
    START_DT = dt.datetime.now()
else:
    if start_str:
        try:
            START_DT = dt.datetime.strptime(start_str, "%Y-%m-%d")
        except ValueError:
            START_DT = df_all["timestamp"].min()
    else:
        START_DT = df_all["timestamp"].min()

# Resolve headers
graph1_headers = [col_header(ws, c) for c in GRAPH1_COLS]
graph2_headers = [col_header(ws, c) for c in GRAPH2_COLS]
graph3_headers = [col_header(ws, c) for c in GRAPH3_COLS]
graph4_headers = [col_header(ws, c) for c in GRAPH4_COLS]
graph5_headers = [col_header(ws, c) for c in GRAPH5_COLS]
graph6_headers = [col_header(ws, c) for c in GRAPH6_COLS]
graph7_label = ws.acell(GRAPH7_LABEL_CELL).value or "Graph 7"
graph7_headers = [col_header(ws, c) for c in GRAPH7_COLS]

print(f"Starting Dash app. Live from {START_DT.date() if not df_all.empty else 'now'} onwards, refresh every 1 minute.")

# ---------- DASH APP ----------

app = Dash(__name__)

# Add basic authentication
VALID = {
    os.environ.get("DASH_USER", "sharma"): 
    os.environ.get("DASH_PASS", "password123")
}

auth = dash_auth.BasicAuth(app, VALID)

# Custom CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }
            
            #react-entry-point > div {
                max-width: 1600px;
                margin: 0 auto;
                background: white;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
                animation: fadeIn 0.5s ease-in;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            h2 {
                color: #1a202c;
                font-weight: 700;
                margin: 0 0 15px 0;
                font-size: 36px;
                letter-spacing: normal;
                line-height: 1.2;
            }
            
            .date-info {
                color: #4a5568;
                font-size: 15px;
                margin-bottom: 30px;
                font-weight: 500;
                line-height: 1.6;
            }
            
            /* Tab container */
            .tabs {
                margin-bottom: 0;
                border-bottom: none;
            }
            
            ._dash-undo-redo {
                display: none;
            }
            
            /* Individual tab styling */
            .tab {
                background-color: transparent !important;
                border: 2px solid transparent !important;
                color: #64748b !important;
                padding: 14px 28px !important;
                font-weight: 600 !important;
                font-size: 15px !important;
                letter-spacing: 0.3px !important;
                border-radius: 10px 10px 0 0 !important;
                margin-right: 8px !important;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                cursor: pointer !important;
            }
            
            .tab:hover {
                background-color: #f1f5f9 !important;
                color: #334155 !important;
                transform: translateY(-2px);
            }
            
            .tab--selected {
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
                color: white !important;
                box-shadow: 0 8px 16px rgba(17, 153, 142, 0.3) !important;
                border: 2px solid transparent !important;
                transform: translateY(-2px);
            }
            
            /* Graph container */
            #tabs-content {
                background: #f8fafc;
                border-radius: 0 0 12px 12px;
                padding: 30px;
                margin-top: 0;
                border: 1px solid #e2e8f0;
                border-top: none;
            }
            
            /* Loading spinner */
            ._dash-loading {
                color: #11998e !important;
            }
            
            /* Plotly graph improvements */
            .js-plotly-plot {
                border-radius: 8px;
                overflow: hidden;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    html.H2("NW Live Dashboard"),
    html.Div(
        f"📅 Start date: {START_DT.date()} | 🔄 Auto-refresh: {REFRESH_MINUTES} minute{'s' if REFRESH_MINUTES != 1 else ''}",
        className='date-info',
        style={
            'padding': '12px 20px',
            'background': 'linear-gradient(to right, #f0fff4, #f0fdf4)',
            'borderRadius': '10px',
            'borderLeft': '4px solid #11998e',
            'display': 'inline-block',
            'marginBottom': '25px'
        }
    ),
    dcc.Tabs(
        id="tabs",
        value='tab-1',
        className='tabs',
        children=[
            dcc.Tab(
                label='📊 Portfolio Overview',
                value='tab-1',
                className='tab',
                selected_className='tab--selected'
            ),
            dcc.Tab(
                label='🎯 Trajectory',
                value='tab-trajectory',
                className='tab',
                selected_className='tab--selected'
            ),
            dcc.Tab(
                label='📈 Detailed Analysis',
                value='tab-2',
                className='tab',
                selected_className='tab--selected'
            ),
        ]
    ),
    html.Div(
        id='trajectory-filter-container',
        style={'marginTop': '20px', 'marginBottom': '10px'},
        children=[
            html.Div([
                html.Label('Show trajectory up to age:', style={
                    'fontWeight': 'bold',
                    'marginRight': '10px',
                    'fontSize': '16px'
                }),
                dcc.Input(
                    id='trajectory-age-filter',
                    type='number',
                    value=100,  # Will be updated dynamically from M4
                    min=1,
                    max=100,
                    step=1,
                    style={'width': '80px', 'padding': '8px', 'border': '2px solid #11998e', 'borderRadius': '4px', 'fontSize': '14px'}
                )
            ], style={
                'padding': '15px 20px',
                'backgroundColor': '#f8fafc',
                'borderRadius': '8px',
                'border': '1px solid #e2e8f0',
                'display': 'none',  # Hidden by default
                'alignItems': 'center'
            }, id='trajectory-filter-inner')
        ]
    ),
    html.Div(id='tabs-content'),
    dcc.Store(id='save-trigger', data=0),
    dcc.Interval(
        id="interval",
        interval=REFRESH_MS,
        n_intervals=0
    )
])

@app.callback(
    [Output('tabs-content', 'children'),
     Output('trajectory-filter-inner', 'style')],
    [Input('tabs', 'value'),
     Input("interval", "n_intervals"),
     Input('trajectory-age-filter', 'value'),
     Input('save-trigger', 'data')]
)
def render_content(tab, n, trajectory_age_filter, save_trigger):
    df = load_df(ws)
    df_range = df[df["timestamp"] >= START_DT].copy()
    # Default values for outputs
    filter_style = {'display': 'none'}
    
    if df_range.empty:
        # return empty fig
        fig = go.Figure()
        fig.update_layout(title="No data in this range yet.")
        return html.Div([dcc.Graph(figure=fig)]), filter_style
    
    if tab == 'tab-1':
        # Tab 1: Graph 1, 2 & NVDA
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=(
                f"Graph 1 ({', '.join(graph1_headers)})",
                f"Graph 2 ({', '.join(graph2_headers)})",
                "NVDA"
            )
        )

        # -------- Graph 1 (row 1) --------
        # Collect data to determine Y-axis range
        graph1_data = []
        for header in graph1_headers:
            if header in df_range.columns:
                y = df_range[header]
                hover_text = [
                    f"{header}: {human_short(v)}" if pd.notna(v) else ""
                    for v in y
                ]
                fig.add_trace(
                    go.Scatter(
                        x=df_range["timestamp"],
                        y=y,
                        mode="lines+markers",
                        name=header,
                        text=hover_text,
                        hovertemplate="%{text}<br>%{x|%d-%b %H:%M}<extra></extra>",
                        legendgroup="g1",
                        showlegend=True
                    ),
                    row=1, col=1
                )
                graph1_data.extend(y.dropna().tolist())
        
        # Add colored zones and threshold lines for Graph 1
        if graph1_data:
            y_min = min(graph1_data)
            y_max = max(graph1_data)
            y_range = y_max - y_min
            y_min_padded = y_min - (y_range * 0.1)
            y_max_padded = y_max + (y_range * 0.1)
            
            add_wealth_zones(fig, y_min_padded, y_max_padded, row=1, col=1)
            add_wealth_threshold_lines(fig, y_min_padded, y_max_padded, row=1, col=1)

        # -------- Graph 2 (row 2) --------
        graph2_data = []
        for header in graph2_headers:
            if header in df_range.columns:
                y = df_range[header]
                hover_text = [
                    f"{header}: {human_short(v)}" if pd.notna(v) else ""
                    for v in y
                ]
                fig.add_trace(
                    go.Scatter(
                        x=df_range["timestamp"],
                        y=y,
                        mode="lines+markers",
                        name=header,
                        text=hover_text,
                        hovertemplate="%{text}<br>%{x|%d-%b %H:%M}<extra></extra>",
                        legendgroup="g2",
                        showlegend=True
                    ),
                    row=2, col=1
                )
                graph2_data.extend(y.dropna().tolist())
        
        # Add colored zones and threshold lines for Graph 2
        if graph2_data:
            y_min = min(graph2_data)
            y_max = max(graph2_data)
            y_range = y_max - y_min
            y_min_padded = y_min - (y_range * 0.1)
            y_max_padded = y_max + (y_range * 0.1)
            
            add_wealth_zones(fig, y_min_padded, y_max_padded, row=2, col=1)
            add_wealth_threshold_lines(fig, y_min_padded, y_max_padded, row=2, col=1)
        
        # -------- Graph 7 (NVDA) (row 3) --------
        graph7_data = []
        for header in graph7_headers:
            if header in df_range.columns:
                y = df_range[header]
                hover_text = [
                    f"NVDA: {human_short(v)}" if pd.notna(v) else ""
                    for v in y
                ]
                fig.add_trace(
                    go.Scatter(
                        x=df_range["timestamp"],
                        y=y,
                        mode="lines+markers",
                        name="NVDA",
                        text=hover_text,
                        hovertemplate="%{text}<br>%{x|%d-%b %H:%M}<extra></extra>",
                        legendgroup="g7",
                        showlegend=True
                    ),
                    row=3, col=1
                )
                graph7_data.extend(y.dropna().tolist())
        
        # Add colored zones and threshold lines for Graph 7
        if graph7_data:
            y_min = min(graph7_data)
            y_max = max(graph7_data)
            y_range = y_max - y_min
            y_min_padded = y_min - (y_range * 0.1)
            y_max_padded = y_max + (y_range * 0.1)
            
            add_wealth_zones(fig, y_min_padded, y_max_padded, row=3, col=1)
            add_wealth_threshold_lines(fig, y_min_padded, y_max_padded, row=3, col=1)
        
        # Axes
        for i in range(1, 4):
            fig.update_xaxes(
                title_text="Time",
                tickformat="%d-%b %H:%M",
                showticklabels=True,
                row=i, col=1
            )
            fig.update_yaxes(title_text="Value (INR)", row=i, col=1)
        
        fig.update_layout(
            title=f"Tab 1: Graph 1, 2 & NVDA (from {START_DT.date()} to {df_range['timestamp'].max().date()})",
            hovermode="x unified",
            height=1000
        )
        
        # Dynamic ticks
        apply_custom_ticks(fig, df_range, graph1_headers, row=1, col=1)
        apply_custom_ticks(fig, df_range, graph2_headers, row=2, col=1)
        apply_custom_ticks(fig, df_range, graph7_headers, row=3, col=1)
        
        # Labels on lines
        add_series_labels(fig, df_range, graph1_headers, xref="x1", yref="y1")
        add_series_labels(fig, df_range, graph2_headers, xref="x2", yref="y2")
        add_series_labels(fig, df_range, graph7_headers, xref="x3", yref="y3")
        
        return html.Div([dcc.Graph(figure=fig)]), filter_style
    
    elif tab == 'tab-trajectory':
        # Tab Trajectory: Plot data from Planned_NW sheet
        # Load config data for settings display
        config_data = load_config_data(ws_planned)
        
        try:
            x_data_full, age_data_full, y_data_full = load_trajectory_data(ws_planned)
            
            if not x_data_full or not y_data_full:
                fig = go.Figure()
                fig.update_layout(
                    title="No trajectory data available",
                    annotations=[{
                        'text': 'No data found where column E=TRUE in Planned_NW sheet (up to age 65)',
                        'xref': 'paper',
                        'yref': 'paper',
                        'x': 0.5,
                        'y': 0.5,
                        'showarrow': False,
                        'font': {'size': 16}
                    }]
                )
                filter_style = {'display': 'none'}
                return html.Div([dcc.Graph(figure=fig)]), filter_style
        except Exception as e:
            print(f"Error in trajectory tab: {e}")
            import traceback
            traceback.print_exc()
            fig = go.Figure()
            fig.update_layout(
                title="Error loading trajectory data",
                annotations=[{
                    'text': f'Error: {str(e)}',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 14, 'color': 'red'}
                }]
            )
            filter_style = {'display': 'none'}
            filter_options = [{'label': 'All Dates', 'value': 'all'}]
            return html.Div([dcc.Graph(figure=fig)]), filter_style
        
        # Filter data based on selected max age (default from Death Age in M4)
        default_max_age = 100  # Fallback default
        try:
            death_age = config_data.get('M4', '')
            if death_age and str(death_age).strip():
                default_max_age = int(float(str(death_age).replace(',', '')))
        except (ValueError, TypeError):
            pass
        
        max_age = trajectory_age_filter if trajectory_age_filter else default_max_age
        
        # Find actual age range in data
        valid_ages = [age for age in age_data_full if age is not None]
        actual_min_age = min(valid_ages) if valid_ages else 0
        actual_max_age = max(valid_ages) if valid_ages else 0
        
        x_data = []
        age_data = []
        y_data = []
        
        for i, age in enumerate(age_data_full):
            if age is None or age <= max_age:
                x_data.append(x_data_full[i])
                age_data.append(age)
                y_data.append(y_data_full[i])
        
        # Show the filter for trajectory tab
        filter_style = {
            'padding': '15px 20px',
            'backgroundColor': '#f8fafc',
            'borderRadius': '8px',
            'border': '1px solid #e2e8f0',
            'display': 'flex',
            'alignItems': 'center'
        }
        
        # Create trajectory plot
        fig = go.Figure()
        
        # Create custom X-axis labels with date and age
        x_labels = []
        for date, age in zip(x_data, age_data):
            if age is not None:
                x_labels.append(f"{date} (Age: {age})")
            else:
                x_labels.append(str(date))
        
        # Create hover text with formatted values
        hover_text = []
        for date, age, y in zip(x_data, age_data, y_data):
            age_str = f"<br>Age: {age}" if age is not None else ""
            hover_text.append(f"Date: {date}{age_str}<br>Value: {human_short(y) if isinstance(y, (int, float)) else y}")
        
        fig.add_trace(
            go.Scatter(
                x=x_labels,  # Use custom labels directly
                y=y_data,
                mode='lines+markers',
                name='Trajectory',
                line=dict(color='black', width=3),
                marker=dict(size=6, color='black'),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            )
        )
        
        # Add colored zones and threshold lines
        if y_data:
            y_min = min(y_data)
            y_max = max(y_data)
            y_range = y_max - y_min
            y_min_padded = y_min - (y_range * 0.1)
            y_max_padded = y_max + (y_range * 0.1)
            
            add_wealth_zones(fig, y_min_padded, y_max_padded)
            add_wealth_threshold_lines(fig, y_min_padded, y_max_padded)
        
        fig.update_layout(
            title=f"Trajectory (Age {actual_min_age}-{actual_max_age} available, showing up to {max_age})",
            xaxis_title="D Column Values (Date/Time)",
            yaxis_title="Z Column Values (INR)",
            hovermode="closest",
            height=700,
            plot_bgcolor='rgba(248, 250, 252, 0.8)',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            tickangle=-90  # Angle dates vertically for better readability
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
        
        # Apply custom ticks for Y-axis
        if y_data:
            y_min = min(y_data)
            y_max = max(y_data)
            y_range = y_max - y_min
            
            # Determine unit for Y-axis
            max_abs = max(abs(y_min), abs(y_max))
            if max_abs >= 1e7:
                unit = 1e7
                suffix = " Cr"
            elif max_abs >= 1e5:
                unit = 1e5
                suffix = " L"
            elif max_abs >= 1e3:
                unit = 1e3
                suffix = "K"
            else:
                unit = 1.0
                suffix = ""
            
            # Create nice tick values
            span_units = (y_max - y_min) / unit if unit > 0 else 1
            target_ticks = 6
            raw_step_units = span_units / target_ticks if span_units > 0 else 1
            
            nice_candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10]
            mag = 10 ** math.floor(math.log10(raw_step_units)) if raw_step_units > 0 else 1
            stepped = [c * mag for c in nice_candidates]
            step_units = min(stepped, key=lambda x: abs(x - raw_step_units))
            
            step = step_units * unit
            tick_min = math.floor(y_min / step) * step
            tick_max = math.ceil(y_max / step) * step
            
            tickvals = []
            v = tick_min
            for _ in range(100):
                if v > tick_max + 1e-9:
                    break
                tickvals.append(v)
                v += step
            
            ticktext = [f"{tv/unit:.2f}{suffix}" for tv in tickvals]
            
            fig.update_yaxes(
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext
            )
        
        # Create compact settings panel showing H2:K9 and L3:M5
        # Editable cells: I2, I4, I5, I6, I7, I9, K4, K9, M3, M4
        
        settings_panel = html.Div([
            html.Div([
                html.H4("⚙️ Settings", style={'margin': '0 0 15px 0', 'color': '#1a202c'}),
                
                # Table layout to show all H2:K9 data
                html.Table([
                    html.Tbody([
                        # Row 2
                        html.Tr([
                            html.Td(config_data.get('H2', 'Year'), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(
                                dcc.Input(id='input-I2', type='number', value=config_data.get('I2', ''), debounce=True,
                                         style={'width': '70px', 'padding': '4px', 'border': '1px solid #11998e', 'borderRadius': '3px', 'fontSize': '13px'}),
                                style={'padding': '8px'}
                            ),
                            html.Td(config_data.get('J2', ''), style={'padding': '8px', 'color': '#666', 'fontSize': '13px'}),
                            html.Td(config_data.get('K2', ''), style={'padding': '8px', 'color': '#666', 'fontSize': '13px'}),
                        ]),
                        # Row 3
                        html.Tr([
                            html.Td(config_data.get('H3', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(config_data.get('I3', ''), style={'padding': '8px', 'color': '#666', 'fontSize': '13px'}),
                            html.Td(config_data.get('J3', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(config_data.get('K3', ''), style={'padding': '8px', 'color': '#666', 'fontSize': '13px'}),
                            html.Td(config_data.get('L3', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(
                                dcc.Input(id='input-M3', type='number', value=config_data.get('M3', ''), debounce=True,
                                         style={'width': '70px', 'padding': '4px', 'border': '1px solid #11998e', 'borderRadius': '3px', 'fontSize': '13px'}),
                                style={'padding': '8px'}
                            ),
                        ]),
                        # Row 4
                        html.Tr([
                            html.Td(config_data.get('H4', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(
                                dcc.Input(id='input-I4', type='number', value=config_data.get('I4', ''), debounce=True,
                                         style={'width': '70px', 'padding': '4px', 'border': '1px solid #11998e', 'borderRadius': '3px', 'fontSize': '13px'}),
                                style={'padding': '8px'}
                            ),
                            html.Td(config_data.get('J4', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(
                                dcc.Input(id='input-K4', type='number', value=config_data.get('K4', ''), debounce=True,
                                         style={'width': '70px', 'padding': '4px', 'border': '1px solid #11998e', 'borderRadius': '3px', 'fontSize': '13px'}),
                                style={'padding': '8px'}
                            ),
                            html.Td(config_data.get('L4', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(
                                dcc.Input(id='input-M4', type='number', value=config_data.get('M4', ''), debounce=True,
                                         style={'width': '90px', 'padding': '4px', 'border': '1px solid #11998e', 'borderRadius': '3px', 'fontSize': '13px'}),
                                style={'padding': '8px'}
                            ),
                        ]),
                        # Row 5
                        html.Tr([
                            html.Td(config_data.get('H5', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(
                                dcc.Input(id='input-I5', type='number', value=config_data.get('I5', ''), debounce=True,
                                         style={'width': '70px', 'padding': '4px', 'border': '1px solid #11998e', 'borderRadius': '3px', 'fontSize': '13px'}),
                                style={'padding': '8px'}
                            ),
                            html.Td(config_data.get('J5', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(config_data.get('K5', ''), style={'padding': '8px', 'color': '#666', 'fontSize': '13px'}),
                            html.Td(config_data.get('L5', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(config_data.get('M5', ''), style={'padding': '8px', 'color': '#666', 'fontSize': '13px'}),
                        ]),
                        # Row 6
                        html.Tr([
                            html.Td(config_data.get('H6', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(
                                dcc.Input(id='input-I6', type='number', value=config_data.get('I6', ''), debounce=True,
                                         style={'width': '70px', 'padding': '4px', 'border': '1px solid #11998e', 'borderRadius': '3px', 'fontSize': '13px'}),
                                style={'padding': '8px'}
                            ),
                            html.Td(config_data.get('J6', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(config_data.get('K6', ''), style={'padding': '8px', 'color': '#666', 'fontSize': '13px'}),
                        ]),
                        # Row 7
                        html.Tr([
                            html.Td(config_data.get('H7', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(
                                dcc.Input(id='input-I7', type='number', value=config_data.get('I7', ''), debounce=True,
                                         style={'width': '70px', 'padding': '4px', 'border': '1px solid #11998e', 'borderRadius': '3px', 'fontSize': '13px'}),
                                style={'padding': '8px'}
                            ),
                            html.Td(config_data.get('J7', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(config_data.get('K7', ''), style={'padding': '8px', 'color': '#666', 'fontSize': '13px'}),
                        ]),
                        # Row 8
                        html.Tr([
                            html.Td(config_data.get('H8', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(config_data.get('I8', ''), style={'padding': '8px', 'color': '#666', 'fontSize': '13px'}),
                            html.Td(config_data.get('J8', ''), style={'padding': '8px', 'color': '#666', 'fontSize': '13px'}),
                            html.Td(config_data.get('K8', ''), style={'padding': '8px', 'color': '#666', 'fontSize': '13px'}),
                        ]),
                        # Row 9
                        html.Tr([
                            html.Td(config_data.get('H9', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(
                                dcc.Input(id='input-I9', type='number', value=config_data.get('I9', ''), debounce=True,
                                         style={'width': '90px', 'padding': '4px', 'border': '1px solid #11998e', 'borderRadius': '3px', 'fontSize': '13px'}),
                                style={'padding': '8px'}
                            ),
                            html.Td(config_data.get('J9', ''), style={'padding': '8px', 'fontWeight': '500', 'color': '#4a5568', 'fontSize': '13px'}),
                            html.Td(
                                dcc.Input(id='input-K9', type='number', value=config_data.get('K9', ''), debounce=True,
                                         style={'width': '90px', 'padding': '4px', 'border': '1px solid #11998e', 'borderRadius': '3px', 'fontSize': '13px'}),
                                style={'padding': '8px'}
                            ),
                        ]),
                    ])
                ], style={'marginTop': '10px'}),
                
                # L7 Status Message with conditional coloring
                html.Div([
                    html.Span(str(config_data.get('L7', '')), 
                             style={
                                 'fontSize': '14px', 
                                 'fontWeight': 'bold',
                                 'color': '#ef4444' if "CAN'T" in str(config_data.get('L7', '')).upper() else '#10b981',
                                 'padding': '8px 12px',
                                 'backgroundColor': 'rgba(239, 68, 68, 0.1)' if "CAN'T" in str(config_data.get('L7', '')).upper() else 'rgba(16, 185, 129, 0.1)',
                                 'borderRadius': '4px',
                                 'display': 'inline-block'
                             })
                ], style={'marginTop': '15px', 'marginBottom': '10px'}),
                
                html.Div(id='save-message', style={'fontSize': '12px', 'marginTop': '10px', 'color': '#10b981'})
            ], style={'padding': '15px 20px', 'backgroundColor': '#f8fafc', 'borderRadius': '8px', 
                     'border': '1px solid #e2e8f0', 'marginBottom': '20px'})
        ])
        
        return html.Div([settings_panel, dcc.Graph(figure=fig)]), filter_style
    
    elif tab == 'tab-2':
        # Tab 2: Graph 3, 4, 5, 6 (reordered to show Q, P, O, R)
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            subplot_titles=(
                f"Graph 3 ({', '.join(graph5_headers)})",
                f"Graph 4 ({', '.join(graph4_headers)})",
                f"Graph 5 ({', '.join(graph3_headers)})",
                f"Graph 6 ({', '.join(graph6_headers)})"
            )
        )

        # -------- Graph 3 (row 1) - showing Q --------
        graph3_data = []
        for header in graph5_headers:
            if header in df_range.columns:
                y = df_range[header]
                hover_text = [
                    f"{header}: {human_short(v)}" if pd.notna(v) else ""
                    for v in y
                ]
                fig.add_trace(
                    go.Scatter(
                        x=df_range["timestamp"],
                        y=y,
                        mode="lines+markers",
                        name=header,
                        text=hover_text,
                        hovertemplate="%{text}<br>%{x|%d-%b %H:%M}<extra></extra>",
                        legendgroup="g3",
                        showlegend=True
                    ),
                    row=1, col=1
                )
                graph3_data.extend(y.dropna().tolist())
        
        # Add colored zones and threshold lines for Graph 3
        if graph3_data:
            y_min = min(graph3_data)
            y_max = max(graph3_data)
            y_range = y_max - y_min
            y_min_padded = y_min - (y_range * 0.1)
            y_max_padded = y_max + (y_range * 0.1)
            
            add_wealth_zones(fig, y_min_padded, y_max_padded, row=1, col=1)
            add_wealth_threshold_lines(fig, y_min_padded, y_max_padded, row=1, col=1)

        # -------- Graph 4 (row 2) - showing P --------
        graph4_data = []
        for header in graph4_headers:
            if header in df_range.columns:
                y = df_range[header]
                hover_text = [
                    f"{header}: {human_short(v)}" if pd.notna(v) else ""
                    for v in y
                ]
                fig.add_trace(
                    go.Scatter(
                        x=df_range["timestamp"],
                        y=y,
                        mode="lines+markers",
                        name=header,
                        text=hover_text,
                        hovertemplate="%{text}<br>%{x|%d-%b %H:%M}<extra></extra>",
                        legendgroup="g4",
                        showlegend=True
                    ),
                    row=2, col=1
                )
                graph4_data.extend(y.dropna().tolist())
        
        # Add colored zones and threshold lines for Graph 4
        if graph4_data:
            y_min = min(graph4_data)
            y_max = max(graph4_data)
            y_range = y_max - y_min
            y_min_padded = y_min - (y_range * 0.1)
            y_max_padded = y_max + (y_range * 0.1)
            
            add_wealth_zones(fig, y_min_padded, y_max_padded, row=2, col=1)
            add_wealth_threshold_lines(fig, y_min_padded, y_max_padded, row=2, col=1)

        # -------- Graph 5 (row 3) - showing O --------
        graph5_data = []
        for header in graph3_headers:
            if header in df_range.columns:
                y = df_range[header]
                hover_text = [
                    f"{header}: {human_short(v)}" if pd.notna(v) else ""
                    for v in y
                ]
                fig.add_trace(
                    go.Scatter(
                        x=df_range["timestamp"],
                        y=y,
                        mode="lines+markers",
                        name=header,
                        text=hover_text,
                        hovertemplate="%{text}<br>%{x|%d-%b %H:%M}<extra></extra>",
                        legendgroup="g5",
                        showlegend=True
                    ),
                    row=3, col=1
                )
                graph5_data.extend(y.dropna().tolist())
        
        # Add colored zones and threshold lines for Graph 5
        if graph5_data:
            y_min = min(graph5_data)
            y_max = max(graph5_data)
            y_range = y_max - y_min
            y_min_padded = y_min - (y_range * 0.1)
            y_max_padded = y_max + (y_range * 0.1)
            
            add_wealth_zones(fig, y_min_padded, y_max_padded, row=3, col=1)
            add_wealth_threshold_lines(fig, y_min_padded, y_max_padded, row=3, col=1)

        # -------- Graph 6 (row 4) - showing R --------
        graph6_data = []
        for header in graph6_headers:
            if header in df_range.columns:
                y = df_range[header]
                hover_text = [
                    f"{header}: {human_short(v)}" if pd.notna(v) else ""
                    for v in y
                ]
                fig.add_trace(
                    go.Scatter(
                        x=df_range["timestamp"],
                        y=y,
                        mode="lines+markers",
                        name=header,
                        text=hover_text,
                        hovertemplate="%{text}<br>%{x|%d-%b %H:%M}<extra></extra>",
                        legendgroup="g6",
                        showlegend=True
                    ),
                    row=4, col=1
                )
                graph6_data.extend(y.dropna().tolist())
        
        # Add colored zones and threshold lines for Graph 6 (special scheme)
        if graph6_data:
            y_min = min(graph6_data)
            y_max = max(graph6_data)
            y_range = y_max - y_min
            y_min_padded = y_min - (y_range * 0.1)
            y_max_padded = y_max + (y_range * 0.1)
            
            # Zone 1: Below 10 Lakh - Yellow
            if y_min_padded < 1e6:
                fig.add_hrect(y0=y_min_padded, y1=min(1e6, y_max_padded),
                    fillcolor="rgba(255, 255, 150, 0.3)", layer="below", line_width=0, row=4, col=1)
            
            # Zone 2: 10 Lakh to 1 Cr - Light Red
            if y_min_padded < 1e7 and y_max_padded > 1e6:
                fig.add_hrect(y0=max(1e6, y_min_padded), y1=min(1e7, y_max_padded),
                    fillcolor="rgba(255, 200, 200, 0.3)", layer="below", line_width=0, row=4, col=1)
            
            # Zone 3: Above 1 Cr - Dark Red
            if y_max_padded > 1e7:
                fig.add_hrect(y0=max(1e7, y_min_padded), y1=y_max_padded,
                    fillcolor="rgba(220, 100, 100, 0.4)", layer="below", line_width=0, row=4, col=1)
            
            # Add threshold lines if they fall within range
            if y_min_padded <= 1e6 <= y_max_padded:
                fig.add_hline(y=1e6, line_dash="dot", line_color="orange", line_width=1.5,
                    annotation_text="10 L", annotation_position="right",
                    annotation_font_size=10, annotation_font_color="orange", row=4, col=1)
            
            if y_min_padded <= 1e7 <= y_max_padded:
                fig.add_hline(y=1e7, line_dash="dash", line_color="red", line_width=2,
                    annotation_text="1 Cr", annotation_position="right",
                    annotation_font_size=11, annotation_font_color="red", row=4, col=1)
        
        # Axes
        for i in range(1, 5):
            fig.update_xaxes(
                title_text="Time",
                tickformat="%d-%b %H:%M",
                showticklabels=True,
                row=i, col=1
            )
            fig.update_yaxes(title_text="Value (INR)", row=i, col=1)
        
        fig.update_layout(
            title=f"Tab 2: Graph 3-6 (from {START_DT.date()} to {df_range['timestamp'].max().date()})",
            hovermode="x unified",
            height=1200
        )
        
        # Dynamic ticks (reordered: Q, P, O, R)
        apply_custom_ticks(fig, df_range, graph5_headers, row=1, col=1)
        apply_custom_ticks(fig, df_range, graph4_headers, row=2, col=1)
        apply_custom_ticks(fig, df_range, graph3_headers, row=3, col=1)
        apply_custom_ticks(fig, df_range, graph6_headers, row=4, col=1)
        
        # Labels on lines (reordered: Q, P, O, R)
        add_series_labels(fig, df_range, graph5_headers, xref="x1", yref="y1")
        add_series_labels(fig, df_range, graph4_headers, xref="x2", yref="y2")
        add_series_labels(fig, df_range, graph3_headers, xref="x3", yref="y3")
        add_series_labels(fig, df_range, graph6_headers, xref="x4", yref="y4")
        
        return html.Div([dcc.Graph(figure=fig)]), filter_style

@app.callback(
    [Output('save-message', 'children'),
     Output('save-trigger', 'data')],
    [Input('input-I2', 'value'),
     Input('input-I4', 'value'),
     Input('input-I5', 'value'),
     Input('input-I6', 'value'),
     Input('input-I7', 'value'),
     Input('input-I9', 'value'),
     Input('input-K4', 'value'),
     Input('input-K9', 'value'),
     Input('input-M3', 'value'),
     Input('input-M4', 'value')],
    [State('save-trigger', 'data')]
)
def save_settings(i2, i4, i5, i6, i7, i9, k4, k9, m3, m4, current_trigger):
    # Check if this is initial load
    ctx = callback_context
    if not ctx.triggered:
        return "", current_trigger or 0
    
    try:
        # Map of cell references to values
        updates = {
            'I2': i2,
            'I4': i4,
            'I5': i5,
            'I6': i6,
            'I7': i7,
            'I9': i9,
            'K4': k4,
            'K9': k9,
            'M3': m3,
            'M4': m4
        }
        
        # Save each cell
        success_count = 0
        for cell_ref, value in updates.items():
            if value is not None and value != '':
                if save_config_data(ws_planned, cell_ref, value):
                    success_count += 1
        
        new_trigger = (current_trigger or 0) + 1
        
        if success_count == len([v for v in updates.values() if v is not None and v != '']):
            message = html.Div([
                html.Span("✓ ", style={'color': '#10b981', 'fontSize': '11px', 'fontWeight': 'bold'}),
                html.Span("Saved", style={'color': '#10b981', 'fontWeight': '600', 'fontSize': '11px'})
            ])
            return message, new_trigger  # Trigger refresh
        else:
            message = html.Div([
                html.Span("⚠ ", style={'color': '#f59e0b', 'fontSize': '11px', 'fontWeight': 'bold'}),
                html.Span(f"Saved {success_count} changes", 
                         style={'color': '#f59e0b', 'fontWeight': '600', 'fontSize': '11px'})
            ])
            return message, new_trigger  # Trigger refresh
    except Exception as e:
        message = html.Div([
            html.Span("✗ ", style={'color': '#ef4444', 'fontSize': '11px', 'fontWeight': 'bold'}),
            html.Span(f"Error: {str(e)}", style={'color': '#ef4444', 'fontWeight': '600', 'fontSize': '11px'})
        ])
        return message, current_trigger or 0

@app.callback(
    Output('trajectory-age-filter', 'value'),
    [Input('tabs', 'value'),
     Input('save-trigger', 'data')]
)
def update_trajectory_age_filter(tab, save_trigger):
    """Update the trajectory age filter to default to Death Age (M4) value."""
    if tab == 'tab-trajectory':
        try:
            config_data = load_config_data(ws_planned)
            
            death_age = config_data.get('M4', '')
            if death_age and str(death_age).strip():
                return int(float(str(death_age).replace(',', '')))
        except Exception as e:
            print(f"Error loading death age for filter: {e}")
            return 100
    return 100

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )
