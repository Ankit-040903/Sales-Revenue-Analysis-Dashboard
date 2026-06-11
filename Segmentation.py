import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import base64
import io
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Initialize Premium Analytics Application Configuration
app = dash.Dash(
    __name__, 
    title="Customer Segmentation Project",
    update_title="Executing Machine Learning Cluster Engine..."
)

# ==========================================
# 📊 UI INTERFACE COMPONENTS (MODULAR FRAMEWORK)
# ==========================================

def render_control_header():
    """Generates the main enterprise dashboard control header navbar."""
    return html.Div(
        style={
            'borderBottom': '1px solid #2d2d2d', 
            'paddingBottom': '20px', 
            'marginBottom': '35px',
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center'
        }, 
        children=[
            html.Div([
                html.H1("Customer Segmentation & Behavioral Analysis System", 
                        style={'color': '#00ADB5', 'fontSize': '26px', 'fontWeight': '700', 'margin': '0', 'letterSpacing': '0.5px'}),
                html.P("Architected by: Ankit Ahirwar | Domain: Customer Analytics & ML Clustering", 
                        style={'color': '#666666', 'fontSize': '13px', 'margin': '5px 0 0 0'})
            ]),
            html.Div(
                html.Span("ML PIPELINE ONLINE", 
                          style={'color': '#00ffcc', 'fontSize': '11px', 'border': '1px solid #00ffcc', 'padding': '5px 12px', 'borderRadius': '4px', 'fontWeight': 'bold', 'letterSpacing': '1px'})
            )
        ]
    )

def render_pipeline_panel():
    """Generates data pipeline controllers and cluster configuration slicers."""
    return html.Div(
        style={
            'display': 'flex', 
            'gap': '20px', 
            'marginBottom': '35px'
        }, 
        children=[
            # Data Pipeline Upload Terminal
            html.Div(style={'flex': '1', 'backgroundColor': '#1A1A1A', 'padding': '20px', 'borderRadius': '8px', 'border': '1px solid #2D2D2D'}, children=[
                html.Label("1. BEHAVIORAL DATA INTEGRATION", style={'fontSize': '11px', 'fontWeight': 'bold', 'color': '#00ADB5', 'letterSpacing': '1px'}),
                dcc.Upload(
                    id='data-stream-uploader',
                    children=html.Div(['Drag Demographics File Here or ', html.A('Browse File System', style={'color': '#00ADB5', 'fontWeight': '700'})]),
                    style={
                        'width': '100%', 'height': '50px', 'lineHeight': '50px',
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '6px',
                        'textAlign': 'center', 'backgroundColor': '#111111', 'borderColor': '#333333',
                        'marginTop': '10px', 'fontSize': '13px', 'color': '#888888'
                    }
                )
            ]),
            # Active Cohort Slicer Dropdown
            html.Div(style={'flex': '1', 'backgroundColor': '#1A1A1A', 'padding': '20px', 'borderRadius': '8px', 'border': '1px solid #2D2D2D'}, children=[
                html.Label("2. INTERACTIVE COHORT SEGMENT FILTER (SLICER)", style={'fontSize': '11px', 'fontWeight': 'bold', 'color': '#00ADB5', 'letterSpacing': '1px'}),
                html.Div(style={'marginTop': '10px'}, children=[
                    dcc.Dropdown(
                        id='cohort-segment-slicer', 
                        multi=True, 
                        placeholder="Filter specific user cohorts...",
                        style={'backgroundColor': '#FFFFFF', 'color': '#000000', 'fontSize': '13px'}
                    )
                ])
            ])
        ]
    )

def render_visualization_matrix():
    """Constructs multi-dimensional visual spaces for segmentation vectors."""
    return html.Div(
        style={
            'display': 'flex', 
            'gap': '25px', 
            'marginBottom': '35px'
        }, 
        children=[
            html.Div([
                html.Div([dcc.Graph(id='k-means-cluster-scatter')], style={'backgroundColor': '#1A1A1A', 'padding': '15px', 'borderRadius': '8px', 'border': '1px solid #2D2D2D'})
            ], style={'width': '50%'}),
            
            html.Div([
                html.Div([dcc.Graph(id='category-preference-bar')], style={'backgroundColor': '#1A1A1A', 'padding': '15px', 'borderRadius': '8px', 'border': '1px solid #2D2D2D'})
            ], style={'width': '50%'})
        ]
    )

# Master Layout Root Architecture Tree
app.layout = html.Div(
    style={
        'backgroundColor': '#0F0F11', 
        'color': '#EEEEEE', 
        'padding': '40px', 
        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    },
    children=[
        render_control_header(),
        render_pipeline_panel(),
        
        html.H3("Core Segment Operational Aggregations", style={'color': '#EEEEEE', 'fontSize': '15px', 'fontWeight': '600', 'marginBottom': '15px', 'letterSpacing': '0.5px'}),
        html.Div(id='metric-aggregation-row', style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '20px', 'marginBottom': '35px'}),
        
        render_visualization_matrix(),
        
        html.H3("Granular Consumer Behavior Grid Ledger", style={'color': '#EEEEEE', 'fontSize': '15px', 'fontWeight': '600', 'marginBottom': '15px'}),
        html.Div(id='ledger-grid-container')
    ]
)

# ==========================================
# ⚙️ ML PIPELINE ENGINE (BACKEND COMPUTE)
# ==========================================

def parse_incoming_bytestream(contents, filename):
    _, content_string = contents.split(',')
    decoded_binary = base64.b64decode(content_string)
    try:
        if filename.endswith('.csv'):
            return pd.read_csv(io.StringIO(decoded_binary.decode('utf-8')))
        return pd.read_excel(io.BytesIO(decoded_binary))
    except Exception as e:
        print(f"Data Pipeline Failure: {e}")
        return None

def compute_machine_learning_clusters(df):
    """Executes advanced K-Means processing on behavior profiles."""
    try:
        # Locating target statistical variables algorithmically
        inc_col = [c for c in df.columns if 'inc' in c.lower() or 'earn' in c.lower()][0]
        score_col = [c for c in df.columns if 'score' in c.lower() or 'spend' in c.lower()][0]
        age_col = [c for c in df.columns if 'age' in c.lower() or 'year' in c.lower()][0]
        
        features = df[[age_col, inc_col, score_col]].dropna()
        
        # Feature transformation scaling vector
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Executing K-Means Cluster Extraction Engine
        km_model = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['Cluster_ID'] = km_model.fit_predict(scaled_features)
        
        # Enterprise Profile Mapping Strategy
        cluster_map = {
            0: 'VIP High Earners',
            1: 'Conservative Core',
            2: 'Young Active Spenders',
            3: 'Budget Value Seekers'
        }
        df['Segment'] = df['Cluster_ID'].map(cluster_map)
        return df
    except Exception as ex:
        print(f"ML Processing Exception: {ex}")
        df['Segment'] = 'Standard Retail Cohort'
        return df

def generate_demographic_dataset():
    """Generates a premium 1-Year customer behavioral matrix vector in INR."""
    np.random.seed(42)
    sample_size = 250
    
    age_vector = np.random.randint(19, 66, size=sample_size)
    income_base = np.random.randint(35000, 160000, size=sample_size)
    spending_index = np.random.randint(5, 100, size=sample_size)
    pref_pool = np.random.choice(['Electronics', 'Apparel', 'Home Decor', 'Groceries'], size=sample_size, p=[0.3, 0.3, 0.2, 0.2])
    
    df = pd.DataFrame({
        'CustomerID': [f"INR-CUST-{1000+i}" for i in range(sample_size)],
        'Age': age_vector,
        'Annual_Income_INR': income_base * 75,  # Scaling to Indian Rupee Base Profile
        'Spending_Score': spending_index,
        'Preference': pref_pool
    })
    
    return compute_machine_learning_clusters(df)

# --- REACTIVE PIPELINE FLOW CALLBACKS ---

@app.callback(
    [Output('cohort-segment-slicer', 'options'),
     Output('cohort-segment-slicer', 'value')],
    [Input('data-stream-uploader', 'contents')],
    [State('data-stream-uploader', 'filename')]
)
def synchronize_slicer_dimensions(contents, filename):
    if contents is None:
        customer_df = generate_demographic_dataset()
    else:
        raw_df = parse_incoming_bytestream(contents, filename)
        customer_df = compute_machine_learning_clusters(raw_df) if raw_df is not None else generate_demographic_dataset()
        
    unique_segments = customer_df['Segment'].dropna().unique().tolist()
    return [{'label': str(x), 'value': str(x)} for x in unique_segments], unique_segments

@app.callback(
    [Output('metric-aggregation-row', 'children'),
     Output('k-means-cluster-scatter', 'figure'),
     Output('category-preference-bar', 'figure'),
     Output('ledger-grid-container', 'children')],
    [Input('data-stream-uploader', 'contents'),
     Input('cohort-segment-slicer', 'value')],
    [State('data-stream-uploader', 'filename')]
)
def execute_segmentation_pipeline(contents, active_segments, filename):
    if contents is None:
        customer_df = generate_demographic_dataset()
    else:
        raw_df = parse_incoming_bytestream(contents, filename)
        if raw_df is None:
            return html.Div("Data integration pipeline error."), {}, {}, ""
        customer_df = compute_machine_learning_clusters(raw_df)

    # Resolve active feature variables dynamically
    inc_col = [c for c in customer_df.columns if 'inc' in c.lower() or 'earn' in c.lower()][0]
    score_col = [c for c in customer_df.columns if 'score' in c.lower() or 'spend' in c.lower()][0]
    pref_col = [c for c in customer_df.columns if 'pref' in c.lower() or 'type' in c.lower() or 'cat' in c.lower()][0]

    # Filter evaluation via Multi-Slicer state
    if active_segments:
        customer_df = customer_df[customer_df['Segment'].isin(active_segments)]

    # Compute Segment Performance Frameworks
    total_cohort_records = len(customer_df)
    mean_annual_income = customer_df[inc_col].mean() if total_cohort_records > 0 else 0
    mean_spending_index = customer_df[score_col].mean() if total_cohort_records > 0 else 0

    metric_cards = [
        html.Div([html.P("TARGET COHORT CAPTURE", style={'color': '#666666', 'fontSize': '11px', 'fontWeight': '700', 'letterSpacing': '1px', 'margin': '0'}), html.H2(f"{total_cohort_records:,} Consumers", style={'color': '#00ADB5', 'margin': '10px 0 0 0', 'fontWeight': '700'})], style={'padding': '22px', 'backgroundColor': '#1A1A1A', 'borderRadius': '8px', 'width': '32%', 'border': '1px solid #2D2D2D', 'borderLeft': '4px solid #00ADB5'}),
        html.Div([html.P("MEAN ANNUAL VALUATION PROFILE", style={'color': '#666666', 'fontSize': '11px', 'fontWeight': '700', 'letterSpacing': '1px', 'margin': '0'}), html.H2(f"₹ {mean_annual_income:,.2f}", style={'color': '#FFD369', 'margin': '10px 0 0 0', 'fontWeight': '700'})], style={'padding': '22px', 'backgroundColor': '#1A1A1A', 'borderRadius': '8px', 'width': '32%', 'border': '1px solid #2D2D2D', 'borderLeft': '4px solid #FFD369'}),
        html.Div([html.P("AGGREGATE SPENDING INDEX", style={'color': '#666666', 'fontSize': '11px', 'fontWeight': '700', 'letterSpacing': '1px', 'margin': '0'}), html.H2(f"{mean_spending_index:.1f} / 100", style={'color': '#4E9F3D', 'margin': '10px 0 0 0', 'fontWeight': '700'})], style={'padding': '22px', 'backgroundColor': '#1A1A1A', 'borderRadius': '8px', 'width': '32%', 'border': '1px solid #2D2D2D', 'borderLeft': '4px solid #4E9F3D'})
    ]

    # Render Plot 1: K-Means Multi-Dimensional Distribution Matrix Space
    cluster_scatter_fig = px.scatter(
        customer_df, x=inc_col, y=score_col, color='Segment', 
        size='Age', hover_data=['CustomerID'],
        title="K-Means Cluster Dimensional Profile Space",
        template="plotly_dark", color_discrete_sequence=px.colors.qualitative.G10
    )
    cluster_scatter_fig.update_layout(plot_bgcolor='#1A1A1A', paper_bgcolor='#1A1A1A', font=dict(family="sans-serif", size=11), xaxis_tickprefix="₹ ")

    # Render Plot 2: Cross-Sectional Purchase Preference Share
    preference_df = customer_df.groupby(['Segment', pref_col]).size().reset_index(name='Volume')
    category_bar_fig = px.bar(
        preference_df, x=pref_col, y='Volume', color='Segment', barmode='group',
        title="Product Preference Distribution Share Matrix",
        template="plotly_dark", color_discrete_sequence=px.colors.qualitative.G10
    )
    category_bar_fig.update_layout(plot_bgcolor='#1A1A1A', paper_bgcolor='#1A1A1A', font=dict(family="sans-serif", size=11))

    # Compile Enterprise Ledger Matrix View
    grid_ledger_layout = dash_table.DataTable(
        data=customer_df.to_dict('records'),
        columns=[{"name": col.upper().replace('_', ' '), "id": col} for col in customer_df.columns],
        page_size=6,
        style_header={'backgroundColor': '#222226', 'color': '#00ADB5', 'fontWeight': '700', 'border': '1px solid #2D2D2D', 'fontSize': '11px'},
        style_cell={'backgroundColor': '#1A1A1A', 'color': '#DDDDDD', 'textAlign': 'left', 'padding': '12px', 'border': '1px solid #2D2D2D', 'fontSize': '13px'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#161618'}]
    )

    return metric_cards, cluster_scatter_fig, category_bar_fig, grid_ledger_layout

if __name__ == '__main__':
    app.run(debug=True, port=8050)