# backtest_dashboard.py
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define custom dark theme
app.layout = html.Div(
    style={
        'backgroundColor': 'black',
        'color': 'white',
        'padding': '20px'
    },
    children=[
        # Title and Buy-Hold Section
        html.Div([
            # Title (left side)
            html.Div([
                html.H1(
                    'Strategy Performance Analysis',
                    style={'marginBottom': '10px'}
                )
            ], style={'display': 'inline-block', 'width': '60%'}),

            # Buy-Hold Metrics (right side)
            html.Div([
                html.Div(
                    id='buy-hold-metrics',
                    style={
                        'backgroundColor': '#1a1a1a',
                        'padding': '10px',
                        'borderRadius': '5px',
                        'border': '1px solid #4a4a4a',
                        'textAlign': 'right'
                    }
                )
            ], style={'display': 'inline-block', 'width': '40%', 'verticalAlign': 'top'})
        ], style={'marginBottom': '20px'}),

        # Filters Section
        html.Div([
            html.Div([
                html.Label('Filter by Composition:',
                           style={'marginRight': '10px', 'color': 'cyan'}),
                dcc.Dropdown(
                    id='composition-filter',
                    style={
                        'backgroundColor': '#1a1a1a',
                        'color': 'black',
                        'width': '300px'
                    }
                )
            ], style={'marginBottom': '20px'}),

            html.Div([
                html.Label('Filter by Strategy Type:',
                           style={'marginRight': '10px', 'color': 'cyan'}),
                dcc.Dropdown(
                    id='strategy-filter',
                    style={
                        'backgroundColor': '#1a1a1a',
                        'color': 'black',
                        'width': '300px'
                    }
                )
            ], style={'marginBottom': '20px'})
        ]),

        # Distribution Plot
        dcc.Graph(
            id='sharpe-distribution',
            style={'backgroundColor': 'black', 'marginBottom': '30px'}
        ),

        # Box Plot
        dcc.Graph(
            id='sharpe-boxplot',
            style={'backgroundColor': 'black', 'marginBottom': '30px'}
        ),

        # Top Parameters Section
        html.Div([
            html.H3('Top 3 Parameter Sets by Strategy and Composition',
                    style={'marginBottom': '15px', 'color': 'cyan'}),
            dash_table.DataTable(
                id='top-params-table',
                style_header={
                    'backgroundColor': '#2a2a2a',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'border': '1px solid #4a4a4a'
                },
                style_cell={
                    'backgroundColor': '#1a1a1a',
                    'color': 'white',
                    'textAlign': 'left',
                    'padding': '10px',
                    'border': '1px solid #4a4a4a'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#262626'
                    },
                    {
                        'if': {'filter_query': '{rank} = 1'},
                        'backgroundColor': '#2d4a4a'
                    }
                ],
                sort_action='native'
            )
        ], style={'marginBottom': '30px'}),

        # Overall Statistics Section
        html.Div([
            html.H3('Strategy Statistics by Composition',
                    style={'marginBottom': '15px', 'color': 'cyan'}),
            dash_table.DataTable(
                id='stats-table',
                style_header={
                    'backgroundColor': '#2a2a2a',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'border': '1px solid #4a4a4a'
                },
                style_cell={
                    'backgroundColor': '#1a1a1a',
                    'color': 'white',
                    'textAlign': 'left',
                    'padding': '10px',
                    'border': '1px solid #4a4a4a'
                },
                style_data_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#262626'
                }],
                sort_action='native'
            )
        ]),

        # Stores
        dcc.Store(id='results-store'),
        dcc.Store(id='buy-hold-store')
    ]
)


def format_buy_hold_metrics(metrics: Dict) -> html.Div:
    """Format buy-hold metrics as a styled text display."""
    if not metrics:
        return html.Div("No buy-hold metrics available")

    value_style = {
        'color': 'cyan',
        'fontWeight': 'bold',
        'marginLeft': '8px',
        'fontFamily': 'monospace'
    }

    return html.Div([
        html.Div('Buy-and-Hold Performance',
                 style={'color': 'cyan', 'fontWeight': 'bold', 'marginBottom': '5px'}),
        html.Div([
            html.Span('SR:', style={'fontFamily': 'monospace'}),
            html.Span(f"{metrics['buy_hold_sharpe']:.3f}", style=value_style),
            html.Span(' | MD:', style={'fontFamily': 'monospace', 'marginLeft': '10px'}),
            html.Span(f"{metrics['buy_hold_max_drawdown'] * 100:.2f}%", style=value_style),
            html.Span(' | TR:', style={'fontFamily': 'monospace', 'marginLeft': '10px'}),
            html.Span(f"{metrics['buy_hold_total_return'] * 100:.2f}%", style=value_style)
        ])
    ])

def get_top_parameters(df: pd.DataFrame, composition_filter=None, strategy_filter=None) -> pd.DataFrame:
    """Get top 3 parameter sets for each strategy type and composition."""
    top_params = []

    # Apply filters if provided
    if composition_filter:
        df = df[df['composition'] == composition_filter]
    if strategy_filter:
        df = df[df['strategy_type'] == strategy_filter]

    # Group by both strategy type and composition
    for strategy in sorted(df['strategy_type'].unique()):
        strategy_df = df[df['strategy_type'] == strategy]

        for composition in sorted(strategy_df['composition'].unique()):
            comp_df = strategy_df[strategy_df['composition'] == composition]
            top_3 = comp_df.nlargest(3, 'sharpe_ratio')

            for rank, (_, row) in enumerate(top_3.iterrows(), 1):
                top_params.append({
                    'Strategy': strategy,
                    'Composition': composition,
                    'Rank': rank,
                    'Window': int(row['window']),
                    'Entry Threshold': round(row['entry_threshold'], 3),
                    'Exit Threshold': round(row['exit_threshold'], 3),
                    'Sharpe Ratio': round(row['sharpe_ratio'], 3),
                    'Max Drawdown': f"{round(row['max_drawdown'] * 100, 2)}%",
                    'Total Return': f"{round(row['total_return'] * 100, 2)}%",
                    'Total Trades': int(row.get('total_trade', 0))
                })

    return pd.DataFrame(top_params)


@app.callback(
    [Output('composition-filter', 'options'),
     Output('composition-filter', 'value'),
     Output('strategy-filter', 'options'),
     Output('strategy-filter', 'value')],
    [Input('results-store', 'data')]
)
def update_filters(results_data):
    if not results_data:
        return [], None, [], None

    df = pd.DataFrame(results_data)

    composition_options = [{'label': c, 'value': c} for c in sorted(df['composition'].unique())]
    strategy_options = [{'label': s, 'value': s} for s in sorted(df['strategy_type'].unique())]

    return composition_options, None, strategy_options, None


@app.callback(
[Output('sharpe-distribution', 'figure'),
     Output('sharpe-boxplot', 'figure'),
     Output('stats-table', 'data'),
     Output('stats-table', 'columns'),
     Output('top-params-table', 'data'),
     Output('top-params-table', 'columns')],
    [Input('composition-filter', 'value'),
     Input('strategy-filter', 'value'),
     Input('results-store', 'data')]
)
def update_analysis(composition_filter, strategy_filter, results_data):
    if not results_data:
        return {}, {}, [], [], [], []

    # Convert to DataFrame
    df = pd.DataFrame(results_data)

    # Apply filters for other components
    filtered_df = df.copy()
    if composition_filter:
        filtered_df = filtered_df[filtered_df['composition'] == composition_filter]
    if strategy_filter:
        filtered_df = filtered_df[filtered_df['strategy_type'] == strategy_filter]

    # Create distribution plot
    dist_fig = create_distribution_plot(filtered_df)

    # Create box plot
    box_fig = create_box_plot(filtered_df)

    # Calculate statistics
    stats = calculate_statistics(filtered_df)

    # Get top parameters
    top_params_df = get_top_parameters(filtered_df, composition_filter, strategy_filter)

    # Create columns for tables
    stats_columns = [{'name': col, 'id': col} for col in stats[0].keys()] if stats else []
    params_columns = [{'name': col, 'id': col} for col in top_params_df.columns] if not top_params_df.empty else []

    return (
        dist_fig,
        box_fig,
        stats,
        stats_columns,
        top_params_df.to_dict('records'),
        params_columns
    )

@app.callback(
    Output('buy-hold-metrics', 'children'),
    [Input('buy-hold-store', 'data')]
)

def update_buy_hold_metrics(buy_hold_data):
    """Update buy-hold metrics display."""
    return format_buy_hold_metrics(buy_hold_data)

def create_distribution_plot(df: pd.DataFrame) -> go.Figure:
    """Create distribution plot for Sharpe ratios."""
    fig = go.Figure()
    colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'pink', 'white', 'red']
    color_idx = 0

    for strategy in sorted(df['strategy_type'].unique()):
        strategy_df = df[df['strategy_type'] == strategy]

        for composition in sorted(strategy_df['composition'].unique()):
            data = strategy_df[strategy_df['composition'] == composition]['sharpe_ratio']
            name = f"{strategy} - {composition}"

            fig.add_trace(go.Histogram(
                x=data,
                name=name,
                opacity=0.7,
                nbinsx=30,
                marker_color=colors[color_idx % len(colors)]
            ))
            color_idx += 1

    fig.update_layout(
        template='plotly_dark',
        title='Distribution of Sharpe Ratios by Strategy and Composition',
        xaxis_title='Sharpe Ratio',
        yaxis_title='Count',
        barmode='overlay',
        showlegend=True,
        plot_bgcolor='black',
        paper_bgcolor='black',
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        )
    )

    return fig


def create_box_plot(df: pd.DataFrame) -> go.Figure:
    """Create box plot for Sharpe ratios."""
    fig = go.Figure()
    colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'pink', 'white', 'red']
    color_idx = 0

    for strategy in sorted(df['strategy_type'].unique()):
        strategy_df = df[df['strategy_type'] == strategy]

        for composition in sorted(strategy_df['composition'].unique()):
            data = strategy_df[strategy_df['composition'] == composition]['sharpe_ratio']
            name = f"{strategy} - {composition}"

            fig.add_trace(go.Box(
                y=data,
                name=name,
                marker_color=colors[color_idx % len(colors)]
            ))
            color_idx += 1

    fig.update_layout(
        template='plotly_dark',
        title='Sharpe Ratio Distribution Statistics',
        yaxis_title='Sharpe Ratio',
        showlegend=True,
        plot_bgcolor='black',
        paper_bgcolor='black'
    )

    return fig


def calculate_statistics(df: pd.DataFrame) -> List[Dict]:
    """Calculate statistics for each strategy and composition combination."""
    stats = []
    for strategy in sorted(df['strategy_type'].unique()):
        strategy_df = df[df['strategy_type'] == strategy]

        for composition in sorted(strategy_df['composition'].unique()):
            data = strategy_df[strategy_df['composition'] == composition]
            stats.append({
                'Strategy': strategy,
                'Composition': composition,
                'Mean Sharpe': round(data['sharpe_ratio'].mean(), 3),
                'Median Sharpe': round(data['sharpe_ratio'].median(), 3),
                'Std Dev': round(data['sharpe_ratio'].std(), 3),
                'Max Sharpe': round(data['sharpe_ratio'].max(), 3),
                'Min Sharpe': round(data['sharpe_ratio'].min(), 3),
                'Count': len(data)
            })
    return stats


def run_dashboard(results: List[Dict], buy_hold_metrics: Dict, port: int = 8050):
    logger.info(f"Starting dashboard with {len(results)} results")
    logger.info(f"Buy-hold metrics: {buy_hold_metrics}")

    try:
        # Store both results and buy-hold metrics
        app.layout.children[-2] = dcc.Store(  # Results store
            id='results-store',
            data=results
        )
        app.layout.children[-1] = dcc.Store(  # Buy-hold store
            id='buy-hold-store',
            data=buy_hold_metrics
        )

        # Run the server
        app.run_server(
            debug=False,
            port=port,
            use_reloader=False
        )

    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        raise


if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)