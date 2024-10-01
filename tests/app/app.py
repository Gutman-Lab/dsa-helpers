from dash import html, Dash, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc
from dsa_helpers.dash.header import get_header
from os import getenv
from dash_ag_grid import AgGrid

table = AgGrid(
    id="table",
    columnDefs=[],
    rowData=[],
)

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
    ],
    title="Test Application",
)

app.layout = html.Div(
    [
        get_header(
            getenv("DSA_API_URL"),
            title="Test Application",
            store_id="user-store",
        ),
        dbc.Row(
            [dbc.Col(dbc.Button("Get items", id="get-item-btn"), width="auto")],
            justify="start",
        ),
        table
    ]
)


@callback(
    [
        Output("table", "columnDefs"),
        Output("table", "rowData"),
    ],
    Input("get-item-btn", "n_clicks"),
    prevent_initial_call=True,
)
def get_items(n_clicks):
    """Get the items."""
    
    

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8060,
        debug=True,
    )
