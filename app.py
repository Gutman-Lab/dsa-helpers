# A test application.
from dash import Dash, html
import dash_bootstrap_components as dbc
from dsa_helpers.dash.header import get_header
import os

os.environ["DSA_API_URL"] = "https://megabrain.neurology.emory.edu/api/v1"

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([get_header("Test")])

if __name__ == "__main__":
    app.run_server(debug=True)
