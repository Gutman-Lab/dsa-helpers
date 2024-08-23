# Callbacks for the header component.
from dash import callback, Output, Input, State, no_update
from girder_client import GirderClient
from os import getenv


@callback(Output("login-btn", "children"), Input("user-store", "data"))
def check_user_store(data):
    # Check if the user store has user info or if no one is logged in.
    return data["user"] if len(data) else "Log in"


@callback(
    [
        Output("login-modal", "is_open", allow_duplicate=True),
        Output("logout-modal", "is_open", allow_duplicate=True),
    ],
    [Input("login-btn", "n_clicks"), State("login-btn", "children")],
    prevent_initial_call=True,
)
def open_login_modal(n_clicks, children):
    # Open login / logout modal.
    if n_clicks:
        if children == "Log in":
            return True, False
        else:
            return False, True

    return False, False


@callback(
    [
        Output("user-store", "data"),
        Output("login-failed", "hidden", allow_duplicate=True),
        Output("login-modal", "is_open", allow_duplicate=True),
        Output("login", "value", allow_duplicate=True),
        Output("password", "value", allow_duplicate=True),
    ],
    [
        Input("log-in-btn", "n_clicks"),
        State("login", "value"),
        State("password", "value"),
    ],
    prevent_initial_call=True,
)
def login(n_clicks, login, password):
    # Try to login.
    gc = GirderClient(apiUrl=getenv("DSA_API_URL"))

    try:
        _ = gc.authenticate(username=login, password=password)

        response = gc.get("token/session")

        user = gc.get("user/me")["login"]

        return {"user": user, "token": response["token"]}, True, False, "", ""
    except:
        return (
            {},
            False,
            True,
            no_update,
            no_update,
        )


@callback(
    [
        Output("login-modal", "is_open", allow_duplicate=True),
        Output("login", "value", allow_duplicate=True),
        Output("password", "value", allow_duplicate=True),
        Output("login-failed", "hidden", allow_duplicate=True),
    ],
    Input("close-login-modal", "n_clicks"),
    prevent_initial_call=True,
)
def close_login_modal(n_clicks):
    if n_clicks:
        return False, "", "", True

    return False, "", "", True


@callback(
    [
        Output("user-store", "data", allow_duplicate=True),
        Output("logout-modal", "is_open", allow_duplicate=True),
    ],
    Input("logout-btn", "n_clicks"),
    prevent_initial_call=True,
)
def logout(n_clicks):
    if n_clicks:
        return {}, False

    return no_update, False
