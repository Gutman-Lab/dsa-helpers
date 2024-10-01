# Callbacks for the header component.
from dash import callback, Output, Input, State, no_update
from girder_client import GirderClient


def get_callbacks(dsa_api_url: str, store_id: str):
    """Setup the callbacks to the DSA API url of interest and the proper store id."""

    @callback(Output(f"{store_id}=login-btn", "children"), Input(store_id, "data"))
    def check_user_store(data):
        # Check if the user store has user info or if no one is logged in.
        return data["user"] if len(data) else "Log in"

    @callback(
        [
            Output(f"{store_id}=login-modal", "is_open", allow_duplicate=True),
            Output(f"{store_id}=logout-modal", "is_open", allow_duplicate=True),
        ],
        [
            Input(f"{store_id}=login-btn", "n_clicks"),
            State(f"{store_id}=login-btn", "children"),
        ],
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
            Output(store_id, "data"),
            Output(f"{store_id}=login-failed", "hidden", allow_duplicate=True),
            Output(f"{store_id}=login-modal", "is_open", allow_duplicate=True),
            Output(f"{store_id}=login", "value", allow_duplicate=True),
            Output(f"{store_id}=password", "value", allow_duplicate=True),
        ],
        [
            Input(f"{store_id}=log-in-btn", "n_clicks"),
            State(f"{store_id}=login", "value"),
            State(f"{store_id}=password", "value"),
        ],
        prevent_initial_call=True,
    )
    def login(n_clicks, login, password):
        # Try to login.
        gc = GirderClient(apiUrl=dsa_api_url)

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
            Output(f"{store_id}=login-modal", "is_open", allow_duplicate=True),
            Output(f"{store_id}=login", "value", allow_duplicate=True),
            Output(f"{store_id}=password", "value", allow_duplicate=True),
            Output(f"{store_id}=login-failed", "hidden", allow_duplicate=True),
        ],
        Input(f"{store_id}=close-login-modal", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_login_modal(n_clicks):
        if n_clicks:
            return False, "", "", True

        return False, "", "", True

    @callback(
        [
            Output(store_id, "data", allow_duplicate=True),
            Output(f"{store_id}=logout-modal", "is_open", allow_duplicate=True),
        ],
        Input(f"{store_id}=logout-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def logout(n_clicks):
        if n_clicks:
            return {}, False

        return no_update, False
