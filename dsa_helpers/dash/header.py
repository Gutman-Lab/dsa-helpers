from dash import html, dcc
import dash_bootstrap_components as dbc
from .header_callbacks import get_callbacks


def get_header(
    dsa_api_url: str,
    title: str = "App",
    store_id: str = "user-store",
    config: dict | None = None,
) -> html.Div:
    """Get the Dash html component that contains the header of the application
    with login capabilities.

    Args:
        dsa_api_url (str): The URL of the DSA API.
        title (str, optional): The title of the application. Defaults to "App".
        config (dict, optional): The configuration of the component.
            Defaults to None which uses default settings.

    Returns:
        html.Div: The header component.

    """
    if config is None:
        config = {}

    get_callbacks(dsa_api_url, store_id)

    login_modal = dbc.Modal(
        [
            dbc.ModalHeader("Log in"),
            dbc.ModalBody(
                [
                    html.Div(
                        "Login or email",
                        style={"margin": 5, "fontWeight": "bold"},
                    ),
                    dbc.Input(
                        id=f"{store_id}=login",
                        type="text",
                        placeholder="Enter login",
                        style={"margin": 5},
                    ),
                    html.Div(
                        "Password",
                        style={
                            "margin": 5,
                            "marginTop": 15,
                            "fontWeight": "bold",
                        },
                    ),
                    dbc.Input(
                        id=f"{store_id}=password",
                        type="password",
                        placeholder="Enter password",
                        style={"margin": 5},
                    ),
                    html.Div(
                        "Login failed.",
                        hidden=True,
                        id=f"{store_id}=login-failed",
                        style={
                            "color": "red",
                            "fontWeight": "bold",
                            "margin": 10,
                        },
                    ),
                ],
            ),
            dbc.ModalFooter(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Close",
                                id=f"{store_id}=close-login-modal",
                                className="me-1",
                                color="light",
                            )
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Login",
                                id=f"{store_id}=log-in-btn",
                                className="me-1",
                                color="primary",
                            )
                        ),
                    ],
                )
            ),
        ],
        is_open=False,
        id=f"{store_id}=login-modal",
    )

    logout_modal = dbc.Modal(
        [
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Log out",
                        id=f"{store_id}=logout-btn",
                        color="danger",
                        className="me-1",
                    ),
                ]
            )
        ],
        is_open=False,
        id=f"{store_id}=logout-modal",
    )

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.H1(
                            title,
                            style={
                                "fontWeight": "bold",
                                "color": config.get("titleColor", "#d9d9d6"),
                                "marginLeft": 5,
                            },
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Log in",
                            id=f"{store_id}=login-btn",
                            color=config.get("backgroundColor", "#012169"),
                            style={
                                "color": config.get("fontColor", "#f2a900"),
                                "fontSize": "1.2rem",
                            },
                            className="me-1",
                        ),
                        width="auto",
                    ),
                ],
                justify="between",
                align="center",
            ),
            login_modal,
            logout_modal,
            dcc.Store(id=store_id, storage_type="local", data={}),
        ],
        style={
            "backgroundColor": config.get("backgroundColor", "#012169"),
            "padding": 5,
        },
    )
