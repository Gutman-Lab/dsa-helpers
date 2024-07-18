# Login / authenticate function for the girder client.
from girder_client import GirderClient


def login(
    api_url: str,
    user: str | None = None,
    password: str | None = None,
    api_key: str | None = None,
) -> GirderClient:
    """Authenticate a girder client with the given credentials or interactively
    if none is given.

    Args:
        api_url (str): The girder API url of the DSA instance.
        user (str | None): The username to authenticate with. Default to None.
        password (str | None): The password to authenticate with. Default to None.
        api_key (str | None): The api key to authenticate with. Default to None.

    Returns:
        girder_client.GirderClient: The authenticated girder client.

    """
    gc = GirderClient(apiUrl=api_url)
    
    if api_key is None:
        if user is not None and password is not None:
            _ = gc.authenticate(username=user, password=password)
        else:
            _ = gc.authenticate(interactive=True, username=user, password=password)
    else:
        _ = gc.authenticate(username=user, password=password, apiKey=api_key)

    return gc