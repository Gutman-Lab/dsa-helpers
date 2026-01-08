import yaml, shutil
from pathlib import Path
from time import perf_counter
from ultralytics import YOLO

from ...utils import convert_to_json_serializable


def train_yolo(
    project_dir: str,
    model_name: str,
    yaml_path: str,
    validate_with_train: bool = True,
    validate_with_val: bool = True,
    pretrained_model: str = "yolo11m",
    epochs: int = 100,
    patience: int = 10,
    batch_size: int = 16,
    imgsz: int = 640,
    device: int | str | list | None = None,
    workers: int = 8,
    lr0: float = 0.01,
    train_kwargs: dict = None,
):
    """Function for training a YOLO model with the Ultralytics API.

    Args:
        project_dir (str): Parent directory the model directory will be
            saved in.
        model_name (str): The name of the model to train.
        yaml_path (str): Path do dataset yaml file that specifies images
            that will be used in train and val, and classes.
        validate_with_train (bool, optional): True to validate on the
            train dataset after training. Defaults to True.
        validate_with_val (bool, optional): True to validate on the
            val dataset after training. Defaults to True.
        pretrained_model (str, optional): The pretrained model to use.
            Local or from Ultralytics model hub. Defaults to "yolo11m".
        epochs (int, optional): The number of epochs to train for.
            Defaults to 100.
        patience (int, optional): The number of epochs to wait before
            early stopping. Defaults to 10.
        batch_size (int, optional): The batch size to use for training.
            Defaults to 16.
        imgsz (int, optional): The size of the images to use for
            training. Defaults to 640.
        device (int | str | list | None, optional): The device to use
            for training. Defaults to None.
        workers (int, optional): The number of workers to use for
            training. Defaults to 8.
        lr0 (float, optional): Initial training learning rate. Defaults
            to 0.01.
        augmentation_kwargs (dict, optional):
            flip_lr (float, optional): The probability of flipping the
                image horizontally. Defaults to 0.5.
            flip_ud (float, optional): The probability of flipping the
                image vertically. Defaults to 0.5.
            degrees (float, optional): The number of degrees to rotate
                the image. Defaults to 0.2.
            shear (float, optional): The amount of shear to apply to the
                image. Defaults to 0.2.
            mosaic (float, optional): The probability of applying a
                mosaic augmentation to the image. Defaults to 0.1.
            hsv_s (float, optional): The saturation of the image.
                Defaults to 0.6.
            hsv_v (float, optional): The intensity of the image.
                Defaults to 0.6.
            scale (float, optional): The amount of zoom to apply to the
                image. Defaults to 0.2.

    Returns:
        model: The trained model.
        results: The results of the training.
        validation_results (dict): The results of the validation.
        train_time (float): Time to train model in seconds.

    """
    # Load the model.
    model = YOLO(pretrained_model)

    if train_kwargs is None:
        train_kwargs = {}

    train_augmentation = {
        "fliplr": train_kwargs.get("hflip", 0.5),
        "flipud": train_kwargs.get("vflip", 0.5),
        "degrees": train_kwargs.get("degrees", 0.2),
        "shear": train_kwargs.get("shear", 0.2),
        "mosaic": train_kwargs.get("mosaic", 0.1),
        "hsv_s": train_kwargs.get("saturation", 0.6),
        "hsv_v": train_kwargs.get("intensity", 0.6),
        "scale": max(
            train_kwargs.get("zoom_in", 0.2), train_kwargs.get("zoom_out", 0.2)
        ),
    }

    # Create the project directory.
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    model_dir = project_dir / model_name

    if model_dir.is_dir():
        raise FileExistsError(
            f"Model {model_name} already exists in {project_dir}."
        )

    # Read the YAML file.
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    nc = yaml_data["nc"]
    names = yaml_data["names"]

    # Train the model.
    start_time = perf_counter()
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        project=str(project_dir),
        name=model_name,
        patience=patience,
        imgsz=imgsz,
        device=device,
        workers=workers,
        lr0=lr0,
        **train_augmentation,
    )
    train_time = perf_counter() - start_time

    validation_results = {}
    runs_dir = project_dir / "runs"

    if validate_with_train:
        runs_dir.mkdir(parents=True, exist_ok=True)
        validation_results["train"] = {}

        # Run validation on each label.
        for i in range(nc):
            print(f"Validating on train for class {names[i]}...")
            metrics = model.val(
                split="train",
                classes=[i],
                project=str(runs_dir),
            )

            validation_results["train"][names[i]] = (
                convert_to_json_serializable(metrics.results_dict)
            )

    if validate_with_val:
        runs_dir.mkdir(parents=True, exist_ok=True)
        validation_results["val"] = {}

        # Run validation on each label.
        for i in range(nc):
            print(f"Validating on val for class {names[i]}...")
            metrics = model.val(
                split="val",
                classes=[i],
                project=str(runs_dir),
            )

            validation_results["val"][names[i]] = convert_to_json_serializable(
                metrics.results_dict
            )

    # Remove the runs dir.
    if runs_dir.is_dir():
        shutil.rmtree(runs_dir)

    return model, results, validation_results, train_time
