import numpy as np
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    data_path: str = r"data/Data_average_new.csv"
    dataset_types: list[str]
    input_columns: list[str]
    target_column: str


FORMULATION_COLUMNS = [
    'Methocel A4C', 'Methocel A4M', 'Ac-Di-Sol', 'Emulsion'
]
PRINTING_CONDITION_COLUMNS = [
    'Pressure', 'Speed'
]

INPUT_COLUMNS = [
    FORMULATION_COLUMNS,
    FORMULATION_COLUMNS + PRINTING_CONDITION_COLUMNS
]

REGRESSION_TARGET_COLUMNS = [
    'Uniformity', 'Disintegration', 'Tanδ', 'Recovery'
]

CLASSIFICATION_TARGET_COLUMNS = [
    "Leakage",
]

DATASET_TYPES = [
    ['Optimization'],
    ['Optimization', 'Mahsa'],
    ['Optimization', 'Robustness'],
    ['Optimization', 'Robustness', 'Mahsa']
]

REGRESSION_DATASETS = [
    DatasetConfig(
        dataset_types=dataset_types,
        input_columns=input_columns,
        target_column=target_column
    )
    for dataset_types in DATASET_TYPES
    for input_columns in INPUT_COLUMNS
    for target_column in REGRESSION_TARGET_COLUMNS
]

CLASSIFICATION_DATASETS = [
    DatasetConfig(
        dataset_types=dataset_types,
        input_columns=input_columns,
        target_column=target_column
    )
    for dataset_types in DATASET_TYPES
    for input_columns in INPUT_COLUMNS
    for target_column in CLASSIFICATION_TARGET_COLUMNS
]


class Target(BaseModel):
    ideal_max: float
    ideal_min: float
    unacceptable_max: float
    unacceptable_min: float
    importance: float = 1.0

    def scaled_target(self, scaler) -> 'Target':
        """Scale the target using the provided scaler."""
        return Target(
            ideal_max=scaler.transform([[self.ideal_max]])[0][0],
            ideal_min=scaler.transform([[self.ideal_min]])[0][0],
            unacceptable_max=scaler.transform([[self.unacceptable_max]])[0][0],
            unacceptable_min=scaler.transform([[self.unacceptable_min]])[0][0],
            importance=self.importance
        )


IDEAL_TARGETS: dict[str, Target] = {
    'Disintegration': Target(
        ideal_max=180,
        ideal_min=0,
        unacceptable_max=900,
        unacceptable_min=0,
        importance=1),
    'Uniformity': Target(
        ideal_max=0,
        ideal_min=0,
        unacceptable_max=7.5,
        unacceptable_min=0,
        importance=1),
    'Tanδ': Target(
        ideal_max=0.1,
        ideal_min=0,
        unacceptable_max=1,
        unacceptable_min=0,
        importance=1),
    'Recovery': Target(
        ideal_max=50,
        ideal_min=30,
        unacceptable_max=100,
        unacceptable_min=0,
        importance=1),
    'Leakage': Target(
        ideal_max=.3,
        ideal_min=0,
        unacceptable_max=1,
        unacceptable_min=0,
        importance=1),

}


GEL_BOUNDS = {
    'Methocel A4C': (120, 360),
    'Methocel A4M': (240, 600),
    'Ac-Di-Sol': (180, 360),
    'Pressure': (25, 50),
    'Speed': (4, 7)
}
