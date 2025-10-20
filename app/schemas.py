from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Literal
import pandas as pd


class CropRequest(BaseModel):
    """
    Crop yield request with quantile-based bin computation using pandas qcut inside validators.
    """

    # --- Original Input Fields ---
    Region: Annotated[str, Field(..., description="Region of cultivation")]
    Soil_Type: Annotated[
        Literal["Loam", "Sandy", "Clay", "Silt", "Peaty", "Chalky"],
        Field(..., description="Dominant soil type"),
    ]
    Crop: Annotated[Literal["Maize", "Rice", "Barley", "Wheat", "Cotton", "Soybean"], Field(..., description="Crop")]
    Rainfall_mm: Annotated[float, Field(..., ge=0, description="Rainfall in mm")]
    Temperature_Celsius: Annotated[float, Field(..., description="Temperature in Celsius")]
    Fertilizer_Used: Annotated[Literal["Yes", "No"], Field(..., description="Fertilizer applied")]
    Irrigation_Used: Annotated[Literal["Yes", "No"], Field(..., description="Irrigation applied")]
    Weather_Condition: Annotated[
        Literal["Sunny", "Cloudy", "Rainy", "Stormy", "Humid", "Dry"],
        Field(..., description="Weather condition"),
    ]
    Days_to_Harvest: Annotated[int, Field(..., ge=1, description="Number of days to harvest")]