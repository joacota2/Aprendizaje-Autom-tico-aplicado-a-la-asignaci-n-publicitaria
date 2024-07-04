from zenml.pipelines import pipeline
from zenml.steps import step
import pandas as pd
from MarketingAttrLib.modeling import AttributionModel
from MarketingAttrLib.data_generator import MamDatasets
import joblib
import os

# Step 1: Data Generation
@step
def generate_data() -> pd.DataFrame:
    dataset = MamDatasets().generate_data()
    return dataset

# Step 2: Model Initialization
@step
def initialize_model(dataset: pd.DataFrame) -> AttributionModel:
    attr_modeling = AttributionModel(dataset)
    return attr_modeling

# Step 3: Calculate Shapley Values
@step
def calculate_shapley(attr_modeling: AttributionModel) -> AttributionModel:
    attr_modeling.calculate_shapley()
    return attr_modeling

# Step 4: Apply Time Decay
@step
def apply_time_decay(attr_modeling: AttributionModel, decay_factor: float = 0.3) -> AttributionModel:
    attr_modeling.time_decay(decay_factor=decay_factor)
    return attr_modeling

# Step 5: Get Results
@step
def get_results(attr_modeling: AttributionModel) -> pd.DataFrame:
    results = attr_modeling.get_results()
    return results

# Step 6: Run All Models
@step
def run_all_models(attr_modeling: AttributionModel) -> pd.DataFrame:
    all_models = attr_modeling.run_all()
    return all_models

# Step 7: Save Model
@step
def save_model(attr_modeling: AttributionModel, model_path: str = "model_mta.pkl") -> str:
    joblib.dump(attr_modeling, model_path)
    return model_path

# Pipeline Definition
@pipeline
def attribution_pipeline(
    generate_data_step,
    initialize_model_step,
    calculate_shapley_step,
    apply_time_decay_step,
    get_results_step,
    run_all_models_step,
    save_model_step
):
    dataset = generate_data_step()
    attr_modeling = initialize_model_step(dataset)
    attr_modeling = calculate_shapley_step(attr_modeling)
    attr_modeling = apply_time_decay_step(attr_modeling)
    results = get_results_step(attr_modeling)
    all_models = run_all_models_step(attr_modeling)
    model_path = save_model_step(attr_modeling)

# Pipeline Execution
def run():
    attribution_p = attribution_pipeline(
        generate_data_step=generate_data(),
        initialize_model_step=initialize_model(),
        calculate_shapley_step=calculate_shapley(),
        apply_time_decay_step=apply_time_decay(),
        get_results_step=get_results(),
        run_all_models_step=run_all_models(),
        save_model_step=save_model()
    )
    attribution_p.run()

if __name__ == "__main__":
    run()
