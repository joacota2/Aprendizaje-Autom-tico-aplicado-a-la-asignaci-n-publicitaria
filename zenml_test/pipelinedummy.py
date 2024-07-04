from zenml.pipelines import pipeline
from zenml.steps import step
import numpy as np

# Step 1: Data Generation
@step
def generate_data() -> np.ndarray:
    # Generate some dummy data
    return np.random.rand(10, 2)  # 10 rows of 2 columns

# Step 2: Data Processing
@step
def process_data(data: np.ndarray) -> np.ndarray:
    # Process the data by normalizing it
    return data / np.max(data)

# Step 3: Data Output
@step
def output_data(data: np.ndarray):
    # Output the data
    print("Processed Data:", data)

# Define the pipeline
@pipeline
def dummy_pipeline(
    data_gen_step,
    data_process_step,
    data_output_step
):
    data = data_gen_step()
    processed_data = data_process_step(data)
    data_output_step(processed_data)

# Main function to run the pipeline
def run():
    pipeline_instance = dummy_pipeline(
        data_gen_step=generate_data(),
        data_process_step=process_data(),
        data_output_step=output_data()
    )
    pipeline_instance.run()

if __name__ == "__main__":
    run()
