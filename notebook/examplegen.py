import os
import tfx
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.components import CsvExampleGen
from tfx.utils.dsl_utils import external_input

# Initialize TFX pipeline context
context = InteractiveContext()

# ExampleGen: Data ingestion
DATA_ROOT = "path/to/your/dataset"  # Update with actual dataset path
example_gen = CsvExampleGen(input=external_input(DATA_ROOT))
context.run(example_gen)