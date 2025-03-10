from feast import Entity, FeatureView, Field, ValueType  # ✅ Import ValueType
from feast.types import Int32, Float32  # ✅ Corrected Data Types
from feast.data_format import ParquetFormat
from feast.infra.offline_stores.file_source import FileSource

# ✅ Correct Entity Definition
customer = Entity(
    name="customer_id",
    value_type=ValueType.INT32,  # ✅ Use ValueType.INT32
    description="Customer identifier",
)

# ✅ Correct Batch Source
customer_source = FileSource(
    path="../data/processed_data.parquet",  # ✅ Correct Path
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
)

# ✅ Correct Feature View
customer_feature_view = FeatureView(
    name="customer_feature_view",
    entities=[customer],  # ✅ Correct reference to entity
    schema=[  # ✅ Use `Int32()` and `Float32()`
        Field(name="age", dtype=Int32),   # ✅ Use Int32
        Field(name="income", dtype=Float32),  # ✅ Use Float32
    ],
    online=True,
    source=customer_source,  # ✅ Ensure correct argument
)