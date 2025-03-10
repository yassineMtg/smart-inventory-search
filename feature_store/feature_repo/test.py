from feast import FeatureStore

# Initialize the Feature Store
store = FeatureStore(repo_path=".")

# Retrieve features
features = store.get_online_features(
    features=["customer_feature_view:age", "customer_feature_view:income"],
    entity_rows=[{"customer_id": 123}]
).to_dict()

print("Retrieved Features:", features)

