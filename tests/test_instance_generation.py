from data import dataset_loader


def test_annals_table_1_random_instance_generator_exists():
    generator = getattr(dataset_loader, "generate_random_instance", None)
    assert callable(generator), (
        "dataset_loader has no random instance generator to validate against "
        "Annals Table 1"
    )
