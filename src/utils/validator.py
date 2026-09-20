import pandas as pd


def validate_predictions(
    df: pd.DataFrame,
    expected_declarations_count: int,
) -> None:
    """Validate the structure and contents of prediction results."""
    required_columns = {"declaration_id", "rank", "regulation_id", "score"}
    missing_columns = required_columns - set(df.columns)

    assert not missing_columns, (
        f"Не хватает колонок: {missing_columns}"
    )

    actual_declarations_count = df["declaration_id"].nunique()
    assert actual_declarations_count == expected_declarations_count, (
        f"Ожидалось {expected_declarations_count} деклараций, "
        f"получено {actual_declarations_count}"
    )

    expected_ranks = list(range(1, 11))

    for declaration_id, group in df.groupby("declaration_id"):
        assert len(group) == 10, (
            f"Декларация {declaration_id} содержит "
            f"{len(group)} строк вместо 10"
        )

        ranks = sorted(group["rank"].tolist())
        assert ranks == expected_ranks, (
            f"Декларация {declaration_id} нарушает ранги 1..10: {ranks}"
        )

        assert group["regulation_id"].nunique() == 10, (
            f"В декларации {declaration_id} есть дубликаты regulation_id"
        )

        assert not group["score"].isna().any(), (
            f"В {declaration_id} найдены NaN score"
        )