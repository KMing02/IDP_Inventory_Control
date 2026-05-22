import pytest

from inventory_control.paper_configs import validate_paper_lost_sales_config


def test_paper_config_validation_requires_fields() -> None:
    with pytest.raises(ValueError, match="missing required"):
        validate_paper_lost_sales_config({"history_length_H": 32})
