import pytest

from inventory_control.paper_configs import validate_paper_lost_sales_config
from inventory_control.paper_evaluation import require_final_test_confirmation


def test_paper_config_validation_requires_fields() -> None:
    with pytest.raises(ValueError, match="missing required"):
        validate_paper_lost_sales_config({"history_length_H": 32})


def test_test_split_requires_final_test_confirmation() -> None:
    require_final_test_confirmation("validation", final_test=False)
    require_final_test_confirmation("test", final_test=True)

    with pytest.raises(ValueError, match="Refusing test-set evaluation"):
        require_final_test_confirmation("test", final_test=False)
