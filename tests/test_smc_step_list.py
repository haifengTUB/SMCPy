import pytest
from smcpy.smc.step_list import StepList
from smcpy.smc.smc_step import SMCStep


@pytest.fixture
def step_list():
    return StepList()


def test_step_list_stores_steps(step_list):
    test_obj_1 = object()
    test_obj_2 = object()
    step_list.add_step(test_obj_1)
    step_list.add_step(test_obj_2)
    assert step_list[0] is test_obj_1
    assert step_list[1] is test_obj_2


def test_step_list_pops_step(step_list):
    test_obj_1 = object()
    test_obj_2 = object()
    step_list.add_step(test_obj_1)
    step_list.add_step(test_obj_2)
    popped_step = step_list.pop_step(0)
    assert popped_step == test_obj_1
    assert step_list[0] is test_obj_2


def test_step_list_trims_self(step_list):
    test_obj_1 = object()
    test_obj_2 = object()
    test_obj_3 = object()
    step_list.add_step(test_obj_1)
    step_list.add_step(test_obj_2)
    step_list.add_step(test_obj_3)
    step_list.trim(1)
    assert len(step_list[:]) == 1
    assert step_list[0] == test_obj_1


def test_compute_bayes_evidence(step_list, mocker):
    mock_step = SMCStep()
    dummy_weights = [1.1, 0.8, 0.1]
    mocker.patch.object(mock_step, 'get_weights', return_value=dummy_weights)

    step_list.add_step(mock_step)
    step_list.add_step(mock_step)

    assert step_list.compute_bayes_evidence() == sum(dummy_weights) ** 2
