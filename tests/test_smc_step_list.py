import pytest
from smcpy.smc.step_list import StepList


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
