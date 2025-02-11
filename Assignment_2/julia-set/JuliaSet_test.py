import pytest
from JuliaSet import calc_pure_python


@pytest.mark.parametrize("width, iterations, expected_sum", [
    (50, 150, 33219980),
    (200, 600, 33219980),
    (1000, 300, 33219980) #this should pass
])

def test_julia_set(width, iterations, expected_sum):
    """
    Test that the Julia set calculation produces the expected sum
    for a 1000x1000 grid with 300 iterations.
    """
    expected_sum = 33219980
    result = calc_pure_python(desired_width=width, max_iterations=iterations)
    assert sum(result) == expected_sum, f"Expected {expected_sum}, but got {sum(result)}"