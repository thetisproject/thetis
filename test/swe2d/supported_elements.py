import pytest

cases = [('rt-dg', 0), ('rt-dg', 1), ('dg-dg', 1), ('dg-cg', 1), ('bdm-dg', 1)]
case_ids = [f'{fam}{deg}' for fam, deg in cases]


@pytest.fixture(params=cases, ids=case_ids)
def element_family_and_degree(request):
    return request.param
