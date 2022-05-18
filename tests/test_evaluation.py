import pytest

testdir = "tests/"

@pytest.mark.evaluation
def test_evaluation_run():
    from climatereconstructionai import evaluate
    evaluate(testdir+"eval-1.inp")

@pytest.mark.evaluation
def test_comp_netcdf():
    import xarray as xr
    ds_ref = xr.open_dataset(testdir+"ref/test-1_infilled.nc")
    ds_run = xr.open_dataset(testdir+"out/evaluation/test-1_infilled.nc")
    assert ds_ref.equals(ds_run)
