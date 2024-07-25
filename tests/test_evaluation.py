import pytest
import os

testdir = "tests/"


@pytest.mark.evaluation
@pytest.mark.parametrize("file", sorted(os.listdir(testdir + "in/evaluation/")))
def test_evaluation_run(file):
    from climatereconstructionai import evaluate
    evaluate('{}in/evaluation/{}'.format(testdir, file))
    # evaluate(testdir + "in/evaluation/minimum-1.inp")


@pytest.mark.evaluation
@pytest.mark.parametrize("file", sorted(os.listdir(testdir + "ref/evaluation/")))
def test_comp_netcdf(file):
    import xarray as xr
    ds_ref = xr.open_dataset('{}ref/evaluation/{}'.format(testdir, file))
    ds_run = xr.open_dataset('{}out/evaluation/{}'.format(testdir, file))
    # assert ds_ref.equals(ds_run)
    xr.testing.assert_allclose(ds_ref, ds_run, rtol=1e-15, atol=1e-8)
