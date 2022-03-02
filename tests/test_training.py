import pytest
import shutil

testdir = "tests/"

@pytest.mark.training
@pytest.mark.parametrize("file", ["train-1.inp","train-2.inp"])
def test_training_run(file):
    from climatereconstructionai import train
    train(testdir+file)

@pytest.mark.training
@pytest.mark.parametrize("model", ["10.pth","15.pth"])
def test_comp_models(model):
    print()
    import torch
    model_ref = torch.load(testdir+"ref/"+model)["model"]
    model_run = torch.load(testdir+"out/training/ckpt/"+model)["model"]
    for k_ref, k_run in zip(model_ref.keys(), model_run.keys()):
        assert k_ref == k_run
        print("* Checking {}...".format(k_ref))
        assert model_ref[k_ref].ne(model_run[k_run]).sum().item() == 0
