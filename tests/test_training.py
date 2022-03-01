import pytest

testdir = "tests/"

@pytest.mark.training
def test_training_run():
    from climatereconstructionai import train
    train(testdir+"train-1.inp")

@pytest.mark.training
def test_comp_models():
    print()
    import torch
    model_ref = torch.load(testdir+"ref/test-1_100.pth")["model"]
    model_run = torch.load(testdir+"out/training/ckpt/100.pth")["model"]
    for k_ref, k_run in zip(model_ref.keys(), model_run.keys()):
        assert k_ref == k_run
        print("* Checking {}...".format(k_ref))
        assert model_ref[k_ref].ne(model_run[k_run]).sum().item() == 0
