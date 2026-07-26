import numpy as np
import pytest
import os

testdir = "tests/"


@pytest.mark.training
@pytest.mark.parametrize("file", sorted(os.listdir(testdir + "in/training/")))
def test_training_run(file):
    from climatereconstructionai import train
    train('{}in/training/{}'.format(testdir, file))
    os.rename('{}out/training/ckpt/final.pth'.format(testdir), '{}out/training/ckpt/{}.pth'.format(testdir, file))


@pytest.mark.training
@pytest.mark.parametrize("model", os.listdir(testdir + "ref/training/"))
def test_comp_models(model):
    import torch
    ckpt_dict = torch.load(testdir + "ref/training/" + model)
    for label in ckpt_dict["labels"]:
        model_ref = ckpt_dict[label]["model"]
        model_run = torch.load(testdir + "out/training/ckpt/" + model)[label]["model"]
        for k_ref, k_run in zip(model_ref.keys(), model_run.keys()):
            assert k_ref == k_run
            print("* Checking {}...".format(k_ref))
            # assert model_ref[k_ref].ne(model_run[k_run]).sum().item() == 0
            assert np.allclose(model_ref[k_ref], model_run[k_run], rtol=1e-15, atol=1e-02)
