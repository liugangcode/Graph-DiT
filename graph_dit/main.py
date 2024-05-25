# These imports are tricky because they use c++, do not move them
import os, shutil
import warnings

import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

import utils
from datasets import dataset
from diffusion_model import Graph_DiT
from metrics.molecular_metrics_train import TrainMolecularMetricsDiscrete
from metrics.molecular_metrics_sampling import SamplingMolecularMetrics

from analysis.visualization import MolecularVisualization

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision("medium")

def remove_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def get_resume(cfg, model_kwargs):
    """Resumes a run. It loads previous config without allowing to update keys (used for testing)."""
    saved_cfg = cfg.copy()
    name = cfg.general.name + "_resume"
    resume = cfg.general.test_only
    batch_size = cfg.train.batch_size
    model = Graph_DiT.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg.train.batch_size = batch_size
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model

def get_resume_adaptive(cfg, model_kwargs):
    """Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split("outputs")[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == "discrete":
        model = Graph_DiT.load_from_checkpoint(
            resume_path, **model_kwargs
        )
    else:
        raise NotImplementedError("Unknown model")

    new_cfg = model.cfg
    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + "_resume"

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


@hydra.main(
    version_base="1.1", config_path="../configs", config_name="config"
)
def main(cfg: DictConfig):

    datamodule = dataset.DataModule(cfg)
    datamodule.prepare_data()
    dataset_infos = dataset.DataInfos(datamodule=datamodule, cfg=cfg)
    train_smiles, reference_smiles = datamodule.get_train_smiles()

    dataset_infos.compute_input_output_dims(datamodule=datamodule)
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

    sampling_metrics = SamplingMolecularMetrics(
        dataset_infos, train_smiles, reference_smiles
    )
    visualization_tools = MolecularVisualization(dataset_infos)

    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
    }

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split("checkpoints")[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split("checkpoints")[0])

    model = Graph_DiT(cfg=cfg, **model_kwargs)
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        accelerator="gpu"
        if torch.cuda.is_available() and cfg.general.gpus > 0
        else "cpu",
        devices=cfg.general.gpus
        if torch.cuda.is_available() and cfg.general.gpus > 0
        else None,
        max_epochs=cfg.train.n_epochs,
        enable_checkpointing=False,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        val_check_interval=cfg.train.val_check_interval,
        strategy="ddp" if cfg.general.gpus > 1 else "auto",
        enable_progress_bar=cfg.general.enable_progress_bar,
        callbacks=[],
        reload_dataloaders_every_n_epochs=0,
        logger=[],
    )

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.save_model:
            trainer.save_checkpoint(f"checkpoints/{cfg.general.name}/last.ckpt")
        trainer.test(model, datamodule=datamodule)
    else:
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)

if __name__ == "__main__":
    main()
