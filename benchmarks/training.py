"""Shared Lightning setup and checkpoint handling for benchmark tasks."""

from pathlib import Path


def dataset_config(config, assets, supervised_file=None):
    import chrombert

    return chrombert.get_preset_dataset_config(
        config["dataset_preset"], basedir=str(assets),
        supervised_file=str(supervised_file) if supervised_file is not None else None,
        **config["dataset"],
    )


def lightning_data(dataset, paths, *, include_test=True):
    import chrombert

    params = {
        "train_params": {"supervised_file": str(paths["train"])},
        "val_params": {"supervised_file": str(paths["valid"])},
    }
    if include_test:
        params["test_params"] = {"supervised_file": str(paths["test"])}
    return chrombert.LitChromBERTFTDataModule(config=dataset, **params)


def excluded_track_options(config, dataset, train_path):
    sample = dataset.init_dataset(supervised_file=str(train_path))[0]
    ignore_index = sample["ignore_index"]
    excluded = {int(index) + 1 for index in ignore_index[0]}
    if not excluded or not excluded.isdisjoint(sample["position_ids"].tolist()):
        raise ValueError("Excluded tracks remain in the tutorial input.")
    return dict(config["model"], ignore=True, ignore_index=ignore_index)


def fit(config, assets, data, model_options, directory, profile=None, *, profile_only=False):
    from contextlib import nullcontext

    import chrombert
    import lightning.pytorch as pl

    training = config["training"]
    model = chrombert.get_preset_model_config(
        config["model_preset"], basedir=str(assets), **model_options,
    ).init_model()
    model.freeze_pretrain(trainable=training["trainable_encoder_layers"])
    train_config = chrombert.finetune.TrainConfig(**training["pl_module"])
    module = train_config.init_pl_module(model)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor=f"{train_config.tag}_validation/{train_config.loss}", mode="min")
    callbacks = [pl.callbacks.LearningRateMonitor()]
    trainer_options = dict(training["trainer"])
    if profile_only:
        trainer_options.update(limit_val_batches=0, num_sanity_val_steps=0, enable_checkpointing=False)
    else:
        callbacks.insert(0, checkpoint)
    with profile.training_callback(pl, model, stop_after_capture=profile_only) if profile else nullcontext() as capture:
        if capture is not None:
            callbacks.append(capture)
        trainer = pl.Trainer(
            max_epochs=1 if config.get("smoke") else train_config.max_epochs,
            accelerator="gpu", precision="bf16-mixed",
            log_every_n_steps=1, fast_dev_run=False, callbacks=callbacks,
            logger=pl.loggers.TensorBoardLogger(str(directory / "lightning"), name=config["task"]),
            **trainer_options,
        )
        trainer.fit(module, data)
    if profile_only:
        if trainer.global_step != 1:
            raise RuntimeError("Expected one optimizer update during profiling.")
        return
    if not checkpoint.best_model_path:
        raise RuntimeError("The tutorial validation callback did not produce a checkpoint.")
    selected = {
        "criterion": checkpoint.monitor,
        "mode": checkpoint.mode,
        "best_score": float(checkpoint.best_model_score),
    }
    final = None
    if "checkpoints/tutorial-final.ckpt" in config["output_checkpoints"]:
        final = directory / "tutorial-final.ckpt"
        module.save_ckpt(str(final))
    return Path(checkpoint.best_model_path), final, selected


def load_inference_model(config, assets, model_options, checkpoint):
    import chrombert

    options = {"dropout": 0, **(model_options or config["model"])}
    if checkpoint is not None:
        options["finetune_ckpt"] = str(checkpoint)
    return chrombert.get_preset_model_config(
        config["model_preset"], basedir=str(assets), **options,
    ).init_model().cuda()
