import wandb

run = wandb.init(
    project="xray-insight",
    name="setup-test"
)

wandb.log({
    "test_metric": 1
})

run.finish()