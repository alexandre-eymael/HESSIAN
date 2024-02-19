import wandb

class WandbLogger:
    def __init__(self, config, mode="online"):
        wandb.init(entity = "lba_mlops", project="mlops", config=config, mode=mode)

    def log(self, metrics):
        wandb.log(metrics)

    def make_image(self, image, caption):
        return wandb.Image(image, caption=caption)

    def create_table(self, columns):
        return wandb.Table(columns = columns)
    
    def log_roc_curve(self, gts, preds):
        wandb.log({"roc": wandb.plot.roc_curve(y_true=gts, y_probas=preds, labels=["healthy"], classes_to_plot=[0])})

    def add_html_to_table(self, table, file_path):
        table.add_data(wandb.Html(file_path))

    def summary(self, key, value):
        wandb.run.summary[key] = value

    def get_summary(self, key):
        return wandb.run.summary[key]

    def finish(self):
        wandb.finish()

    def get_name(self):
        return wandb.run.name
    
    def start_watch(self, model, log_freq=5):
        wandb.watch(model, log_freq=log_freq)
    
    def get_config(self):
        return wandb.config