import os
import glob
import torch
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from jiwer import cer

# Model
ENCODER_MODEL_PATH = "/project/lt200324-optmul/pluem/model/trocr-base-handwritten"
DECODER_MODEL_PATH = "/project/lt200324-optmul/pluem/model/huggingface_electra-small-25000-no-grad-small"
TRAINING_DATA_PATH = "/project/lt200324-optmul/pluem/version_3/train_dataset_final.csv"

# Data and Parameter
DATA_NROWS = None
TEST_SIZE = 0.01
BATCH_SIZE = 32
DEVICES = 4  # Number of GPUs per node
NUM_NODES = 8  # Number of nodes
STRATEGY = "ddp_find_unused_parameters_true"
EPOCHS = 10
ACCUMULATE_GRAD = 1
LOG_STEPS = 100

# Checkpoint
LOAD_FROM_CHECKPOINT_PATH = None
CHECKPOINT_PATH = "/project/lt200324-optmul/pluem/version_3/lightning_checkpoints/"
SAVE_MODEL_PATH = "/project/lt200324-optmul/pluem/version_3/saved_models"

class IAMDataset(Dataset):
    """
    Custom Dataset for loading images and target text for TrOCR model.

    Args:
        df (pd.DataFrame): Dataframe containing 'image_path' and 'text' columns.
        processor (TrOCRProcessor): Pre-trained processor for image and text.
        max_target_length (int, optional): Maximum token length for the target text. Defaults to 128.
    """

    def __init__(self, df, processor, max_target_length=128):
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["image_path"][idx]
        text = self.df["text"][idx]

        # Load and process the image
        image = Image.open(file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Tokenize the text
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_target_length
        ).input_ids

        # Replace padding token ids with -100 to ignore in loss calculation
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
        }
        return encoding


def compute_metrics(pred_ids, label_ids, processor):
    """
    Compute the Character Error Rate (CER) between predictions and labels.

    Args:
        pred_ids (torch.Tensor): Predicted token ids.
        label_ids (torch.Tensor): Ground truth token ids.
        processor (TrOCRProcessor): Pre-trained processor for decoding the tokens.

    Returns:
        float: The Character Error Rate (CER) score.
    """
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    return cer(label_str, pred_str)


class TrOCRLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for TrOCR fine-tuning.

    Args:
        model (VisionEncoderDecoderModel): The TrOCR model to be fine-tuned.
        processor (TrOCRProcessor): Processor for image and text handling.
        train_dataset (IAMDataset): Dataset for training.
        eval_dataset (IAMDataset): Dataset for evaluation.
        lr (float, optional): Learning rate for the optimizer. Defaults to 5e-5.
    """

    def __init__(
        self, model=None, processor=None, train_dataset=None, eval_dataset=None, lr=5e-5
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.lr = lr

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        outputs = self.model.generate(pixel_values)
        cer_score = compute_metrics(outputs, labels, self.processor)
        self.log("val_cer", cer_score, prog_bar=True, logger=True)
        return cer_score

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        total_steps = len(self.train_dataloader()) * self.trainer.max_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=BATCH_SIZE, num_workers=4)

    def save_pretrained(self, output_dir):
        """
        Save the trained model and processor to a specified directory.

        Args:
            output_dir (str): Directory path where the model and processor will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)


def find_best_checkpoint(checkpoint_dir):
    """
    Find the best checkpoint by sorting checkpoints based on validation CER.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files.

    Returns:
        str: The path to the best checkpoint, or None if no checkpoint is found.
    """
    checkpoint_files = glob.glob(
        os.path.join(checkpoint_dir, "best-model-epoch=*.ckpt")
    )
    if not checkpoint_files:
        return None

    sorted_checkpoints = sorted(
        checkpoint_files,
        key=lambda x: (
            float(x.split("val_cer=")[1].split(".ckpt")[0]),  # CER score
            -int(
                x.split("epoch=")[1].split("-")[0]
            ),  # Epoch (higher is better if CER is tied)
        ),
    )

    best_checkpoint = sorted_checkpoints[0]
    print(f"Best checkpoint: {best_checkpoint}")
    return best_checkpoint


def main():
    vision_encoder = VisionEncoderDecoderModel.from_pretrained(
        ENCODER_MODEL_PATH
    ).encoder
    decoder_config = AutoConfig.from_pretrained(DECODER_MODEL_PATH)
    decoder_config.add_cross_attention = True
    decoder_config.is_decoder = True
    decoder_model = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL_PATH, config=decoder_config
    )

    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL_PATH)
    processor = TrOCRProcessor.from_pretrained(ENCODER_MODEL_PATH, tokenizer=tokenizer)

    model = VisionEncoderDecoderModel(encoder=vision_encoder, decoder=decoder_model)
    model.config.update(
        {
            "decoder_start_token_id": processor.tokenizer.cls_token_id,
            "pad_token_id": processor.tokenizer.pad_token_id,
            "vocab_size": model.config.decoder.vocab_size,
            "eos_token_id": processor.tokenizer.sep_token_id,
            "max_length": 64,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "length_penalty": 2.0,
            "num_beams": 4,
        }
    )

    # Load and prepare data
    if DATA_NROWS:
        df = pd.read_csv(TRAINING_DATA_PATH, nrows=DATA_NROWS)
    else:
        df = pd.read_csv(TRAINING_DATA_PATH)

    train_df, test_df = train_test_split(df, test_size=TEST_SIZE)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_dataset = IAMDataset(train_df, processor)
    eval_dataset = IAMDataset(test_df, processor)

    # Setup PyTorch Lightning model
    trocr_module = TrOCRLightningModule(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if LOAD_FROM_CHECKPOINT_PATH:
        # Load checkpoint
        checkpoint = torch.load(LOAD_FROM_CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        trocr_module.model = model

    # Checkpoints for saving best models
    checkpoint_callback = ModelCheckpoint(
        monitor="val_cer",
        mode="min",
        save_top_k=3,
        dirpath=CHECKPOINT_PATH,
        filename="best-model-{epoch:02d}-{val_cer:.4f}",
    )

    # WandB logger setup
    wandb_logger = WandbLogger(project="trocr-handwritten", log_model="all")

    # Trainer configuration with multi-GPU and multi-node support
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=DEVICES,
        num_nodes=NUM_NODES,
        strategy=STRATEGY,
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback],
        precision=16,
        accumulate_grad_batches=ACCUMULATE_GRAD,
        log_every_n_steps=LOG_STEPS,
        logger=wandb_logger,
    )

    trainer.fit(
        trocr_module,
        ckpt_path=LOAD_FROM_CHECKPOINT_PATH if LOAD_FROM_CHECKPOINT_PATH else None,
    )

    # Save the best model
    best_checkpoint_path = find_best_checkpoint(CHECKPOINT_PATH)
    if best_checkpoint_path:
        best_model = TrOCRLightningModule.load_from_checkpoint(
            best_checkpoint_path, model=model, processor=processor
        )
        best_model_save_path = os.path.join(
            SAVE_MODEL_PATH, f"best_{os.path.basename(best_checkpoint_path)}"
        )
        best_model.save_pretrained(best_model_save_path)
        print(f"Best model saved to: {best_model_save_path}")
    else:
        print("No best model found to save.")

    # Save the final model
    final_model_save_path = os.path.join(
        SAVE_MODEL_PATH,
        f"final_model-epoch={trainer.current_epoch:02d}-step={trainer.global_step}",
    )
    trocr_module.save_pretrained(final_model_save_path)
    print(f"Final model saved to: {final_model_save_path}")


if __name__ == "__main__":
    main()
