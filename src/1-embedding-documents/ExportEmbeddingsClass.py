import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# IMPORTS ######################################################################
from datasets import Dataset
from gc import collect as gc_collect
from transformers import AutoModelForSequenceClassification
from torch import Tensor, no_grad
from torch.cuda import is_available as cuda_available
from torch.cuda import synchronize, ipc_collect, empty_cache
from tqdm import tqdm


# SCRIPTS ######################################################################
def clean():
    """ """
    empty_cache()
    if cuda_available():
        synchronize()
        ipc_collect()
    gc_collect()


class ExportEmbeddings:
    """ """

    def __init__(self, ds: Dataset, model_name: str, device: str | None = None) -> None:
        """ """
        self.__ds: Dataset = ds

        if device is None:
            self.device = "cuda" if cuda_available() else "cpu"
        else:
            self.device = device
        self.__model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_name, trust_remote_code = True)
            .to(device=self.device)
        )

    def __get_embeddings(self, batch_size: int, additional_tags: list[str]) -> Tensor:
        with no_grad():
            ds_embeddings = Dataset.from_dict(
                {"embedding": [], **{tag: [] for tag in additional_tags}}
            )

            batch_parameters = {"drop_last_batch": True}

            for batch in tqdm(self.__ds.batch(batch_size, **batch_parameters)):
                model_input = {
                    key: Tensor(batch[key]).int().to(device=self.device)
                    for key in ["input_ids", "attention_mask"]
                }
                batch_embeddings = (
                    self.__model.base_model(**model_input)
                    .last_hidden_state[:, 0, :]
                    .squeeze()
                )

                for i in range(batch_size):
                    ds_embeddings = ds_embeddings.add_item(
                        {
                            "embedding": batch_embeddings[i, :].reshape((-1,)).tolist(),
                            **{tag: batch[tag][i] for tag in additional_tags},
                        }
                    )
        return ds_embeddings

    def routine(
        self, batch_size: int = 4, additional_tags: list[str] = [], folder: str = None
    ) -> None:
        try:
            ds_embeddings = self.__get_embeddings(batch_size, additional_tags)
            if folder is None:
                ds_embeddings.save_to_disk("embeddings")
            else:
                ds_embeddings.save_to_disk(folder)
        except Exception as e:
            print(e)
        finally:
            del self.__ds, self.__model
            clean()
