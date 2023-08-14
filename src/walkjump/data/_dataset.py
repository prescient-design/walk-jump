from dataclasses import InitVar, dataclass, field

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from walkjump.constants import ALPHABET_AHO
from walkjump.utils import token_string_to_tensor


@dataclass
class AbDataset(Dataset):
    df: pd.DataFrame
    alphabet_or_token_list: InitVar[LabelEncoder | list[str]] = ALPHABET_AHO
    alphabet: LabelEncoder = field(init=False)

    def __post_init__(self, alphabet_or_token_list: LabelEncoder | list[str]):
        self.alphabet = (
            alphabet_or_token_list
            if isinstance(alphabet_or_token_list, LabelEncoder)
            else LabelEncoder().fit(alphabet_or_token_list)
        )
        self.df.reset_index(drop=True, inplace=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.df.loc[index]
        tensor_h = token_string_to_tensor(row.fv_heavy_aho, self.alphabet)
        tensor_l = token_string_to_tensor(row.fv_light_aho, self.alphabet)
        return torch.cat([tensor_h, tensor_l])
