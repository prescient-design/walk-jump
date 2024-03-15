# discrete Walk-Jump Sampling (dWJS)

This is the official open source repository for [discrete Walk-Jump Sampling](https://arxiv.org/abs/2306.12360) developed by [ncfrey](https://github.com/ncfrey), [djberenberg](https://github.com/djberenberg), [kleinhenz](https://github.com/kleinhenz), and [saeedsaremi](https://github.com/saeedsaremi), from [Prescient Design, a Genentech accelerator.](https://gene.com/prescient)

## Setup
Assuming you have [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed, clone the repository, navigate inside, and run:
```bash
./scripts/install.sh
```

## Training
The entrypoint `walkjump_train` is the main driver for training and accepts parameters using Hydra syntax.
The available parameters for configuration can be found by running `train` --help or by looking in the `src/walkjump/hydra_config` directory

## Sampling
The entrypoint `walkjump_sample` is the main driver for training and accepts parameters using Hydra syntax.
The available parameters for configuration can be found by running `sample` --help or by looking in the `src/walkjump/hydra_config` directory

## Evaluation

### Large molecule descriptors
Use the [LargeMoleculeDescriptors](src/walkjump/metrics/_large_molecule_descriptors.py) class to compute descriptors for large molecules (proteins, antibodies, etc.) and see the [code for computing Wasserstein distances between samples and reference distributions](src/walkjump/metrics/_get_batch_descriptors.py) for evaluating sample quality.

### Distributional conformity score (DCS)
See the [DCS code](src/walkjump/conformity/_conformity_score.py) and [DCS README](src/walkjump/conformity/README.md) to evaluate samples.

## Contributing

We welcome contributions. If you would like to submit pull requests, please make sure you base your pull requests off the latest version of the `main` branch.

## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


## Citations
If you use the code and/or model, please cite:
```
@article{frey2023protein,
      title={Protein Discovery with Discrete Walk-Jump Sampling},
      author={Nathan C. Frey and Daniel Berenberg and Karina Zadorozhny and Joseph Kleinhenz and Julien Lafrance-Vanasse and Isidro Hotzel and Yan Wu and Stephen Ra and Richard Bonneau and Kyunghyun Cho and Andreas Loukas and Vladimir Gligorijevic and Saeed Saremi},
      year={2023},
      eprint={2306.12360},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM}
}
```
