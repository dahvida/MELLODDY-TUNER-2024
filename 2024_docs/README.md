# How to use the extended functionalities of MELLODDY-TUNER-2024

To run the pipeline with different descriptors, simply run `prepare_4_training` as usual, modifying the `fingerprint` dictionary in the JSON as shown in [example_parameters.json](./example_parameters.json). The following options are available for calculation:
- `rdkit_custom`
- `rdkit_all`
- `mordred_custom`
- `mordred_all`

To postprocess the computed descriptors, you need to run `normalize_descriptors.py`, using `normalize_descs.json` as an argument.

Once you have run `normalize_descriptors.py`, you can proceed with training as usual.